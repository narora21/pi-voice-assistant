import asyncio
import logging
import random

import numpy as np

from src.config.schema import AppConfig
from src.core.session import Session
from src.core.state import AssistantState, validate_transition
from src.services.agent import AgentService
from src.services.audio_capture import AudioCaptureService
from src.services.audio_playback import AudioPlaybackService
from src.services.stt import SpeechToTextService
from src.services.tts import TextToSpeechService
from src.services.wake_word import WakeWordDetector
from src.core.signal_bus import SignalBus
from src.util.chunk_batcher import ChunkBatcher
from src.util.prompt_loader import PromptLoader

logger = logging.getLogger(__name__)


class Orchestrator:
    """State machine orchestrating the voice assistant loop."""

    def __init__(
        self,
        args: dict,
        config: AppConfig,
        wake_word: WakeWordDetector,
        audio_capture: AudioCaptureService,
        stt: SpeechToTextService,
        agent: AgentService,
        tts: TextToSpeechService,
        audio_playback: AudioPlaybackService,
        session: Session,
        signal_bus: SignalBus,
    ) -> None:
        self._args = args
        self._config = config
        self._wake_word = wake_word
        self._audio_capture = audio_capture
        self._stt = stt
        self._agent = agent
        self._tts = tts
        self._playback = audio_playback
        self._session = session
        self._signal_bus = signal_bus
        self._state = AssistantState.WAITING
        self._running = False
        self._skip_greeting = False
        self._played_init = False
        self._pending_chunks: asyncio.Queue[str | None] = asyncio.Queue()

        # Temporary data passed between states
        self._audio_buffer: np.ndarray | None = None
        self._pending_text: str = ""
        self._pending_response: str = ""

    async def start(self) -> None:
        """Start all services and begin the main loop."""
        await self._wake_word.start()
        await self._audio_capture.start()
        await self._stt.start()
        await self._agent.start()
        await self._tts.start()
        await self._playback.start()

        # Init sound bytes
        self.init_byte = await self._tts.synthesize(self._config.sound_bytes.init_byte)
        self.greeting_bytes = [await self._tts.synthesize(byte) for byte in self._config.sound_bytes.greeting_bytes]
        self.thinking_bytes = [await self._tts.synthesize(byte) for byte in self._config.sound_bytes.thinking_bytes]

        self._running = True
        logger.info("Orchestrator started")
        await self._run_loop()


    async def stop(self) -> None:
        """Gracefully stop all services."""
        self._running = False
        await self._playback.stop()
        await self._tts.stop()
        await self._agent.stop()
        await self._stt.stop()
        await self._audio_capture.stop()
        await self._wake_word.stop()
        try:
            import sounddevice as sd
            sd._terminate()
            sd._initialize()
        except:
            logger.warning("Unable to force PortAudio to release device")
        logger.info("Orchestrator stopped")

    def _transition_to(self, target: AssistantState) -> None:
        validate_transition(self._state, target)
        logger.info(f"State: {self._state.name} -> {target.name}")
        self._state = target

    async def _run_loop(self) -> None:
        while self._running:
            try:
                match self._state:
                    case AssistantState.WAITING:
                        await self._handle_waiting()
                    case AssistantState.LISTENING:
                        await self._handle_listening()
                    case AssistantState.TRANSCRIBING:
                        await self._handle_transcribing()
                    case AssistantState.THINKING:
                        await self._handle_thinking()
                    case AssistantState.SPEAKING:
                        await self._handle_speaking()
            except Exception:
                logger.exception(f"Error in state {self._state.name}, resetting to WAITING")
                self._state = AssistantState.WAITING

    async def _handle_waiting(self) -> None:
        """Listen for wake word on audio frames."""
        if self._args.no_wake_wait:
            logger.info("Skipping wake word wait...")
            self._transition_to(AssistantState.LISTENING)
            return
        logger.info("Waiting for wake word...")
        self._audio_capture.start_capture()
        if not self._played_init:
            await self._playback.play(self.init_byte) # Signal ready for voice
            self._played_init = True
        async for frame in self._audio_capture.stream_frames():
            if not self._running:
                break
            if await self._wake_word.detect(frame):
                self._transition_to(AssistantState.LISTENING)
                break
        self._audio_capture.stop_capture()

    async def _handle_listening(self) -> None:
        """Capture audio until silence or timeout."""
        logger.info("Listening for speech...")
        frames: list[np.ndarray] = []
        silence_frames = 0
        max_frames = int(10 * 1000 / self._config.audio.frame_duration_ms)  # 10s max
        silence_threshold = int(1.5 * 1000 / self._config.audio.frame_duration_ms)  # 1.5s silence
        speech_detected = False

        if not self._skip_greeting:
            await asyncio.sleep(0.3)
            await self._playback.play(random.choice(self.greeting_bytes))
        self._skip_greeting = False

        # Reset wake word detector to clear any stale audio state
        await self._wake_word.reset()
        self._audio_capture.start_capture()
        async for frame in self._audio_capture.stream_frames():
            if not self._running:
                return

            frames.append(frame)

            # Detect silence
            if self._audio_capture.is_silent_frame(frame):
                silence_frames += 1
            else:
                speech_detected = True
                silence_frames = 0

            # End on silence after speech
            if speech_detected and silence_frames >= silence_threshold:
                break

            # Timeout with no speech
            if len(frames) >= max_frames:
                break
        self._audio_capture.stop_capture()

        if not speech_detected:
            logger.info("No speech detected, returning to WAITING")
            self._transition_to(AssistantState.WAITING)
            return

        self._audio_buffer = np.concatenate(frames)
        self._transition_to(AssistantState.TRANSCRIBING)

    async def _handle_transcribing(self) -> None:
        """Transcribe captured audio to text."""
        if self._audio_buffer is None:
            self._transition_to(AssistantState.WAITING)
            return
        
        logger.info("Transcribing speech input...")
        await self._playback.play(random.choice(self.thinking_bytes))

        text = await self._stt.transcribe(self._audio_buffer)
        self._audio_buffer = None

        if not text or len(text) < 2:
            logger.info("Empty transcription, returning to WAITING")
            self._transition_to(AssistantState.WAITING)
            return

        logger.info(f"User said: {text!r}")
        self._pending_text = text
        self._transition_to(AssistantState.THINKING)

    async def _handle_thinking(self) -> None:
        """Run the agent loop and accumulate response."""
        tool_registry = self._agent._tools
        system_prompt = PromptLoader.load_system_prompt(self._config.agent, tool_registry)
        if not self._session.is_active:
            self._session.start(system_prompt)

        response_text = ""
        
        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        playback_task = asyncio.create_task(self._playback_worker(audio_queue))
        synthesis_task = asyncio.create_task(self._synthesis_worker(audio_queue))
        
        batcher = ChunkBatcher(
            min_chars=self._config.tts.batch_min_chars,
            max_chars=self._config.tts.batch_max_chars,
        )
        
        async for chunk in self._agent.run(self._pending_text, self._session):
            response_text += chunk
            
            for batch in batcher.add(chunk):
                await self._pending_chunks.put(batch)
        
        # Flush remaining text
        remaining = batcher.flush()
        if remaining:
            await self._pending_chunks.put(remaining)
        
        # Signal end
        await self._pending_chunks.put(None)
        
        await synthesis_task
        await playback_task

        self._pending_text = ""

        if not response_text.strip():
            logger.warning("Agent returned empty response")
            self._transition_to(AssistantState.WAITING)
            return

        logger.info(f"Agent response: {response_text!r}")
        self._session.touch()

        self._transition_to(AssistantState.WAITING)


    async def _synthesis_worker(self, audio_queue: asyncio.Queue[bytes | None]) -> None:
        """Synthesize text chunks and queue audio for playback."""
        while True:
            text_chunk = await self._pending_chunks.get()
            if text_chunk is None:
                # Signal playback worker to stop
                await audio_queue.put(None)
                return
            
            audio_bytes = await self._tts.synthesize(text_chunk)
            if audio_bytes:
                await audio_queue.put(audio_bytes)


    async def _playback_worker(self, audio_queue: asyncio.Queue[bytes | None]) -> None:
        """Play audio chunks as they become available."""
        while True:
            audio_bytes = await audio_queue.get()
            if audio_bytes is None:
                return
            
            await self._playback.play(audio_bytes)
