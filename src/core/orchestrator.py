import asyncio
import logging

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
        self._state = AssistantState.WAITING
        self._running = False

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
        async for frame in self._audio_capture.stream_frames():
            if not self._running:
                return
            if await self._wake_word.detect(frame):
                self._transition_to(AssistantState.LISTENING)
                return

    async def _handle_listening(self) -> None:
        """Capture audio until silence or timeout."""
        logger.info("Listening for speech...")
        frames: list[np.ndarray] = []
        silence_frames = 0
        max_frames = int(10 * 1000 / self._config.audio.frame_duration_ms)  # 10s max
        silence_threshold = int(1.5 * 1000 / self._config.audio.frame_duration_ms)  # 1.5s silence
        energy_threshold = 500  # RMS threshold for speech detection
        speech_detected = False

        async for frame in self._audio_capture.stream_frames():
            if not self._running:
                return

            frames.append(frame)
            rms = np.sqrt(np.mean(frame.astype(np.float32) ** 2))

            if rms > energy_threshold:
                speech_detected = True
                silence_frames = 0
            else:
                silence_frames += 1

            # End on silence after speech
            if speech_detected and silence_frames >= silence_threshold:
                break

            # Timeout with no speech
            if len(frames) >= max_frames:
                break

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
        if not self._session.is_active:
            self._session.start(self._config.agent.system_prompt)

        response_text = ""
        async for chunk in self._agent.run(self._pending_text, self._session):
            response_text += chunk
            # TODO: Make agent speak as it streams instead of after it finishes

        self._pending_text = ""

        if not response_text.strip():
            logger.warning("Agent returned empty response")
            self._transition_to(AssistantState.WAITING)
            return

        logger.info(f"Agent response: {response_text!r}")
        self._pending_response = response_text
        self._transition_to(AssistantState.SPEAKING)

    async def _handle_speaking(self) -> None:
        """Synthesize and play the response."""
        audio_bytes = await self._tts.synthesize(self._pending_response)
        self._pending_response = ""

        if audio_bytes:
            await self._playback.play(audio_bytes)

        self._session.touch()

        if self._session.is_active:
            self._transition_to(AssistantState.LISTENING)
        else:
            self._transition_to(AssistantState.WAITING)
