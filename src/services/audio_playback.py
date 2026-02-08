import asyncio
import logging
from typing import Protocol

import numpy as np

from src.config.schema import AudioConfig

logger = logging.getLogger(__name__)


class AudioPlaybackService(Protocol):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def play(self, audio_data: bytes) -> None: ...


class MockAudioPlaybackService:
    """Mock playback that logs duration instead of playing audio."""

    def __init__(self, sample_rate: int = 22050) -> None:
        self._sample_rate = sample_rate

    async def start(self) -> None:
        logger.info("MockAudioPlayback started")

    async def stop(self) -> None:
        logger.info("MockAudioPlayback stopped")

    async def play(self, audio_data: bytes) -> None:
        num_samples = len(audio_data) // 2  # int16 = 2 bytes per sample
        duration = num_samples / self._sample_rate
        logger.info("MockPlayback: would play %.2fs of audio", duration)
        await asyncio.sleep(duration)


class LiveAudioPlaybackService:
    """
    Real audio playback using sounddevice (PortAudio).

    Designed for TTS output (Piper) at 22050 Hz, int16 PCM mono.
    """

    def __init__(self, config: AudioConfig) -> None:
        self._config = config
        self._loop: asyncio.AbstractEventLoop | None = None
        self._device: int | None = None
        self._playback_sample_rate: int = 22050  # Piper fixed output rate
        self._volume = config.playback_volume

    async def start(self) -> None:
        import sounddevice as sd

        self._loop = asyncio.get_running_loop()
        self._device = self._resolve_device(sd)

        logger.info(
            "LiveAudioPlayback started (device=%s, rate=%d)",
            self._device,
            self._playback_sample_rate,
        )

    async def stop(self) -> None:
        # No persistent stream to close (using sd.play)
        logger.info("LiveAudioPlayback stopped")

    async def play(self, audio_data: bytes) -> None:
        """
        Play raw PCM int16 mono audio at 22050 Hz.
        volume: 0.0 (mute) → 1.0 (normal) → >1.0 (boost, may clip)
        """

        if not audio_data:
            return

        if self._loop is None:
            raise RuntimeError("Playback service must be started before calling play()")

        import sounddevice as sd

        # Convert bytes → int16 numpy array
        audio = np.frombuffer(audio_data, dtype=np.int16)

        # Convert to float32 in range [-1.0, 1.0]
        audio_float = audio.astype(np.float32) / 32768.0

        # Apply volume scaling
        audio_float *= self._volume

        # Clip to prevent overflow distortion
        np.clip(audio_float, -1.0, 1.0, out=audio_float)

        # Convert back to int16
        audio_scaled = (audio_float * 32768.0).astype(np.int16)

        def _play_blocking():
            sd.play(
                audio_scaled,
                samplerate=self._playback_sample_rate,
                device=self._device,
            )
            sd.wait()

        await self._loop.run_in_executor(None, _play_blocking)

    
    async def play_file(self, path: str, volume: float = 1.0) -> None:
        """Play a WAV file at its native sample rate."""
        import wave
        import numpy as np
        import sounddevice as sd

        with wave.open(path, "rb") as wav:
            if wav.getsampwidth() != 2:
                raise ValueError("Only 16-bit WAV files supported")
            
            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
        
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        
        # Convert stereo to mono if needed
        if channels == 2:
            audio = audio.reshape(-1, 2).mean(axis=1)
        
        # Apply volume scaling
        volume *= self._volume
        volume = max(0.0, min(1.0, volume))
        audio = audio * volume
        
        audio = np.clip(audio, -32768, 32767).astype(np.int16)
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: sd.play(audio, samplerate=sample_rate, blocking=True)
        )


    def _resolve_device(self, sd: object) -> int | None:
        """
        Resolve config device string to a sounddevice output device index.
        Matches by substring, same logic as capture service.
        """
        device_str = self._config.playback_device
        if device_str is None:
            return None

        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            # Ensure device supports output
            if dev["max_output_channels"] > 0 and device_str.lower() in dev["name"].lower():
                logger.info("Matched output audio device: [%d] %s", i, dev["name"])
                return i

        available = [
            d["name"]
            for d in devices
            if d["max_output_channels"] > 0
        ]

        raise RuntimeError(
            f"Audio output device not found: {device_str!r}. "
            f"Available output devices: {available}"
        )
