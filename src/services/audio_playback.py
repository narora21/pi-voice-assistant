import asyncio
import logging
from typing import Protocol

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
        logger.info(f"MockPlayback: would play {duration:.2f}s of audio")
        await asyncio.sleep(duration)
