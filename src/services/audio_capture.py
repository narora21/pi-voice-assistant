import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Protocol

import numpy as np

from src.config.schema import AudioConfig

logger = logging.getLogger(__name__)


class AudioCaptureService(Protocol):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    def stream_frames(self) -> AsyncIterator[np.ndarray]: ...


class MockAudioCaptureService:
    """Mock audio capture that yields silent frames at real-time rate."""

    def __init__(self, config: AudioConfig) -> None:
        self._config = config
        self._running = False
        self._frame_size = int(config.sample_rate * config.frame_duration_ms / 1000)

    async def start(self) -> None:
        logger.info("MockAudioCapture started (silent frames)")
        self._running = True

    async def stop(self) -> None:
        self._running = False
        logger.info("MockAudioCapture stopped")

    async def stream_frames(self) -> AsyncIterator[np.ndarray]:
        while self._running:
            yield np.zeros(self._frame_size, dtype=np.int16)
            await asyncio.sleep(self._config.frame_duration_ms / 1000.0)
