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


class LiveAudioCaptureService:
    """Real audio capture using sounddevice (PortAudio)."""

    def __init__(self, config: AudioConfig) -> None:
        self._config = config
        self._running = False
        self._frame_size = int(config.sample_rate * config.frame_duration_ms / 1000)
        self._stream: object | None = None
        self._queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue(maxsize=64)
        self._loop: asyncio.AbstractEventLoop | None = None

    async def start(self) -> None:
        import sounddevice as sd

        self._loop = asyncio.get_event_loop()
        device = self._resolve_device(sd)

        def _audio_callback(
            indata: bytes, frames: int, time_info: object, status: object
        ) -> None:
            if status:
                self._loop.call_soon_threadsafe(
                    logger.warning, "Audio capture status: %s", status
                )
            frame = np.frombuffer(indata, dtype=np.int16).copy()
            try:
                self._queue.put_nowait(frame)
            except asyncio.QueueFull:
                pass  # Drop frame rather than block audio thread

        try:
            self._stream = sd.RawInputStream(
                samplerate=self._config.sample_rate,
                blocksize=self._frame_size,
                device=device,
                channels=self._config.channels,
                dtype="int16",
                callback=_audio_callback,
            )
            self._stream.start()
            self._running = True
            logger.info(
                "LiveAudioCapture started (device=%s, rate=%d, frame_size=%d)",
                device,
                self._config.sample_rate,
                self._frame_size,
            )
        except sounddevice.PortAudioError:
            logging.error("An error ocurred while starting audio capture. Please check the device connection and make sure all channels are detectable")
            raise

    async def stop(self) -> None:
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        await self._queue.put(None)
        logger.info("LiveAudioCapture stopped")

    async def stream_frames(self) -> AsyncIterator[np.ndarray]:
        while self._running:
            frame = await self._queue.get()
            if frame is None:
                return
            yield frame

    def _resolve_device(self, sd: object) -> int | None:
        """Resolve config device string to a sounddevice device index."""
        device_str = self._config.device
        if device_str is None:
            return None

        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if device_str.lower() in dev["name"].lower():
                logger.info("Matched audio device: [%d] %s", i, dev["name"])
                return i

        available = [d["name"] for d in devices]
        raise RuntimeError(
            f"Audio device not found: {device_str!r}. Available devices: {available}"
        )
