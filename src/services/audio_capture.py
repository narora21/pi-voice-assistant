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
        self._queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue(maxsize=128)
        self._loop: asyncio.AbstractEventLoop | None = None

    async def start(self) -> None:
        import sounddevice as sd

        self._loop = asyncio.get_event_loop()
        device = self._resolve_device(sd)

        def _audio_callback(
            indata: bytes, frames: int, time_info: object, status: object
        ) -> None:
            if status:
                # Schedule warning on the event loop instead of logging directly
                self._loop.call_soon_threadsafe(
                    lambda: logger.warning("Audio capture status: %s", status)
                )
            frame = np.frombuffer(indata, dtype=np.int16).copy()
            try:
                self._queue.put_nowait(frame)
            except asyncio.QueueFull:
                pass # Drop the frame

        try:
            self._stream = sd.RawInputStream(
                samplerate=self._config.sample_rate,
                blocksize=self._frame_size,
                device=device,
                channels=self._config.channels,
                dtype="int16",
                callback=_audio_callback,
            )
            logger.info(
                "LiveAudioCapture started (device=%s, rate=%d, frame_size=%d)",
                device,
                self._config.sample_rate,
                self._frame_size,
            )
        except Exception:
            logger.error(
                "An error occurred while starting audio capture. "
                "Please check the device connection and make sure all channels are detectable"
            )
            raise

    async def stop(self) -> None:
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        # Signal consumer to exit
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            pass
        logger.info("LiveAudioCapture stopped")

    async def stream_frames(self) -> AsyncIterator[np.ndarray]:
        if not self._running:
            logger.warning("Cannot stream audio capture frames, stream must be started first")
            return
        while self._running:
            try:
                # Use wait_for to not block on audio callback hogging async loop
                frame = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                if frame is None:
                    return
                yield frame
            except asyncio.TimeoutError:
                # Just loop back and check _running again
                continue

    def start_capture(self) -> None:
        if self._stream is None:
            logger.warning("Audio input stream is None")
            return
        # Empty the queue
        self._drain_queue()
        self._stream.start()
        self._running = True
        logger.info("Audio capture started")

    def stop_capture(self) -> None:
        if self._stream is None:
            logger.warning("Audio input stream is None")
            return
        self._running = False
        self._stream.stop()
        self._drain_queue()
        logger.info("Audio capture stopped")

    def _drain_queue(self) -> None:
        """Empty the queue without blocking."""
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

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

    def is_silent_frame(self, frame: np.ndarray) -> bool:
        energy_threshold = self._config.energy_threshold
        rms = np.sqrt(np.mean(frame.astype(np.float32) ** 2))
        return rms <= energy_threshold