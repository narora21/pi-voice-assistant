import asyncio
import logging

import numpy as np

from src.config.schema import STTConfig

logger = logging.getLogger(__name__)


class TranscriptionError(Exception):
    pass


class SpeechToTextService:
    """Speech-to-text using faster-whisper."""

    def __init__(self, config: STTConfig) -> None:
        self._config = config
        self._model: object | None = None

    async def start(self) -> None:
        from faster_whisper import WhisperModel

        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(
            None,
            lambda: WhisperModel(
                self._config.model_size,
                device=self._config.device,
                compute_type=self._config.compute_type,
            ),
        )
        logger.info(f"STT model loaded: {self._config.model_size}")

    async def stop(self) -> None:
        self._model = None
        logger.info("STT model unloaded")

    async def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe int16 PCM audio to text."""
        if self._model is None:
            raise TranscriptionError("STT model not loaded")

        # faster-whisper expects float32 normalized to [-1.0, 1.0]
        audio_float = audio.astype(np.float32) / 32768.0

        loop = asyncio.get_event_loop()
        try:
            segments, _info = await loop.run_in_executor(
                None,
                lambda: self._model.transcribe(  # type: ignore[union-attr]
                    audio_float,
                    beam_size=self._config.beam_size,
                    language=self._config.language,
                    vad_filter=self._config.vad_filter,
                ),
            )
            # Segments is a generator - consume it in executor too
            text = await loop.run_in_executor(
                None,
                lambda: " ".join(seg.text.strip() for seg in segments),
            )
        except Exception as e:
            raise TranscriptionError(f"Transcription failed: {e}") from e

        logger.info(f"Transcribed: {text!r}")
        return text.strip()
