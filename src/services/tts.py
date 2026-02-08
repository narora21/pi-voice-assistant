import asyncio
import io
import logging
import shutil
import subprocess
import wave

from src.config.schema import TTSConfig
from src.config.utils import get_tts_model_path

logger = logging.getLogger(__name__)


class SynthesisError(Exception):
    pass


class TextToSpeechService:
    """Text-to-speech using piper-tts (Python package or CLI fallback)."""

    def __init__(self, config: TTSConfig) -> None:
        self._config = config
        self._voice: object | None = None
        self._use_cli: bool = False

    async def start(self) -> None:
        try:
            from piper.voice import PiperVoice

            loop = asyncio.get_running_loop()

            self._voice = await loop.run_in_executor(
                None,
                lambda: PiperVoice.load(get_tts_model_path(self._config)),
            )
            logger.info(f"Piper voice loaded (Python): {get_tts_model_path(self._config)}")
        except (ImportError, Exception) as e:
            logger.warning(f"piper-tts Python package failed: {e}. Trying CLI fallback.")
            if shutil.which("piper"):
                self._use_cli = True
                logger.info("Using piper CLI for TTS")
            else:
                logger.warning("Piper not available. TTS will return empty audio.")

    async def stop(self) -> None:
        self._voice = None

    async def synthesize(self, text: str) -> bytes:
        """Synthesize text to raw int16 PCM bytes."""
        if not text.strip():
            return b""

        if self._voice is not None:
            return await self._synthesize_python(text)
        elif self._use_cli:
            return await self._synthesize_cli(text)
        else:
            logger.warning(f"TTS unavailable, skipping: {text!r}")
            return b""

    async def _synthesize_python(self, text: str) -> bytes:
        loop = asyncio.get_running_loop()

        try:
            def _run() -> bytes:
                # Lazy import to avoid hard dependency at module import time
                from piper.config import SynthesisConfig

                if self._voice is None:
                    raise SynthesisError("Piper voice not initialized")

                syn_config = SynthesisConfig(
                    speaker_id=self._config.speaker_id,
                    length_scale=self._config.length_scale,
                    noise_scale=self._config.noise_scale,
                    noise_w_scale=self._config.noise_w,
                )

                # synthesize() returns Iterable[AudioChunk]
                chunks = self._voice.synthesize(text, syn_config)

                # Concatenate raw int16 PCM bytes
                audio_bytes = b"".join(
                    chunk.audio_int16_bytes for chunk in chunks
                )

                return audio_bytes

            return await loop.run_in_executor(None, _run)

        except Exception as e:
            raise SynthesisError(f"Piper synthesis failed: {e}") from e


    async def _synthesize_cli(self, text: str) -> bytes:
        try:
            proc = await asyncio.create_subprocess_exec(
                "piper",
                "--model", get_tts_model_path(self._config),
                "--output-raw",
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate(input=text.encode())
            if proc.returncode != 0:
                raise SynthesisError(f"Piper CLI error: {stderr.decode()}")
            return stdout
        except FileNotFoundError:
            raise SynthesisError("Piper CLI not found")
