import asyncio
import io
import logging
import shutil
import subprocess
import wave

from src.config.schema import TTSConfig

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

            loop = asyncio.get_event_loop()
            self._voice = await loop.run_in_executor(
                None,
                lambda: PiperVoice.load(self._config.model_path),
            )
            logger.info(f"Piper voice loaded (Python): {self._config.model_path}")
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
        loop = asyncio.get_event_loop()
        try:
            def _run() -> bytes:
                buf = io.BytesIO()
                with wave.open(buf, "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(22050)
                    self._voice.synthesize(  # type: ignore[union-attr]
                        text,
                        wav,
                        speaker_id=self._config.speaker_id,
                        length_scale=self._config.length_scale,
                        noise_scale=self._config.noise_scale,
                        noise_w=self._config.noise_w,
                    )
                # Return raw PCM (skip WAV header)
                buf.seek(44)
                return buf.read()

            return await loop.run_in_executor(None, _run)
        except Exception as e:
            raise SynthesisError(f"Piper synthesis failed: {e}") from e

    async def _synthesize_cli(self, text: str) -> bytes:
        try:
            proc = await asyncio.create_subprocess_exec(
                "piper",
                "--model", self._config.model_path,
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
