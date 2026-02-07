import asyncio
import logging
import os
import platform
from typing import Protocol

import numpy as np

from src.config.schema import WakeWordConfig

logger = logging.getLogger(__name__)


class WakeWordDetector(Protocol):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def detect(self, audio_frame: np.ndarray) -> bool: ...


class OpenWakeWordService:
    """Wake word detection using openwakeword."""

    def __init__(self, config: WakeWordConfig) -> None:
        self._config = config
        self._model: object | None = None
        self._disabled = False
        if platform.system() != "Linux":
            logger.info("Wake word disabled (not running on Linux).")
            self._disabled = True


    async def start(self) -> None:
        if self._disabled:
            return 
        
        try:
            from openwakeword.model import Model
        except ImportError as e:
            logger.error(f"Unable to import open wake word model on platform {platform.system()}")
            raise e

        model_path = os.path.join(self._config.models_dir, self._config.model_name)
        model_args = dict()
        if self._config.vad_threshold:
            model_args = { "vad_threshold": self._config.vad_threshold }
        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(
            None,
            lambda: Model(
                wakeword_models=[model_path],
                **model_args
            ),
        )
        logger.info(f"Wake word model loaded: {self._config.model_name}")

    async def stop(self) -> None:
        self._model = None
        logger.info("Wake word model unloaded")

    async def detect(self, audio_frame: np.ndarray) -> bool:
        if self._model is None or self._disabled:
            return False

        loop = asyncio.get_event_loop()
        prediction = await loop.run_in_executor(
            None, self._model.predict, audio_frame  # type: ignore[union-attr]
        )

        for name, score in prediction.items():
            if score >= self._config.threshold:
                logger.info(f"Wake word detected: {name} (score={score:.3f})")
                return True
        return False
