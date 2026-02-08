# src/config/loader.py
import os
from pathlib import Path
from typing import Any

import yaml

from src.config.schema import (
    AgentConfig,
    AppConfig,
    AudioConfig,
    LoggingConfig,
    SessionConfig,
    STTConfig,
    TTSConfig,
    WakeWordConfig,
)
from src.config.secrets import Secrets, load_secrets

_SECTION_CLASSES = {
    "wake_word": WakeWordConfig,
    "audio": AudioConfig,
    "stt": STTConfig,
    "agent": AgentConfig,
    "tts": TTSConfig,
    "session": SessionConfig,
    "logging": LoggingConfig,
}

_ENV_OVERRIDES: dict[str, tuple[str, str]] = {
    "AGENT_SYSTEM_PROMPT": ("agent", "system_prompt"),
}


def _apply_env_overrides(data: dict[str, Any]) -> None:
    """Override YAML values with environment variables where mapped."""
    for env_var, (section, key) in _ENV_OVERRIDES.items():
        value = os.environ.get(env_var)
        if value is not None:
            data.setdefault(section, {})[key] = value


def load_config(
    config_path: Path = Path("config.yaml"),
    env_path: Path = Path(".env"),
) -> tuple[AppConfig, Secrets]:
    """Load YAML config and secrets, return both."""
    secrets = load_secrets(env_path)

    data: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

    _apply_env_overrides(data)

    sections: dict[str, Any] = {}
    for name, cls in _SECTION_CLASSES.items():
        section_data = data.get(name, {})
        if section_data:
            sections[name] = cls(**section_data)
        else:
            sections[name] = cls()

    return AppConfig(**sections), secrets