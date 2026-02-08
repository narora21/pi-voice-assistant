import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.config.loader import load_config
from src.config.schema import AppConfig


def test_load_default_config():
    """Loading with no files produces valid defaults."""
    config = load_config(
        config_path=Path("/nonexistent/config.yaml"),
        env_path=Path("/nonexistent/.env"),
    )
    assert isinstance(config, AppConfig)
    assert config.agent.model == "qwen2.5:1.5b"
    assert config.stt.model_size == "base.en"
    assert config.session.idle_timeout_seconds == 30.0


def test_load_yaml_config():
    """YAML values override defaults."""
    data = {
        "agent": {"model": "llama3.2:3b", "temperature": 0.5},
        "stt": {"model_size": "tiny.en"},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        tmp_path = Path(f.name)

    try:
        config = load_config(config_path=tmp_path, env_path=Path("/nonexistent/.env"))
        assert config.agent.model == "llama3.2:3b"
        assert config.agent.temperature == 0.5
        assert config.stt.model_size == "tiny.en"
        # Non-overridden values keep defaults
        assert config.wake_word.model_name == "hey_jarvis_v0.1.tflite"
    finally:
        tmp_path.unlink()


def test_env_override(monkeypatch):
    """Environment variables override YAML values."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://custom:1234")
    config = load_config(
        config_path=Path("/nonexistent/config.yaml"),
        env_path=Path("/nonexistent/.env"),
    )
    assert config.agent.base_url == "http://custom:1234"


def test_load_env_file():
    """Values from .env file are loaded."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
        f.write('OLLAMA_BASE_URL=http://from-env:5678\n')
        env_path = Path(f.name)

    # Clear any existing env var so .env file takes effect
    old_val = os.environ.pop("OLLAMA_BASE_URL", None)
    try:
        config = load_config(
            config_path=Path("/nonexistent/config.yaml"),
            env_path=env_path,
        )
        assert config.agent.base_url == "http://from-env:5678"
    finally:
        env_path.unlink()
        if old_val is not None:
            os.environ["OLLAMA_BASE_URL"] = old_val
        else:
            os.environ.pop("OLLAMA_BASE_URL", None)


def test_frozen_config():
    """Config dataclasses are immutable."""
    config = load_config(
        config_path=Path("/nonexistent/config.yaml"),
        env_path=Path("/nonexistent/.env"),
    )
    with pytest.raises(AttributeError):
        config.agent = None  # type: ignore[misc]
