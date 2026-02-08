import platform
from pathlib import Path

from src.config.loader import load_config
from src.config.schema import AppConfig, AudioConfig, TTSConfig, WakeWordConfig
from src.config.utils import get_wake_word_model_dir

# Piper model URLs from https://github.com/rhasspy/piper/blob/master/VOICES.md
PIPER_MODELS = {
    "en_US-lessac-medium": {
        "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "json": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
    },
    "en_US-amy-medium": {
        "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx",
        "json": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx.json",
    },
}


def download_tts_model(config: TTSConfig):
    """Download Piper TTS model if not present."""
    import urllib.request

    model_path = Path(config.model_path)
    json_path = Path(f"{config.model_path}.json")

    # Extract model name from path (e.g., "models/tts/en_US-lessac-medium.onnx" -> "en_US-lessac-medium")
    model_name = model_path.stem  # removes .onnx

    if model_name not in PIPER_MODELS:
        print(f"Unknown Piper model: {model_name}. Skipping download.")
        print(f"Available models: {list(PIPER_MODELS.keys())}")
        return

    urls = PIPER_MODELS[model_name]

    # Create directory if needed
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Download ONNX model
    if not model_path.exists():
        print(f"Downloading Piper model: {model_name}...")
        urllib.request.urlretrieve(urls["onnx"], model_path)
        print(f"  Saved to {model_path}")
    else:
        print(f"Piper model already exists: {model_path}")

    # Download JSON config
    if not json_path.exists():
        print(f"Downloading Piper model config...")
        urllib.request.urlretrieve(urls["json"], json_path)
        print(f"  Saved to {json_path}")
    else:
        print(f"Piper config already exists: {json_path}")


def download_wake_word_model(config: WakeWordConfig):
    try:
        import openwakeword
    except ImportError as e:
        print(f"Unable to import open wake word model on platform {platform.system()}. Skipping...")
        return

    if config.models_path:
        model_dir = get_wake_word_model_dir(config)
        print(f"Downloading model: {config.model_name} to {model_dir}")
        openwakeword.utils.download_models([config.model_name], model_dir)
    else:
        print(f"Downloading model: {config.model_name}")
        openwakeword.utils.download_models([config.model_name])


def verify_audio_device(config: AudioConfig):
    try:
        import sounddevice as sd
    except ImportError:
        print(f"sounddevice not available on {platform.system()}. Skipping audio device check...")
        return

    if config.device is None:
        default = sd.query_devices(kind="input")
        print(f"No audio device configured. Default input: {default['name']}")
        return

    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if config.device.lower() in dev["name"].lower() and dev["max_input_channels"] > 0:
            print(f"Audio device found: [{i}] {dev['name']}")
            return

    available = [d["name"] for d in devices if d["max_input_channels"] > 0]
    print(f"WARNING: Audio device '{config.device}' not found. Available: {available}")


def main() -> None:
    config: AppConfig = load_config()
    download_wake_word_model(config.wake_word)
    download_tts_model(config.tts)
    verify_audio_device(config.audio)


if __name__ == '__main__':
    main()