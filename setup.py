import platform

from src.config.loader import load_config
from src.config.schema import AppConfig, AudioConfig, WakeWordConfig
from src.config.utils import get_wake_word_model_dir


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
    verify_audio_device(config.audio)


if __name__ == '__main__':
    main()