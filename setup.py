import platform

from src.config.loader import load_config
from src.config.schema import AppConfig, WakeWordConfig
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


def main() -> None:
    config: AppConfig = load_config()
    download_wake_word_model(config.wake_word)


if __name__ == '__main__':
    main()