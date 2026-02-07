import platform

from src.config.loader import load_config
from src.config.schema import AppConfig, WakeWordConfig


def download_wake_word_model(config: WakeWordConfig):
    try:
        import openwakeword
    except ImportError as e:
        print(f"Unable to import open wake word model on platform {platform.system()}. Skipping...")
        return

    print(f"Downloading model: {config.model_name}")
    if config.models_dir:
        openwakeword.utils.download_models([config.model_name], config.models_dir)
    else:
        openwakeword.utils.download_models([config.model_name])


def main() -> None:
    config: AppConfig = load_config()
    download_wake_word_model(config.wake_word)


if __name__ == '__main__':
    main()