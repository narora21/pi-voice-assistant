import os

from src.config.schema import WakeWordConfig

def get_wake_word_model_dir(config: WakeWordConfig) -> str:
    return os.path.join(config.models_path, config.model_name)

def get_wake_word_model_path(config: WakeWordConfig) -> str:
    model_dir = get_wake_word_model_dir(config)
    return os.path.join(model_dir, config.model_name + "." + config.model_extension)

def get_wake_word_melspec_path(config: WakeWordConfig) -> str:
    model_dir = get_wake_word_model_dir(config)
    return os.path.join(model_dir, "melspectrogram." + config.model_extension)

def get_wake_word_embedding_path(config: WakeWordConfig) -> str:
    model_dir = get_wake_word_model_dir(config)
    return os.path.join(model_dir, "embedding_model." + config.model_extension)
