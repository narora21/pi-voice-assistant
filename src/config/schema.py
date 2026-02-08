from dataclasses import dataclass, field

from src.util.prompt_loader import PromptLoader


@dataclass(frozen=True)
class WakeWordConfig:
    model_name: str = "hey_jarvis_v0.1"
    models_path: str = "models/wake_word/"
    model_extension: str = "tflite"
    threshold: float = 0.5
    vad_threshold: float | None = None


@dataclass(frozen=True)
class SoundBytesConfig:
    greeting: str = "Hello."
    thinking: str = "One moment please."


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    frame_duration_ms: int = 80
    channels: int = 1
    capture_device: str | None = None
    playback_device: str | None = None
    energy_threshold: int = 500 # For silence sound threshold
    playback_volume: int = 0.5 # Between 0-1 (above 1 may cause distortion)


@dataclass(frozen=True)
class STTConfig:
    model_size: str = "base.en"
    device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 5
    language: str = "en"
    vad_filter: bool = True


@dataclass(frozen=True)
class AgentConfig:
    model: str = "qwen2.5:1.5b"
    system_prompt: str = PromptLoader.load_system_prompt()
    max_tool_rounds: int = 5
    temperature: float = 0.7
    stream: bool = True
    num_ctx: int = 2048
    num_thread: int = 4


@dataclass(frozen=True)
class TTSConfig:
    model_name: str = "en_US-lessac-medium"
    models_path: str = "models/tts/en_US-lessac-medium"
    model_extension: str = "onnx"
    batch_min_chars: int = 50   # Don't speak until at least this many chars
    batch_max_chars: int = 200  # Force speak at this limit
    speaker_id: int = 0
    length_scale: float = 1.0
    noise_scale: float = 0.667
    noise_w: float = 0.8


@dataclass(frozen=True)
class SessionConfig:
    idle_timeout_seconds: float = 30.0
    max_history_messages: int = 50


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    file: str = "logs/assistant.log"
    max_bytes: int = 5_242_880
    backup_count: int = 3


@dataclass(frozen=True)
class AppConfig:
    wake_word: WakeWordConfig = field(default_factory=WakeWordConfig)
    sound_bytes: SoundBytesConfig = field(default_factory=SoundBytesConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
