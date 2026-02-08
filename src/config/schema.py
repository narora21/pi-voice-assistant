from dataclasses import dataclass, field


@dataclass(frozen=True)
class WakeWordConfig:
    model_name: str = "hey_jarvis_v0.1"
    models_path: str = "models/wake_word/"
    model_extension: str = "tflite"
    threshold: float = 0.5
    vad_threshold: float | None = None


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
    system_prompt: str = (
        "You are a helpful voice assistant running on a Raspberry Pi. "
        "Keep responses concise and conversational. "
        "When using tools, explain what you're doing briefly."
    )
    max_tool_rounds: int = 5
    temperature: float = 0.7
    stream: bool = True


@dataclass(frozen=True)
class TTSConfig:
    model_name: str = "en_US-lessac-medium"
    models_path: str = "models/tts/en_US-lessac-medium"
    model_extension: str = "onnx"
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
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
