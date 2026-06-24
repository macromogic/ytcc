from dataclasses import dataclass, field


@dataclass
class Config:
    chunk_duration_sec: int = 300
    sample_rate: int = 16000
    channels: int = 1
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    language: str | None = None  # None = auto-detect
