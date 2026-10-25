from pathlib import Path

import mlx_whisper

from ytcc.asr.base import ASRBackend
from ytcc.models import Segment

# Short model names → mlx-community repos on Hugging Face.
# Anything containing "/" is passed through as a repo id or local path.
_MODEL_REPOS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}


class MLXWhisperBackend(ASRBackend):
    """Apple Silicon backend: runs Whisper on the Metal GPU via MLX."""

    def __init__(self, model_size: str = "base", language: str | None = None) -> None:
        if "/" in model_size:
            self._repo = model_size
        elif model_size in _MODEL_REPOS:
            self._repo = _MODEL_REPOS[model_size]
        else:
            raise ValueError(
                f"Unknown model {model_size!r} for mlx; "
                f"use one of {', '.join(_MODEL_REPOS)} or a Hugging Face repo id."
            )
        self._language = language

    def transcribe(self, audio_path: Path) -> list[Segment]:
        result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=self._repo,
            language=self._language,
            # No built-in VAD like faster-whisper's vad_filter; these limit
            # hallucinations that repeat across silent stretches.
            condition_on_previous_text=False,
            hallucination_silence_threshold=2.0,
            word_timestamps=True,  # required by hallucination_silence_threshold
        )
        return [
            Segment(start=s["start"], end=s["end"], text=s["text"].strip())
            for s in result["segments"]
            if s["text"].strip()
        ]
