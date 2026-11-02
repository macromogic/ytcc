import importlib
from contextlib import contextmanager

import mlx_whisper
from mlx_whisper.tokenizer import get_tokenizer
import numpy as np

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

    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        language: str | None = None,
        prompt: str | None = None,
    ) -> None:
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
        self.prompt = prompt

    def transcribe(self, audio: np.ndarray) -> list[Segment]:
        with _prompt_every_decode(self.prompt):
            result = mlx_whisper.transcribe(
                audio,
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

    def count_tokens(self, text: str) -> int:
        # Same BPE vocabulary for every multilingual Whisper model; only the
        # language special tokens differ. mlx_whisper encodes " " + prompt.strip().
        return len(get_tokenizer(multilingual=True, num_languages=100).encode(" " + text.strip()))


@contextmanager
def _prompt_every_decode(prompt: str | None):
    """
    Give every 30 s decode the prompt, like faster-whisper's hotwords.

    mlx_whisper's initial_prompt reaches only the first decode when
    condition_on_previous_text is off, and turning that on makes Whisper drop
    and repeat phrases. mlx_whisper has no option for this, so wrap the
    DecodingOptions it builds for each decode.
    """
    module = importlib.import_module("mlx_whisper.transcribe")
    original = getattr(module, "DecodingOptions", None)
    if prompt is None or original is None:
        yield
        return

    def with_prompt(*args, **kwargs):
        if not kwargs.get("prompt"):
            kwargs["prompt"] = prompt
        return original(*args, **kwargs)

    module.DecodingOptions = with_prompt
    try:
        yield
    finally:
        module.DecodingOptions = original
