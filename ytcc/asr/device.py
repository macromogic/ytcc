import platform
import sys

import click

from ytcc.asr.base import ASRBackend


def _cuda_available() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def _mlx_available() -> bool:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx_whisper  # noqa: F401
        return True
    except Exception:
        return False


def resolve_device(requested: str) -> str:
    """
    Return the device to use for inference.

    "auto"  → cuda if available, else mlx on Apple Silicon, else cpu
    "cuda"  → cuda; warn and fall back to cpu if unavailable
    "mlx"   → mlx (Apple Silicon GPU); warn and fall back to cpu if unavailable
    "cpu"   → cpu unconditionally
    """
    if requested == "cpu":
        return "cpu"

    if requested in ("auto", "cuda") and _cuda_available():
        return "cuda"
    if requested in ("auto", "mlx") and _mlx_available():
        return "mlx"

    if requested in ("cuda", "mlx"):
        click.echo(f"Warning: {requested} not available, falling back to CPU.", err=True)
    return "cpu"


def create_backend(
    device: str,
    model_size: str,
    compute_type: str,
    language: str | None,
) -> ASRBackend:
    """Build the ASR backend for a device returned by resolve_device()."""
    if device == "mlx":
        from ytcc.asr.mlx_whisper_backend import MLXWhisperBackend
        try:
            return MLXWhisperBackend(model_size=model_size, language=language)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc

    from ytcc.asr.faster_whisper_backend import FasterWhisperBackend
    return FasterWhisperBackend(
        model_size=model_size,
        device=device,
        compute_type=compute_type,
        language=language,
    )
