import click


def resolve_device(requested: str) -> str:
    """
    Return the device to use for inference.

    "auto"  → cuda if available, else cpu
    "cuda"  → cuda; warn and fall back to cpu if unavailable
    "cpu"   → cpu unconditionally
    """
    if requested == "cpu":
        return "cpu"

    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass

    if requested == "cuda":
        click.echo("Warning: CUDA not available, falling back to CPU.", err=True)
    return "cpu"
