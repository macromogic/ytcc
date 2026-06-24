import subprocess
from pathlib import Path

from ytcc.config import Config


def extract_chunks(source: Path, chunks_dir: Path, config: Config) -> list[Path]:
    """Convert source file to normalized WAV chunks. Returns sorted list of chunk paths."""
    chunks_dir.mkdir(parents=True, exist_ok=True)

    output_pattern = str(chunks_dir / "chunk_%03d.wav")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(source),
        "-vn",                              # drop video stream
        "-ac", str(config.channels),
        "-ar", str(config.sample_rate),
        "-f", "segment",
        "-segment_time", str(config.chunk_duration_sec),
        output_pattern,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")

    chunks = sorted(chunks_dir.glob("chunk_*.wav"))
    if not chunks:
        raise RuntimeError("ffmpeg produced no output chunks.")
    return chunks
