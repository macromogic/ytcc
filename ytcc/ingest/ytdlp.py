import subprocess
from pathlib import Path


def download_audio(url: str, out_dir: Path) -> Path:
    """Download best audio from URL into out_dir. Returns path to the downloaded file."""
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "yt-dlp",
        "-f", "bestaudio",
        "--no-playlist",
        "-o", str(out_dir / "source_audio.%(ext)s"),
        url,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed:\n{result.stderr}")

    matches = list(out_dir.glob("source_audio.*"))
    if not matches:
        raise RuntimeError("yt-dlp produced no output file.")
    return matches[0]
