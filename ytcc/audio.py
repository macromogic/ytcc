import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


def probe_duration(source: Path) -> float:
    """Return the media duration in seconds."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(source),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{result.stderr}")
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"ffprobe reported no duration for {source}") from exc


def decode(source: Path, start: float, length: float | None) -> np.ndarray:
    """Decode [start, start + length) of source, or to its end, to 16 kHz mono float32 in memory."""
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error",
        "-ss", f"{start:.3f}",
        *(["-t", f"{length:.3f}"] if length is not None else []),
        "-i", str(source),
        "-vn",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-f", "s16le",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr.decode(errors='replace')}")
    return np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def find_cut(audio: np.ndarray, target: float, search: float, frame: float = 0.1) -> float:
    """
    Return the quietest point (seconds) within target ± search.

    Cutting a window in a pause instead of at a fixed time keeps words and
    short sentences from being split across two windows.
    """
    frame_len = int(frame * SAMPLE_RATE)
    lo = max(int((target - search) * SAMPLE_RATE), 0)
    hi = min(int((target + search) * SAMPLE_RATE), len(audio))
    n_frames = (hi - lo) // frame_len
    if n_frames < 1:
        return min(target, len(audio) / SAMPLE_RATE)

    frames = audio[lo : lo + n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    # Smooth over ~0.5 s so a single quiet frame inside a word doesn't win
    rms = np.convolve(np.pad(rms, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    # Among equally quiet frames (e.g. a long silence), prefer the one nearest target
    centre = (target * SAMPLE_RATE - lo) / frame_len
    rms += 1e-9 * np.abs(np.arange(n_frames) - centre)
    best = int(np.argmin(rms))
    return (lo + best * frame_len + frame_len / 2) / SAMPLE_RATE


def split_at_pauses(audio: np.ndarray, target: float, search: float) -> list[tuple[float, np.ndarray]]:
    """Split audio into pieces of about target seconds, cut in pauses. Returns (offset, piece) pairs."""
    pieces = []
    offset = 0
    while len(audio) - offset > (target + search) * SAMPLE_RATE:
        cut = offset + int(find_cut(audio[offset:], target, search) * SAMPLE_RATE)
        pieces.append((offset / SAMPLE_RATE, audio[offset:cut]))
        offset = cut
    pieces.append((offset / SAMPLE_RATE, audio[offset:]))
    return pieces
