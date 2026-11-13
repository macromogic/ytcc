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


# Pauses shorter than this stay in the speech-only audio: they are where
# Whisper ends its segments, and without them it runs utterances together into
# segments that skip speech (faster-whisper's own VAD keeps up to 2 s too)
KEEP_PAUSE_SEC = 2.0


class SpeechMap:
    """Maps times in speech-only audio (see speech_only) back to the original audio."""

    def __init__(self, regions: list[tuple[float, float]]) -> None:
        # (start in speech-only audio, start in original, length), seconds
        self._pieces: list[tuple[float, float, float]] = []
        offset = 0.0
        for a, b in regions:
            self._pieces.append((offset, a, b - a))
            offset += b - a

    def _piece(self, t: float, is_end: bool) -> int:
        for i, (start, _, length) in enumerate(self._pieces):
            if t < start + length or (is_end and t <= start + length):
                return i
        return len(self._pieces) - 1

    def original(self, t: float, is_end: bool = False) -> float:
        """
        Original time of t. A time on the joint of two regions is the end of the
        first when is_end, else the start of the second.
        """
        if not self._pieces:
            return t
        start, orig, length = self._pieces[self._piece(t, is_end)]
        # Clamped: Whisper's last timestamp can overshoot the audio
        return orig + min(max(t - start, 0.0), length)

    def span(self, start: float, end: float) -> tuple[float, float]:
        """
        Original (start, end) of a segment, its end cut at the end of the region
        it starts in, so it never stretches over the silence that was left out.
        """
        if not self._pieces:
            return start, end
        first = self._piece(start, is_end=False)
        if self._piece(end, is_end=True) > first:
            _, orig, length = self._pieces[first]
            return self.original(start), orig + length
        return self.original(start), self.original(end, is_end=True)


def speech_only(audio: np.ndarray, regions: list[tuple[float, float]]) -> tuple[np.ndarray, SpeechMap]:
    """
    Join the speech regions of audio, leaving out pauses of KEEP_PAUSE_SEC or more.

    Whisper spends no time on long silence and gets no chance to write text
    over it. Timestamps from the result go back through the returned SpeechMap.
    """
    rate = SAMPLE_RATE
    merged: list[list[float]] = []
    for a, b in regions:
        if merged and a - merged[-1][1] < KEEP_PAUSE_SEC:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    spans = [(int(a * rate), min(int(b * rate), len(audio))) for a, b in merged]
    spans = [(a, b) for a, b in spans if b > a]
    joined = np.concatenate([audio[a:b] for a, b in spans]) if spans else audio[:0]
    return joined, SpeechMap([(a / rate, b / rate) for a, b in spans])


def speech_regions(audio: np.ndarray) -> list[tuple[float, float]]:
    """Return the (start, end) seconds of speech in audio, found with Silero VAD."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    stamps = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=500))
    return [(t["start"] / SAMPLE_RATE, t["end"] / SAMPLE_RATE) for t in stamps]
