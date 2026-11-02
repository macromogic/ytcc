import math
from collections.abc import Callable
from pathlib import Path

import click
import numpy as np

from ytcc import audio
from ytcc.asr.base import ASRBackend
from ytcc.storage.state import State, Window, save_state
from ytcc.storage.transcript_writer import append_segments, fmt_time, render_transcript

# How far around the nominal window length to look for a pause to cut at
SEARCH_SEC = 30.0

Decoder = Callable[[Path, float, float], np.ndarray]


def run(
    out_dir: Path,
    state: State,
    backend: ASRBackend,
    *,
    retry_failed: bool = False,
    decode: Decoder = audio.decode,
) -> None:
    """
    Transcribe state.audio from state.position to the end, one window at a time.

    Each window is decoded straight from the source into memory, so no audio
    is written to disk. State is saved after every window; an interrupted run
    continues from state.position.
    """
    source = Path(state.audio)

    if retry_failed:
        for window in state.failed:
            click.echo(f"Retrying {_span(window.start, window.end)} ...", nl=False)
            _transcribe(out_dir, state, backend, decode, source, window)

    target = state.options.window_sec
    total = max(math.ceil(state.duration / target), 1)
    while state.position < state.duration:
        start = state.position
        index = round(start / target) + 1
        click.echo(f"[{index}/~{total}] ", nl=False)

        remaining = state.duration - start
        try:
            if remaining <= target + SEARCH_SEC:
                # Last window: read to the end (+1 s in case the probed duration is short)
                samples = decode(source, start, remaining + 1)
                end = state.duration
            else:
                samples = decode(source, start, target + SEARCH_SEC)
                cut = audio.find_cut(samples, target, SEARCH_SEC)
                samples = samples[: int(cut * audio.SAMPLE_RATE)]
                end = start + cut
        except Exception as exc:
            end = min(start + target, state.duration)
            click.echo(f"{_span(start, end)} FAILED to decode: {exc}", err=True)
            _record(out_dir, state, Window(start, end, "failed", error=str(exc)), end)
            continue

        window = Window(start, end, "pending")
        click.echo(f"{_span(start, end)} ...", nl=False)
        _transcribe(out_dir, state, backend, decode, source, window, samples)


def _transcribe(
    out_dir: Path,
    state: State,
    backend: ASRBackend,
    decode: Decoder,
    source: Path,
    window: Window,
    samples: np.ndarray | None = None,
) -> None:
    try:
        if samples is None:
            samples = decode(source, window.start, window.end - window.start)
        segments = backend.transcribe(samples)
    except Exception as exc:
        click.echo(f" FAILED: {exc}", err=True)
        window.status = "failed"
        window.error = str(exc)
        _record(out_dir, state, window, window.end)
        return

    for seg in segments:
        seg.start += window.start
        seg.end += window.start
    # Persist segments before advancing state, so a crash can at worst repeat a window
    append_segments(out_dir, segments)
    window.status = "done"
    window.segments = len(segments)
    window.error = None
    _record(out_dir, state, window, window.end)
    render_transcript(out_dir)
    click.echo(f" done ({len(segments)} segments)")


def _record(out_dir: Path, state: State, window: Window, end: float) -> None:
    if not any(w is window for w in state.windows):  # retried windows are already listed
        state.windows.append(window)
    state.windows.sort(key=lambda w: w.start)
    state.position = max(state.position, end)
    save_state(out_dir, state)


def _span(start: float, end: float) -> str:
    return f"{fmt_time(start)}–{fmt_time(end)}"
