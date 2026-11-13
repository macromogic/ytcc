import math
import re
from collections.abc import Callable
from pathlib import Path

import click
import numpy as np

from ytcc import audio
from ytcc.asr.base import ASRBackend
from ytcc.models import Segment
from ytcc.storage.state import State, Window, save_state
from ytcc.storage.transcript_writer import (
    append_segments,
    fmt_time,
    read_segments,
    render_transcript,
    write_segments,
)

# How far around the nominal window length to look for a pause to cut at
SEARCH_SEC = 30.0

# The language is detected per window from its speech alone. Whisper's own
# detection looks at the first 30 s, which in a meeting track is often silence
# or the other side talking; a wrong guess garbles or translates the window.
# Windows with less speech than this keep the previous window's language.
DETECT_SPEECH_SEC = 10.0

# Whisper writes text over silence ("Thank you."); drop segments that are
# mostly outside the speech the VAD found
MIN_SPEECH_RATIO = 0.2
MIN_SPEECH_SEC = 1.0
# Subtitle credits from Whisper's training data
HALLUCINATIONS = re.compile(r"请不吝点赞|点赞.{0,4}订阅|订阅.{0,4}转发|字幕(由|by|提供|制作|製作)|明镜与点点|Amara\.org", re.I)

# Whisper's own signals for decodes that went wrong over speech, where the VAD
# check above can't help. Measured on real meetings (mlx, large-v3-turbo), whose
# no_speech_prob is always 0 and so is not used:
# - Whisper retries a decode at higher temperatures while its text compresses
#   better than 2.4 (a repetition loop); above that, every retry failed too
#   ("Just..." 22 times, ratio 15.9).
MAX_COMPRESSION_RATIO = 2.4
# - A loop that compresses poorly, because Whisper split it into many short
#   segments ("Am I" 9 times at avg_logprob -2.67). Low log-probability alone is
#   no test: real speech in a mixed-language meeting scores as low, and
#   repeated backchannels ("嗯", "对") as low as -1.5.
LOOP_MIN_REPEATS = 3
LOOP_MAX_LOGPROB = -2.0

Decoder = Callable[[Path, float, float | None], np.ndarray]
SpeechFinder = Callable[[np.ndarray], list[tuple[float, float]]]


def run(
    out_dir: Path,
    state: State,
    backend: ASRBackend,
    *,
    retry_failed: bool = False,
    decode: Decoder = audio.decode,
    speech: SpeechFinder = audio.speech_regions,
) -> None:
    """
    Transcribe state.audio from state.position to the end, one window at a time.

    Each window is decoded straight from the source into memory, so no audio
    is written to disk. State is saved after every window; an interrupted run
    continues from state.position.
    """
    source = Path(state.audio)
    language = state.options.language
    _drop_unfinished_segments(out_dir, state)

    if retry_failed:
        for window in state.failed:
            click.echo(f"Retrying {_span(window.start, window.end)} ...", nl=False)
            language = _transcribe(out_dir, state, backend, decode, speech, source, window, language)

    target = state.options.window_sec
    total = max(math.ceil(state.duration / target), 1)
    while state.position < state.duration:
        start = state.position
        index = round(start / target) + 1
        click.echo(f"[{index}/~{total}] ", nl=False)

        remaining = state.duration - start
        try:
            last = remaining <= target + SEARCH_SEC
            # The last window reads to the end of the file, whatever the probed
            # duration says (it is only an estimate for some formats)
            samples = decode(source, start, None if last else target + SEARCH_SEC)
            got = len(samples) / audio.SAMPLE_RATE
            if last or got < target + SEARCH_SEC - 1:
                if got == 0:  # the file ended before the probed duration
                    state.duration = start
                    save_state(out_dir, state)
                    click.echo("end of audio")
                    break
                end = start + got
                state.duration = end
            else:
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
        language = _transcribe(out_dir, state, backend, decode, speech, source, window, language, samples)


def _transcribe(
    out_dir: Path,
    state: State,
    backend: ASRBackend,
    decode: Decoder,
    speech: SpeechFinder,
    source: Path,
    window: Window,
    language: str | None,
    samples: np.ndarray | None = None,
) -> str | None:
    """Transcribe one window. Returns the language used, for the next window."""
    try:
        if samples is None:
            samples = decode(source, window.start, window.end - window.start)
        regions = speech(samples)
        if regions:
            if not state.options.language:
                language = _detect_language(backend, samples, regions, language)
            speech_audio, speech_map = audio.speech_only(samples, regions)
            segments = backend.transcribe(speech_audio, language)
            for seg in segments:
                seg.start, seg.end = speech_map.span(seg.start, seg.end)
            segments = _drop_loops([s for s in segments if _is_speech(s, regions)])
        else:
            segments = []  # don't let Whisper hallucinate over silence
    except Exception as exc:
        click.echo(f" FAILED: {exc}", err=True)
        window.status = "failed"
        window.error = str(exc)
        _record(out_dir, state, window, window.end)
        return language

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
    click.echo(f" done ({len(segments)} segments, {language})" if regions else " no speech")
    return language


def _drop_unfinished_segments(out_dir: Path, state: State) -> None:
    """
    Remove segments outside the windows state.json records as done.

    Segments are appended before the state is saved, so a run stopped between
    the two leaves a window's segments on disk while the window itself is
    transcribed again; without this its lines would appear twice.
    """
    segments = read_segments(out_dir)
    done = [(w.start, w.end) for w in state.windows if w.status == "done"]
    # Stored times are rounded to 3 decimals. Half-open: a segment at 0.0 s of
    # the unfinished window starts exactly where the last done window ends.
    kept = [s for s in segments if any(a - 0.01 <= s.start < b - 0.001 for a, b in done)]
    if len(kept) < len(segments):
        write_segments(out_dir, kept)
        render_transcript(out_dir)
        click.echo(f"Removed {len(segments) - len(kept)} segment(s) of an unfinished window.")


def _detect_language(
    backend: ASRBackend, samples: np.ndarray, regions: list[tuple[float, float]], previous: str | None
) -> str:
    rate = audio.SAMPLE_RATE
    clip = np.concatenate([samples[int(a * rate) : int(b * rate)] for a, b in regions])[: 30 * rate]
    if previous and len(clip) < DETECT_SPEECH_SEC * rate:
        return previous
    return backend.detect_language(clip)[0]


def _is_speech(segment: Segment, regions: list[tuple[float, float]]) -> bool:
    if HALLUCINATIONS.search(segment.text):
        return False
    if segment.compression_ratio is not None and segment.compression_ratio > MAX_COMPRESSION_RATIO:
        return False
    overlap = sum(max(0.0, min(b, segment.end) - max(a, segment.start)) for a, b in regions)
    return overlap >= min(MIN_SPEECH_SEC, MIN_SPEECH_RATIO * (segment.end - segment.start))


def _drop_loops(segments: list[Segment]) -> list[Segment]:
    """Remove runs of LOOP_MIN_REPEATS or more identical, low-confidence segments."""
    kept: list[Segment] = []
    i = 0
    while i < len(segments):
        j = i
        while j < len(segments) and segments[j].text == segments[i].text:
            j += 1
        run = segments[i:j]
        loop = len(run) >= LOOP_MIN_REPEATS and all(
            s.avg_logprob is not None and s.avg_logprob < LOOP_MAX_LOGPROB for s in run
        )
        if not loop:
            kept.extend(run)
        i = j
    return kept


def _record(out_dir: Path, state: State, window: Window, end: float) -> None:
    if not any(w is window for w in state.windows):  # retried windows are already listed
        state.windows.append(window)
    state.windows.sort(key=lambda w: w.start)
    state.position = max(state.position, end)
    save_state(out_dir, state)


def _span(start: float, end: float) -> str:
    return f"{fmt_time(start)}–{fmt_time(end)}"
