import functools
import json
from datetime import datetime
from pathlib import Path

import click

from ytcc import audio, glossary, meeting, pipeline, recorder
from ytcc.asr.base import PROMPT_TOKENS, ASRBackend
from ytcc.asr.device import create_backend, resolve_device
from ytcc.ingest.ytdlp import download_audio
from ytcc.storage.state import Options, State, load_state, save_state

DEVICES = click.Choice(["auto", "cpu", "cuda", "mlx"])
DEFAULT_MODEL = "large-v3-turbo"
TRACK_DIRS = {meeting.ME: "tracks/me", meeting.OTHERS: "tracks/others", "": "tracks/main"}


@click.group()
def cli() -> None:
    """Long-form audio/video transcription tool."""


def _is_url(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")


def asr_options(f):
    """Options shared by `transcribe` and `meeting`."""

    @click.option("--model", default=DEFAULT_MODEL, show_default=True, help="Whisper model size")
    @click.option("--device", type=DEVICES, default="auto", show_default=True, help="mlx = Apple Silicon GPU")
    @click.option("--compute-type", default="int8", show_default=True, help="Quantization type (cpu/cuda)")
    @click.option("--language", default=None, help="Language code (e.g. zh, en). Default: auto-detect")
    @click.option("--prompt", default=None, help="Extra Whisper prompt, e.g. names or topic")
    @click.option(
        "--glossary", "glossary_path", type=click.Path(path_type=Path), default=None,
        help=f"Glossary file  [default: {glossary.default_path()}]",
    )
    @click.option("--no-glossary", is_flag=True, help="Don't use the glossary")
    @click.option(
        "--window", "--chunk-duration", "window", default=300, show_default=True,
        help="Target window length in seconds; windows end at the nearest pause",
    )
    @functools.wraps(f)
    def wrapper(*args, model, device, compute_type, language, prompt, glossary_path, no_glossary, window, **kwargs):
        options = Options(
            model=model,
            device=resolve_device(device),
            compute_type=compute_type,
            language=language,
            prompt=prompt,  # glossary terms are added by _backend()
            window_sec=window,
        )
        glossary_file = None if no_glossary else (glossary_path or glossary.default_path())
        return f(*args, options=options, glossary_file=glossary_file, **kwargs)

    return wrapper


def _new_out_dir(out: Path) -> None:
    if out.exists() and any(p.name != ".DS_Store" for p in out.iterdir()):
        raise click.ClickException(
            f"{out} is not empty. Use a new output directory, or `ytcc resume {out}`."
        )
    out.mkdir(parents=True, exist_ok=True)


def _backend(options: Options, glossary_file: Path | None = None) -> ASRBackend:
    """
    Build the backend. With glossary_file (new runs only), add its terms to
    options.prompt, as many as fit Whisper's prompt limit.
    """
    click.echo(f"Using device: {options.device}, model: {options.model}")
    backend = create_backend(options.device, options.model, options.compute_type, options.language)
    terms = glossary.load_terms(glossary_file) if glossary_file else []
    if glossary_file and terms:
        try:
            options.prompt, dropped = glossary.fit_prompt(
                terms, options.prompt, backend.count_tokens, PROMPT_TOKENS
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"Glossary: {len(terms) - len(dropped)} of {len(terms)} term(s) in the Whisper prompt")
        if dropped:
            shown = ", ".join(dropped[:5]) + (f" and {len(dropped) - 5} more" if len(dropped) > 5 else "")
            click.echo(
                f"Warning: over Whisper's {PROMPT_TOKENS}-token prompt limit; left out: {shown}. "
                "Terms near the top of the glossary are kept first.",
                err=True,
            )
    elif options.prompt and backend.count_tokens(options.prompt) > PROMPT_TOKENS:
        raise click.ClickException(f"--prompt is longer than Whisper's {PROMPT_TOKENS}-token limit.")
    backend.prompt = options.prompt
    return backend


def _start_track(out_dir: Path, source: str, options: Options) -> State:
    """Create the state for a new track in out_dir, downloading the audio if source is a URL."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if _is_url(source):
        click.echo(f"Downloading audio from {source} ...")
        audio_path = download_audio(source, out_dir)
        click.echo(f"  → {audio_path.name}")
    else:
        audio_path = Path(source).resolve()
        if not audio_path.exists():
            raise click.ClickException(f"File not found: {source}")
    state = State(
        source=source,
        audio=str(audio_path),
        duration=audio.probe_duration(audio_path),
        options=options,
    )
    save_state(out_dir, state)
    return state


def _run_track(out_dir: Path, state: State, backend: ASRBackend, retry_failed: bool = False) -> None:
    pipeline.run(out_dir, state, backend, retry_failed=retry_failed)
    # Downloaded audio is kept until every window succeeded, for resume and retries
    if state.complete and _is_url(state.source):
        Path(state.audio).unlink(missing_ok=True)


@cli.command()
@click.argument("source", type=str)
@click.option("--out", required=True, type=click.Path(path_type=Path), help="Output directory (new or empty)")
@asr_options
def transcribe(source: str, out: Path, options: Options, glossary_file: Path | None) -> None:
    """Transcribe SOURCE audio/video file or URL."""
    _new_out_dir(out)
    # Built before ingest so a bad --model fails fast instead of after the download
    backend = _backend(options, glossary_file)
    state = _start_track(out, source, options)
    _run_track(out, state, backend)
    _finish(out, [state])


@cli.command(name="meeting")
@click.argument("source", type=str, required=False)
@click.option("--out", required=True, type=click.Path(path_type=Path), help="Output directory (new or empty)")
@click.option("--me", "me_track", type=click.Path(exists=True, path_type=Path), help="Microphone track")
@click.option("--others", "others_track", type=click.Path(exists=True, path_type=Path), help="System audio track")
@click.option("--title", default=None, help="Meeting title  [default: output directory name]")
@click.option("--started-at", default=None, help="Start time, ISO 8601  [default: recording's creation time]")
@click.option("--summarize", is_flag=True, help="Write summary.md with `claude -p` (sends the transcript to Anthropic)")
@click.option("--summary-model", default=None, help="Claude model for --summarize")
@asr_options
def meeting_cmd(
    source: str | None,
    out: Path,
    me_track: Path | None,
    others_track: Path | None,
    title: str | None,
    started_at: str | None,
    summarize: bool,
    summary_model: str | None,
    options: Options,
    glossary_file: Path | None,
) -> None:
    """
    Transcribe a meeting recording, optionally with separate tracks.

    Pass one SOURCE (file or URL), or --me and --others for a recording with
    the microphone and the call audio on separate tracks; the transcript then
    labels lines "Me" or "Others".
    """
    if source and (me_track or others_track):
        raise click.UsageError("Pass either SOURCE or --me/--others, not both.")
    if source:
        sources = {"": source}
    elif me_track and others_track:
        sources = {meeting.ME: str(me_track), meeting.OTHERS: str(others_track)}
    else:
        raise click.UsageError("Pass SOURCE, or both --me and --others.")

    _new_out_dir(out)
    backend = _backend(options, glossary_file)

    first = next(iter(sources.values()))
    if started_at is None:
        started = datetime.now() if _is_url(first) else meeting.recorded_at(Path(first))
        started_at = started.isoformat(timespec="minutes")
    mtg = meeting.Meeting(
        title=title or out.name,
        started_at=started_at,
        tracks={label: TRACK_DIRS[label] for label in sources},
    )
    meeting.save_meeting(out, mtg)

    states = []
    for label, src in sources.items():
        if label:
            click.echo(f"\n== Track: {label}")
        track_dir = out / mtg.tracks[label]
        state = _start_track(track_dir, src, options)
        _run_track(track_dir, state, backend)
        states.append(state)

    _finish_meeting(out, mtg, states, summarize, summary_model, glossary_file)


@cli.command()
@click.argument("out_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--retry-failed", is_flag=True, help="Transcribe failed windows again")
@click.option("--device", type=DEVICES, default=None, help="Override the device of the original run")
@click.option("--summarize", is_flag=True, help="Meetings: write summary.md once complete")
@click.option("--summary-model", default=None)
def resume(out_dir: Path, retry_failed: bool, device: str | None, summarize: bool, summary_model: str | None) -> None:
    """Continue an interrupted transcription in OUT_DIR with its original settings."""
    mtg = meeting.load_meeting(out_dir)
    track_dirs = [out_dir / d for d in mtg.tracks.values()] if mtg else [out_dir]

    states = []
    for track_dir in track_dirs:
        try:
            state = load_state(track_dir)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        if state is None:
            raise click.ClickException(f"No state.json in {track_dir}. Run `ytcc transcribe` first.")
        states.append(state)

    backend = None
    for track_dir, state in zip(track_dirs, states):
        if state.position >= state.duration and not (retry_failed and state.failed):
            continue
        if backend is None:
            if device:
                state.options.device = resolve_device(device)
            backend = _backend(state.options)
        if mtg:
            click.echo(f"\n== Track: {track_dir.name}")
        _run_track(track_dir, state, backend, retry_failed)

    if backend is None:
        click.echo("Nothing left to transcribe." + ("" if retry_failed else " (use --retry-failed for failed windows)"))
    if mtg:
        _finish_meeting(out_dir, mtg, states, summarize, summary_model, glossary.default_path())
    else:
        _finish(out_dir, states)


@cli.command(name="summarize")
@click.argument("out_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--model", default=None, help="Claude model")
@click.option(
    "--glossary", "glossary_path", type=click.Path(path_type=Path), default=None,
    help=f"Glossary for correcting names and terms  [default: {glossary.default_path()}]",
)
@click.option("--no-glossary", is_flag=True, help="Don't use the glossary")
def summarize_cmd(out_dir: Path, model: str | None, glossary_path: Path | None, no_glossary: bool) -> None:
    """Write OUT_DIR/summary.md from its transcript with `claude -p` (sends it to Anthropic)."""
    if not (out_dir / "transcript.md").exists():
        raise click.ClickException(f"No transcript.md in {out_dir}.")
    _summarize(out_dir, model, None if no_glossary else (glossary_path or glossary.default_path()))


@cli.command()
@click.argument("out_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def clean(out_dir: Path) -> None:
    """Delete downloaded audio and leftover chunks in OUT_DIR (keeps transcript and state)."""
    removed = 0
    for path in [*out_dir.glob("source_audio.*"), *out_dir.glob("tracks/*/source_audio.*")]:
        path.unlink(missing_ok=True)
        removed += 1
    chunks_dir = out_dir / "chunks"  # left by older ytcc versions
    if chunks_dir.exists():
        for chunk in chunks_dir.glob("chunk_*.wav"):
            chunk.unlink()
            removed += 1
        if not any(chunks_dir.iterdir()):
            chunks_dir.rmdir()
    click.echo(f"Removed {removed} file(s).")


@cli.group()
def record() -> None:
    """Record meetings on macOS (microphone + call audio)."""


@record.command(name="start")
@click.option("--title", default=None, help="Meeting title, also used in the folder name")
@click.option(
    "--dir", "base_dir", type=click.Path(file_okay=False, path_type=Path), default=None,
    help=f"Where recordings go  [default: $YTCC_MEETINGS_DIR or {recorder.default_meetings_dir()}]",
)
@click.option("--summarize", is_flag=True, help="Summarize with `claude -p` after transcribing (sends it to Anthropic)")
@click.option("--echo-cancel", is_flag=True, help="Remove speaker sound from the mic track (when not using headphones)")
def record_start(title: str | None, base_dir: Path | None, summarize: bool, echo_cancel: bool) -> None:
    """Start recording the microphone and the call audio."""
    try:
        rec = recorder.start(title, base_dir or recorder.default_meetings_dir(), summarize, echo_cancel)
    except recorder.RecorderError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Recording into {rec.dir}")


@record.command(name="stop")
@click.option("--no-transcribe", is_flag=True, help="Only save the audio")
def record_stop(no_transcribe: bool) -> None:
    """Stop recording and transcribe in the background (notifies when done)."""
    try:
        rec = recorder.stop(transcribe=not no_transcribe)
    except recorder.RecorderError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Saved {rec.dir}")
    if not no_transcribe:
        click.echo(f"Transcribing in the background; log: {Path(rec.dir) / 'transcribe.log'}")


@record.command(name="status")
def record_status() -> None:
    """Print the current recording as JSON."""
    click.echo(json.dumps(recorder.status(), ensure_ascii=False))


@record.command(name="detect")
def record_detect() -> None:
    """Print apps using the microphone as JSON."""
    try:
        click.echo(json.dumps(recorder.detect(), ensure_ascii=False))
    except recorder.RecorderError as exc:
        raise click.ClickException(str(exc)) from exc


@record.command(name="process", hidden=True)
@click.argument("rec_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def record_process(rec_dir: Path) -> None:
    """Transcribe a stopped recording (run by `record stop`)."""
    raise SystemExit(recorder.process(rec_dir))


def _finish_meeting(
    out_dir: Path,
    mtg: meeting.Meeting,
    states: list[State],
    summarize: bool,
    summary_model: str | None,
    glossary_file: Path | None,
) -> None:
    meeting.merge(out_dir, mtg, max(s.duration for s in states))
    complete = _finish(out_dir, states, exit_on_failure=False)
    if summarize:
        if complete:
            _summarize(out_dir, summary_model, glossary_file)
        else:
            click.echo("Skipping summary: transcript is incomplete.", err=True)
    if not complete:
        raise SystemExit(1)


def _summarize(out_dir: Path, model: str | None, glossary_file: Path | None) -> None:
    click.echo("Summarizing with claude ...")
    entries = glossary.load_entries(glossary_file) if glossary_file else []
    try:
        path = meeting.summarize(out_dir, model, entries)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"  Summary   : {path}")


def _finish(out_dir: Path, states: list[State], exit_on_failure: bool = True) -> bool:
    """Print the outcome; return whether every track is complete."""
    complete = all(s.complete for s in states)
    click.echo("\nDone." if complete else "\nIncomplete.")
    for s in states:
        done = sum(1 for w in s.windows if w.status == "done")
        line = f"  {Path(s.source).name if not _is_url(s.source) else s.source}: {done} window(s) done"
        if s.failed:
            line += f", {len(s.failed)} failed (ytcc resume --retry-failed)"
        elif s.position < s.duration:
            line += " so far (ytcc resume)"
        click.echo(line)
    click.echo(f"  Transcript: {out_dir / 'transcript.md'}")
    click.echo(f"  Segments  : {out_dir / 'segments.jsonl'}")
    if not complete and exit_on_failure:
        raise SystemExit(1)
    return complete
