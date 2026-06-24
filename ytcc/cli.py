from pathlib import Path

import click

from ytcc.asr.device import resolve_device
from ytcc.asr.faster_whisper_backend import FasterWhisperBackend
from ytcc.chunking.chunk_manager import process_chunks
from ytcc.config import Config
from ytcc.ingest.ffmpeg import extract_chunks
from ytcc.storage.cleanup import delete_chunk
from ytcc.storage.state import State, load_state, save_state


@click.group()
def cli() -> None:
    """Long-form audio/video transcription tool."""


@cli.command()
@click.argument("source", type=click.Path(exists=True, path_type=Path))
@click.option("--out", required=True, type=click.Path(path_type=Path), help="Output directory")
@click.option("--model", default="base", show_default=True, help="Whisper model size")
@click.option("--device", default="auto", show_default=True, help="auto, cpu, or cuda")
@click.option("--compute-type", default="int8", show_default=True, help="Quantization type")
@click.option("--language", default=None, help="Language code (e.g. zh, en). Default: auto-detect")
@click.option("--chunk-duration", default=300, show_default=True, help="Chunk length in seconds")
def transcribe(
    source: Path,
    out: Path,
    model: str,
    device: str,
    compute_type: str,
    language: str | None,
    chunk_duration: int,
) -> None:
    """Transcribe SOURCE audio/video file."""
    out.mkdir(parents=True, exist_ok=True)
    chunks_dir = out / "chunks"
    device = resolve_device(device)
    click.echo(f"Using device: {device}")

    config = Config(
        chunk_duration_sec=chunk_duration,
        whisper_model=model,
        whisper_device=device,
        whisper_compute_type=compute_type,
        language=language,
    )

    click.echo(f"Extracting chunks from {source} ...")
    chunk_paths = extract_chunks(source, chunks_dir, config)
    click.echo(f"  → {len(chunk_paths)} chunks created")

    state = State(
        source=str(source),
        total_chunks=len(chunk_paths),
        chunk_duration_sec=chunk_duration,
        processed_chunks=[],
        failed_chunks=[],
    )
    save_state(out, state)

    backend = FasterWhisperBackend(
        model_size=model,
        device=device,
        compute_type=compute_type,
        language=language,
    )

    process_chunks(chunk_paths, out, state, backend)

    _print_summary(out, state)


@cli.command()
@click.argument("out_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--model", default="base", show_default=True)
@click.option("--device", default="auto", show_default=True)
@click.option("--compute-type", default="int8", show_default=True)
@click.option("--language", default=None)
def resume(
    out_dir: Path,
    model: str,
    device: str,
    compute_type: str,
    language: str | None,
) -> None:
    """Resume an interrupted transcription in OUT_DIR."""
    state = load_state(out_dir)
    if state is None:
        raise click.ClickException("No state.json found. Run `ytcc transcribe` first.")

    chunks_dir = out_dir / "chunks"
    chunk_paths = sorted(chunks_dir.glob("chunk_*.wav"))
    if not chunk_paths:
        click.echo("No remaining chunks to process.")
        return

    device = resolve_device(device)
    click.echo(f"Using device: {device}")

    backend = FasterWhisperBackend(
        model_size=model,
        device=device,
        compute_type=compute_type,
        language=language,
    )

    process_chunks(chunk_paths, out_dir, state, backend)
    _print_summary(out_dir, state)


@cli.command()
@click.argument("out_dir", type=click.Path(exists=True, path_type=Path))
def clean(out_dir: Path) -> None:
    """Delete leftover chunks in OUT_DIR (keeps transcript and state)."""
    chunks_dir = out_dir / "chunks"
    removed = 0
    if chunks_dir.exists():
        for chunk in chunks_dir.glob("chunk_*.wav"):
            delete_chunk(chunk)
            removed += 1
        chunks_dir.rmdir() if not any(chunks_dir.iterdir()) else None
    click.echo(f"Removed {removed} chunk file(s).")


def _print_summary(out_dir: Path, state: State) -> None:
    click.echo("\nDone.")
    click.echo(f"  Processed : {len(state.processed_chunks)}/{state.total_chunks} chunks")
    if state.failed_chunks:
        click.echo(f"  Failed    : {len(state.failed_chunks)} chunk(s) → {out_dir / 'failed_chunks'}")
    click.echo(f"  Transcript: {out_dir / 'transcript.md'}")
    click.echo(f"  Segments  : {out_dir / 'segments.jsonl'}")
