from pathlib import Path

import click

from ytcc.asr.base import ASRBackend
from ytcc.storage.cleanup import delete_chunk, move_to_failed
from ytcc.storage.state import State, save_state
from ytcc.storage.transcript_writer import append_segments


def _chunk_index(chunk_path: Path) -> int:
    """Extract numeric index from chunk filename, e.g. chunk_007.wav -> 7."""
    return int(chunk_path.stem.split("_")[1])


def process_chunks(
    chunk_paths: list[Path],
    out_dir: Path,
    state: State,
    backend: ASRBackend,
) -> None:
    failed_dir = out_dir / "failed_chunks"
    processed_set = set(state.processed_chunks)

    remaining = [c for c in chunk_paths if c.name not in processed_set]
    total = len(remaining)

    for idx, chunk_path in enumerate(remaining, start=1):
        click.echo(f"[{idx}/{total}] Transcribing {chunk_path.name} ...", nl=False)

        # Global time offset: derived from chunk filename, not loop index,
        # so resume works correctly after partial runs.
        offset = _chunk_index(chunk_path) * state.chunk_duration_sec

        try:
            segments = backend.transcribe(chunk_path)

            for seg in segments:
                seg.start += offset
                seg.end += offset

            # Safe write order: persist data before deleting audio
            append_segments(out_dir, segments, chunk_path.name)
            state.processed_chunks.append(chunk_path.name)
            save_state(out_dir, state)
            delete_chunk(chunk_path)

            click.echo(f" done ({len(segments)} segments)")

        except Exception as exc:
            click.echo(f" FAILED: {exc}", err=True)
            move_to_failed(chunk_path, failed_dir)
            state.failed_chunks.append(chunk_path.name)
            save_state(out_dir, state)
