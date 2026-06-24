import json
from pathlib import Path

from ytcc.models import Segment


def append_segments(out_dir: Path, segments: list[Segment], chunk_name: str) -> None:
    """Write segments to transcript.md and segments.jsonl, both flushed to disk."""
    _append_markdown(out_dir / "transcript.md", segments)
    _append_jsonl(out_dir / "segments.jsonl", segments, chunk_name)


def _append_markdown(path: Path, segments: list[Segment]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"[{_fmt(seg.start)}] {seg.text}\n")
        f.flush()


def _append_jsonl(path: Path, segments: list[Segment], chunk_name: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        for seg in segments:
            record = {
                "chunk": chunk_name,
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def _fmt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
