import json
from pathlib import Path

from ytcc.models import Segment


def append_segments(out_dir: Path, segments: list[Segment]) -> None:
    """Append segments to segments.jsonl, the source of truth for transcript.md."""
    with (out_dir / "segments.jsonl").open("a", encoding="utf-8") as f:
        for seg in segments:
            f.write(json.dumps(_record(seg), ensure_ascii=False) + "\n")
        f.flush()


def read_segments(out_dir: Path) -> list[Segment]:
    path = out_dir / "segments.jsonl"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        segments = [Segment(**json.loads(line)) for line in f if line.strip()]
    # Retried windows are appended after later ones, so order by time
    return sorted(segments, key=lambda s: s.start)


def write_segments(out_dir: Path, segments: list[Segment]) -> None:
    """Replace segments.jsonl with segments."""
    path = out_dir / "segments.jsonl"
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for seg in segments:
            f.write(json.dumps(_record(seg), ensure_ascii=False) + "\n")
    tmp.replace(path)


def render_transcript(out_dir: Path, header: str = "") -> None:
    """Rewrite transcript.md from segments.jsonl."""
    lines = []
    for seg in read_segments(out_dir):
        speaker = f"{seg.speaker}: " if seg.speaker else ""
        lines.append(f"[{fmt_time(seg.start)}] {speaker}{seg.text}\n")
    path = out_dir / "transcript.md"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(header + "".join(lines), encoding="utf-8")
    tmp.replace(path)


def _record(seg: Segment) -> dict:
    record = {"start": round(seg.start, 3), "end": round(seg.end, 3), "text": seg.text}
    if seg.speaker:
        record["speaker"] = seg.speaker
    return record


def fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
