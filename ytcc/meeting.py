import difflib
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ytcc.models import Segment
from ytcc.storage.transcript_writer import fmt_time, read_segments, render_transcript, write_segments

MEETING_FILE = "meeting.json"
ME = "Me"
OTHERS = "Others"

# A "Me" segment is treated as echo when this share of its text also appears in
# overlapping "Others" speech (speakers playing the call into the microphone).
_ECHO_MATCH = 0.6
_ECHO_SLACK_SEC = 1.0


@dataclass
class Meeting:
    title: str
    started_at: str  # ISO 8601, local time
    tracks: dict[str, str]  # speaker label ("" when unlabeled) → track dir, relative to the meeting dir


def load_meeting(out_dir: Path) -> Meeting | None:
    path = out_dir / MEETING_FILE
    if not path.exists():
        return None
    return Meeting(**json.loads(path.read_text(encoding="utf-8")))


def save_meeting(out_dir: Path, meeting: Meeting) -> None:
    (out_dir / MEETING_FILE).write_text(
        json.dumps(meeting.__dict__, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def recorded_at(path: Path) -> datetime:
    """Best guess at when a local recording started: its creation time."""
    st = path.stat()
    return datetime.fromtimestamp(getattr(st, "st_birthtime", st.st_mtime))


def merge(out_dir: Path, meeting: Meeting, duration: float) -> None:
    """Merge track transcripts into the meeting's segments.jsonl and transcript.md."""
    by_speaker = {}
    for label, track_dir in meeting.tracks.items():
        segments = read_segments(out_dir / track_dir)
        for seg in segments:
            seg.speaker = label or None
        by_speaker[label] = segments

    if ME in by_speaker and OTHERS in by_speaker:
        by_speaker[ME] = drop_echo(by_speaker[ME], by_speaker[OTHERS])

    merged = sorted((s for segs in by_speaker.values() for s in segs), key=lambda s: s.start)
    write_segments(out_dir, merged)
    render_transcript(out_dir, header=_header(meeting, duration))


def drop_echo(mine: list[Segment], others: list[Segment]) -> list[Segment]:
    """Remove "Me" segments that repeat what "Others" said at the same time."""
    kept = []
    for seg in mine:
        overlapping = [
            o for o in others
            if o.start < seg.end + _ECHO_SLACK_SEC and o.end > seg.start - _ECHO_SLACK_SEC
        ]
        if not overlapping or _shared(seg.text, " ".join(o.text for o in overlapping)) < _ECHO_MATCH:
            kept.append(seg)
    return kept


def _shared(text: str, reference: str) -> float:
    """Share of text's characters that also appear, in order, in reference."""
    a, b = _normalize(text), _normalize(reference)
    if not a:
        return 1.0
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return sum(block.size for block in matcher.get_matching_blocks()) / len(a)


def _normalize(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.lower())


def _header(meeting: Meeting, duration: float) -> str:
    lines = [
        f"# {meeting.title}",
        "",
        f"- Started: {meeting.started_at}",
        f"- Duration: {fmt_time(duration)}",
    ]
    if ME in meeting.tracks:
        lines.append(f"- Speakers: {ME} = recording microphone; {OTHERS} = everyone else in the call")
    return "\n".join(lines) + "\n\n"


def summarize(out_dir: Path, model: str | None = None, glossary: list[str] = ()) -> Path:
    """
    Write summary.md from transcript.md with Claude Code in print mode.

    glossary entries ("term" or "term — note") let Claude correct names and
    terms the transcript got wrong; unlike the Whisper prompt it has no size limit.
    """
    claude = shutil.which("claude")
    if claude is None:
        raise RuntimeError("`claude` is not on PATH; cannot summarize.")
    prompt = (Path(__file__).parent / "prompts" / "summary.md").read_text(encoding="utf-8")
    transcript = (out_dir / "transcript.md").read_text(encoding="utf-8")
    if glossary:
        prompt += "\n## Glossary\n\n" + "".join(f"- {entry}\n" for entry in glossary)

    cmd = [claude, "-p", prompt, "--tools", "", "--no-session-persistence"]
    if model:
        cmd += ["--model", model]
    result = subprocess.run(cmd, input=transcript, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"claude failed:\n{result.stderr or result.stdout}")

    path = out_dir / "summary.md"
    path.write_text(f"# Summary: {meeting_title(out_dir)}\n\n{result.stdout.strip()}\n", encoding="utf-8")
    return path


def meeting_title(out_dir: Path) -> str:
    meeting = load_meeting(out_dir)
    return meeting.title if meeting else out_dir.name
