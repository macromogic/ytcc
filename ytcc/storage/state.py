import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Options:
    """Transcription settings, stored so `resume` continues with the same ones."""

    model: str
    device: str
    compute_type: str
    language: str | None
    prompt: str | None
    window_sec: int


@dataclass
class Window:
    start: float
    end: float
    status: str  # "done" or "failed"
    segments: int = 0
    error: str | None = None


@dataclass
class State:
    source: str  # what the user passed: file path or URL
    audio: str  # local file that is decoded (the source, or audio downloaded from it)
    duration: float
    options: Options
    position: float = 0.0  # seconds transcribed (or attempted) so far
    windows: list[Window] = field(default_factory=list)

    @property
    def failed(self) -> list[Window]:
        return [w for w in self.windows if w.status == "failed"]

    @property
    def complete(self) -> bool:
        return self.position >= self.duration and not self.failed


def load_state(out_dir: Path) -> State | None:
    state_file = out_dir / "state.json"
    if not state_file.exists():
        return None
    data = json.loads(state_file.read_text(encoding="utf-8"))
    if "total_chunks" in data:
        raise ValueError(
            f"{state_file} was written by an older chunk-based ytcc and cannot be resumed; "
            "transcribe again into a new directory."
        )
    data["options"] = Options(**data["options"])
    data["windows"] = [Window(**w) for w in data["windows"]]
    return State(**data)


def save_state(out_dir: Path, state: State) -> None:
    state_file = out_dir / "state.json"
    tmp = state_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(state), indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(state_file)  # atomic rename — avoids partial writes
