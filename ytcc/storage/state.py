import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class State:
    source: str
    total_chunks: int
    chunk_duration_sec: int
    processed_chunks: list[str]
    failed_chunks: list[str]


def load_state(out_dir: Path) -> State | None:
    state_file = out_dir / "state.json"
    if not state_file.exists():
        return None
    data = json.loads(state_file.read_text(encoding="utf-8"))
    return State(**data)


def save_state(out_dir: Path, state: State) -> None:
    state_file = out_dir / "state.json"
    tmp = state_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(state), indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(state_file)  # atomic rename — avoids partial writes
