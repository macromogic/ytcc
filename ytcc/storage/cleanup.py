import shutil
from pathlib import Path


def delete_chunk(chunk_path: Path) -> None:
    chunk_path.unlink(missing_ok=True)


def move_to_failed(chunk_path: Path, failed_dir: Path) -> None:
    failed_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(chunk_path), failed_dir / chunk_path.name)
