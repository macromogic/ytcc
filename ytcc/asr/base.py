from abc import ABC, abstractmethod
from pathlib import Path

from ytcc.models import Segment


class ASRBackend(ABC):
    @abstractmethod
    def transcribe(self, audio_path: Path) -> list[Segment]:
        """Transcribe audio file. Returned timestamps are chunk-local (start from 0)."""
        ...
