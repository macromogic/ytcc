from pathlib import Path

from faster_whisper import WhisperModel

from ytcc.asr.base import ASRBackend
from ytcc.models import Segment


class FasterWhisperBackend(ASRBackend):
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = None,
    ) -> None:
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._language = language

    def transcribe(self, audio_path: Path) -> list[Segment]:
        segments, _ = self._model.transcribe(
            str(audio_path),
            language=self._language,
            vad_filter=True,  # skip silent regions
        )
        return [
            Segment(start=s.start, end=s.end, text=s.text.strip())
            for s in segments
            if s.text.strip()
        ]
