import numpy as np
from faster_whisper import WhisperModel

from ytcc.asr.base import ASRBackend
from ytcc.models import Segment


class FasterWhisperBackend(ASRBackend):
    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = None,
        prompt: str | None = None,
    ) -> None:
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._language = language
        self.prompt = prompt

    def transcribe(self, audio: np.ndarray) -> list[Segment]:
        segments, _ = self._model.transcribe(
            audio,
            language=self._language,
            vad_filter=True,  # skip silent regions
            # hotwords is prepended to every 30 s decode; initial_prompt would
            # only reach the first one once the prompt resets
            hotwords=self.prompt,
        )
        return [
            Segment(start=s.start, end=s.end, text=s.text.strip())
            for s in segments
            if s.text.strip()
        ]

    def count_tokens(self, text: str) -> int:
        # faster-whisper encodes hotwords as " " + hotwords.strip()
        return len(self._model.hf_tokenizer.encode(" " + text.strip(), add_special_tokens=False).ids)
