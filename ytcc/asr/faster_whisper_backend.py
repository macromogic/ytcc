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
        prompt: str | None = None,
    ) -> None:
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.prompt = prompt

    def transcribe(self, audio: np.ndarray, language: str | None) -> list[Segment]:
        segments, _ = self._model.transcribe(
            audio,
            language=language,
            # No vad_filter: the pipeline passes speech only (audio.speech_only)
            # hotwords is prepended to every 30 s decode; initial_prompt would
            # only reach the first one once the prompt resets
            hotwords=self.prompt,
            # Off, as in the mlx backend. On, faster-whisper puts up to 223
            # tokens of previous text after the hotwords, which with a full
            # glossary leaves almost none of the 448-token context for new
            # text; conditioning is also what sends Whisper into repetition loops.
            condition_on_previous_text=False,
        )
        return [
            Segment(
                start=s.start,
                end=s.end,
                text=s.text.strip(),
                avg_logprob=s.avg_logprob,
                no_speech_prob=s.no_speech_prob,
                compression_ratio=s.compression_ratio,
            )
            for s in segments
            if s.text.strip()
        ]

    def detect_language(self, audio: np.ndarray) -> dict[str, float]:
        _, _, probabilities = self._model.detect_language(audio)
        return dict(probabilities)

    def count_tokens(self, text: str) -> int:
        # faster-whisper encodes hotwords as " " + hotwords.strip()
        return len(self._model.hf_tokenizer.encode(" " + text.strip(), add_special_tokens=False).ids)
