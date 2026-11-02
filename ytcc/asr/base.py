from abc import ABC, abstractmethod

import numpy as np

from ytcc.models import Segment

# Whisper's decoder context is 448 tokens; the prompt may use at most half of it
# (minus the start-of-previous token). Longer prompts are cut silently.
PROMPT_TOKENS = 448 // 2 - 1


class ASRBackend(ABC):
    prompt: str | None = None  # Whisper prompt (glossary terms etc.) for every decode

    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> list[Segment]:
        """
        Transcribe 16 kHz mono float32 audio.

        Returned timestamps are window-local (start from 0).
        """
        ...

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Number of tokens text takes as a Whisper prompt."""
        ...
