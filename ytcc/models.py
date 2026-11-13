from dataclasses import dataclass


@dataclass
class Segment:
    start: float  # global timestamp in seconds
    end: float    # global timestamp in seconds
    text: str
    speaker: str | None = None  # set only when tracks are merged, e.g. "Me" / "Others"
    # Whisper's own confidence in the segment, for filtering hallucinations.
    # Set by the ASR backends only; not stored in segments.jsonl.
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None
