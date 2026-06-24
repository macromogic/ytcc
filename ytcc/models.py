from dataclasses import dataclass


@dataclass
class Segment:
    start: float  # global timestamp in seconds
    end: float    # global timestamp in seconds
    text: str
