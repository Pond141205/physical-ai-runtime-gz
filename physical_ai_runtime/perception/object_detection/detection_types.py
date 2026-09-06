from dataclasses import dataclass
from typing import Tuple


@dataclass
class Detection:
    object_id: str
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]
