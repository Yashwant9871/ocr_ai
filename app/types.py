from dataclasses import dataclass
from typing import Tuple

BBox = Tuple[int, int, int, int]


@dataclass
class Detection:
    bbox: BBox
    confidence: float
    class_id: int
