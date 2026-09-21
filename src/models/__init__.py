"""Typed records shared by detection and occupancy."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

type Frame = NDArray[np.uint8]


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int


@dataclass(frozen=True)
class Occupancy:
    spot_id: str
    occupied: bool
    coverage: float
