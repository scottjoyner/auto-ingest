from __future__ import annotations
from datetime import datetime
from typing import List
import numpy as np
from birdcam.models import Detection
from birdcam.detector.base import Detector

class MockDetector(Detector):
    def __init__(self, sequence: list[list[Detection]] | None = None):
        self.sequence = sequence or []
        self._i = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self._i < len(self.sequence):
            out = self.sequence[self._i]
            self._i += 1
            return out
        return []
