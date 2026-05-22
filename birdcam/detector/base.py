from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
import numpy as np
from birdcam.models import Detection

class Detector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray):
        raise NotImplementedError
