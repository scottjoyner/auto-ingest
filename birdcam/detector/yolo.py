from __future__ import annotations
from datetime import datetime
import numpy as np
from birdcam.models import Detection
from birdcam.detector.base import Detector

class YoloDetector(Detector):
    def __init__(self, model_name_or_path: str, target_class: str = "bird", confidence: float = 0.4):
        self.target_class = target_class
        self.confidence = confidence
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_name_or_path)
        except Exception:
            self.model = None

    def detect(self, frame: np.ndarray):
        if self.model is None:
            return []
        out = []
        res = self.model(frame, verbose=False)
        for r in res:
            names = r.names
            for b in r.boxes:
                cls_name = names[int(b.cls.item())]
                conf = float(b.conf.item())
                if cls_name == self.target_class and conf >= self.confidence:
                    x1,y1,x2,y2 = [int(v) for v in b.xyxy[0].tolist()]
                    out.append(Detection(cls_name, conf, (x1,y1,x2,y2), datetime.utcnow()))
        return out
