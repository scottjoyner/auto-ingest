from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

BBox = Tuple[int, int, int, int]

@dataclass
class Detection:
    label: str
    confidence: float
    bbox: BBox
    timestamp: datetime

@dataclass
class EventSummary:
    event_id: str
    camera_id: str
    start_time: datetime
    end_time: datetime
    confidence_max: float
    confidence_avg: float
    detection_count: int
    clip_path: str
    thumbnail_path: str
    metadata_path: str
    protected: bool = False
    detections: List[Detection] = field(default_factory=list)

