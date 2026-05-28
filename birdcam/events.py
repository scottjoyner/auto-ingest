from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional
from birdcam.models import Detection

@dataclass
class EventState:
    id: str
    start: datetime
    last_seen: datetime
    detections: list[Detection] = field(default_factory=list)

class EventManager:
    def __init__(self, merge_seconds: int, cooldown_seconds: int, persistence_frames: int = 1):
        self.merge_window = timedelta(seconds=merge_seconds)
        self.cooldown = timedelta(seconds=cooldown_seconds)
        self.persistence_frames = persistence_frames
        self.current: EventState | None = None
        self.last_event_end: datetime | None = None
        self.pending = 0

    def update(self, now: datetime, detections: list[Detection]) -> tuple[str, EventState | None]:
        has_bird = len(detections) > 0
        if has_bird:
            self.pending += 1
        else:
            self.pending = 0

        if self.current is None and has_bird and self.pending >= self.persistence_frames:
            if self.last_event_end and now - self.last_event_end < self.cooldown:
                return "cooldown", None
            self.current = EventState(id=now.strftime("evt-%Y%m%d%H%M%S%f"), start=now, last_seen=now, detections=[])

        if self.current and has_bird:
            self.current.last_seen = now
            self.current.detections.extend(detections)
            return "active", self.current

        if self.current and (now - self.current.last_seen) > self.merge_window:
            done = self.current
            self.last_event_end = now
            self.current = None
            return "finalized", done
        return "idle", None
