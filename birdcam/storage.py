from __future__ import annotations
from datetime import datetime
from pathlib import Path
import json
import cv2

class Storage:
    def __init__(self, root: str):
        self.root = Path(root)

    def _day_dir(self, kind: str, ts: datetime):
        d = self.root / kind / ts.strftime("%Y/%m/%d")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def event_paths(self, event_id: str, ts: datetime):
        return (
            self._day_dir("clips", ts) / f"{event_id}.mp4",
            self._day_dir("thumbs", ts) / f"{event_id}.jpg",
            self._day_dir("metadata", ts) / f"{event_id}.json",
        )

    def write_metadata(self, path: Path, data: dict):
        path.write_text(json.dumps(data, indent=2))

    def write_thumbnail(self, path: Path, frame):
        cv2.imwrite(str(path), frame)
