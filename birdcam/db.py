from __future__ import annotations
import sqlite3
from datetime import datetime
from pathlib import Path
from birdcam.models import EventSummary, Detection

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  event_id TEXT PRIMARY KEY,
  camera_id TEXT,
  start_time TEXT,
  end_time TEXT,
  start_epoch_ms INTEGER,
  end_epoch_ms INTEGER,
  confidence_max REAL,
  confidence_avg REAL,
  detection_count INTEGER,
  clip_path TEXT,
  thumbnail_path TEXT,
  metadata_path TEXT,
  protected INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS detections (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id TEXT,
  ts TEXT,
  confidence REAL,
  label TEXT,
  bbox TEXT
);
CREATE TABLE IF NOT EXISTS clips (
  event_id TEXT PRIMARY KEY,
  clip_path TEXT,
  bytes INTEGER,
  created_at TEXT
);
"""

class DB:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def add_event(self, event: EventSummary):
        c = self.conn.cursor()
        c.execute("INSERT OR REPLACE INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            event.event_id, event.camera_id, event.start_time.isoformat(), event.end_time.isoformat(),
            int(event.start_time.timestamp()*1000), int(event.end_time.timestamp()*1000),
            event.confidence_max, event.confidence_avg, event.detection_count, event.clip_path,
            event.thumbnail_path, event.metadata_path, int(event.protected)
        ))
        for d in event.detections:
            c.execute("INSERT INTO detections(event_id,ts,confidence,label,bbox) VALUES (?,?,?,?,?)", (
                event.event_id, d.timestamp.isoformat(), d.confidence, d.label, str(d.bbox)
            ))
        c.execute("INSERT OR REPLACE INTO clips(event_id,clip_path,bytes,created_at) VALUES(?,?,?,?)", (
            event.event_id, event.clip_path, Path(event.clip_path).stat().st_size if Path(event.clip_path).exists() else 0,
            datetime.utcnow().isoformat()
        ))
        self.conn.commit()
