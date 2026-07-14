from datetime import datetime, timedelta
from pathlib import Path

import pytest

pytest.importorskip("cv2")
import numpy as np
import cv2

from birdcam.config import load_settings
from birdcam.events import EventManager
from birdcam.db import DB
from birdcam.models import Detection, EventSummary
from birdcam.retention import apply_retention


def test_config_loading(tmp_path):
    cfg = tmp_path / 'c.yaml'
    cfg.write_text('camera_id: c1\nstream_url: f.mp4\nstorage_root: /tmp/root\n')
    s = load_settings(str(cfg))
    assert s.camera_id == 'c1'


def test_event_debounce_merge():
    mgr = EventManager(merge_seconds=1, cooldown_seconds=1, persistence_frames=2)
    now = datetime.utcnow()
    d = [Detection('bird', 0.9, (0,0,1,1), now)]
    assert mgr.update(now, d)[0] == 'idle'
    assert mgr.update(now + timedelta(milliseconds=20), d)[0] == 'active'


def test_db_insert(tmp_path):
    db = DB(str(tmp_path / 'a.sqlite'))
    now = datetime.utcnow()
    e = EventSummary('e1','c1',now,now,0.9,0.8,1,'/tmp/a.mp4','/tmp/a.jpg','/tmp/a.json')
    db.add_event(e)
    row = db.conn.execute('select event_id from events where event_id="e1"').fetchone()
    assert row[0] == 'e1'


def test_retention_order(tmp_path):
    cdir = tmp_path / 'clips/2026/01/01'; cdir.mkdir(parents=True)
    p1 = cdir / 'old.mp4'; p1.write_bytes(b'a'*100)
    p2 = cdir / 'new.mp4'; p2.write_bytes(b'a'*100)
    removed = apply_retention(str(tmp_path), max_gb=0.0000001, max_age_days=999)
    assert removed


def test_roi_crop_shape():
    frame = np.zeros((100,200,3), dtype=np.uint8)
    roi = frame[10:40,20:70]
    assert roi.shape == (30,50,3)


def test_mock_file_mode_path(tmp_path):
    out = tmp_path / 'x.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w = cv2.VideoWriter(str(out), fourcc, 5, (32,32))
    for _ in range(3):
        w.write(np.zeros((32,32,3), dtype=np.uint8))
    w.release()
    assert out.exists()
