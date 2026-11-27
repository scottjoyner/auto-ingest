"""Re-adding one birdcam event must not duplicate or retain stale detections.

Hermetic: only a tiny throwaway SQLite file under pytest ``tmp_path``. No
camera, OpenCV, Neo4j, network, service, credential or live-file IO.
"""
from datetime import datetime, timezone

from birdcam.db import DB
from birdcam.models import Detection, EventSummary


def _event(event_id, detections, *, label="bird"):
    ts = datetime(2026, 11, 27, 12, 0, 0, tzinfo=timezone.utc)
    return EventSummary(
        event_id, "cam1", ts, ts, 0.9, 0.85, len(detections),
        "/nonexistent/clip.mp4", "/nonexistent/thumb.jpg",
        "/nonexistent/meta.json", detections=detections,
    )


def _dets(*labels):
    ts = datetime(2026, 11, 27, 12, 0, 0, tzinfo=timezone.utc)
    return [Detection(label, 0.9, (0, 0, 1, 1), ts) for label in labels]


def _detection_rows(db, event_id):
    return db.conn.execute(
        "SELECT label FROM detections WHERE event_id=? ORDER BY id", (event_id,)
    ).fetchall()


def test_readd_does_not_duplicate_detections(tmp_path):
    """Defect: INSERT OR REPLACE of the event left the old detection rows behind."""
    db = DB(str(tmp_path / "a.sqlite"))
    event = _event("e1", _dets("a", "b"))
    db.add_event(event)
    assert len(_detection_rows(db, "e1")) == 2
    db.add_event(event)  # reprocessed / replayed event with the same id
    assert len(_detection_rows(db, "e1")) == 2


def test_readd_replaces_stale_detection_payloads(tmp_path):
    """Defect: a corrected re-add kept both the stale and the new detection."""
    db = DB(str(tmp_path / "b.sqlite"))
    db.add_event(_event("e2", _dets("old")))
    db.add_event(_event("e2", _dets("new")))
    assert [r[0] for r in _detection_rows(db, "e2")] == ["new"]


def test_first_add_stores_every_detection(tmp_path):
    """Control: untouched first-add behavior."""
    db = DB(str(tmp_path / "c.sqlite"))
    db.add_event(_event("e3", _dets("a", "b", "c")))
    assert [r[0] for r in _detection_rows(db, "e3")] == ["a", "b", "c"]


def test_readd_of_one_event_leaves_other_events_alone(tmp_path):
    """Control: the cleanup must be scoped to the re-added event id only."""
    db = DB(str(tmp_path / "d.sqlite"))
    db.add_event(_event("e4", _dets("keep")))
    db.add_event(_event("e5", _dets("one", "two")))
    db.add_event(_event("e5", _dets("three")))
    assert [r[0] for r in _detection_rows(db, "e4")] == ["keep"]
    assert [r[0] for r in _detection_rows(db, "e5")] == ["three"]


def test_event_and_clip_rows_stay_singular_on_readd(tmp_path):
    """Control: those tables already used INSERT OR REPLACE and stay unchanged."""
    db = DB(str(tmp_path / "e.sqlite"))
    event = _event("e6", _dets("a"))
    db.add_event(event)
    db.add_event(event)
    assert db.conn.execute("SELECT count(*) FROM events").fetchone()[0] == 1
    assert db.conn.execute("SELECT count(*) FROM clips").fetchone()[0] == 1
