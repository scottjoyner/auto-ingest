"""Regression tests for birdcam EventManager merge-window boundaries (r37).

Pure, hermetic: fixed datetimes and Detection instances, no clock, DB, camera,
file or network IO. A detection arriving more than ``merge_seconds`` after the
event's last sighting is a new visit and must finalize the prior event rather
than being folded into it.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from birdcam.events import EventManager
from birdcam.models import Detection

T0 = datetime(2026, 11, 27, 0, 0, 0)


def _det(t: datetime) -> Detection:
    return Detection("bird", 0.9, (0, 0, 10, 10), t)


def _at(seconds: int) -> datetime:
    return T0 + timedelta(seconds=seconds)


def test_gap_beyond_merge_window_finalizes_prior_event():
    """DEFECT: a 19 s gap (window 8 s) must not extend the same event."""
    mgr = EventManager(merge_seconds=8, cooldown_seconds=10, persistence_frames=1)
    _, first = mgr.update(_at(0), [_det(_at(0))])
    _, second = mgr.update(_at(1), [_det(_at(1))])
    status, done = mgr.update(_at(20), [_det(_at(20))])

    assert first is not None and second is not None
    assert second.id == first.id  # within-window sightings still merge
    assert status == "finalized", "gap past merge window must finalize the old event"
    assert done is not None and done.id == first.id
    assert len(done.detections) == 2, "distant detection must not be folded in"
    assert done.detections[-1].timestamp == _at(1)


def test_within_window_detections_still_merge_control():
    """Control: sightings inside the merge window remain one active event."""
    mgr = EventManager(merge_seconds=8, cooldown_seconds=10, persistence_frames=1)
    _, first = mgr.update(_at(0), [_det(_at(0))])
    status, active = mgr.update(_at(5), [_det(_at(5))])

    assert status == "active"
    assert active is not None and active.id == first.id
    assert len(active.detections) == 2


def test_empty_tick_past_window_finalizes_control():
    """Control: the existing empty-tick finalization path is unchanged."""
    mgr = EventManager(merge_seconds=8, cooldown_seconds=10, persistence_frames=1)
    _, first = mgr.update(_at(0), [_det(_at(0))])
    status, done = mgr.update(_at(9), [])

    assert status == "finalized"
    assert done is not None and done.id == first.id
    assert len(done.detections) == 1


def test_persistence_frames_gate_control():
    """Control: the first event still waits for the persistence frames."""
    mgr = EventManager(merge_seconds=1, cooldown_seconds=1, persistence_frames=2)
    now = T0 + timedelta(milliseconds=20)
    assert mgr.update(T0, [_det(T0)])[0] == "idle"
    assert mgr.update(now, [_det(now)])[0] == "active"


def test_cooldown_applies_after_gap_finalization():
    """A fresh sighting just after a gap finalization is still in cooldown."""
    mgr = EventManager(merge_seconds=8, cooldown_seconds=10, persistence_frames=1)
    mgr.update(_at(0), [_det(_at(0))])
    assert mgr.update(_at(20), [_det(_at(20))])[0] == "finalized"
    status, event = mgr.update(_at(21), [_det(_at(21))])
    assert status == "cooldown"
    assert event is None
