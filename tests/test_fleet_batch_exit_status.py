"""Fleet batch fan-out must not report success when tasks were not created.

A date-root batch is the hand-off from auto-ingest to AssistX: the operator (or a
scheduler) reads the exit status to decide whether the work was queued. Returning
0 after every POST failed made a completely unqueued batch look successful, so
the failure was only visible in a stderr line nobody gates on.
"""
from __future__ import annotations

import argparse

from auto_ingest import fleet_batch


def _args(**overrides):
    values = {
        "assistx_url": "http://assistx.invalid",
        "auth_user": "admin",
        "auth_pass": None,
        "items": ["clip_a.mp4", "clip_b.mp4"],
        "input": None,
        "glob": "*",
        "capabilities": "yolo",
        "command": None,
        "yolo_command": None,
        "title_prefix": "batch",
        "batch_id": None,
        "priority": "background",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_run_batch_reports_total_failure(monkeypatch, capsys):
    """Every POST rejected -> non-zero status, not a green 'done'."""

    def fail(*_a, **_k):
        raise RuntimeError("assistx unavailable")

    monkeypatch.setattr(fleet_batch, "create_task", fail)

    rc = fleet_batch.run_batch(_args())

    assert rc == 1
    assert "0/2 tasks created" in capsys.readouterr().err


def test_run_batch_reports_partial_failure(monkeypatch):
    """A dropped item must also fail the batch, not just a total outage."""
    attempts = []

    def flaky(*a, **_k):
        attempts.append(a[2])
        if len(attempts) == 2:
            raise RuntimeError("500 from AssistX")
        return {"task_id": "t-1"}

    monkeypatch.setattr(fleet_batch, "create_task", flaky)

    assert fleet_batch.run_batch(_args()) == 1
    assert len(attempts) == 2  # one bad item must not abort the rest of the batch


def test_run_batch_still_succeeds_when_all_created(monkeypatch):
    """Guard against over-correcting: a fully queued batch stays green."""
    monkeypatch.setattr(
        fleet_batch, "create_task", lambda *_a, **_k: {"task_id": "t-1"}
    )
    assert fleet_batch.run_batch(_args()) == 0
