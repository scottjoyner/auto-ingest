"""Acceptance gate must fail closed when a READY task has zero checks (sweep39x).

The READY->Run path hydrates ``acceptance_json``; malformed JSON or a missing
list hydrates to ``[]``.  ``run_acceptance`` then returns an empty result list and
the old ``all(...) if results else True`` marked the run and the task DONE without
a single check.  These tests pin the contract with only the Neo4j boundary stubbed:
the real hydration, executor and acceptance helpers still run, and every artifact
is written under pytest's tmp_path.
"""

from __future__ import annotations

import os

# The module reads ARTIFACTS_DIR at import time; never let it default to ./artifacts.
os.environ.setdefault(
    "ARTIFACTS_DIR", os.path.join(os.environ.get("TMP", "/tmp"), "sweep39x-artifacts")
)

import summarize_from_segments as sfs  # noqa: E402

ORIG_GET_TASK = sfs.q_get_task


class _Result:
    def __init__(self, rec):
        self._rec = rec

    def single(self):
        return self._rec


class _Session:
    def __init__(self, rec):
        self._rec = rec

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, *args, **kwargs):
        return _Result(self._rec)


class _Driver:
    def __init__(self, rec):
        self._rec = rec

    def session(self, *args, **kwargs):
        return _Session(self._rec)


def _execute(monkeypatch, tmp_path, raw_task):
    """Run the real execute_ready decision path with a stubbed Neo4j boundary."""
    rec = {"task": raw_task, "s": None, "t": None}
    monkeypatch.setattr(sfs, "ARTIFACTS_DIR", str(tmp_path))

    # Real hydration of acceptance_json (including the malformed-JSON fallback).
    hydrated = ORIG_GET_TASK(_Session(rec), raw_task["id"])["task"]

    captured = {}

    def fake_driver():
        return _Driver(rec)

    def fake_pick(session, limit):
        return [raw_task["id"]]

    def fake_get_task(session, task_id):
        return {"task": hydrated, "summary": None, "transcription": None}

    def fake_attach(session, task_id, run):
        captured["attached"] = run

    def fake_finish(session, task_id, run_id, success, manifest):
        captured["success"] = success
        captured["status"] = "DONE" if success else "FAILED"
        captured["results"] = manifest["acceptance_results"]

    monkeypatch.setattr(sfs, "neo_driver", fake_driver)
    monkeypatch.setattr(sfs, "q_pick_ready", fake_pick)
    monkeypatch.setattr(sfs, "q_get_task", fake_get_task)
    monkeypatch.setattr(sfs, "q_attach_run", fake_attach)
    monkeypatch.setattr(sfs, "q_finish_run", fake_finish)

    sfs.execute_ready(1)
    return captured


def test_malformed_acceptance_json_cannot_mark_run_done(monkeypatch, tmp_path):
    """DEFECT: malformed acceptance_json hydrates to [] and used to pass vacuously."""
    out = _execute(
        monkeypatch,
        tmp_path,
        {
            "id": "t-malformed",
            "title": "x",
            "acceptance_json": "{not-valid-json",
        },
    )
    assert out["results"] == []
    assert out["success"] is False
    assert out["status"] == "FAILED"


def test_missing_acceptance_list_cannot_mark_run_done(monkeypatch, tmp_path):
    out = _execute(monkeypatch, tmp_path, {"id": "t-absent", "title": "x"})
    assert out["results"] == []
    assert out["success"] is False
    assert out["status"] == "FAILED"


def test_explicitly_empty_acceptance_list_cannot_mark_run_done(monkeypatch, tmp_path):
    out = _execute(monkeypatch, tmp_path, {"id": "t-empty", "title": "x", "acceptance": []})
    assert out["results"] == []
    assert out["success"] is False
    assert out["status"] == "FAILED"


def test_single_passing_check_still_marks_run_done(monkeypatch, tmp_path):
    out = _execute(
        monkeypatch,
        tmp_path,
        {
            "id": "t-pass",
            "title": "x",
            "acceptance": [
                {"type": "file_exists", "args": {"path": "artifacts/{TASK_ID}/output.txt"}}
            ],
        },
    )
    assert len(out["results"]) == 1
    assert out["results"][0]["passed"] is True
    assert out["success"] is True
    assert out["status"] == "DONE"


def test_single_failing_check_still_marks_run_failed(monkeypatch, tmp_path):
    out = _execute(
        monkeypatch,
        tmp_path,
        {
            "id": "t-fail",
            "title": "x",
            "acceptance": [
                {"type": "file_exists", "args": {"path": "artifacts/{TASK_ID}/absent.txt"}}
            ],
        },
    )
    assert len(out["results"]) == 1
    assert out["results"][0]["passed"] is False
    assert out["success"] is False
    assert out["status"] == "FAILED"


def test_unknown_acceptance_type_still_fails(monkeypatch, tmp_path):
    out = _execute(
        monkeypatch,
        tmp_path,
        {
            "id": "t-unknown",
            "title": "x",
            "acceptance": [{"type": "not_a_real_check", "args": {}}],
        },
    )
    assert out["results"][0]["passed"] is False
    assert out["success"] is False
    assert out["status"] == "FAILED"
