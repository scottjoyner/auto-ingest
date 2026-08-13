from __future__ import annotations

import argparse
import importlib
import sys
from types import SimpleNamespace

import pytest

from auto_ingest import fleet_batch


def test_event_fallback_envelope_and_emit(monkeypatch):
    # Force the standalone mirror path even if assistx happens to be installed.
    for name in list(sys.modules):
        if name == "auto_ingest.events" or name.startswith("assistx"):
            sys.modules.pop(name, None)
    monkeypatch.setitem(sys.modules, "assistx", None)
    events = importlib.import_module("auto_ingest.events")
    assert events._USING_CANONICAL is False

    envelope = events.EventEnvelope(
        schema_version="v1",
        source_repo="repo",
        event_type="thing.happened",
        payload={"x": 1},
        links=[{"kind": "evidence"}],
    )
    data = envelope.to_dict()
    assert data["schema_version"] == "v1"
    assert data["source_repo"] == "repo"
    assert data["event_type"] == "thing.happened"
    assert data["payload"] == {"x": 1}
    assert data["links"] == [{"kind": "evidence"}]
    assert data["correlation_id"]
    assert data["ts"].endswith("+00:00")

    emitted = events.emit("ingest.done", {"ok": True})
    assert emitted.source_repo == events.SOURCE_REPO
    assert emitted.schema_version == events.SCHEMA_VERSION
    assert emitted.payload == {"ok": True}

    with pytest.raises(ValueError, match="valid UUID"):
        events.EventEnvelope("v1", "repo", "bad", correlation_id="not-a-uuid")


def _args(**overrides):
    values = {
        "items": [],
        "input": None,
        "glob": "*",
        "auth_user": "admin",
        "auth_pass": "",
        "capabilities": "script, media",
        "assistx_url": "http://assistx:8000",
        "command": None,
        "yolo_command": None,
        "title_prefix": "batch",
        "batch_id": None,
        "priority": "background",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_build_items_deduplicates_and_scans(tmp_path):
    (tmp_path / "b.mp4").write_text("b", encoding="utf-8")
    (tmp_path / "a.mp4").write_text("a", encoding="utf-8")
    args = _args(
        items=[str(tmp_path / "a.mp4"), "explicit"],
        input=str(tmp_path),
        glob="*.mp4",
    )
    assert fleet_batch.build_items(args) == [
        str(tmp_path / "a.mp4"),
        "explicit",
        str(tmp_path / "b.mp4"),
    ]


def test_create_task_builds_expected_http_payload(monkeypatch):
    seen = {}

    class Response:
        def raise_for_status(self):
            seen["raised"] = True

        def json(self):
            return {"task_id": "t1"}

    class Requests:
        @staticmethod
        def post(url, json, auth, timeout):
            seen.update(url=url, json=json, auth=auth, timeout=timeout)
            return Response()

    monkeypatch.setattr(fleet_batch, "requests", Requests)
    result = fleet_batch.create_task(
        "http://assistx:8000",
        ("u", "p"),
        "title",
        ["yolo"],
        command="run x",
        yolo_command="vision x",
        payload={"item": "x"},
        correlation_id="cid",
        priority="high",
        task_type="custom",
    )
    assert result == {"task_id": "t1"}
    assert seen["url"].endswith("/api/tasks")
    assert seen["auth"] == ("u", "p")
    assert seen["timeout"] == 30
    assert seen["json"]["status"] == "READY"
    assert seen["json"]["required_capabilities"] == ["yolo"]
    assert seen["json"]["payload"]["command"] == "run x"
    assert seen["json"]["payload"]["yolo_command"] == "vision x"
    assert seen["json"]["correlation_id"] == "cid"
    assert seen["raised"] is True

    monkeypatch.setattr(fleet_batch, "requests", None)
    with pytest.raises(RuntimeError, match="requests is required"):
        fleet_batch.create_task("x", None, "t", [])


def test_run_batch_empty_success_and_partial_failure(monkeypatch, capsys):
    assert fleet_batch.run_batch(_args()) == 1
    assert "no items found" in capsys.readouterr().err

    created = []

    def fake_create(*args, **kwargs):
        item = kwargs["payload"]["item"]
        created.append((args, kwargs))
        if item.endswith("bad"):
            raise RuntimeError("boom")
        return {"task_id": "ok"}

    monkeypatch.setattr(fleet_batch, "create_task", fake_create)
    args = _args(
        items=["/tmp/good", "/tmp/bad"],
        auth_pass="secret",
        command="process {item}",
        yolo_command="detect {item}",
        title_prefix="ingest",
        batch_id="batch-1",
        priority="urgent",
    )
    assert fleet_batch.run_batch(args) == 0
    err = capsys.readouterr().err
    assert "created ok" in err
    assert "FAILED /tmp/bad" in err
    assert "done: 1/2" in err
    first_args, first_kwargs = created[0]
    assert first_args[1] == ("admin", "secret")
    assert first_args[2] == "ingest good"
    assert first_args[3] == ["script", "media"]
    assert first_kwargs["command"] == "process /tmp/good"
    assert first_kwargs["yolo_command"] == "detect /tmp/good"
    assert first_kwargs["payload"]["batch_id"] == "batch-1"
    assert first_kwargs["priority"] == "urgent"


def test_fleet_main_parses_and_exits(monkeypatch):
    monkeypatch.setattr(fleet_batch, "run_batch", lambda args: 7)
    monkeypatch.setattr(
        sys,
        "argv",
        ["fleet-batch", "--capabilities", "script", "--items", "a"],
    )
    with pytest.raises(SystemExit) as exc:
        fleet_batch.main()
    assert exc.value.code == 7
