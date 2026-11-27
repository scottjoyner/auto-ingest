"""Exercise actual local executor output and offline receipt validation."""
import ast
import datetime
import json
import os
import pathlib
import time
import uuid
from contextlib import nullcontext
from typing import Any, Dict, Optional

import pytest

from auto_ingest.executor_evidence import (
    inspect_local_output,
    publish_local_output_receipt,
)


def executor(tmp_path, **extra):
    source = pathlib.Path(__file__).resolve().parents[1] / "summarize_from_segments.py"
    names = {"ensure_artifacts", "execute_task_minimal", "execute_ready"}
    fns = [n for n in ast.parse(source.read_text()).body
           if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = dict(ARTIFACTS_DIR=str(tmp_path), pathlib=pathlib, time=time,
              datetime=datetime, os=os, uuid=uuid, Dict=Dict, Any=Any,
              Optional=Optional) | extra
    exec(compile(ast.Module(body=fns, type_ignores=[]), str(source), "exec"), ns)
    return ns


def test_real_local_output_survives_unacknowledged_finalization(tmp_path):
    state = {"status": "READY", "owner": None}

    class Driver:
        def session(self):
            return nullcontext(object())

    def pick(session, limit):
        return ["task"] if state["status"] == "READY" else []

    def attach(session, tid, run):
        state.update(status="RUNNING", owner=run["id"])

    def finish(*args):
        raise OSError("finalization acknowledgement unavailable")

    ns = executor(tmp_path, neo_driver=Driver, q_pick_ready=pick,
                  q_get_task=lambda s, tid: {"task": {"id": tid, "title": "café"}},
                  q_attach_run=attach, q_finish_run=finish,
                  run_acceptance=lambda task, tid, *, execution_steps: [{"passed": True}])
    with pytest.raises(OSError):
        ns["execute_ready"](1)
    rid = state["owner"]
    directory = tmp_path / "task"
    output = (directory / "output.txt").read_bytes()
    receipt = (directory / f"run-{rid}.json").read_bytes()
    assert b"caf\xc3\xa9" in output
    assert inspect_local_output(receipt, output, task_id="task", run_id=rid) == {
        "classification": "LOCAL_OUTPUT_BYTES_MATCH", "replay_allowed": False}
    assert state["status"] == "RUNNING"
    ns["execute_ready"](1)
    assert (directory / "output.txt").read_bytes() == output
    assert len(list(directory.glob("run-*.json"))) == 1


def test_prior_receipt_cannot_match_replaced_output(tmp_path):
    ns = executor(tmp_path)
    first, second = str(uuid.uuid4()), str(uuid.uuid4())
    ns["execute_task_minimal"]({"id": "task"}, run_id=first)
    directory = tmp_path / "task"
    receipt = (directory / f"run-{first}.json").read_bytes()
    ns["execute_task_minimal"]({"id": "task"}, run_id=second)
    output = (directory / "output.txt").read_bytes()
    assert inspect_local_output(receipt, output, task_id="task", run_id=first) == {
        "classification": "LOCAL_OUTPUT_UNVERIFIED", "replay_allowed": False}
    preserved = (directory / f"run-{first}.txt").read_bytes()
    assert inspect_local_output(receipt, preserved, task_id="task", run_id=first)[
        "classification"] == "LOCAL_OUTPUT_BYTES_MATCH"
    assert (directory / f"run-{second}.txt").read_bytes() == output
    current = (directory / f"run-{second}.json").read_bytes()
    assert inspect_local_output(current, output, task_id="task", run_id=second)[
        "classification"] == "LOCAL_OUTPUT_BYTES_MATCH"


def test_create_only_and_tamper_rejection(tmp_path):
    rid = str(uuid.uuid4())
    output = f"local\nRUN: {rid}\n".encode()
    target = publish_local_output_receipt(tmp_path, task_id="task", run_id=rid,
                                          output=output)
    receipt = target.read_bytes()
    with pytest.raises(FileExistsError):
        publish_local_output_receipt(tmp_path, task_id="task", run_id=rid,
                                     output=b"replacement")
    assert target.read_bytes() == receipt
    assert list(tmp_path.iterdir()) == [target]
    invalid = [b"{", b"x" * 16385,
               receipt.replace(b'"schema":', b'"schema": "duplicate", "schema":'),
               json.dumps(dict(json.loads(receipt), bytes=True)).encode()]
    for data in invalid:
        assert inspect_local_output(data, output, task_id="task", run_id=rid)[
            "classification"] == "LOCAL_OUTPUT_UNVERIFIED"
    assert inspect_local_output(receipt, output+b"x", task_id="task", run_id=rid)[
        "classification"] == "LOCAL_OUTPUT_UNVERIFIED"
    assert inspect_local_output(receipt, output, task_id="other", run_id=rid)[
        "classification"] == "LOCAL_OUTPUT_UNVERIFIED"
    assert inspect_local_output(receipt, output, task_id="task",
                                run_id=str(uuid.uuid4()))["replay_allowed"] is False


def test_no_remote_or_completion_assertion(tmp_path):
    rid = str(uuid.uuid4())
    ns = executor(tmp_path)
    manifest = ns["execute_task_minimal"]({"id": "task"}, run_id=rid)
    receipt = json.loads((tmp_path / "task" / f"run-{rid}.json").read_bytes())
    assert receipt["scope"] == "LOCAL_OUTPUT_ONLY"
    assert "success" not in receipt and "status" not in receipt
    assert manifest["steps"][-1]["op"] == "local_output_receipt"


@pytest.mark.parametrize("rid", ["../escape", "not-uuid", ""])
def test_invalid_run_id_cannot_publish(tmp_path, rid):
    with pytest.raises(ValueError):
        publish_local_output_receipt(tmp_path, task_id="task", run_id=rid, output=b"")
    assert list(tmp_path.iterdir()) == []


def test_duplicate_run_preserves_existing_bytes_and_alias(tmp_path):
    ns = executor(tmp_path)
    rid = str(uuid.uuid4())
    ns["execute_task_minimal"]({"id": "task", "title": "first"}, run_id=rid)
    directory = tmp_path / "task"
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(FileExistsError):
        ns["execute_task_minimal"]({"id": "task", "title": "second"}, run_id=rid)
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


def test_preservation_failure_does_not_overwrite_alias(tmp_path, monkeypatch):
    from auto_ingest import executor_evidence
    ns = executor(tmp_path)
    ns["execute_task_minimal"]({"id": "task"})
    directory = tmp_path / "task"
    before = (directory / "output.txt").read_bytes()

    def fail(*args):
        raise OSError("publication failed")

    monkeypatch.setattr(executor_evidence.os, "link", fail)
    with pytest.raises(OSError):
        ns["execute_task_minimal"]({"id": "task"}, run_id=str(uuid.uuid4()))
    assert (directory / "output.txt").read_bytes() == before
    assert list(directory.iterdir()) == [directory / "output.txt"]
