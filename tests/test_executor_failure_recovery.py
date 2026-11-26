"""A task execution failure must not strand the remaining selected tasks."""
import ast
import datetime
import uuid
from contextlib import nullcontext
from pathlib import Path

import pytest


@pytest.mark.parametrize("failure_stage", ["execute", "acceptance", "rejected", None])
def test_selected_tasks_reach_terminal_states(tmp_path, failure_stage):
    source = Path(__file__).resolve().parents[1] / "summarize_from_segments.py"
    tree = ast.parse(source.read_text())
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "execute_ready")
    tasks = {"first": "READY", "second": "READY"}
    runs = {}
    executions = []
    checks = []

    class Driver:
        def session(self):
            return nullcontext(object())

    def pick(session, limit):
        selected = [k for k, v in tasks.items() if v == "READY"][:limit]
        return selected

    def attach(session, tid, run):
        assert tasks[tid] == "READY"
        tasks[tid] = "RUNNING"
        assert tid not in runs
        runs[tid] = {"id": run["id"], "status": "RUNNING"}

    def execute(task, *, run_id):
        assert uuid.UUID(run_id)
        tid = task["id"]
        executions.append(tid)
        if tid == "first" and failure_stage == "execute":
            raise OSError("private details should not be persisted")
        directory = tmp_path / tid
        directory.mkdir()
        (directory / "output.txt").write_text(tid)
        return {"artifacts_dir": str(directory), "steps": [{"op": "write_file"}]}

    def acceptance(task, tid):
        checks.append(tid)
        if tid == "first" and failure_stage == "acceptance":
            raise ValueError("private details should not be persisted")
        return [{"passed": not (tid == "first" and failure_stage == "rejected")}]

    def finish(session, tid, rid, success, manifest):
        assert runs[tid]["id"] == rid
        assert runs[tid]["status"] == "RUNNING"
        tasks[tid] = "DONE" if success else "FAILED"
        runs[tid].update(status=tasks[tid], manifest=manifest)

    ns = dict(neo_driver=Driver, q_pick_ready=pick,
              q_get_task=lambda session, tid: {"task": {"id": tid}},
              q_attach_run=attach, execute_task_minimal=execute,
              run_acceptance=acceptance, q_finish_run=finish,
              datetime=datetime, uuid=uuid)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), ns)
    ns["execute_ready"](2)
    assert tasks == {"first": "FAILED" if failure_stage else "DONE", "second": "DONE"}
    assert executions == ["first", "second"]
    assert checks == (["second"] if failure_stage == "execute" else ["first", "second"])
    assert len(runs) == 2
    assert (tmp_path / "second" / "output.txt").read_text() == "second"
    assert "private details" not in repr(runs)
    if failure_stage in ("execute", "acceptance"):
        error = runs["first"]["manifest"]["execution_error"]
        assert error == {"stage": failure_stage,
                         "type": "OSError" if failure_stage == "execute" else "ValueError"}
    if failure_stage == "acceptance":
        assert runs["first"]["manifest"]["steps"] == [{"op": "write_file"}]
    # Terminal tasks are not re-selected by the same executor.
    ns["execute_ready"](2)
    assert executions == ["first", "second"]
