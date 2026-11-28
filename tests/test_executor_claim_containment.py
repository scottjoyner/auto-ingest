"""Do not reserve unstarted tasks when a selected task cannot complete."""
import ast
import datetime
import uuid
from contextlib import nullcontext
from pathlib import Path

import pytest


@pytest.mark.parametrize("failure", ["attach", "finalize", "interrupt", None])
def test_claim_only_current_task(failure):
    source = Path(__file__).resolve().parents[1] / "summarize_from_segments.py"
    fn = next(n for n in ast.parse(source.read_text()).body
              if isinstance(n, ast.FunctionDef) and n.name == "execute_ready")
    tasks = {str(i): "READY" for i in range(4)}
    calls = []
    executed = []

    class Driver:
        def session(self):
            return nullcontext(object())

    def pick(session, limit):
        calls.append(limit)
        selected = [k for k, v in tasks.items() if v == "READY"][:limit]
        return selected

    def attach(session, tid, run):
        if failure == "attach":
            raise OSError("attachment uncertain")
        assert tasks[tid] == "READY"
        tasks[tid] = "RUNNING"

    def execute(task, *, run_id):
        assert uuid.UUID(run_id)
        executed.append(task["id"])
        if failure == "interrupt":
            raise KeyboardInterrupt()
        return {"artifacts_dir": "synthetic"}

    def finish(session, tid, rid, success, manifest):
        if failure == "finalize":
            raise OSError("finalization uncertain")
        tasks[tid] = "DONE"

    ns = dict(neo_driver=Driver, q_pick_ready=pick,
              q_get_task=lambda s, tid: {"task": {"id": tid}},
              q_attach_run=attach, execute_task_minimal=execute,
              run_acceptance=lambda task, tid, *, execution_steps: [{"passed": True}],
              q_finish_run=finish, datetime=datetime, uuid=uuid)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), ns)
    if failure:
        with pytest.raises(KeyboardInterrupt if failure == "interrupt" else OSError):
            ns["execute_ready"](3)
        expected = "READY" if failure == "attach" else "RUNNING"
        assert tasks == {"0": expected, "1": "READY", "2": "READY", "3": "READY"}
        assert calls == [1]
        assert executed == ([] if failure == "attach" else ["0"])
    else:
        ns["execute_ready"](3)
        assert tasks == {"0": "DONE", "1": "DONE", "2": "DONE", "3": "READY"}
        assert calls == [1, 1, 1]
        assert executed == ["0", "1", "2"]
