"""A task must not execute before its run attachment is acknowledged."""
import ast
import json
from pathlib import Path
from typing import Any, Dict

import pytest


@pytest.mark.parametrize("outcome", ["confirmed", "missing", "duplicate", "wrong_id", "deferred"])
def test_attach_requires_confirmed_run(outcome):
    source = Path(__file__).resolve().parents[1] / "summarize_from_segments.py"
    fn = next(n for n in ast.parse(source.read_text()).body
              if isinstance(n, ast.FunctionDef) and n.name == "q_attach_run")

    class Result:
        consumed = False

        def single(self, *, strict=False):
            if outcome in ("missing", "duplicate") and strict:
                raise RuntimeError("unexpected record cardinality")
            if outcome == "missing":
                return None
            return {"rid": "wrong" if outcome == "wrong_id" else "run"}

        def consume(self):
            if outcome == "deferred":
                raise OSError("commit acknowledgement unavailable")
            self.consumed = True

    result = Result()

    class Session:
        def run(self, query, **params):
            assert params["tid"] == "task"
            assert params["rid"] == "run"
            assert params["status"] == "RUNNING"
            assert json.loads(params["manifest_json"]) == {"steps": []}
            return result

    ns = dict(json=json, Dict=Dict, Any=Any)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), ns)
    run = {"id": "run", "started_at": "2026-11-25T00:00:00Z",
           "status": "RUNNING", "manifest": {"steps": []}}
    if outcome == "confirmed":
        assert ns["q_attach_run"](Session(), "task", run) == "run"
        assert result.consumed
    else:
        with pytest.raises((RuntimeError, OSError)):
            ns["q_attach_run"](Session(), "task", run)
        assert not result.consumed
