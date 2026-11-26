"""The executor must not announce completion on an unconfirmed graph update."""
import ast
import datetime
import json
from pathlib import Path
from typing import Any, Dict

import pytest


@pytest.mark.parametrize("success", [True, False])
@pytest.mark.parametrize("outcome", ["confirmed", "missing", "duplicate", "deferred"])
def test_finalize_requires_confirmed_update(success, outcome):
    source = Path(__file__).resolve().parents[1] / "summarize_from_segments.py"
    fn = next(n for n in ast.parse(source.read_text()).body
              if isinstance(n, ast.FunctionDef) and n.name == "q_finish_run")
    expected = "DONE" if success else "FAILED"

    class Result:
        consumed = False

        def single(self, *, strict=False):
            assert strict
            if outcome in ("missing", "duplicate"):
                raise RuntimeError("unexpected record cardinality")
            return {"tid": "task", "rid": "run", "status": expected}

        def consume(self):
            if outcome == "deferred":
                raise OSError("commit acknowledgement unavailable")
            self.consumed = True

    result = Result()

    class Session:
        def begin_transaction(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def commit(self):
            assert result.consumed

        def run(self, query, **params):
            assert params["tid"] == "task"
            assert params["rid"] == "run"
            assert params["tstatus"] == params["rstatus"] == expected
            assert params["success"] is success
            assert json.loads(params["manifest_json"]) == {"steps": []}
            return result

    ns = dict(datetime=datetime, json=json, Dict=Dict, Any=Any)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), ns)
    if outcome == "confirmed":
        ns["q_finish_run"](Session(), "task", "run", success, {"steps": []})
        assert result.consumed
    else:
        with pytest.raises((RuntimeError, OSError)):
            ns["q_finish_run"](Session(), "task", "run", success, {"steps": []})
        assert not result.consumed
