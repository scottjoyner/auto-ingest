"""Persisted acceptance corruption must never become implicit success."""
import ast
import datetime
import json
import pathlib
import unittest
import uuid
from contextlib import nullcontext
from typing import Any, Dict, List, Optional


class AcceptanceContractTests(unittest.TestCase):
    def setUp(self):
        source = pathlib.Path(__file__).resolve().parents[1] / "summarize_from_segments.py"
        names = {"q_get_task", "execute_ready"}
        nodes = [n for n in ast.parse(source.read_text()).body
                 if isinstance(n, ast.FunctionDef) and n.name in names]
        self.assertEqual({n.name for n in nodes}, names)
        self.ns = dict(json=json, datetime=datetime, uuid=uuid,
                       Any=Any, Dict=Dict, List=List, Optional=Optional)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), self.ns)
        self.calls = []

    def session(self, task):
        class Result:
            def single(self):
                return {"task": task, "s": None, "t": None}

        class Session:
            def run(self, query, **params):
                return Result()

        return Session()

    def test_valid_and_explicit_empty(self):
        for checks in ([], [{"type": "http_ok", "args": {"url": "http://127.0.0.1:1"}}],
                       [{"type": "file_exists"}]):
            task = {"id": "fixture", "acceptance_json": json.dumps(checks)}
            result = self.ns["q_get_task"](self.session(task), "fixture")
            self.assertEqual(result["task"]["acceptance"], checks)

    def test_legacy_absent_field_preserved(self):
        task = {"id": "fixture"}
        self.assertEqual(self.ns["q_get_task"](self.session(task), "fixture")["task"], task)

    def test_corrupt_contracts_rejected_without_payload_disclosure(self):
        invalid = ["{private", "null", "false", "0", '""', "{}", "[null]", "[1]",
                   '[{"type":""}]', '[{"type":true}]', '[{"type":"http_ok","args":null}]',
                   '[{"type":"http_ok","type":"file_exists"}]',
                   '[{"type":"http_ok","args":{"url":"a","url":"b"}}]',
                   '[{"type":"http_ok","args":{"value":NaN}}]', 0, [], None]
        for raw in invalid:
            with self.subTest(raw=repr(raw)):
                task = {"id": "fixture", "acceptance_json": raw}
                with self.assertRaisesRegex(ValueError, "^invalid persisted acceptance contract$"):
                    self.ns["q_get_task"](self.session(task), "fixture")

    def test_invalid_task_never_claims_dispatches_or_finalizes(self):
        outer = self

        class Driver:
            def session(self):
                return nullcontext(outer.session({"id": "fixture", "acceptance_json": "{"}))

        def unexpected(*args, **kwargs):
            self.calls.append("unexpected operation")
            raise AssertionError("invalid task must remain unclaimed")

        self.ns.update(neo_driver=Driver, q_pick_ready=lambda s, limit: ["fixture"],
                       q_attach_run=unexpected, execute_task_minimal=unexpected,
                       run_acceptance=unexpected, q_finish_run=unexpected)
        with self.assertRaisesRegex(ValueError, "^invalid persisted acceptance contract$"):
            self.ns["execute_ready"](1)
        self.assertEqual(self.calls, [])

    def test_corrupt_candidate_does_not_starve_next_valid_task(self):
        outer = self
        selected = []
        claimed = []
        finished = []

        class Driver:
            def session(self):
                return nullcontext(object())

        def pick(session, limit, *, excluded=()):
            selected.append(tuple(excluded))
            return ["good"] if excluded else ["bad"]

        def read(session, tid):
            if tid == "bad":
                return original(outer.session({"id": tid, "acceptance_json": "{"}), tid)
            return {"task": {"id": tid, "acceptance": []}}

        original = self.ns["q_get_task"]
        self.ns.update(neo_driver=Driver, q_pick_ready=pick, q_get_task=read,
                       q_attach_run=lambda s, tid, run: claimed.append(tid),
                       execute_task_minimal=lambda task, **kw: {"steps": []},
                       run_acceptance=lambda *a, **kw: [],
                       q_finish_run=lambda s, tid, rid, success, manifest: finished.append((tid, success)))
        with self.assertRaisesRegex(ValueError, "^invalid persisted acceptance contract$"):
            self.ns["execute_ready"](2)
        self.assertEqual(selected, [(), ("bad",)])
        self.assertEqual(claimed, ["good"])
        self.assertEqual(finished, [("good", True)])


if __name__ == "__main__":
    unittest.main()
