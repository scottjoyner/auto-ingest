"""HTTP observation reuse, isolated worker and loopback fixtures."""
import ast
import datetime
import os
import pathlib
import tempfile
import threading
import time
import unittest
import uuid
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import requests


class HttpObservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.hits = []
        hits = self.hits

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                hits.append(self.path)
                self.send_response(503 if self.path == "/bad" else 200)
                self.end_headers()
                self.wfile.write(b"fixture")

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def stop():
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.addCleanup(stop)
        self.base = f"http://127.0.0.1:{server.server_port}"
        source = pathlib.Path(__file__).resolve().parents[1] / "summarize_from_segments.py"
        names = {"ensure_artifacts", "execute_task_minimal", "execute_ready",
                 "run_acceptance", "acc_http_ok"}
        nodes = [n for n in ast.parse(source.read_text()).body
                 if isinstance(n, ast.FunctionDef) and n.name in names]
        self.assertEqual({n.name for n in nodes}, names)
        self.ns = dict(ARTIFACTS_DIR=self.temp.name, pathlib=pathlib, time=time,
                       datetime=datetime, os=os, uuid=uuid, requests=requests,
                       Any=Any, Dict=Dict, List=List, Optional=Optional)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), self.ns)
        self.ns["ACCEPTANCE_FUNCS"] = {"http_ok": self.ns["acc_http_ok"]}

    def task(self, *paths):
        return {"id": "task", "acceptance": [
            {"type": "http_ok", "args": {"url": self.base + p}} for p in paths]}

    def check(self, paths, expected):
        task = self.task(*paths)
        manifest = self.ns["execute_task_minimal"](task, run_id=str(uuid.uuid4()))
        self.assertEqual(self.hits, list(dict.fromkeys(paths)))
        result = self.ns["run_acceptance"](task, "task", execution_steps=manifest["steps"])
        self.assertEqual([r["passed"] for r in result], expected)
        self.assertEqual(self.hits, list(dict.fromkeys(paths)))

    def test_success_once(self):
        self.check(["/ok"], [True])

    def test_http_failure_once(self):
        self.check(["/bad"], [False])

    def test_repeated_url_once(self):
        self.check(["/ok", "/ok"], [True, True])

    def test_distinct_urls(self):
        self.check(["/ok", "/bad", "/ok"], [True, False, True])

    def test_timeout_not_retried(self):
        task = self.task("/timeout")
        with patch.object(requests, "get", side_effect=requests.Timeout("private detail")) as get:
            manifest = self.ns["execute_task_minimal"](task, run_id=str(uuid.uuid4()))
            result = self.ns["run_acceptance"](task, "task", execution_steps=manifest["steps"])
            self.assertFalse(result[0]["passed"])
            self.assertEqual(get.call_count, 1)
            self.assertNotIn("private detail", repr(manifest))

    def test_missing_invalid_failed_observations(self):
        task = self.task("/ok")
        for steps in ([], [{"op": "http_get", "url": self.base+"/ok", "status": "200"}],
                      [{"op": "http_get", "url": self.base+"/ok", "status": 200, "error": "failed"}]):
            result = self.ns["run_acceptance"](task, "task", execution_steps=steps)
            self.assertFalse(result[0]["passed"])
        self.assertEqual(self.hits, [])

    def test_finalization_uncertainty_no_replay(self):
        task = self.task("/ok")
        state = {"status": "READY"}

        class Driver:
            def session(self):
                return nullcontext(object())

        def attach(session, tid, run):
            state.update(status="RUNNING", run_id=run["id"])

        def finish(session, tid, rid, success, manifest):
            self.assertTrue(success)
            self.assertTrue(manifest["acceptance_results"][0]["passed"])
            raise OSError("synthetic finalization uncertainty")

        self.ns.update(neo_driver=Driver, q_pick_ready=lambda s, limit: (
            ["task"] if state["status"] == "READY" else []),
            q_get_task=lambda s, tid: {"task": task}, q_attach_run=attach, q_finish_run=finish)
        with self.assertRaises(OSError):
            self.ns["execute_ready"](1)
        self.ns["execute_ready"](1)
        self.assertEqual(self.hits, ["/ok"])
        self.assertEqual(state["status"], "RUNNING")
        self.assertTrue((pathlib.Path(self.temp.name) / "task" /
                         f'run-{state["run_id"]}.json').exists())


if __name__ == "__main__":
    unittest.main()
