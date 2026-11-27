"""Offline HTTP reconciliation and abrupt exit at real worker boundary."""
import json
import pathlib
import subprocess
import sys
import tempfile
import threading
import unittest
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

from auto_ingest.executor_evidence import (
    inspect_http_observation,
    publish_http_intent,
    publish_http_outcome,
)


class ReconciliationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = pathlib.Path(temp.name)
        self.args = dict(task_id="synthetic", run_id=str(uuid.uuid4()),
                         url="http://127.0.0.1:1/synthetic")

    def inspect(self, intent, outcome, classification, status=None, **overrides):
        result = inspect_http_observation(intent, outcome, **(self.args | overrides))
        self.assertEqual(result, dict(classification=classification, status=status,
                                     remote_completion_verified=False, replay_allowed=False))

    def pair(self, status):
        intent = publish_http_intent(self.directory, **self.args).read_bytes()
        outcome = publish_http_outcome(self.directory, **self.args, status=status).read_bytes()
        return intent, outcome

    def test_absence_is_not_non_dispatch_proof(self):
        self.inspect(None, None, "HTTP_NO_EVIDENCE")

    def test_intent_only(self):
        intent = publish_http_intent(self.directory, **self.args).read_bytes()
        self.inspect(intent, None, "HTTP_INTENT_ONLY_UNCERTAIN")

    def test_success_observation(self):
        self.inspect(*self.pair(200), "HTTP_LOCAL_RESPONSE_OBSERVED", 200)

    def test_failure_observation(self):
        self.inspect(*self.pair(503), "HTTP_LOCAL_RESPONSE_OBSERVED", 503)

    def test_uncertain_outcome(self):
        self.inspect(*self.pair(None), "HTTP_OUTCOME_UNCERTAIN")

    def test_orphan_wrong_identity_and_corruption(self):
        intent, outcome = self.pair(200)
        for i, o, args in [(None, outcome, {}), (intent+b"x", outcome, {}),
                           (intent, outcome, {"task_id": "other"}),
                           (intent, outcome, {"run_id": str(uuid.uuid4())}),
                           (intent, outcome, {"url": self.args["url"]+"x"}),
                           (intent, b"x"*16385, {}), (intent, b"{", {}),
                           (intent, outcome.replace(b'"status": 200', b'"status": true'), {}),
                           (intent, outcome.replace(b'"status": 200', b'"status": 999'), {}),
                           (intent, outcome.replace(b'"status": 200', b'"status": 200, "status": 200'), {}),
                           (intent, outcome.replace(b'"scope":', b'"extra": false, "scope":'), {}),
                           (intent, json.dumps(json.loads(outcome)).encode(), {})]:
            with self.subTest(args=args, length=len(o)):
                self.inspect(i, o, "HTTP_EVIDENCE_UNVERIFIED", **args)

    def test_actual_worker_exit_after_http_before_outcome(self):
        hits = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                hits.append(self.path)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"fixture")

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.args["url"] = f"http://127.0.0.1:{server.server_port}/synthetic"
        root = pathlib.Path(__file__).resolve().parents[1]
        code = """import ast, datetime, os, pathlib, sys, time
from typing import Any, Dict, Optional
import requests
from auto_ingest import executor_evidence
source = pathlib.Path('summarize_from_segments.py')
nodes = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef)
         and n.name in {'ensure_artifacts', 'execute_task_minimal'}]
ARTIFACTS_DIR = sys.argv[1]
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'))
def stop(*args, **kwargs):
    os._exit(77)
executor_evidence.publish_http_outcome = stop
execute_task_minimal({'id': 'synthetic', 'acceptance': [
    {'type': 'http_ok', 'args': {'url': sys.argv[3]}}]}, run_id=sys.argv[2])
"""
        try:
            child = subprocess.run([sys.executable, "-c", code, str(self.directory),
                                    self.args["run_id"], self.args["url"]],
                                   cwd=root, capture_output=True, timeout=15)
            self.assertEqual(child.returncode, 77, child.stderr.decode(errors="replace"))
            directory = self.directory / "synthetic"
            intent = next(directory.glob("*-intent.json")).read_bytes()
            self.assertEqual(list(directory.glob("*-outcome.json")), [])
            self.assertEqual(hits, ["/synthetic"])
            self.inspect(intent, None, "HTTP_INTENT_ONLY_UNCERTAIN")
            self.assertEqual(hits, ["/synthetic"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
