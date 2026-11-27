"""Bounded offline operator report, no external services."""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
import uuid

from auto_ingest.executor_evidence import publish_http_intent, publish_http_outcome
from auto_ingest.executor_http_report import report


class HttpReportTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = pathlib.Path(temp.name)
        self.args = dict(task_id="synthetic", run_id=str(uuid.uuid4()),
                         url="http://127.0.0.1:1/path?token=synthetic-private")

    def cli(self, payload, directory=None):
        return subprocess.run(
            [sys.executable, "-m", "auto_ingest.executor_http_report",
             "--artifact-dir", str(directory or self.directory)],
            input=payload, capture_output=True, timeout=5)

    def test_observed_response_and_no_plaintext_target(self):
        intent = publish_http_intent(self.directory, **self.args)
        outcome = publish_http_outcome(self.directory, **self.args, status=200)
        before = {p.name: p.read_bytes() for p in (intent, outcome)}
        result = self.cli(json.dumps(self.args).encode())
        self.assertEqual(result.returncode, 0, result.stderr)
        obj = json.loads(result.stdout)
        self.assertEqual(obj["classification"], "HTTP_LOCAL_RESPONSE_OBSERVED")
        self.assertEqual(obj["status"], 200)
        self.assertFalse(obj["replay_allowed"])
        self.assertFalse(obj["remote_completion_verified"])
        self.assertNotIn(b"synthetic-private", result.stdout)
        self.assertEqual(before, {p.name: p.read_bytes() for p in (intent, outcome)})

    def test_missing_and_intent_only(self):
        self.assertEqual(report(self.directory, **self.args)["classification"], "HTTP_NO_EVIDENCE")
        publish_http_intent(self.directory, **self.args)
        result = report(self.directory, **self.args)
        self.assertEqual(result["classification"], "HTTP_INTENT_ONLY_UNCERTAIN")
        self.assertFalse(result["replay_allowed"])

    def test_only_exact_names_read(self):
        (self.directory / "unrelated").write_bytes(b"x" * 20000)
        self.assertEqual(report(self.directory, **self.args)["classification"], "HTTP_NO_EVIDENCE")

    def test_oversize_evidence_fails_closed(self):
        intent = publish_http_intent(self.directory, **self.args)
        intent.write_bytes(b"x" * 16385)
        with self.assertRaises(ValueError):
            report(self.directory, **self.args)

    def test_symlink_and_fifo_rejected(self):
        intent = publish_http_intent(self.directory, **self.args)
        other = self.directory / "other"
        intent.rename(other)
        intent.symlink_to(other)
        with self.assertRaises(OSError):
            report(self.directory, **self.args)
        intent.unlink()
        os.mkfifo(intent)
        with self.assertRaises(ValueError):
            report(self.directory, **self.args)

    def test_hardlink_and_directory_symlink_rejected(self):
        intent = publish_http_intent(self.directory, **self.args)
        os.link(intent, self.directory / "alias")
        with self.assertRaises(ValueError):
            report(self.directory, **self.args)
        with tempfile.TemporaryDirectory() as parent:
            link = pathlib.Path(parent) / "link"
            link.symlink_to(self.directory, target_is_directory=True)
            with self.assertRaises(OSError):
                report(link, **self.args)

    def test_invalid_input_no_success_output(self):
        for payload in (b"{", b"x"*16385, b'{"url":"a","url":"b"}',
                        json.dumps(dict(self.args, extra=True)).encode()):
            result = self.cli(payload)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, b"")
            self.assertIn(b"no replay authorization", result.stderr)
        result = self.cli(json.dumps(self.args).encode(), self.directory / "missing")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, b"")


if __name__ == "__main__":
    unittest.main()
