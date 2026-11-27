"""CLI admission and reporting behavior without a live service."""
import io
import json
import os
import unittest
import uuid
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import MagicMock, patch

from auto_ingest.executor_operation_cli import _DatabaseDriver, main


class Input:
    def __init__(self, data):
        self.buffer = io.BytesIO(data)


class OperationCliTests(unittest.TestCase):
    def setUp(self):
        self.args = ["--artifact-dir", "/fixture", "--bolt-port", "12345",
                     "--database", "fixture", "--confirm-read-only-admission"]
        self.identity = {"task_id": "test", "run_id": str(uuid.uuid4()),
                         "url": "http://127.0.0.1:1/private"}
        self.credentials = {"EXECUTOR_REPORT_USER": "fixture_reader",
                            "EXECUTOR_REPORT_PASSWORD": "fixture_only"}
        self.factory = MagicMock()

    def invoke(self, data=None, env=None, argv=None):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.dict(os.environ, self.credentials if env is None else env, clear=True), \
                patch("sys.stdin", Input(json.dumps(self.identity).encode() if data is None else data)), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(self.args if argv is None else argv, driver_factory=self.factory)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_valid_explicit_driver_and_redaction(self):
        report = {"replay_allowed": False, "activation_allowed": False}
        with patch("auto_ingest.executor_operation_cli.reconcile_operation", return_value=report) as call:
            code, out, err = self.invoke()
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out), report)
        self.assertEqual(err, "")
        self.assertNotIn("private", out)
        self.factory.assert_called_once_with("bolt://127.0.0.1:12345",
            auth=("fixture_reader","fixture_only"), connection_timeout=3,
            connection_acquisition_timeout=5, max_transaction_retry_time=0)
        self.assertEqual(call.call_args.args[0].database, "fixture")
        self.assertEqual(call.call_args.kwargs, self.identity)

    def test_invalid_inputs_no_connection(self):
        for data in (b"{}", b"x"*16385, b"{", b'{"url":"a","url":"b"}',
                     json.dumps(dict(self.identity, run_id="invalid")).encode()):
            with self.subTest(data=data[:20]):
                code, out, err = self.invoke(data)
                self.assertEqual((code,out),(2,""))
                self.assertIn("no replay",err)
        self.factory.assert_not_called()

    def test_no_default_credentials_or_invalid_port(self):
        self.assertEqual(self.invoke(env={})[:2], (2,""))
        args=list(self.args)
        args[args.index("--bolt-port")+1]="0"
        self.assertEqual(self.invoke(argv=args)[:2], (2,""))
        self.factory.assert_not_called()

    def test_driver_error_redacted(self):
        self.factory.side_effect=RuntimeError("private credential fixture_only")
        code,out,err=self.invoke()
        self.assertEqual((code,out),(2,""))
        self.assertNotIn("fixture_only",err)
        self.assertNotIn("private",err)

    def test_database_wrapper_only_read(self):
        driver=MagicMock()
        bound=_DatabaseDriver(driver,"fixture")
        bound.session(default_access_mode="READ")
        driver.session.assert_called_once_with(database="fixture",default_access_mode="READ")
        with self.assertRaises(RuntimeError):
            bound.session(default_access_mode="WRITE")
