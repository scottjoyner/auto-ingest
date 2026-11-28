"""Exact graph/file binding and no-authorization regression tests."""
import json
import pathlib
import tempfile
import unittest
import uuid
from contextlib import nullcontext

from auto_ingest.executor_evidence import (
    publish_http_intent,
    publish_http_outcome,
    publish_local_output_receipt,
    publish_run_output,
)
from auto_ingest.executor_operation_report import _RUN, _TASK, reconcile_operation


class Result(list):
    def consume(self):
        return self

    plan = {"operatorType": "NodeIndexSeek"}


class Driver:
    def __init__(self, tasks, runs, indexed=True):
        self.tasks, self.runs, self.indexed = tasks, runs, indexed
        self.calls = []

    def session(self, **kwargs):
        assert kwargs == {"default_access_mode": "READ"}
        return nullcontext(self)

    def begin_transaction(self, **kwargs):
        assert kwargs == {"timeout": 5}
        return nullcontext(self)

    def run(self, query, **params):
        self.calls.append(query)
        if query.startswith("EXPLAIN "):
            r = Result()
            r.plan = {"operatorType": "NodeIndexSeek" if self.indexed else "NodeByLabelScan"}
            return r
        if query == _TASK:
            return Result(self.tasks)
        if query == _RUN:
            return Result(self.runs)
        raise AssertionError("unexpected query")

    def commit(self):
        pass


class OperationReportTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = pathlib.Path(temp.name)
        self.rid = str(uuid.uuid4())
        self.args = dict(task_id="task", run_id=self.rid, url="http://127.0.0.1:1/private")
        self.task = dict(identity="fixture", status="RUNNING", owner=self.rid,
                         acceptance=json.dumps([{"type":"http_ok","args":{"url":self.args["url"]}}]))
        self.run = dict(rid=self.rid, status="RUNNING")

    def report(self, driver=None, directory=None):
        result = reconcile_operation(driver or Driver([self.task], [self.run]),
                                     directory or self.dir, **self.args)
        self.assertFalse(result["replay_allowed"])
        self.assertFalse(result["activation_allowed"])
        self.assertFalse(result["remote_completion_verified"])
        self.assertFalse(result["atomic_snapshot"])
        self.assertNotIn(self.args["url"], json.dumps(result))
        return result

    def test_intent_only_and_local_receipt(self):
        output = f"fixture\nRUN: {self.rid}\n".encode()
        local = dict(task_id="task", run_id=self.rid, output=output)
        publish_run_output(self.dir, **local)
        publish_local_output_receipt(self.dir, **local)
        publish_http_intent(self.dir, **self.args)
        before = {p.name:p.read_bytes() for p in self.dir.iterdir()}
        result = self.report()
        self.assertEqual(result["graph_classification"], "RUNNING_OWNER_MATCH")
        self.assertEqual(result["local_output"], "LOCAL_OUTPUT_BYTES_MATCH")
        self.assertEqual(result["http"]["classification"], "HTTP_INTENT_ONLY_UNCERTAIN")
        self.assertEqual(before, {p.name:p.read_bytes() for p in self.dir.iterdir()})

    def test_terminal_observation(self):
        self.task.update(status="DONE", owner=None)
        self.run.update(status="DONE")
        publish_http_intent(self.dir, **self.args)
        publish_http_outcome(self.dir, **self.args, status=200)
        result = self.report()
        self.assertEqual(result["graph_classification"], "TERMINAL_RUN_MATCH")
        self.assertEqual(result["http"]["status"], 200)
        self.assertEqual(result["local_output"], "LOCAL_OUTPUT_UNVERIFIED")

    def test_ambiguous_and_foreign_skip_disk(self):
        for tasks, runs, expected in (
            ([], [], "TASK_NOT_FOUND"),
            ([self.task]*2, [self.run], "AMBIGUOUS_TASK"),
            ([self.task], [dict(self.run,rid=str(uuid.uuid4()))], "RUN_LINK_UNVERIFIED"),
            ([self.task], [self.run]*2, "RUN_LINK_UNVERIFIED"),
            ([dict(self.task,owner="foreign")], [self.run], "GRAPH_STATE_MISMATCH"),
            ([dict(self.task,acceptance="[]")], [self.run], "HTTP_CONTRACT_UNVERIFIED"),
        ):
            with self.subTest(expected=expected):
                result = self.report(Driver(tasks,runs), self.dir/"missing")
                self.assertEqual(result["graph_classification"],expected)
                self.assertEqual(result["local_output"],"NOT_READ")

    def test_missing_index_fails_before_data(self):
        driver = Driver([self.task],[self.run],indexed=False)
        with self.assertRaises(RuntimeError):
            self.report(driver)
        self.assertEqual(len(driver.calls),1)

    def test_missing_evidence_is_not_proof(self):
        result=self.report()
        self.assertEqual(result["http"]["classification"],"HTTP_NO_EVIDENCE")

    def test_oversize_fails_closed(self):
        (self.dir/f"run-{self.rid}.txt").write_bytes(b"x"*16385)
        with self.assertRaises(ValueError):
            self.report()
