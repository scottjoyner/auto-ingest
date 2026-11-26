"""Dependency-free tests for the cutover probe."""
import unittest
from types import SimpleNamespace

from auto_ingest.executor_cutover import PROBES, inspect_cutover


class Result:
    def __init__(self, rows=(), operator=None):
        self.rows = rows
        self.operator = operator

    def __iter__(self):
        return iter(self.rows)

    def consume(self):
        return SimpleNamespace(plan={"operatorType": "ProduceResults", "children": [
            {"operatorType": self.operator, "children": []}]})


class Driver:
    def __init__(self, rows=None, unsafe=None, fail_commit=False):
        self.rows = rows or {}
        self.unsafe = unsafe
        self.fail_commit = fail_commit
        self.calls = []

    def session(self, *, default_access_mode):
        assert default_access_mode == "READ"
        return self

    def begin_transaction(self, *, timeout):
        assert timeout == 5
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def commit(self):
        if self.fail_commit:
            raise OSError("uncertain read completion")

    def run(self, query, *, limit):
        explain = query.startswith("EXPLAIN ")
        raw = query.removeprefix("EXPLAIN ")
        name = next(name for name, text in PROBES.items() if text == raw)
        self.calls.append((explain, name, limit))
        assert not any(word in raw.split() for word in ("CREATE", "SET", "DELETE", "MERGE"))
        if explain:
            return Result(operator="NodeByLabelScan" if name == self.unsafe else "NodeIndexSeek")
        return Result(rows=self.rows.get(name, []))


class CutoverTests(unittest.TestCase):
    def test_empty_never_authorizes_activation(self):
        driver = Driver()
        report = inspect_cutover(driver)
        self.assertEqual(report["graph_status"], "NO_BLOCKERS_OBSERVED")
        self.assertEqual(report["legacy_writer_quiescence"], "UNVERIFIED")
        self.assertFalse(report["activation_allowed"])
        self.assertFalse(report["replay_allowed"])
        self.assertEqual([x[0] for x in driver.calls], [True]*3+[False]*3)

    def test_each_blocker(self):
        for name in PROBES:
            with self.subTest(name=name):
                report = inspect_cutover(Driver(rows={name: [{"identity": "node"}]}))
                self.assertEqual(report["graph_status"], "BLOCKERS_OBSERVED")
                self.assertEqual(report["findings"][name], {"observed": 1, "truncated": False})
                self.assertFalse(report["activation_allowed"])

    def test_truncation(self):
        report = inspect_cutover(Driver(rows={"RUNNING_RUN": [
            {"identity": str(i)} for i in range(3)]}), sample_limit=2)
        self.assertEqual(report["findings"]["RUNNING_RUN"], {"observed": 2, "truncated": True})

    def test_unsafe_plan_prevents_execution(self):
        for name in PROBES:
            with self.subTest(name=name):
                driver = Driver(unsafe=name)
                with self.assertRaisesRegex(RuntimeError, "index-only"):
                    inspect_cutover(driver)
                self.assertTrue(all(x[0] for x in driver.calls))

    def test_invalid_limits(self):
        for limit in (0, 51, True, 1.5):
            with self.subTest(limit=limit):
                driver = Driver()
                with self.assertRaises(ValueError):
                    inspect_cutover(driver, sample_limit=limit)
                self.assertEqual(driver.calls, [])

    def test_invalid_responses(self):
        for rows in ([{"identity": ""}], [{"identity": "a"}]*2,
                     [{"identity": str(i)} for i in range(22)]):
            with self.subTest(rows=rows), self.assertRaises(RuntimeError):
                inspect_cutover(Driver(rows={"RUNNING_TASK": rows}))

    def test_read_commit_failure(self):
        with self.assertRaises(OSError):
            inspect_cutover(Driver(fail_commit=True))
