"""Hermetic tests for exact-ID orphan candidate inspection."""
from datetime import datetime, timedelta, timezone

import pytest

from auto_ingest.executor_reconcile import inspect_tasks

NOW = datetime(2026, 11, 26, tzinfo=timezone.utc)
STALE = timedelta(hours=1)


def row(**overrides):
    return dict(identity="node", task_id="task", status="RUNNING", owner="run",
                updated_at="2026-11-25T22:00:00Z", run_id="run",
                run_status="RUNNING") | overrides


class Result:
    def __init__(self, rows, failure):
        self.rows = rows
        self.failure = failure

    def __iter__(self):
        return iter(self.rows)

    def consume(self):
        if self.failure:
            raise OSError("read acknowledgement lost")


class Driver:
    def __init__(self, rows, failure=False):
        self.rows = rows
        self.failure = failure
        self.calls = []

    def session(self, *, default_access_mode):
        assert default_access_mode == "READ"
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, *, task_id):
        assert "MATCH (t:Task {id:$task_id})" in query
        assert "LIMIT 3" in query
        assert not any(word in query.split() for word in
                       ("SET", "CREATE", "MERGE", "DELETE", "REMOVE"))
        self.calls.append(task_id)
        return Result(self.rows.get(task_id, []), self.failure)


@pytest.mark.parametrize(("rows", "expected"), [
    ([], "TASK_NOT_FOUND"),
    ([row()], "STALE_RUNNING_REVIEW"),
    ([row(updated_at="2026-11-25T23:00:00Z")], "STALE_RUNNING_REVIEW"),
    ([row(updated_at="2026-11-25T23:30:00Z")], "RUNNING_OBSERVED"),
    ([row(owner=None, run_id=None, run_status=None)], "UNOWNED_RUNNING_REVIEW"),
    ([row(run_id="foreign")], "OWNER_RUN_MISMATCH_REVIEW"),
    ([row(run_status="DONE")], "OWNER_RUN_MISMATCH_REVIEW"),
    ([row(run_id=None, run_status=None)], "OWNER_RUN_MISMATCH_REVIEW"),
    ([row(), row()], "AMBIGUOUS_TASK_OR_RUN"),
    ([row(), row(identity="other")], "AMBIGUOUS_TASK_OR_RUN"),
    ([row(status="DONE")], "INCONSISTENT_NONRUNNING"),
    ([row(status="DONE", owner=None, run_id=None, run_status=None)], "NOT_RUNNING"),
    ([row(status="OTHER", owner=None, run_id=None, run_status=None)], "UNKNOWN_TASK_STATE"),
    ([row(updated_at=None)], "CLOCK_OR_TIMESTAMP_UNKNOWN"),
    ([row(updated_at="invalid")], "CLOCK_OR_TIMESTAMP_UNKNOWN"),
    ([row(updated_at="2026-11-25T22:00:00")], "CLOCK_OR_TIMESTAMP_UNKNOWN"),
    ([row(updated_at="2026-11-26T00:00:01Z")], "CLOCK_OR_TIMESTAMP_UNKNOWN"),
    ([row(task_id="wrong")], "IDENTITY_MISMATCH"),
    ([row(identity=None)], "IDENTITY_MISMATCH"),
])
def test_classifications_never_authorize_replay(rows, expected):
    driver = Driver({"task": rows})
    assert inspect_tasks(driver, ["task"], now=NOW, stale_after=STALE) == [
        {"task_id": "task", "classification": expected, "replay_allowed": False}]
    assert driver.calls == ["task"]


@pytest.mark.parametrize("ids", [[], ["x"] * 51, ["x", "x"], [""], [None],
                                  ["x" * 257], "task"])
def test_invalid_scope_never_queries(ids):
    driver = Driver({})
    with pytest.raises(ValueError):
        inspect_tasks(driver, ids, now=NOW, stale_after=STALE)
    assert driver.calls == []


@pytest.mark.parametrize(("now", "age"), [
    (NOW.replace(tzinfo=None), STALE), (NOW, timedelta(0)), (NOW, -STALE)])
def test_invalid_time_never_queries(now, age):
    driver = Driver({})
    with pytest.raises(ValueError):
        inspect_tasks(driver, ["task"], now=now, stale_after=age)
    assert driver.calls == []


def test_bounded_exact_ids_and_deferred_failure():
    ids = [f"task-{i}" for i in range(50)]
    driver = Driver({})
    report = inspect_tasks(driver, ids, now=NOW, stale_after=STALE)
    assert len(report) == 50 and driver.calls == ids
    driver = Driver({}, failure=True)
    with pytest.raises(OSError):
        inspect_tasks(driver, ids, now=NOW, stale_after=STALE)
    assert driver.calls == ids[:1]


def test_unexpected_cardinality_fails_closed():
    driver = Driver({"task": [row()] * 4})
    with pytest.raises(RuntimeError):
        inspect_tasks(driver, ["task"], now=NOW, stale_after=STALE)
