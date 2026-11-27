"""Hermetic contract test for ``auto_ingest.shorts.db_retry.retry``.

A permanently pressured database must be reported as failed as soon as the last
attempt fails. The retry loop currently sleeps *after* that last failure too,
even though no further attempt follows it, so callers wait an extra
``base_wait * attempts`` seconds before learning the operation failed.

No live Neo4j and no real sleeping: both the driver factory and the clock are
replaced with in-process fakes. Only the retry loop's backoff schedule is under
test.
"""
from __future__ import annotations

from types import SimpleNamespace

from auto_ingest.shorts import db_retry


class _Transient(RuntimeError):
    """Exception whose message marks it as safe to retry."""


class _FakeDriver:
    def close(self) -> None:
        pass


class _FakeGraphDatabase:
    @staticmethod
    def driver(*args, **kwargs):
        return _FakeDriver()


def _run_all_transient(monkeypatch, *, attempts: int, base_wait: float):
    waits = []
    # Replace the clock and the driver factory; nothing here touches a socket.
    monkeypatch.setattr(db_retry, "time", SimpleNamespace(sleep=waits.append))
    monkeypatch.setattr(db_retry, "GraphDatabase", _FakeGraphDatabase)

    calls = []

    def always_transient(_drv):
        calls.append(1)
        raise _Transient("ServiceUnavailable")

    result = db_retry.retry(
        always_transient, attempts=attempts, base_wait=base_wait
    )
    return result, len(calls), waits


def test_retry_does_not_sleep_after_the_final_attempt(monkeypatch):
    result, calls, waits = _run_all_transient(
        monkeypatch, attempts=3, base_wait=2.0
    )
    assert result is None
    assert calls == 3
    # Three attempts leave exactly two gaps to wait through. The third failure
    # ends the loop, so sleeping a third time only delays the failure report.
    assert waits == [2.0, 4.0]


def test_retry_with_no_attempts_never_sleeps(monkeypatch):
    result, calls, waits = _run_all_transient(
        monkeypatch, attempts=0, base_wait=2.0
    )
    assert result is None
    assert calls == 0
    assert waits == []
