"""db_retry resilience tests (pure, no live Neo4j)."""
import time
from unittest import mock

from auto_ingest.shorts import db_retry


class _MemErr(Exception):
    pass


def _make_driver(raise_on):
    states = {"n": 0, "raise_on": list(raise_on)}

    class _FakeDriver:
        def __init__(self):
            self.calls = 0
            self.closed = False

        def session(self, **kw):
            return self

        def run(self, *a, **k):
            self.calls += 1
            states["n"] += 1
            if states["raise_on"] and states["n"] <= len(states["raise_on"]):
                raise states["raise_on"][states["n"] - 1]
            return "ok"

        def close(self):
            self.closed = True

    d = _FakeDriver()
    d._states = states
    return d


def _patch(monkeypatch, drv):
    monkeypatch.setattr(db_retry, "GraphDatabase", mock.MagicMock(
        driver=mock.MagicMock(return_value=drv)))
    monkeypatch.setattr(time, "sleep", lambda *_: None)


def test_is_transient_detects_markers():
    assert db_retry.is_transient(_MemErr("MemoryPoolOutOfMemoryError boom"))
    assert db_retry.is_transient(_MemErr("neo4j.exceptions.TransientError x"))
    assert db_retry.is_transient(_MemErr("ServiceUnavailable"))
    assert not db_retry.is_transient(ValueError("nope"))


def test_retry_succeeds_after_transient(monkeypatch):
    drv = _make_driver([_MemErr("TransientError"), _MemErr("ServiceUnavailable")])
    _patch(monkeypatch, drv)
    out = db_retry.retry(lambda d: d.session().run(), attempts=5)
    assert out == "ok"
    assert drv.calls == 3
    assert drv.closed


def test_retry_gives_up_after_non_transient(monkeypatch):
    drv = _make_driver([ValueError("fatal")])
    _patch(monkeypatch, drv)
    out = db_retry.retry(lambda d: d.session().run(), attempts=3)
    assert out is None
    assert drv.calls == 1


def test_retry_gives_up_after_attempts(monkeypatch):
    drv = _make_driver([_MemErr("TransientError")] * 10)
    _patch(monkeypatch, drv)
    out = db_retry.retry(lambda d: d.session().run(), attempts=3)
    assert out is None
    assert drv.calls == 3
