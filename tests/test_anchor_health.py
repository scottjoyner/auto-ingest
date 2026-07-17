"""
Tests for auto_ingest.speaker_health (read-only Scott anchor health check).

No live Neo4j, no moviepy, works under system python + pytest. The health
logic is exercised against a fake neo4j driver object so we assert the count
helper returns an int and the assertion fires when the count is wrong.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import auto_ingest.speaker_health as sh  # noqa: E402


class _FakeRecord:
    def __init__(self, n):
        self._n = n

    def __getitem__(self, key):
        return self._n


class _FakeResult:
    def __init__(self, count):
        self._count = count

    def single(self):
        return _FakeRecord(self._count)


class _FakeSession:
    def __init__(self, count):
        self._count = count
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False

    def run(self, _query):
        return _FakeResult(self._count)


class _FakeDriver:
    def __init__(self, count):
        self._count = count
        self.closed = False

    def session(self, database=None):
        return _FakeSession(self._count)

    def close(self):
        self.closed = True


def test_count_returns_int_one():
    drv = _FakeDriver(1)
    n = sh.count_is_me_global_speakers(driver=drv)
    assert isinstance(n, int)
    assert n == 1
    # Provided driver is NOT closed by the helper.
    assert drv.closed is False


def test_count_returns_int_many():
    drv = _FakeDriver(4)
    n = sh.count_is_me_global_speakers(driver=drv)
    assert n == 4


def test_count_returns_int_zero():
    drv = _FakeDriver(0)
    assert sh.count_is_me_global_speakers(driver=drv) == 0


def test_check_anchor_health_ok():
    with mock.patch.object(sh, "count_is_me_global_speakers", return_value=1):
        assert sh.check_anchor_health(expected=1) == 1


def test_check_anchor_health_bad_raises():
    with mock.patch.object(sh, "count_is_me_global_speakers", return_value=4):
        with pytest.raises(AssertionError):
            sh.check_anchor_health(expected=1)


def test_check_anchor_health_custom_expected():
    with mock.patch.object(sh, "count_is_me_global_speakers", return_value=0):
        assert sh.check_anchor_health(expected=0) == 0
        with pytest.raises(AssertionError):
            sh.check_anchor_health(expected=1)
