"""
Tests for the new per-trip / co-occurrence MCP tools in scott_graph_mcp.

Verifies the tool functions are importable and callable. The DB-touching body
is wrapped by ``@_safe``; we monkeypatch ``_driver`` so no live Neo4j is
contacted and assert the tools return a plain string result.

NOTE: scott_graph_mcp imports the optional ``mcp`` SDK at module load, which is
not installed in CI/test interpreters. We therefore import it lazily inside the
tests (and skip if the SDK is absent) so collection never fails.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class _FakeRows:
    def __init__(self, rows):
        self._rows = rows

    def data(self):
        return self._rows


class _FakeSession:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, **params):
        return _FakeRows([])


class _FakeDriver:
    def session(self, database=None):
        return _FakeSession()

    def close(self):
        pass


def _load():
    try:
        import scott_graph_mcp as mcp_mod  # noqa: E402
    except ImportError:
        pytest.skip("mcp SDK not installed")
    return mcp_mod


@pytest.fixture
def fake_driver():
    mcp_mod = _load()
    with mock.patch.object(mcp_mod, "_driver", return_value=_FakeDriver()):
        yield mcp_mod


def test_find_speaker_importable_and_callable(fake_driver):
    assert callable(fake_driver.find_speaker)
    out = fake_driver.find_speaker("Scott")
    assert isinstance(out, str)


def test_who_was_with_importable_and_callable(fake_driver):
    assert callable(fake_driver.who_was_with)
    out = fake_driver.who_was_with("2026-06-14")
    assert isinstance(out, str)


def test_find_speaker_empty_name(fake_driver):
    assert "give a speaker name" in fake_driver.find_speaker("")


def test_who_was_with_empty_date(fake_driver):
    assert "give a date" in fake_driver.who_was_with("")
