"""
Tests for the new per-trip / co-occurrence MCP tools in scott_graph_mcp.

Verifies the tool functions are importable and callable. The DB-touching body
is wrapped by ``@_safe``; we monkeypatch ``_driver`` so no live Neo4j is
contacted and assert the tools return a plain string result.

NOTE: scott_graph_mcp imports the optional ``mcp`` SDK at module load. When the
SDK is absent we stand in for its registration decorator only; the tool bodies
under test are the real functions, called with a fake Neo4j driver.
"""
from __future__ import annotations

import importlib
import sys
import types
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


def _load(monkeypatch):
    try:
        return importlib.import_module("scott_graph_mcp")
    except ModuleNotFoundError as exc:
        if exc.name != "mcp":
            raise

    # Only the SDK's @tool() bookkeeping is stubbed; no tool logic is replaced.
    # Install the real server with `pip install -e ".[mcp]"`.
    mcp = types.ModuleType("mcp")
    server = types.ModuleType("mcp.server")
    fastmcp = types.ModuleType("mcp.server.fastmcp")

    class FastMCP:
        def __init__(self, name=None):
            self.name = name

        def tool(self, fn=None):
            return fn if fn is not None else (lambda f: f)

        def run(self, *args, **kwargs):
            raise RuntimeError("mcp SDK required to serve: pip install -e '.[mcp]'")

    fastmcp.FastMCP = FastMCP
    server.fastmcp = fastmcp
    mcp.server = server
    for name, module in (("mcp", mcp), ("mcp.server", server),
                         ("mcp.server.fastmcp", fastmcp)):
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules.pop("scott_graph_mcp", None)
    return importlib.import_module("scott_graph_mcp")


@pytest.fixture
def fake_driver(monkeypatch):
    mcp_mod = _load(monkeypatch)
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
