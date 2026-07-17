"""Tests for observability helpers: neo4j watchdog + logging setup.

Runs under system python3:  python3 -m pytest tests/test_observability.py
No live Neo4j required — the watchdog is exercised against an unreachable URI
so it returns False without touching any real graph.
"""
from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_watchdog_imports_and_returns_bool():
    wd = _load(REPO / "scripts" / "neo4j_watchdog.py", "neo4j_watchdog")
    # Point at an unreachable port; must return False (never raise).
    ok = wd.check_neo4j(uri="bolt://127.0.0.1:59999", user="neo4j",
                        password="nope", db="neo4j", alert=False)
    assert isinstance(ok, bool)
    assert ok is False


def test_watchdog_classify_error():
    wd = _load(REPO / "scripts" / "neo4j_watchdog.py", "neo4j_watchdog")
    assert wd.classify_error(RuntimeError("MemoryPoolOutOfMemory: boom")) == "oom"
    assert wd.classify_error(RuntimeError("ServiceUnavailable: gone")) == "down"
    assert wd.classify_error(ValueError("something else")) == "error"


def test_watchdog_send_alert_noop_without_url(monkeypatch):
    wd = _load(REPO / "scripts" / "neo4j_watchdog.py", "neo4j_watchdog")
    monkeypatch.delenv("WATCHDOG_ALERT_URL", raising=False)
    assert wd.send_alert("down", "x") is False


def test_setup_logging_creates_rotating_handler(tmp_path):
    sl = _load(REPO / "tools" / "setup_logging.py", "setup_logging")
    path = sl.setup_logging(name="test-log", log_dir=tmp_path, console=False)
    assert path is not None
    assert path.exists() or path.parent.exists()
    logging.getLogger("shorts").info("hello rotating world")
    # A RotatingFileHandler for our file should be attached to root.
    import logging.handlers as h
    root = logging.getLogger()
    assert any(isinstance(hd, h.RotatingFileHandler)
               and getattr(hd, "baseFilename", "").endswith("test-log.log")
               for hd in root.handlers)
    # Clean up the handler so we don't leak into other tests.
    for hd in list(root.handlers):
        if isinstance(hd, h.RotatingFileHandler):
            root.removeHandler(hd)
            hd.close()
