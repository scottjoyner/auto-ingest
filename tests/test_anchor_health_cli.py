"""Tests for the ``scripts/check_anchor_health.py`` validator's ``--json`` mode.

The JSON mode is consumed by CI gates, so it must emit parseable JSON on the
failure paths too -- exactly when the gate needs a machine-readable verdict.
No live Neo4j: the two imported helpers are monkeypatched, and only the CLI's
own rendering/exit-status logic runs.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from importlib.machinery import SourceFileLoader
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_anchor_health.py"

_loader = SourceFileLoader("check_anchor_health_cli_under_test", str(SCRIPT_PATH))
_spec = importlib.util.spec_from_loader(_loader.name, _loader)
check_anchor_health_cli = importlib.util.module_from_spec(_spec)
_loader.exec_module(check_anchor_health_cli)


def _run_main(*argv: str) -> tuple[int, str]:
    """Run the validator's main() with the given argv and capture stdout."""
    with mock.patch.object(sys, "argv", [str(SCRIPT_PATH), *argv]):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = check_anchor_health_cli.main()
    return rc, buf.getvalue().strip()


def test_assertion_failure_json_is_parseable_and_carries_facts():
    def unhealthy(expected=1):
        raise AssertionError("Expected exactly 1 'is_me' GlobalSpeaker node, found 4.")

    with mock.patch.object(check_anchor_health_cli, "check_anchor_health", unhealthy), \
         mock.patch.object(check_anchor_health_cli, "count_is_me_global_speakers",
                           return_value=4):
        rc, out = _run_main("--json")

    assert rc == 1, "unhealthy anchor must still exit non-zero"
    data = json.loads(out)  # must not raise: this is a machine-readable gate
    assert data["ok"] is False
    assert data["expected"] == 1
    assert data["actual"] == 4
    assert "Expected exactly 1" in data["error"]


def test_assertion_failure_json_with_unknown_count_is_parseable():
    def unhealthy(expected=1):
        raise AssertionError("anchor unhealthy")

    with mock.patch.object(check_anchor_health_cli, "check_anchor_health", unhealthy), \
         mock.patch.object(check_anchor_health_cli, "count_is_me_global_speakers",
                           side_effect=RuntimeError("cannot recount")):
        rc, out = _run_main("--json")

    assert rc == 1
    data = json.loads(out)
    assert data["ok"] is False
    assert data["actual"] is None  # previously the literal string "null" leaked in
    assert "anchor unhealthy" in data["error"]


def test_connection_error_json_is_parseable():
    with mock.patch.object(check_anchor_health_cli, "check_anchor_health",
                           side_effect=RuntimeError("db down")):
        rc, out = _run_main("--json")

    assert rc == 2, "connection/query failures keep their distinct exit code"
    data = json.loads(out)
    assert data["ok"] is False
    assert "db down" in data["error"]


def test_success_json_remains_parseable():
    with mock.patch.object(check_anchor_health_cli, "check_anchor_health",
                           return_value=1):
        rc, out = _run_main("--json")

    assert rc == 0
    data = json.loads(out)
    assert data == {"ok": True, "expected": 1, "actual": 1}


def test_text_failure_exit_code_unchanged():
    def unhealthy(expected=1):
        raise AssertionError("anchor unhealthy")

    with mock.patch.object(check_anchor_health_cli, "check_anchor_health", unhealthy), \
         mock.patch.object(check_anchor_health_cli, "count_is_me_global_speakers",
                           return_value=4):
        rc, out = _run_main()

    assert rc == 1
    assert "ANCHOR UNHEALTHY" in out
