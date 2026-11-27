"""Hermetic regression tests for the link-speakers re-anchor exit status.

N11 re-anchors the canonical ``is_me`` GlobalSpeaker after a successful link
pass. If that re-anchor fails, a caller of ``auto-ingest link-speakers`` (or
``run-all``) must see a non-zero exit: otherwise the command reports success
while the identity invariant it was asked to restore is unverified.

No Neo4j, no model load, no live filesystem: every subprocess boundary is a
monkeypatched fake that returns a controlled status code.
"""
from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

CLI_PATH = Path(__file__).resolve().parent.parent / "bin" / "auto-ingest"


@pytest.fixture
def cli():
    """Load the CLI entrypoint in isolation, like test_auto_ingest_cli does."""
    name = "autoingest_cli_anchor_rc"
    loader = SourceFileLoader(name, str(CLI_PATH))
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod


class _LinkArgs:
    """Minimal args namespace for cmd_link_speakers."""

    limit = 0
    max_speakers = 0
    priority = None
    faiss = False
    more_aggressive = False
    rank_and_label = False
    promote = False
    promote_min_weight = 0.0
    dry_run = False
    state_file = ""


class _RunAllArgs:
    """Minimal args namespace for cmd_run_all with only the link stage on."""

    skip_copy = True
    skip_diarize = True
    skip_ingest = True
    skip_classify = True
    skip_link = False
    skip_yolo = True
    limit = 0
    max_speakers = 0
    faiss = False
    more_aggressive = False
    rank_and_label = False
    promote = False
    promote_min_weight = 0.0
    dry_run = False
    state_file = ""


def _fake_runs(cli, monkeypatch, anchor_rc: int) -> None:
    """Stub neo4j_env + both subprocess helpers; only the anchor returns anchor_rc."""
    monkeypatch.setattr(cli, "neo4j_env", lambda: {})

    def fake_run_module(mod, args, extra_env=None):  # noqa: ARG001
        return 0  # the linker itself succeeded in every case here

    def fake_run(script, args, extra_env=None):  # noqa: ARG001
        if script == "scripts/anchor_speaker_me.py":
            return anchor_rc
        return 0

    monkeypatch.setattr(cli, "_run_module", fake_run_module)
    monkeypatch.setattr(cli, "_run", fake_run)


def test_link_speakers_anchor_failure_propagates(cli, monkeypatch):
    """A failed re-anchor must not be reported as a successful link pass."""
    _fake_runs(cli, monkeypatch, anchor_rc=7)
    assert cli.cmd_link_speakers(_LinkArgs()) == 7


def test_run_all_propagates_anchor_failure(cli, monkeypatch):
    """run-all must surface a failed re-anchor through its own exit code."""
    _fake_runs(cli, monkeypatch, anchor_rc=7)
    assert cli.cmd_run_all(_RunAllArgs()) == 7


def test_link_speakers_still_succeeds_when_anchor_succeeds(cli, monkeypatch):
    """Guard against over-correcting: all-zero remains zero."""
    _fake_runs(cli, monkeypatch, anchor_rc=0)
    assert cli.cmd_link_speakers(_LinkArgs()) == 0
