"""Sweep44: local `--root` ingest must not claim Nextcloud provenance.

Hermetic: the CLI module is imported from source, its process launcher and
Neo4j env builder are monkeypatched, and no script is executed. No DB,
network, credentials, mounts or live fleet state are involved.
"""
from __future__ import annotations

import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CLI_PATH = REPO / "bin" / "auto-ingest"


@pytest.fixture
def cli(monkeypatch):
    loader = SourceFileLoader("auto_ingest_cli_sweep44", str(CLI_PATH))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    def fake_run(script, args, extra_env=None):
        calls.append((script, list(args)))
        return 0

    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(module, "neo4j_env", lambda: {})
    module.calls = calls
    return module


def _forwarded_source(cli):
    assert len(cli.calls) == 1, cli.calls
    argv = cli.calls[0][1]
    i = argv.index("--source")
    return argv[i + 1]


def test_local_root_defaults_to_local_source(cli, tmp_path):
    args = cli.build_parser().parse_args(
        ["ingest", "--root", str(tmp_path), "--dry-run"])
    assert cli.cmd_ingest(args) == 0
    assert cli.calls[0][0] == "ingest_media.py"
    assert _forwarded_source(cli) == "local"


def test_explicit_source_still_wins_for_local_root(cli, tmp_path):
    args = cli.build_parser().parse_args(
        ["ingest", "--root", str(tmp_path), "--source", "dashcam"])
    assert cli.cmd_ingest(args) == 0
    assert _forwarded_source(cli) == "dashcam"


def test_nextcloud_mode_keeps_nextcloud_source(cli):
    args = cli.build_parser().parse_args(
        ["ingest", "--nextcloud-url", "https://example.invalid/dav"])
    assert cli.cmd_ingest(args) == 0
    assert _forwarded_source(cli) == "nextcloud"
