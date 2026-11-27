"""Regression tests for run-all forwarding the linker's --promote-min-weight.

The run-all command delegates its link stage to ``cmd_link_speakers`` through
``_link_args``. That helper copies ``args.promote_min_weight`` from the parsed
namespace. If the run-all subparser does not define the flag, every ordinary
``auto-ingest run-all`` invocation (which links by default) dies with an
unhandled ``AttributeError`` instead of running the pipeline.

Hermetic: only argparse namespaces and in-memory fakes; no Neo4j, no subprocess,
no filesystem writes beyond pytest's own tmp dirs.
"""

import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

CLI_PATH = Path(__file__).resolve().parent.parent / "bin" / "auto-ingest"


@pytest.fixture
def cli():
    loader = SourceFileLoader("autoingest_cli_runall", str(CLI_PATH))
    spec = importlib.util.spec_from_loader("autoingest_cli_runall", loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["autoingest_cli_runall"] = mod
    loader.exec_module(mod)
    return mod


def test_run_all_link_args_has_promote_min_weight(cli):
    """The default run-all namespace must satisfy _link_args (no crash)."""
    args = cli.build_parser().parse_args(["run-all"])
    link = cli._link_args(args)
    assert link.promote_min_weight == 0.0


def test_run_all_forwards_promote_min_weight(cli):
    """An explicit threshold on run-all reaches the linker namespace."""
    args = cli.build_parser().parse_args(
        ["run-all", "--promote-min-weight", "42.5"]
    )
    link = cli._link_args(args)
    assert link.promote_min_weight == 42.5
    assert link.promote is False  # control: flag itself still opt-in


def test_run_all_promote_with_weight_reaches_linker(cli, monkeypatch):
    """cmd_run_all passes the parsed weight to the linker, rc stays clean."""
    monkeypatch.setattr(cli, "neo4j_env", lambda: {
        "NEO4J_URI": "bolt://x", "NEO4J_USER": "u",
        "NEO4J_PASSWORD": "p", "NEO4J_DB": "db",
    })
    seen = {}

    def fake_link(link_args):
        seen["link_args"] = link_args
        return 0

    monkeypatch.setattr(cli, "cmd_link_speakers", fake_link)
    args = cli.build_parser().parse_args(
        ["run-all", "--skip-copy", "--skip-diarize", "--skip-ingest",
         "--skip-classify", "--skip-yolo", "--promote", "--promote-min-weight", "7.25"]
    )
    rc = cli.cmd_run_all(args)
    assert rc == 0
    assert seen["link_args"].promote is True
    assert seen["link_args"].promote_min_weight == 7.25


def test_link_speakers_promote_min_weight_unchanged(cli):
    """Control: the link-speakers parser still owns the flag with its default."""
    args = cli.build_parser().parse_args(["link-speakers"])
    assert args.promote_min_weight == 0.0
