"""Regression: `auto-ingest run-all` must expose the linker's promote-weight knob.

`cmd_run_all()` delegates to `cmd_link_speakers(_link_args(args))`, and
`_link_args()` reads `args.promote_min_weight`.  The `run-all` subparser must
therefore define that option exactly like the `link-speakers` subparser does;
otherwise the parsed namespace lacks the attribute and every real `run-all`
invocation raises AttributeError before the link stage can run.  The existing
CLI tests build the namespace by hand, so they never exercised the real parser.

All tests here are hermetic: they load the CLI from source, parse argv, and
monkeypatch the subprocess runners.  No Neo4j, network, model load, or IO.
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


def test_run_all_default_namespace_builds_link_args(cli):
    """A real `run-all` parse must produce a namespace `_link_args` can read."""
    args = cli.build_parser().parse_args(["run-all"])
    linked = cli._link_args(args)
    assert linked.promote_min_weight == 0.0


def test_run_all_parser_accepts_and_forwards_promote_min_weight(cli):
    """The documented --promote-min-weight knob must be usable from run-all."""
    args = cli.build_parser().parse_args(
        ["run-all", "--promote", "--promote-min-weight", "1.5"]
    )
    assert args.promote_min_weight == 1.5
    assert cli._link_args(args).promote_min_weight == 1.5


def test_cmd_run_all_with_real_parser_forwards_weight_without_crashing(cli, monkeypatch):
    """End-to-end: parsed run-all args reach the linker without AttributeError."""
    monkeypatch.setattr(cli, "neo4j_env", lambda: {
        "NEO4J_URI": "bolt://x", "NEO4J_USER": "u",
        "NEO4J_PASSWORD": "p", "NEO4J_DB": "db",
    })
    calls = []

    def fake_run_module(mod, args, extra_env=None):
        calls.append(("module", mod, list(args)))
        return 0

    def fake_run(script, args, extra_env=None):
        calls.append(("script", script, list(args)))
        return 0

    monkeypatch.setattr(cli, "_run_module", fake_run_module)
    monkeypatch.setattr(cli, "_run", fake_run)

    args = cli.build_parser().parse_args([
        "run-all", "--skip-copy", "--skip-diarize", "--skip-ingest",
        "--skip-classify", "--skip-yolo", "--promote",
        "--promote-min-weight", "1.5",
    ])
    rc = cli.cmd_run_all(args)
    assert rc == 0
    linker = [c for c in calls if c[0] == "module" and
              c[1] == "auto_ingest.diarize.link_global_speakers"]
    assert len(linker) == 1
    lw = linker[0][2]
    assert "--promote" in lw
    idx = lw.index("--promote-min-weight")
    assert lw[idx + 1] == "1.5"
