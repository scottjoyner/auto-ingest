"""Regression: `run-all` must run all three independent media copiers.

The copiers handle audio / dashcam / bodycam media going to different
destinations; they are complementary, not fallbacks. A failure in one must not
silently skip the others. All tests are hermetic: the CLI is loaded from source
and its module-level helpers are monkeypatched, so no filesystem, network, DB,
credential or model access occurs.
"""
import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

CLI_PATH = Path(__file__).resolve().parent.parent / "bin" / "auto-ingest"
COPY_SCRIPTS = ("./audio_copy.sh", "./dashcam_copy.sh", "./bodycam_copy.sh")


@pytest.fixture
def cli():
    loader = SourceFileLoader("autoingest_cli_copy_stage", str(CLI_PATH))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = mod
    loader.exec_module(mod)
    return mod


def _run_all_args(cli):
    """Real parser namespace with only the copy stage enabled."""
    return cli.build_parser().parse_args([
        "run-all",
        "--skip-diarize", "--skip-ingest", "--skip-classify",
        "--skip-link", "--skip-yolo",
    ])


def _patch(cli, monkeypatch, codes, calls):
    monkeypatch.setattr(cli, "neo4j_env", lambda: {
        "NEO4J_URI": "bolt://x", "NEO4J_USER": "u",
        "NEO4J_PASSWORD": "p", "NEO4J_DB": "neo4j",
    })

    def fake_run(script, args, extra_env=None):
        calls.append(script)
        return codes.get(script, 0)

    monkeypatch.setattr(cli, "_run", fake_run)


def test_first_copier_failure_does_not_skip_the_others(cli, monkeypatch):
    """audio_copy fails; dashcam and bodycam must still be invoked."""
    calls = []
    _patch(cli, monkeypatch, {"./audio_copy.sh": 3}, calls)

    rc = cli.cmd_run_all(_run_all_args(cli))

    assert calls == list(COPY_SCRIPTS), (
        f"all three copiers must run even when the first fails; got {calls}"
    )
    assert rc == 3, "the first failure must still be reported as the exit status"


def test_later_copier_failure_is_reported_after_all_run(cli, monkeypatch):
    """A failure in a later copier is aggregated once all three have run."""
    calls = []
    _patch(cli, monkeypatch, {"./dashcam_copy.sh": 4}, calls)

    rc = cli.cmd_run_all(_run_all_args(cli))

    assert calls == list(COPY_SCRIPTS)
    assert rc == 4


def test_all_copiers_succeed(cli, monkeypatch):
    """Success path unchanged: all three run and the stage exits 0."""
    calls = []
    _patch(cli, monkeypatch, {}, calls)

    rc = cli.cmd_run_all(_run_all_args(cli))

    assert calls == list(COPY_SCRIPTS)
    assert rc == 0
