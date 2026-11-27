"""run-all must not silently drop independent stages when one fails.

`cmd_run_all` used to chain the three copy sources (and the two classification
stages) with `or`. `subprocess` returns 0 on success, so `or` continued past a
*successful* stage but short-circuited on the first *failure*: a failed audio
copy silently suppressed both the dashcam and bodycam copies, and a failed
precompute silently suppressed lyrics classification. `step()` then prints
"(continuing)" and moves on, so the only trace was one generic warning - exactly
the sort of omission that looks like a quiet stretch of road.

The existing sequence test mocks every stage to succeed, so it never exercised
the short-circuit. These tests fail to zero in on the failure path.
"""
from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

CLI_PATH = Path(__file__).resolve().parent.parent / "bin" / "auto-ingest"


@pytest.fixture
def cli(monkeypatch):
    loader = SourceFileLoader("autoingest_cli_omission", str(CLI_PATH))
    spec = importlib.util.spec_from_loader("autoingest_cli_omission", loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["autoingest_cli_omission"] = mod
    loader.exec_module(mod)
    # Keep the test hermetic: no config.yaml, no env, no Neo4j.
    monkeypatch.setattr(mod, "neo4j_env", lambda: {})
    return mod


def _args(**over):
    class A:
        pass

    a = A()
    a.skip_copy = False
    a.skip_diarize = True
    a.skip_ingest = True
    a.skip_classify = True
    a.skip_link = True
    a.skip_yolo = True
    a.limit = 0
    a.max_speakers = 0
    a.faiss = False
    a.more_aggressive = False
    a.rank_and_label = False
    a.promote = False
    a.promote_min_weight = 0.0
    a.dry_run = False
    a.state_file = ""
    for k, v in over.items():
        setattr(a, k, v)
    return a


def test_every_copy_source_runs_when_the_first_one_fails(cli, monkeypatch):
    calls = []

    def fake_run(script, args, extra_env=None):
        calls.append(script)
        return 7 if script == "./audio_copy.sh" else 0

    monkeypatch.setattr(cli, "_run", fake_run)

    rc = cli.cmd_run_all(_args())

    # The other two media sources are independent of the audio copy; dropping
    # them because audio failed is a silent omission of data.
    assert calls == ["./audio_copy.sh", "./dashcam_copy.sh", "./bodycam_copy.sh"]
    # The overall status must still report the failure.
    assert rc == 7


def test_lyrics_classification_runs_when_precompute_fails(cli, monkeypatch):
    calls = []

    def fake_run_module(mod, args, extra_env=None):
        calls.append(mod)
        return 5 if mod == "01_precompute_music_segments" else 0

    monkeypatch.setattr(cli, "_run_module", fake_run_module)

    rc = cli.cmd_run_all(_args(skip_copy=True, skip_classify=False))

    assert calls == ["01_precompute_music_segments", "02_classify_lyrics"]
    assert rc == 5
