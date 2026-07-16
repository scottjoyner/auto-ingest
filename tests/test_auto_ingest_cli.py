import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

CLI_PATH = Path(__file__).resolve().parent.parent / "bin" / "auto-ingest"


def _load_cli():
    loader = SourceFileLoader("autoingest_cli", str(CLI_PATH))
    spec = importlib.util.spec_from_loader("autoingest_cli", loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["autoingest_cli"] = mod
    loader.exec_module(mod)
    return mod


@pytest.fixture
def cli():
    return _load_cli()


# ---------------------------------------------------------------------------
# build_parser
# ---------------------------------------------------------------------------
def test_build_parser_has_all_subcommands(cli):
    p = cli.build_parser()
    choices = list(p._subparsers._group_actions[0].choices.keys())  # type: ignore[attr-defined]
    assert set(choices) == {
        "caps", "status", "link-speakers", "whoami", "tiktok",
        "worker", "ingest", "run-all", "shorts",
    }


# ---------------------------------------------------------------------------
# neo4j_env
# ---------------------------------------------------------------------------
def test_neo4j_env_from_config(cli, monkeypatch):
    monkeypatch.delenv("NEO4J_URI", raising=False)
    monkeypatch.delenv("NEO4J_USER", raising=False)
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    monkeypatch.delenv("NEO4J_DB", raising=False)

    fake_cfg = {"uri": "bolt://db:7687", "user": "neo", "password": "secret"}

    import auto_ingest_config

    monkeypatch.setattr(auto_ingest_config, "get_neo4j_config", lambda: fake_cfg)

    env = cli.neo4j_env()
    assert env["NEO4J_URI"] == "bolt://db:7687"
    assert env["NEO4J_USER"] == "neo"
    assert env["NEO4J_PASSWORD"] == "secret"
    assert env["NEO4J_DB"] == "neo4j"


def test_neo4j_env_env_overrides_config(cli, monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "bolt://env:7687")
    monkeypatch.setenv("NEO4J_USER", "envuser")
    monkeypatch.setenv("NEO4J_PASSWORD", "envpass")

    import auto_ingest_config

    monkeypatch.setattr(
        auto_ingest_config, "get_neo4j_config", lambda: {"uri": "bolt://cfg:7687",
                                                         "user": "cfg", "password": "cfgpass"}
    )

    env = cli.neo4j_env()
    assert env["NEO4J_URI"] == "bolt://env:7687"
    assert env["NEO4J_USER"] == "envuser"
    assert env["NEO4J_PASSWORD"] == "envpass"


# ---------------------------------------------------------------------------
# cmd_caps
# ---------------------------------------------------------------------------
def test_cmd_caps(cli, capfd):
    rc = cli.cmd_caps(object())
    out = capfd.readouterr().out
    assert rc == 0
    assert "HOST:" in out
    assert "GPU backend:" in out


# ---------------------------------------------------------------------------
# cmd_status
# ---------------------------------------------------------------------------
class _FakeResult:
    def __init__(self, c):
        self._c = c

    def __getitem__(self, key):
        return self._c


class _FakeSession:
    def __init__(self, counts):
        self._counts = counts
        self._i = 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def run(self, q):
        self._i += 1
        return self

    def single(self):
        return _FakeResult(c=self._counts[self._i - 1])


class _FakeDriver:
    def __init__(self, counts):
        self._counts = counts

    def session(self, **kwargs):
        return _FakeSession(self._counts)

    def close(self):
        pass


def test_cmd_status_ok(cli, monkeypatch, capfd):
    counts = [10, 3, 2, 4, 5, 1, 7]
    driver = _FakeDriver(counts)

    fake_neo4j = type("neo4j", (), {"GraphDatabase": type("GD", (), {"driver": staticmethod(lambda *a, **k: driver)})})
    monkeypatch.setitem(sys.modules, "neo4j", fake_neo4j)

    rc = cli.cmd_status(object())
    out = capfd.readouterr().out
    assert rc == 0
    assert "Speaker" in out
    assert "GlobalSpeaker" in out
    assert "10" in out


def test_cmd_status_driver_failure(cli, monkeypatch, capfd):
    def boom(*a, **k):
        raise RuntimeError("connection refused")

    fake_neo4j = type("neo4j", (), {"GraphDatabase": type("GD", (), {"driver": staticmethod(boom)})})
    monkeypatch.setitem(sys.modules, "neo4j", fake_neo4j)

    rc = cli.cmd_status(object())
    err = capfd.readouterr().err
    assert rc == 1
    assert "neo4j query failed" in err


# ---------------------------------------------------------------------------
# cmd_run_all
# ---------------------------------------------------------------------------
def _make_run_all_args(**over):
    class A:
        pass

    a = A()
    a.skip_copy = False
    a.skip_diarize = False
    a.skip_ingest = False
    a.skip_classify = False
    a.skip_link = False
    a.skip_yolo = False
    a.limit = 0
    a.max_speakers = 0
    a.faiss = False
    a.more_aggressive = False
    a.rank_and_label = False
    a.dry_run = False
    a.state_file = ""
    for k, v in over.items():
        setattr(a, k, v)
    return a


def test_cmd_run_all_sequence(cli, monkeypatch):
    calls = []

    def fake_run_module(mod, args, extra_env=None):
        calls.append(("module", mod, list(args)))
        return 0

    def fake_run(script, args, extra_env=None):
        calls.append(("script", script, list(args)))
        return 0

    monkeypatch.setattr(cli, "_run_module", fake_run_module)
    monkeypatch.setattr(cli, "_run", fake_run)

    rc = cli.cmd_run_all(_make_run_all_args())
    assert rc == 0

    scripts = [c for c in calls if c[0] == "script"]
    modules = [c for c in calls if c[0] == "module"]

    assert ("script", "./run_ingest_all.sh", []) in scripts
    assert ("script", "./audio_copy.sh", []) in scripts
    assert ("script", "./dashcam_copy.sh", []) in scripts
    assert ("script", "./bodycam_copy.sh", []) in scripts

    mod_names = [c[1] for c in modules]
    assert "speakers" in mod_names
    assert "speakers_reconcile" in mod_names
    assert "01_precompute_music_segments" in mod_names
    assert "02_classify_lyrics" in mod_names
    assert "auto_ingest.diarize.link_global_speakers" in mod_names
    assert "auto_ingest.dashcam.yolo_embeddings" in mod_names


def test_cmd_run_all_skip_flags(cli, monkeypatch):
    calls = []

    def fake_run_module(mod, args, extra_env=None):
        calls.append(("module", mod))
        return 0

    def fake_run(script, args, extra_env=None):
        calls.append(("script", script))
        return 0

    monkeypatch.setattr(cli, "_run_module", fake_run_module)
    monkeypatch.setattr(cli, "_run", fake_run)

    args = _make_run_all_args(skip_copy=True, skip_diarize=True, skip_ingest=True,
                              skip_classify=True, skip_yolo=True)
    rc = cli.cmd_run_all(args)
    assert rc == 0
    # Only the link stage should run
    assert calls == [("module", "auto_ingest.diarize.link_global_speakers")]


# ---------------------------------------------------------------------------
# cmd_link_speakers
# ---------------------------------------------------------------------------
def _make_link_args(**over):
    class A:
        pass

    a = A()
    a.limit = 0
    a.max_speakers = 0
    a.priority = None
    a.faiss = False
    a.more_aggressive = False
    a.rank_and_label = False
    a.dry_run = False
    a.state_file = ""
    for k, v in over.items():
        setattr(a, k, v)
    return a


def test_cmd_link_speakers_flags(cli, monkeypatch):
    captured = {}

    def fake_run_module(mod, args, extra_env=None):
        captured["mod"] = mod
        captured["args"] = list(args)
        return 0

    monkeypatch.setattr(cli, "_run_module", fake_run_module)

    rc = cli.cmd_link_speakers(_make_link_args())
    assert rc == 0
    assert captured["mod"] == "auto_ingest.diarize.link_global_speakers"
    args = captured["args"]
    assert "--global-prefilter" in args
    assert "--skip-already-linked" in args
    assert "--holdout" in args
    assert "--priority-name" in args
    idx = args.index("--priority-name")
    assert args[idx + 1] == "Scott|Kipnerter"


def test_cmd_link_speakers_priority_override(cli, monkeypatch):
    captured = {}

    def fake_run_module(mod, args, extra_env=None):
        captured["args"] = list(args)
        return 0

    monkeypatch.setattr(cli, "_run_module", fake_run_module)

    rc = cli.cmd_link_speakers(_make_link_args(priority="Alice|Bob", faiss=True,
                                              more_aggressive=True, rank_and_label=True,
                                              dry_run=True, limit=5,
                                              max_speakers=10, state_file="s.json"))
    assert rc == 0
    args = captured["args"]
    idx = args.index("--priority-name")
    assert args[idx + 1] == "Alice|Bob"
    assert "--faiss-prefilter" in args
    assert "--global-include-tentative" in args
    assert "--rank-and-label" in args
    assert "--dry-run" in args
    assert "--limit-speakers" in args and args[args.index("--limit-speakers") + 1] == "5"
    assert "--max-speakers" in args and args[args.index("--max-speakers") + 1] == "10"
    assert "--state-file" in args and args[args.index("--state-file") + 1] == "s.json"
    # more_aggressive widens global-thresh
    idx_t = args.index("--global-thresh")
    assert args[idx_t + 1] == "0.74"
