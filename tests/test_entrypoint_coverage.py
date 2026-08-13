from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from auto_ingest import transcript_entrypoint, yolo_entrypoint


@pytest.mark.parametrize(("raw", "expected"), [("0", 0), ("7", 7), (None, 4)])
def test_env_int(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("UNIT_INT", raising=False)
    else:
        monkeypatch.setenv("UNIT_INT", raw)
    assert transcript_entrypoint._env_int("UNIT_INT", 4) == expected


def test_env_int_rejects_negative(monkeypatch):
    monkeypatch.setenv("UNIT_INT", "-1")
    with pytest.raises(ValueError, match="non-negative"):
        transcript_entrypoint._env_int("UNIT_INT", 4)


def test_transcript_build_command_defaults_and_optional_flags(monkeypatch):
    import auto_ingest_config as cfg

    monkeypatch.setattr(cfg, "get_fileserver_root", lambda: "/fs")
    monkeypatch.setattr(cfg, "get_dashcam_root", lambda: "/dash")
    monkeypatch.setattr(cfg, "get_audio_root", lambda: "/audio")
    monkeypatch.setattr(
        cfg,
        "get_neo4j_config",
        lambda: {"uri": "bolt://graph", "user": "neo", "password": "secret"},
    )
    for name in (
        "FILESERVER_ROOT",
        "DASHCAM_ROOT",
        "AUDIO_ROOT",
        "NEO4J_URI",
        "NEO4J_USER",
        "NEO4J_PASSWORD",
        "SCAN_ROOTS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LIMIT", "2")
    monkeypatch.setenv("DRY_RUN", "1")
    monkeypatch.setenv("FORCE", "1")

    cmd, env = transcript_entrypoint.build_command()
    assert cmd[:4] == [sys.executable, "-u", "-m", "auto_ingest.ingest.transcripts"]
    assert cmd[-4:] == ["--limit", "2", "--dry-run", "--force"]
    assert env["FILESERVER_ROOT"] == "/fs"
    assert env["DASHCAM_ROOT"] == "/dash"
    assert env["AUDIO_ROOT"] == "/audio"
    assert env["NEO4J_URI"] == "bolt://graph"
    assert env["NEO4J_USER"] == "neo"
    assert env["NEO4J_PASSWORD"] == "secret"
    assert "/fs/bodycam" in env["SCAN_ROOTS"]


def test_transcript_build_command_env_overrides(monkeypatch):
    import auto_ingest_config as cfg

    monkeypatch.setattr(cfg, "get_fileserver_root", lambda: pytest.fail("unused"))
    monkeypatch.setattr(cfg, "get_dashcam_root", lambda: pytest.fail("unused"))
    monkeypatch.setattr(cfg, "get_audio_root", lambda: pytest.fail("unused"))
    monkeypatch.setattr(
        cfg,
        "get_neo4j_config",
        lambda: {"uri": "x", "user": "x", "password": "x"},
    )
    monkeypatch.setenv("FILESERVER_ROOT", "/efs")
    monkeypatch.setenv("DASHCAM_ROOT", "/edash")
    monkeypatch.setenv("AUDIO_ROOT", "/eaudio")
    monkeypatch.setenv("NEO4J_URI", "bolt://env")
    monkeypatch.setenv("NEO4J_USER", "env-user")
    monkeypatch.setenv("NEO4J_PASSWORD", "env-pass")
    monkeypatch.setenv("SCAN_ROOTS", "/one,/two")
    monkeypatch.setenv("LIMIT", "0")
    monkeypatch.setenv("DRY_RUN", "0")
    monkeypatch.setenv("FORCE", "0")
    cmd, env = transcript_entrypoint.build_command()
    assert "--limit" not in cmd
    assert "--dry-run" not in cmd
    assert "--force" not in cmd
    assert env["SCAN_ROOTS"] == "/one,/two"
    assert env["NEO4J_PASSWORD"] == "env-pass"


def test_transcript_main_prefixes_and_calls(monkeypatch):
    monkeypatch.setattr(
        transcript_entrypoint,
        "build_command",
        lambda: (["python", "job"], {"X": "1"}),
    )
    monkeypatch.setenv("IONICE", "ionice -c2")
    monkeypatch.setenv("NICE", "nice -n 3")
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    captured = {}

    def fake_call(cmd, env):
        captured.update(cmd=cmd, env=env)
        return 7

    monkeypatch.setattr(transcript_entrypoint.subprocess, "call", fake_call)
    assert transcript_entrypoint.main() == 7
    assert captured["cmd"] == [
        "ionice",
        "-c2",
        "nice",
        "-n",
        "3",
        "python",
        "job",
    ]
    assert captured["env"] == {"X": "1"}
    with pytest.raises(ValueError):
        transcript_entrypoint.main(["unexpected"])


def test_transcript_main_without_optional_tools(monkeypatch):
    monkeypatch.setattr(
        transcript_entrypoint, "build_command", lambda: (["python", "job"], {})
    )
    monkeypatch.setattr("shutil.which", lambda _name: None)
    monkeypatch.setattr(
        transcript_entrypoint.subprocess,
        "call",
        lambda cmd, env: 0 if cmd == ["python", "job"] and env == {} else 9,
    )
    assert transcript_entrypoint.main() == 0


def test_yolo_internal_argv_config_and_env(monkeypatch):
    import auto_ingest_config as cfg

    monkeypatch.setattr(
        cfg,
        "get_neo4j_config",
        lambda: {"uri": "bolt://cfg", "user": "u", "password": "p"},
    )
    for name in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    args = yolo_entrypoint.build_internal_argv(["--limit", "4"])
    assert args[args.index("--neo4j-uri") + 1] == "bolt://cfg"
    assert args[args.index("--neo4j-user") + 1] == "u"
    assert args[args.index("--neo4j-pass") + 1] == "p"
    assert args[-2:] == ["--limit", "4"]

    monkeypatch.setenv("NEO4J_URI", "bolt://env")
    monkeypatch.setenv("NEO4J_USER", "eu")
    monkeypatch.setenv("NEO4J_PASSWORD", "ep")
    args = yolo_entrypoint.build_internal_argv()
    assert args[args.index("--neo4j-uri") + 1] == "bolt://env"
    assert args[args.index("--neo4j-pass") + 1] == "ep"


def test_yolo_main_restores_sys_argv(monkeypatch):
    import auto_ingest.dashcam as package

    fake = SimpleNamespace(main=lambda: None)
    monkeypatch.setitem(sys.modules, "auto_ingest.dashcam.yolo_embeddings", fake)
    monkeypatch.setattr(package, "yolo_embeddings", fake, raising=False)
    monkeypatch.setattr(
        yolo_entrypoint,
        "build_internal_argv",
        lambda extra=None: ["internal", *(extra or [])],
    )
    original = list(sys.argv)
    assert yolo_entrypoint.main(["--x"]) == 0
    assert sys.argv == original

    def boom():
        assert sys.argv == ["internal"]
        raise RuntimeError("boom")

    fake.main = boom
    with pytest.raises(RuntimeError, match="boom"):
        yolo_entrypoint.main()
    assert sys.argv == original
