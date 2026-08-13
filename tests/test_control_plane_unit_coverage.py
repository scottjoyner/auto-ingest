from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from auto_ingest import full_pipeline, metrics, operations, profile_runner, readiness, resources
from auto_ingest import transcript_entrypoint, worker_loop, yolo_entrypoint


class _ClosableDriver:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _Result:
    def __init__(self, *, single=None, data=None):
        self._single = single
        self._data = [] if data is None else data

    def single(self):
        return self._single

    def data(self):
        return self._data

    def consume(self):
        return self


class _Session:
    def __init__(self, results):
        self.results = iter(results)
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def run(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return next(self.results)


class _DriverWithSessions(_ClosableDriver):
    def __init__(self, sessions):
        super().__init__()
        self.sessions = iter(sessions)

    def session(self, *args, **kwargs):
        return next(self.sessions)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("0", 0), ("7", 7), (None, 4)],
)
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
    assert cmd[-5:] == ["--limit", "2", "--dry-run", "--force"][-5:]
    assert "--dry-run" in cmd and "--force" in cmd
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
    monkeypatch.setattr(transcript_entrypoint, "build_command", lambda: (["python", "job"], {"X": "1"}))
    monkeypatch.setenv("IONICE", "ionice -c2")
    monkeypatch.setenv("NICE", "nice -n 3")
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    captured = {}

    def fake_call(cmd, env):
        captured.update(cmd=cmd, env=env)
        return 7

    monkeypatch.setattr(transcript_entrypoint.subprocess, "call", fake_call)
    assert transcript_entrypoint.main() == 7
    assert captured["cmd"] == ["ionice", "-c2", "nice", "-n", "3", "python", "job"]
    assert captured["env"] == {"X": "1"}
    with pytest.raises(ValueError):
        transcript_entrypoint.main(["unexpected"])


def test_transcript_main_without_optional_tools(monkeypatch):
    monkeypatch.setattr(transcript_entrypoint, "build_command", lambda: (["python", "job"], {}))
    monkeypatch.setattr("shutil.which", lambda _name: None)
    monkeypatch.setattr(transcript_entrypoint.subprocess, "call", lambda cmd, env: 0 if cmd == ["python", "job"] else 9)
    assert transcript_entrypoint.main() == 0


def test_yolo_internal_argv_config_and_env(monkeypatch):
    import auto_ingest_config as cfg

    monkeypatch.setattr(cfg, "get_neo4j_config", lambda: {"uri": "bolt://cfg", "user": "u", "password": "p"})
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
    monkeypatch.setattr(yolo_entrypoint, "build_internal_argv", lambda extra=None: ["internal", *(extra or [])])
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


def test_resource_memory_parse_and_fallback(monkeypatch):
    monkeypatch.setattr(Path, "read_text", lambda self, encoding=None: "MemTotal: 1 kB\nMemAvailable: 4096 kB\n")
    assert resources._memory_available_mb() == 4
    monkeypatch.setattr(Path, "read_text", lambda self, encoding=None: "MemAvailable: nope kB\n")
    assert resources._memory_available_mb() == 0
    monkeypatch.setattr(Path, "read_text", lambda self, encoding=None: (_ for _ in ()).throw(OSError("nope")))
    assert resources._memory_available_mb() == 0


def test_resource_snapshot_handles_missing_cpu_and_load(monkeypatch, tmp_path):
    monkeypatch.setattr(resources.os, "cpu_count", lambda: None)
    monkeypatch.setattr(resources.os, "getloadavg", lambda: (_ for _ in ()).throw(OSError("no load")))
    monkeypatch.setattr(resources, "_memory_available_mb", lambda: 123)
    monkeypatch.setattr(resources.shutil, "disk_usage", lambda _p: SimpleNamespace(free=5 * 1024**3))
    snap = resources.snapshot(tmp_path)
    assert snap.cpu_count == 1
    assert snap.load1 == 0.0
    assert snap.memory_available_mb == 123
    assert snap.disk_free_gb == 5.0
    assert snap.load_per_cpu == 0.0


def test_metrics_collect_validation_and_values(monkeypatch):
    with pytest.raises(ValueError):
        metrics.collect_metrics(None, stale_after_sec=0)
    session = _Session(
        [
            _Result(data=[{"state": "RUNNING", "count": 2}]),
            _Result(single={"count": 2}),
            _Result(single={"count": 1}),
            _Result(single={"count": 9}),
        ]
    )
    driver = _DriverWithSessions([session])
    result = metrics.collect_metrics(driver, stale_after_sec=10)
    assert result == {
        "jobs_by_state": {"RUNNING": 2},
        "active_leases": 2,
        "stale_jobs": 1,
        "artifacts": 9,
    }


def test_metrics_collect_handles_empty_single_results():
    session = _Session([_Result(data=[]), _Result(single=None), _Result(single=None), _Result(single=None)])
    result = metrics.collect_metrics(_DriverWithSessions([session]))
    assert result["active_leases"] == 0
    assert result["stale_jobs"] == 0
    assert result["artifacts"] == 0


def test_metrics_main_closes_driver(monkeypatch, capsys):
    driver = _ClosableDriver()
    monkeypatch.setattr(metrics, "_driver", lambda: driver)
    monkeypatch.setattr(metrics, "collect_metrics", lambda d, stale_after_sec: {"jobs_by_state": {}})
    assert metrics.main(["--stale-after-sec", "5"]) == 0
    assert "auto_ingest_jobs" in capsys.readouterr().out
    assert driver.closed


def test_readiness_healthy_and_unhealthy(monkeypatch):
    ok_session = _Session([_Result(single={"ok": 1})])
    driver = _DriverWithSessions([ok_session])
    monkeypatch.setattr(readiness, "audit_schema", lambda _d: {"ok": True})
    monkeypatch.setattr(readiness, "collect_metrics", lambda _d: {"stale_jobs": 0})
    report = readiness.readiness(driver)
    assert report["ready"] is True
    assert report["metrics"] == {"stale_jobs": 0}

    bad_session = _Session([_Result(single=None)])
    report = readiness.readiness(_DriverWithSessions([bad_session]))
    assert report["ready"] is False
    assert report["metrics"] == {}


def test_readiness_main_output_modes(monkeypatch, capsys):
    for payload, args, expected_rc in [
        ({"ready": True}, [], 0),
        ({"ready": False}, [], 1),
        ({"ready": True, "x": 1}, ["--json"], 0),
    ]:
        driver = _ClosableDriver()
        monkeypatch.setattr(readiness, "_driver", lambda d=driver: d)
        monkeypatch.setattr(readiness, "readiness", lambda _d, p=payload: p)
        assert readiness.main(args) == expected_rc
        assert driver.closed
    output = capsys.readouterr().out
    assert "ready" in output
    assert "not ready" in output
    assert '"x": 1' in output


def test_profile_tasks_and_bound_runner(monkeypatch):
    custom_tasks = (SimpleNamespace(name="a"),)
    monkeypatch.setitem(profile_runner.PROFILES, "sync", custom_tasks)
    assert profile_runner.profile_tasks("sync") is custom_tasks
    monkeypatch.setattr(full_pipeline, "build_tasks", lambda: (SimpleNamespace(name="full"),))
    assert profile_runner.profile_tasks("full")[0].name == "full"

    calls = []
    monkeypatch.setattr(profile_runner, "default_job_key", lambda p: f"key:{p}")
    monkeypatch.setattr(profile_runner, "ensure_job", lambda d, k, p: calls.append(("ensure", k, p)))
    monkeypatch.setattr(profile_runner, "bind_plan", lambda d, k, t: calls.append(("bind", k, t)))
    monkeypatch.setattr(profile_runner, "run_profile", lambda d, p, **kw: calls.append(("run", p, kw)) or 6)
    assert profile_runner.run_bound_profile(object(), "sync", owner="w") == 6
    assert calls[0] == ("ensure", "key:sync", "sync")
    assert calls[1][0] == "bind"
    assert calls[2][2]["owner"] == "w"
    with pytest.raises(ValueError, match="unknown profile"):
        profile_runner.run_bound_profile(object(), "missing")


def test_profile_runner_main_closes_driver(monkeypatch):
    driver = _ClosableDriver()
    monkeypatch.setattr(profile_runner, "_driver", lambda: driver)
    monkeypatch.setattr(profile_runner, "run_bound_profile", lambda *a, **k: 3)
    assert profile_runner.main(["--profile", "sync"]) == 3
    assert driver.closed


def test_full_pipeline_state_root_and_main(monkeypatch, tmp_path):
    monkeypatch.setenv("STATE_ROOT", str(tmp_path / "state"))
    assert full_pipeline.state_root().is_dir()
    driver = _ClosableDriver()
    monkeypatch.setattr(full_pipeline, "_driver", lambda: driver)
    monkeypatch.setattr(full_pipeline, "default_job_key", lambda: "full:key")
    monkeypatch.setattr(full_pipeline, "build_tasks", lambda: (SimpleNamespace(name="one"),))
    calls = []
    monkeypatch.setattr(full_pipeline, "ensure_job", lambda *a: calls.append("ensure"))
    monkeypatch.setattr(full_pipeline, "bind_plan", lambda *a: calls.append("bind"))
    monkeypatch.setattr(full_pipeline, "run_profile", lambda *a, **k: calls.append("run") or 0)
    assert full_pipeline.main() == 0
    assert calls == ["ensure", "bind", "run"]
    assert driver.closed
    with pytest.raises(ValueError):
        full_pipeline.main(["bad"])


def test_operations_helpers(monkeypatch):
    retry_session = _Session([_Result(single={"key": "k"})])
    assert operations.retry_job(_DriverWithSessions([retry_session]), "k") is True
    retry_none = _Session([_Result(single=None)])
    assert operations.retry_job(_DriverWithSessions([retry_none]), "k") is False

    with pytest.raises(ValueError):
        operations.quarantine_job(None, "k", "  ")
    q_session = _Session([_Result(single={"key": "k"})])
    assert operations.quarantine_job(_DriverWithSessions([q_session]), "k", "reason") is True

    monkeypatch.setattr(operations, "lifecycle", lambda _d, _k: None)
    assert operations.inspect_job(object(), "missing") is None
    monkeypatch.setattr(operations, "lifecycle", lambda _d, _k: {"state": "DONE"})
    a_session = _Session([_Result(data=[{"artifact_id": "a"}])])
    assert operations.inspect_job(_DriverWithSessions([a_session]), "k") == {
        "job": {"state": "DONE"},
        "artifacts": [{"artifact_id": "a"}],
    }


def test_operations_main_all_dispatches(monkeypatch, tmp_path, capsys):
    class Finding:
        def __init__(self, classification):
            self.classification = classification
            self.detail = "x"

    cases = [
        (["inspect", "--job-key", "k"], 0),
        (["retry", "--job-key", "k"], 0),
        (["quarantine", "--job-key", "k", "--reason", "r"], 0),
        (["reconcile", "--artifact-root", str(tmp_path)], 1),
        (["schema-audit"], 0),
        (["schema-ensure"], 0),
    ]
    monkeypatch.setattr(operations, "inspect_job", lambda *_: {"job": {"state": "DONE"}})
    monkeypatch.setattr(operations, "retry_job", lambda *_: True)
    monkeypatch.setattr(operations, "quarantine_job", lambda *_: True)
    monkeypatch.setattr(operations, "reconcile", lambda *_: [Finding("ORPHAN_FILE")])
    monkeypatch.setattr(operations, "audit_schema", lambda *_: {"ok": True})
    monkeypatch.setattr(operations, "ensure_schema", lambda *_: {"ok": True})
    for argv, expected in cases:
        driver = _ClosableDriver()
        monkeypatch.setattr(operations, "_driver", lambda d=driver: d)
        assert operations.main(argv) == expected
        assert driver.closed
    assert "ORPHAN_FILE" in capsys.readouterr().out


def test_operations_main_failure_return_codes(monkeypatch, tmp_path):
    monkeypatch.setattr(operations, "retry_job", lambda *_: False)
    monkeypatch.setattr(operations, "quarantine_job", lambda *_: False)
    monkeypatch.setattr(operations, "audit_schema", lambda *_: {"ok": False})
    monkeypatch.setattr(operations, "ensure_schema", lambda *_: {"ok": False})
    monkeypatch.setattr(operations, "reconcile", lambda *_: [])
    for argv, expected in [
        (["retry", "--job-key", "k"], 2),
        (["quarantine", "--job-key", "k", "--reason", "r"], 2),
        (["schema-audit"], 1),
        (["schema-ensure"], 1),
        (["reconcile", "--artifact-root", str(tmp_path)], 0),
    ]:
        driver = _ClosableDriver()
        monkeypatch.setattr(operations, "_driver", lambda d=driver: d)
        assert operations.main(argv) == expected
        assert driver.closed


def test_worker_state_root_and_run_cycle_admission(monkeypatch, tmp_path):
    monkeypatch.setenv("STATE_ROOT", str(tmp_path / "state"))
    assert worker_loop.state_root().is_dir()
    policy = resources.ResourcePolicy()
    monkeypatch.setattr(worker_loop, "snapshot", lambda _p: SimpleNamespace())
    monkeypatch.setattr(worker_loop, "admission", lambda _s, _p: (False, ["busy"]))
    assert worker_loop.run_cycle(object(), policy=policy, resource_path=tmp_path) == 4


def test_worker_run_cycle_binds_and_runs(monkeypatch, tmp_path):
    policy = resources.ResourcePolicy()
    monkeypatch.setattr(worker_loop, "snapshot", lambda _p: SimpleNamespace())
    monkeypatch.setattr(worker_loop, "admission", lambda _s, _p: (True, []))
    monkeypatch.setattr(worker_loop, "build_worker_tasks", lambda: (SimpleNamespace(name="x"),))
    calls = []
    monkeypatch.setattr(worker_loop, "ensure_job", lambda *a: calls.append("ensure"))
    monkeypatch.setattr(worker_loop, "bind_plan", lambda *a: calls.append("bind"))
    monkeypatch.setattr(worker_loop, "run_profile", lambda *a, **k: calls.append("run") or 2)
    assert worker_loop.run_cycle(object(), policy=policy, resource_path=tmp_path, now=600) == 2
    assert calls == ["ensure", "bind", "run"]


def test_worker_main_once_and_stop_cleanup(monkeypatch, tmp_path):
    monkeypatch.setenv("STATE_ROOT", str(tmp_path))
    driver = _ClosableDriver()
    monkeypatch.setattr(worker_loop, "_driver", lambda: driver)
    monkeypatch.setattr(worker_loop, "run_cycle", lambda *a, **k: 4)
    assert worker_loop.main(["--once", "--resource-path", str(tmp_path)]) == 4
    assert driver.closed

    stop = tmp_path / "worker.stop"
    stop.write_text("stop", encoding="utf-8")
    driver2 = _ClosableDriver()
    monkeypatch.setattr(worker_loop, "_driver", lambda: driver2)
    assert worker_loop.main(["--resource-path", str(tmp_path)]) == 0
    assert not stop.exists()
    assert driver2.closed


def test_worker_main_rejects_nonpositive_sleep():
    with pytest.raises(SystemExit):
        worker_loop.main(["--sleep-sec", "0", "--once"])
