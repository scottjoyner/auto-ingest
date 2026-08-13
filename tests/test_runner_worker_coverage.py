from __future__ import annotations

from types import SimpleNamespace

import pytest

from auto_ingest import full_pipeline, profile_runner, resources, worker_loop


class _ClosableDriver:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_profile_tasks_and_bound_runner(monkeypatch):
    custom_tasks = (SimpleNamespace(name="a"),)
    monkeypatch.setitem(profile_runner.PROFILES, "sync", custom_tasks)
    assert profile_runner.profile_tasks("sync") is custom_tasks
    monkeypatch.setattr(
        full_pipeline, "build_tasks", lambda: (SimpleNamespace(name="full"),)
    )
    assert profile_runner.profile_tasks("full")[0].name == "full"

    calls = []
    monkeypatch.setattr(profile_runner, "default_job_key", lambda p: f"key:{p}")
    monkeypatch.setattr(
        profile_runner,
        "ensure_job",
        lambda d, k, p: calls.append(("ensure", k, p)),
    )
    monkeypatch.setattr(
        profile_runner,
        "bind_plan",
        lambda d, k, t: calls.append(("bind", k, t)),
    )
    monkeypatch.setattr(
        profile_runner,
        "run_profile",
        lambda d, p, **kw: calls.append(("run", p, kw)) or 6,
    )
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


def test_full_pipeline_state_root_key_and_main(monkeypatch, tmp_path):
    monkeypatch.setenv("STATE_ROOT", str(tmp_path / "state"))
    assert full_pipeline.state_root().is_dir()
    monkeypatch.setenv("FULL_PIPELINE_WINDOW_SEC", "60")
    assert full_pipeline.default_job_key(120) == "pipeline:full:2"
    monkeypatch.setenv("FULL_PIPELINE_WINDOW_SEC", "59")
    with pytest.raises(ValueError, match="at least 60"):
        full_pipeline.default_job_key(120)

    driver = _ClosableDriver()
    monkeypatch.setattr(full_pipeline, "_driver", lambda: driver)
    monkeypatch.setattr(full_pipeline, "default_job_key", lambda: "full:key")
    monkeypatch.setattr(
        full_pipeline, "build_tasks", lambda: (SimpleNamespace(name="one"),)
    )
    calls = []
    monkeypatch.setattr(full_pipeline, "ensure_job", lambda *a: calls.append("ensure"))
    monkeypatch.setattr(full_pipeline, "bind_plan", lambda *a: calls.append("bind"))
    monkeypatch.setattr(
        full_pipeline, "run_profile", lambda *a, **k: calls.append("run") or 0
    )
    assert full_pipeline.main() == 0
    assert calls == ["ensure", "bind", "run"]
    assert driver.closed
    with pytest.raises(ValueError):
        full_pipeline.main(["bad"])


def test_worker_state_root_and_task_shapes(monkeypatch, tmp_path):
    monkeypatch.setenv("STATE_ROOT", str(tmp_path / "state"))
    monkeypatch.setenv("DROP_ROOT", "/drop")
    monkeypatch.setenv("CONTENT", "0")
    assert worker_loop.state_root().is_dir()
    tasks = worker_loop.build_worker_tasks()
    assert [task.name for task in tasks[:3]] == [
        "fallback-queue",
        "speaker-link",
        "dashcam-compress",
    ]
    assert "content" not in [task.name for task in tasks]


def test_worker_run_cycle_admission_denied(monkeypatch, tmp_path):
    policy = resources.ResourcePolicy()
    monkeypatch.setattr(worker_loop, "snapshot", lambda _p: SimpleNamespace())
    monkeypatch.setattr(worker_loop, "admission", lambda _s, _p: (False, ["busy"]))
    assert worker_loop.run_cycle(object(), policy=policy, resource_path=tmp_path) == 4


def test_worker_run_cycle_binds_and_runs(monkeypatch, tmp_path):
    policy = resources.ResourcePolicy()
    monkeypatch.setattr(worker_loop, "snapshot", lambda _p: SimpleNamespace())
    monkeypatch.setattr(worker_loop, "admission", lambda _s, _p: (True, []))
    monkeypatch.setattr(
        worker_loop, "build_worker_tasks", lambda: (SimpleNamespace(name="x"),)
    )
    calls = []
    monkeypatch.setattr(worker_loop, "ensure_job", lambda *a: calls.append("ensure"))
    monkeypatch.setattr(worker_loop, "bind_plan", lambda *a: calls.append("bind"))
    monkeypatch.setattr(
        worker_loop, "run_profile", lambda *a, **k: calls.append("run") or 2
    )
    assert (
        worker_loop.run_cycle(object(), policy=policy, resource_path=tmp_path, now=600)
        == 2
    )
    assert calls == ["ensure", "bind", "run"]


def test_worker_main_once_stop_cleanup_and_validation(monkeypatch, tmp_path):
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

    with pytest.raises(SystemExit):
        worker_loop.main(["--sleep-sec", "0", "--once"])
