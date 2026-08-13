from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from auto_ingest import metrics, operations, readiness, resources


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

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def run(self, _query, **_kwargs):
        return next(self.results)


class _DriverWithSessions(_ClosableDriver):
    def __init__(self, sessions):
        super().__init__()
        self.sessions = iter(sessions)

    def session(self, *args, **kwargs):
        return next(self.sessions)


def test_resource_memory_parse_and_fallback(monkeypatch):
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self, encoding=None: "MemTotal: 1 kB\nMemAvailable: 4096 kB\n",
    )
    assert resources._memory_available_mb() == 4
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self, encoding=None: "MemAvailable: nope kB\n",
    )
    assert resources._memory_available_mb() == 0

    def fail_read(*_args, **_kwargs):
        raise OSError("nope")

    monkeypatch.setattr(Path, "read_text", fail_read)
    assert resources._memory_available_mb() == 0


def test_resource_snapshot_handles_missing_cpu_and_load(monkeypatch, tmp_path):
    monkeypatch.setattr(resources.os, "cpu_count", lambda: None)

    def fail_load():
        raise OSError("no load")

    monkeypatch.setattr(resources.os, "getloadavg", fail_load)
    monkeypatch.setattr(resources, "_memory_available_mb", lambda: 123)
    monkeypatch.setattr(
        resources.shutil,
        "disk_usage",
        lambda _p: SimpleNamespace(free=5 * 1024**3),
    )
    snap = resources.snapshot(tmp_path)
    assert snap.cpu_count == 1
    assert snap.load1 == 0.0
    assert snap.memory_available_mb == 123
    assert snap.disk_free_gb == 5.0
    assert snap.load_per_cpu == 0.0


def test_metrics_collect_validation_and_values():
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
    result = metrics.collect_metrics(_DriverWithSessions([session]), stale_after_sec=10)
    assert result == {
        "jobs_by_state": {"RUNNING": 2},
        "active_leases": 2,
        "stale_jobs": 1,
        "artifacts": 9,
    }


def test_metrics_collect_handles_empty_single_results():
    session = _Session(
        [
            _Result(data=[]),
            _Result(single=None),
            _Result(single=None),
            _Result(single=None),
        ]
    )
    result = metrics.collect_metrics(_DriverWithSessions([session]))
    assert result["active_leases"] == 0
    assert result["stale_jobs"] == 0
    assert result["artifacts"] == 0


def test_metrics_main_closes_driver(monkeypatch, capsys):
    driver = _ClosableDriver()
    monkeypatch.setattr(metrics, "_driver", lambda: driver)
    monkeypatch.setattr(
        metrics,
        "collect_metrics",
        lambda _d, stale_after_sec: {"jobs_by_state": {}},
    )
    assert metrics.main(["--stale-after-sec", "5"]) == 0
    assert "auto_ingest_jobs" in capsys.readouterr().out
    assert driver.closed


def test_readiness_healthy_and_unhealthy(monkeypatch):
    ok_session = _Session([_Result(single={"ok": 1})])
    monkeypatch.setattr(readiness, "audit_schema", lambda _d: {"ok": True})
    monkeypatch.setattr(readiness, "collect_metrics", lambda _d: {"stale_jobs": 0})
    report = readiness.readiness(_DriverWithSessions([ok_session]))
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


def test_operations_helpers(monkeypatch):
    assert operations.retry_job(
        _DriverWithSessions([_Session([_Result(single={"key": "k"})])]), "k"
    )
    assert not operations.retry_job(
        _DriverWithSessions([_Session([_Result(single=None)])]), "k"
    )
    with pytest.raises(ValueError):
        operations.quarantine_job(None, "k", "  ")
    assert operations.quarantine_job(
        _DriverWithSessions([_Session([_Result(single={"key": "k"})])]),
        "k",
        "reason",
    )

    monkeypatch.setattr(operations, "lifecycle", lambda _d, _k: None)
    assert operations.inspect_job(object(), "missing") is None
    monkeypatch.setattr(operations, "lifecycle", lambda _d, _k: {"state": "DONE"})
    session = _Session([_Result(data=[{"artifact_id": "a"}])])
    assert operations.inspect_job(_DriverWithSessions([session]), "k") == {
        "job": {"state": "DONE"},
        "artifacts": [{"artifact_id": "a"}],
    }


def test_operations_main_dispatch_and_failures(monkeypatch, tmp_path, capsys):
    class Finding:
        def __init__(self, classification):
            self.classification = classification
            self.detail = "x"

    monkeypatch.setattr(operations, "inspect_job", lambda *_: {"job": {"state": "DONE"}})
    monkeypatch.setattr(operations, "retry_job", lambda *_: True)
    monkeypatch.setattr(operations, "quarantine_job", lambda *_: True)
    monkeypatch.setattr(operations, "reconcile", lambda *_: [Finding("ORPHAN_FILE")])
    monkeypatch.setattr(operations, "audit_schema", lambda *_: {"ok": True})
    monkeypatch.setattr(operations, "ensure_schema", lambda *_: {"ok": True})
    cases = [
        (["inspect", "--job-key", "k"], 0),
        (["retry", "--job-key", "k"], 0),
        (["quarantine", "--job-key", "k", "--reason", "r"], 0),
        (["reconcile", "--artifact-root", str(tmp_path)], 1),
        (["schema-audit"], 0),
        (["schema-ensure"], 0),
    ]
    for argv, expected in cases:
        driver = _ClosableDriver()
        monkeypatch.setattr(operations, "_driver", lambda d=driver: d)
        assert operations.main(argv) == expected
        assert driver.closed
    assert "ORPHAN_FILE" in capsys.readouterr().out

    monkeypatch.setattr(operations, "retry_job", lambda *_: False)
    monkeypatch.setattr(operations, "quarantine_job", lambda *_: False)
    monkeypatch.setattr(operations, "audit_schema", lambda *_: {"ok": False})
    monkeypatch.setattr(operations, "ensure_schema", lambda *_: {"ok": False})
    monkeypatch.setattr(operations, "reconcile", lambda *_: [])
    failures = [
        (["retry", "--job-key", "k"], 2),
        (["quarantine", "--job-key", "k", "--reason", "r"], 2),
        (["schema-audit"], 1),
        (["schema-ensure"], 1),
        (["reconcile", "--artifact-root", str(tmp_path)], 0),
    ]
    for argv, expected in failures:
        driver = _ClosableDriver()
        monkeypatch.setattr(operations, "_driver", lambda d=driver: d)
        assert operations.main(argv) == expected
        assert driver.closed
