"""Phase C: the campaign lock and the competing-writer race.

The investigation found `sync-service` and `ingest.crontab:5` running
`sync_from_legacy_drop.sh` -- `rsync --archive --ignore-existing` into
`$DASHCAM_ROOT` / `$AUDIO_ROOT` / `$BODYCAM_ROOT`, the same roots a campaign
writes to, with no lock and no campaign awareness. These tests pin the primitive
that resolves it, and pin that preflight reports the hazard rather than papering
over it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from custody_helpers import campaign, destination, evidence, write_bundle

from auto_ingest.custody.cli import EXIT_GATE_CLOSED, main
from auto_ingest.custody.lock import (
    LockUnavailable,
    campaign_lock,
    competing_activity,
    is_locked,
    uncoordinated_writers,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI = REPO_ROOT / "bin" / "auto-ingest"


@pytest.fixture(autouse=True)
def isolated_lock_root(tmp_path, monkeypatch):
    """Never touch the real /tmp lock dir from a test."""
    root = tmp_path / "locks"
    monkeypatch.setenv("CUSTODY_LOCK_ROOT", str(root))
    return root


# ---------------------------------------------------------------------------
# the lock itself
# ---------------------------------------------------------------------------
def test_lock_is_exclusive(isolated_lock_root):
    first = campaign_lock("primary:fileserver/dashcam")
    assert first.acquire() is True
    second = campaign_lock("primary:fileserver/dashcam")
    assert second.acquire() is False
    first.release()
    assert second.acquire() is True
    second.release()


def test_lock_reports_its_holder_pid(isolated_lock_root):
    lock = campaign_lock("x")
    lock.acquire()
    try:
        assert lock.holder_pid() == os.getpid()
    finally:
        lock.release()
    assert lock.holder_pid() is None


def test_lock_releases_on_exception(isolated_lock_root):
    """A crashed executor must not leave the destination blocked forever."""
    lock = campaign_lock("y")
    with pytest.raises(ValueError):
        with lock:
            raise ValueError("executor exploded")
    assert is_locked("y") is False


def test_context_manager_raises_when_busy(isolated_lock_root):
    held = campaign_lock("z")
    held.acquire()
    try:
        with pytest.raises(LockUnavailable):
            with campaign_lock("z"):
                pass
    finally:
        held.release()


def test_different_destinations_do_not_block_each_other(isolated_lock_root):
    a = campaign_lock("primary:fileserver/dashcam")
    b = campaign_lock("primary:fileserver/audio")
    assert a.acquire() and b.acquire()
    a.release()
    b.release()


def test_is_locked_does_not_take_the_lock(isolated_lock_root):
    lock = campaign_lock("probe")
    assert is_locked("probe") is False
    lock.acquire()
    assert is_locked("probe") is True
    lock.release()
    assert is_locked("probe") is False


def test_lock_key_is_path_safe(isolated_lock_root):
    lock = campaign_lock("primary:fileserver/dashcam")
    assert ":" not in lock.path.name
    assert "/" not in lock.path.name


def test_two_threads_race_for_one_lock(isolated_lock_root):
    results = []
    ready = threading.Barrier(4)

    def contend():
        ready.wait(timeout=10)
        lock = campaign_lock("contended")
        results.append(lock.acquire())
        if lock.held:
            import time
            time.sleep(0.05)
            lock.release()

    threads = [threading.Thread(target=contend) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
    assert sum(results) >= 1
    assert results.count(False) == 4 - sum(results)


def test_a_lock_held_by_another_process_is_detected(isolated_lock_root):
    """flock is per-open-file-description, so a real second process must be seen."""
    script = "\n".join([
        "import sys, time",
        f"sys.path.insert(0, {str(REPO_ROOT)!r})",
        "from auto_ingest.custody.lock import campaign_lock",
        "lock = campaign_lock('crossproc')",
        "assert lock.acquire()",
        "print('held', flush=True)",
        "time.sleep(30)",
    ])
    proc = subprocess.Popen([sys.executable, "-c", script],
                            stdout=subprocess.PIPE, text=True,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    try:
        assert proc.stdout.readline().strip() == "held"
        assert is_locked("crossproc") is True
        assert campaign_lock("crossproc").acquire() is False
    finally:
        proc.kill()
        proc.wait(timeout=10)


# ---------------------------------------------------------------------------
# competing activity
# ---------------------------------------------------------------------------
def test_the_legacy_sync_is_always_reported(isolated_lock_root):
    """A standing hazard, so it does not depend on probing a path."""
    activity = competing_activity()
    assert "legacy_drop_sync" in uncoordinated_writers(activity)
    entry = next(a for a in activity if a.kind == "legacy_drop_sync")
    assert entry.honours_lock is False
    assert "sync_from_legacy_drop.sh" in entry.detail
    assert "separate change" in entry.remedy


def test_a_queued_drop_root_is_reported(isolated_lock_root, tmp_path):
    drop = tmp_path / "drop"
    drop.mkdir()
    (drop / "audio.job").write_text("", encoding="utf-8")
    activity = competing_activity(drop_root=drop)
    entry = next(a for a in activity if a.kind == "ingest_worker_queue")
    assert "queued_jobs=1" in entry.detail
    assert entry.honours_lock is False


def test_an_empty_drop_root_is_still_reported(isolated_lock_root, tmp_path):
    """The worker exists whether or not its queue is momentarily empty."""
    drop = tmp_path / "drop"
    drop.mkdir()
    activity = competing_activity(drop_root=drop)
    assert "ingest_worker_queue" in uncoordinated_writers(activity)


def test_a_missing_drop_root_is_still_reported(isolated_lock_root, tmp_path):
    activity = competing_activity(drop_root=tmp_path / "absent")
    entry = next(a for a in activity if a.kind == "ingest_worker_queue")
    assert "queued_jobs=0" in entry.detail


# ---------------------------------------------------------------------------
# preflight integration
# ---------------------------------------------------------------------------
def test_preflight_refuses_on_the_uncoordinated_writer(tmp_path, capsys):
    """The honest answer: this cannot be cleared from inside the package."""
    bundle = write_bundle(tmp_path / "b", campaign(), evidence())
    code = main(["preflight", "--bundle", str(bundle), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == EXIT_GATE_CLOSED
    assert payload["safe_to_execute"] is False
    assert "no_uncoordinated_writers" in payload["failed"]
    assert "no_uncoordinated_writers" in {c["name"] for c in payload["checks"]}
    assert any(a["kind"] == "legacy_drop_sync"
               for a in payload["competing_activity"])


def test_preflight_reports_a_held_lock(tmp_path, capsys):
    bundle = write_bundle(tmp_path / "b", campaign(), evidence())
    held = campaign_lock("primary:fileserver/dashcam")
    assert held.acquire() is True
    try:
        main(["preflight", "--bundle", str(bundle), "--json"])
        payload = json.loads(capsys.readouterr().out)
        check = next(c for c in payload["checks"]
                     if c["name"] == "destination_not_locked_by_another_campaign")
        assert check["ok"] is False
        assert "locked=True" in check["detail"]
    finally:
        held.release()


def test_preflight_is_clear_once_nothing_else_is_wrong(tmp_path, capsys, monkeypatch):
    """With a resolvable destination, the uncoordinated writer is the only FAIL."""
    dest = tmp_path / "dest"
    dest.mkdir()
    bundle = write_bundle(
        tmp_path / "b",
        campaign(dest=destination(host_path=str(dest), mounted=True)),
        evidence(),
    )
    monkeypatch.setattr(
        "auto_ingest.custody.cli.competing_activity",
        lambda **kw: (),
    )
    code = main(["preflight", "--bundle", str(bundle), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["failed"] == ["release_gate_open"]
    assert payload["safe_to_execute"] is False
    assert code == EXIT_GATE_CLOSED


def test_preflight_text_names_the_competing_writer(tmp_path, capsys):
    bundle = write_bundle(tmp_path / "b", campaign(), evidence())
    main(["preflight", "--bundle", str(bundle)])
    out = capsys.readouterr().out
    assert "competing     legacy_drop_sync" in out


def test_the_lock_module_writes_only_its_own_directory(isolated_lock_root):
    """An advisory lock must not touch the destination it protects."""
    import ast
    import inspect

    from auto_ingest.custody import lock as lock_mod

    tree = ast.parse(inspect.getsource(lock_mod))
    assert lock_mod.DEFAULT_LOCK_ROOT == "/tmp/auto_ingest_custody"
    # only os.mkdir, and only under the lock root
    mkdirs = [n for n in ast.walk(tree)
              if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "mkdir"]
    assert mkdirs, "expected the lock root to be created"
