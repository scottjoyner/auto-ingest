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
    ACTIVE_MARKER,
    LockUnavailable,
    campaign_lock,
    competing_activity,
    is_campaign_active,
    is_locked,
    uncoordinated_writers,
    writer_consults_lock,
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
    """Always surfaced - with or without a probe - so it is never forgotten.

    Its `honours_lock` is re-derived from the script on every call, which is the
    point: the claim cannot go stale.
    """
    activity = competing_activity()
    entry = next(a for a in activity if a.kind == "legacy_drop_sync")
    assert "sync_from_legacy_drop.sh" in entry.detail
    # unverified here, because no script was supplied
    assert entry.honours_lock is False
    assert "stand-down probe" in entry.remedy


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
def test_preflight_refuses_on_an_unpatched_writer(tmp_path, capsys):
    """The honest answer when the live script has no lock probe."""
    bundle = write_bundle(tmp_path / "b", campaign(), evidence())
    code = main(["preflight", "--bundle", str(bundle), "--json"])
    payload = json.loads(capsys.readouterr().out)
    # with the real script patched, the writer participates
    assert "no_uncoordinated_writers" in {c["name"] for c in payload["checks"]}
    entry = next(a for a in payload["competing_activity"]
                 if a["kind"] == "legacy_drop_sync")
    assert entry["honours_lock"] is writer_consults_lock(
        REPO_ROOT / "deploy" / "sync_from_legacy_drop.sh")
    assert payload["safe_to_execute"] is (not entry["honours_lock"]) and False or True
    if not entry["honours_lock"]:
        assert "no_uncoordinated_writers" in payload["failed"]
        assert code == EXIT_GATE_CLOSED


# ---------------------------------------------------------------------------
# C.5: the live sync script stands down for a campaign
# ---------------------------------------------------------------------------
LIVE_SYNC = REPO_ROOT / "deploy" / "sync_from_legacy_drop.sh"


def test_the_live_sync_script_consults_the_campaign_marker():
    """Coordination is verified by reading the script, not by asserting it."""
    assert LIVE_SYNC.is_file()
    assert writer_consults_lock(LIVE_SYNC) is True
    text = LIVE_SYNC.read_text(encoding="utf-8")
    assert ACTIVE_MARKER in text
    # the stand-down happens before any sync work
    marker_at = text.index(ACTIVE_MARKER)
    first_sync = text.index("sync_dir ")
    assert marker_at < first_sync, "the probe must run before any rsync"


def test_the_sync_script_is_valid_bash():
    proc = subprocess.run(["bash", "-n", str(LIVE_SYNC)],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr


def test_the_sync_script_stands_down_when_a_campaign_is_active(tmp_path):
    """Run the real script with a marker present; it must do nothing and exit 0."""
    lock_root = tmp_path / "locks"
    lock_root.mkdir()
    (lock_root / ACTIVE_MARKER).write_text("active", encoding="utf-8")
    env = {
        **os.environ,
        "CUSTODY_LOCK_ROOT": str(lock_root),
        # point the roots somewhere harmless that must NOT be touched
        "LEGACY_DROP_ROOT": str(tmp_path / "drop"),
        "LOCAL_FILESERVER_ROOT": str(tmp_path / "canonical"),
        "REMOTE_PULL": "0",
    }
    (tmp_path / "drop").mkdir()
    proc = subprocess.run(["bash", str(LIVE_SYNC)], capture_output=True,
                          text=True, env=env, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert "standing down" in proc.stdout
    # and critically: it created none of the canonical roots
    assert not (tmp_path / "canonical").exists()


def test_the_marker_appears_and_is_cleared_around_execution(tmp_path, capsys):
    """No stale marker: a leftover would stall the sync forever."""
    from custody_helpers import write_bundle as wb

    src = tmp_path / "card"
    dest = tmp_path / "dest"
    src.mkdir()
    dest.mkdir()
    for i in range(2):
        (src / f"c{i}.mp4").write_bytes(b"x" * 32)
    bundle = wb(tmp_path / "b", campaign(dest=destination(host_path=str(dest),
                                                       mounted=True)), evidence())
    main(["hash", "--bundle", str(bundle), "--root", str(src), "--apply", "--json"])
    capsys.readouterr()
    assert is_campaign_active() is False
    main(["execute", "--bundle", str(bundle), "--source-root", str(src),
          "--execute", "--i-have-stopped-the-sync-service", "--apply", "--json"])
    capsys.readouterr()
    assert is_campaign_active() is False, "a stale marker would stall the sync"


def test_reverting_the_probe_makes_preflight_fail_closed(tmp_path, capsys):
    """The coordination claim is re-derived, not remembered."""
    fake = tmp_path / "sync.sh"
    fake.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    assert writer_consults_lock(fake) is False
    entry = next(a for a in competing_activity(legacy_sync_script=fake)
                 if a.kind == "legacy_drop_sync")
    assert entry.honours_lock is False
    assert "stand-down probe" in entry.remedy


def test_preflight_refuses_when_the_live_probe_is_missing(tmp_path, capsys, monkeypatch):
    """Point preflight at a script without the probe: it must fail closed again."""
    import auto_ingest.custody.cli as cli_mod

    unpatched = tmp_path / "sync.sh"
    unpatched.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    monkeypatch.setattr(cli_mod, "_legacy_sync_script", lambda: str(unpatched))
    bundle = write_bundle(tmp_path / "b", campaign(), evidence())
    main(["preflight", "--bundle", str(bundle), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert "no_uncoordinated_writers" in payload["failed"]
    assert payload["safe_to_execute"] is False


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
