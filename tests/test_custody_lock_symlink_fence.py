"""The campaign lock must never follow a symlink planted at its path.

`CampaignLock.acquire()` opened the lock file with O_CREAT and no O_NOFOLLOW.
A symlink at that path was followed, then ftruncate/write rewrote its target —
an unrelated file outside the lock root — while the caller believed it had
taken a campaign lock.
"""
from __future__ import annotations

import os

import pytest

from auto_ingest.custody.lock import campaign_lock


def test_a_symlinked_lock_path_is_refused_and_the_target_is_untouched(tmp_path, monkeypatch):
    lock_root = tmp_path / "locks"
    lock_root.mkdir()
    monkeypatch.setenv("CUSTODY_LOCK_ROOT", str(lock_root))

    victim = tmp_path / "operator-file.txt"
    victim.write_bytes(b"do not truncate\n")

    lock = campaign_lock("primary:fileserver/dashcam")
    lock.path.symlink_to(victim)

    with pytest.raises(OSError):
        lock.acquire()
    assert victim.read_bytes() == b"do not truncate\n"
    assert lock.held is False


def test_a_normal_lock_file_still_acquires(tmp_path, monkeypatch):
    lock_root = tmp_path / "locks"
    lock_root.mkdir()
    monkeypatch.setenv("CUSTODY_LOCK_ROOT", str(lock_root))
    lock = campaign_lock("primary:fileserver/dashcam")
    assert lock.acquire() is True
    try:
        assert lock.held is True
        assert lock.path.read_text(encoding="ascii").strip() == str(os.getpid())
    finally:
        lock.release()
