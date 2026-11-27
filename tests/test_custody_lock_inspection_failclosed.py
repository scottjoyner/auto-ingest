"""A lock we cannot inspect must not be reported as unlocked.

`is_locked` feeds two gates: `preflight`'s "destination_not_locked_by_another_campaign"
check and `execute`'s refusal list. It used to answer False on every OSError,
so a lock file that existed but could not be opened (permissions, wrong type,
an unexpected flock failure) read as "no other campaign holds this destination"
and both gates passed. "Could not check" is not "checked and unlocked".
"""
from __future__ import annotations

import errno
import os

import pytest

from auto_ingest.custody import lock as lock_mod
from auto_ingest.custody.lock import campaign_lock, is_locked

KEY = "primary:fileserver/dashcam"


def test_a_missing_lock_file_is_genuinely_unlocked():
    assert is_locked(KEY) is False


def test_a_lock_path_that_is_not_a_regular_file_fails_closed():
    """A directory or device at the lock path cannot be flock-probed."""
    path = campaign_lock(KEY).path
    path.mkdir(parents=True)
    assert is_locked(KEY) is True


def test_an_unopenable_lock_file_fails_closed(monkeypatch):
    path = campaign_lock(KEY).path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")

    real_open = os.open

    def refusing_open(target, *args, **kwargs):
        if os.fspath(target) == os.fspath(path):
            raise PermissionError(errno.EACCES, "permission denied")
        return real_open(target, *args, **kwargs)

    monkeypatch.setattr(lock_mod.os, "open", refusing_open)
    assert is_locked(KEY) is True


def test_an_unexpected_flock_error_fails_closed(monkeypatch):
    path = campaign_lock(KEY).path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")

    def odd_flock(_fd, _op):
        raise OSError(errno.EPERM, "operation not permitted")

    monkeypatch.setattr(lock_mod.fcntl, "flock", odd_flock)
    assert is_locked(KEY) is True


def test_a_held_lock_is_still_detected():
    held = campaign_lock(KEY)
    assert held.acquire() is True
    try:
        assert is_locked(KEY) is True
    finally:
        held.release()
    assert is_locked(KEY) is False
