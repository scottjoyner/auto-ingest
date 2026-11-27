"""Preflight must not call an unreadable job queue 'no competing jobs'.

`_queued_jobs` swallowed OSError from `os.listdir` and returned an empty tuple;
the `no_competing_jobs` check then read that as a clean pass. A queue directory
that could not be listed is NO DATA, not an empty queue.
"""
from __future__ import annotations

import errno

import pytest

from auto_ingest.custody import cli


def _raise(*_a, **_kw):
    raise OSError(errno.EACCES, "permission denied")


def test_unreadable_queue_directory_is_no_data(monkeypatch):
    monkeypatch.setattr(cli.os, "listdir", _raise)
    assert cli._queued_jobs("/some/job/dir") is None
    assert cli._no_competing_jobs(None) is None


def test_readable_queue_directory_keeps_its_verdict(tmp_path):
    (tmp_path / "a.job").write_bytes(b"")
    (tmp_path / "b.txt").write_bytes(b"")
    queued = cli._queued_jobs(str(tmp_path))
    assert queued == ("a.job",)
    assert cli._no_competing_jobs(queued) is False
    assert cli._no_competing_jobs(()) is True
