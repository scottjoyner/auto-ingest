"""An uninspectable release audit must not be declared clean.

`audit_ends_unterminated` decides whether a killed release's partial JSON line
is closed off before the next record is appended. Returning False on a stat/open
error means the next audit row is glued onto the damaged line, and the release
history reads as a clean run with two records lost.
"""
from __future__ import annotations

import errno
from types import SimpleNamespace

import pytest

from auto_ingest.custody import release_source
from auto_ingest.custody.release_source import audit_ends_unterminated


class _Unstatable:
    def is_file(self):
        return True

    def stat(self):
        raise OSError(errno.EACCES, "permission denied")

    def open(self, *args, **kwargs):
        raise AssertionError("stat failure must not reach open")


class _Unreadable:
    def is_file(self):
        return True

    def stat(self):
        return SimpleNamespace(st_size=128)

    def open(self, *args, **kwargs):
        raise OSError(errno.EACCES, "permission denied")


def test_real_audit_tails_are_still_classified_correctly(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_bytes(b"")
    assert audit_ends_unterminated(empty) is False

    ended = tmp_path / "ended.jsonl"
    ended.write_bytes(b'{"k":"v"}\n')
    assert audit_ends_unterminated(ended) is False

    partial = tmp_path / "partial.jsonl"
    partial.write_bytes(b'{"k":"v')
    assert audit_ends_unterminated(partial) is True


@pytest.mark.parametrize("fake", [_Unstatable(), _Unreadable()])
def test_an_uninspectable_audit_is_treated_as_unterminated(monkeypatch, fake):
    monkeypatch.setattr(release_source, "Path", lambda _path: fake)
    assert audit_ends_unterminated("placeholder") is True
