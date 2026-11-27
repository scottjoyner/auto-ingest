"""A ledger tail we cannot inspect must not be declared clean.

`ends_unterminated` decides whether `hash_source` closes off a damaged line
before appending. A killed producer leaves a partial JSON record; if the probe
answers "no partial line" because it could not stat or read the file, the next
record is glued onto that partial line and both objects are lost together.
"Could not inspect" must take the repairing branch, not the clean branch.
"""
from __future__ import annotations

import errno
from types import SimpleNamespace

from auto_ingest.custody.hashing import ends_unterminated


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


def test_real_ledger_tails_are_still_classified_correctly(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_bytes(b"")
    assert ends_unterminated(empty) is False

    ended = tmp_path / "ended.jsonl"
    ended.write_bytes(b'{"k":"v"}\n')
    assert ends_unterminated(ended) is False

    partial = tmp_path / "partial.jsonl"
    partial.write_bytes(b'{"k":"v')
    assert ends_unterminated(partial) is True


def test_an_unstatable_ledger_is_treated_as_unterminated():
    assert ends_unterminated(_Unstatable()) is True


def test_an_unreadable_ledger_is_treated_as_unterminated():
    assert ends_unterminated(_Unreadable()) is True
