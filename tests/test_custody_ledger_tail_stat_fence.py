"""A ledger tail we cannot stat must not be declared clean.

`ends_unterminated` used `Path.is_file()`, which swallows permission/OSError and
returns False, and returned False on stat/open failures. Both collapse "could
not inspect the tail" into "no partial line", so the next digest record is
appended onto a possibly damaged line and two objects are lost.
"""
from __future__ import annotations

import errno
from types import SimpleNamespace

import pytest

from auto_ingest.custody.hashing import ends_unterminated


class _Unstatable:
    def is_file(self):
        raise OSError(errno.EACCES, "permission denied")

    def stat(self):
        raise AssertionError("unreachable")

    def open(self, *args, **kwargs):
        raise AssertionError("unreachable")


class _Unopenable:
    def is_file(self):
        return True

    def stat(self):
        return SimpleNamespace(st_size=128)

    def open(self, *args, **kwargs):
        raise OSError(errno.EACCES, "permission denied")


def test_real_ledger_tails_keep_their_verdicts(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_bytes(b"")
    assert ends_unterminated(empty) is False

    ended = tmp_path / "ended.jsonl"
    ended.write_bytes(b'{"k":"v"}\n')
    assert ends_unterminated(ended) is False

    partial = tmp_path / "partial.jsonl"
    partial.write_bytes(b'{"k":"v')
    assert ends_unterminated(partial) is True


@pytest.mark.parametrize("fake", [_Unstatable(), _Unopenable()])
def test_uninspectable_ledger_tail_is_treated_as_unterminated(fake):
    assert ends_unterminated(fake) is True
