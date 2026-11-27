"""Ledger rows with malformed or negative sizes must not be coherent custody.

`LedgerRecord.from_dict` coerced a malformed size to 0 and kept negative sizes,
and `summarize_ledger` counted such rows as verified with a byte total that
could be zero or negative while still reporting `coherent=True`. An operator
could read a corrupt ledger as complete proof.
"""
from __future__ import annotations

import json

from auto_ingest.custody.ledger import summarize_ledger


def _row(key, size, status="verified"):
    return json.dumps({"key": key, "size": size, "status": status,
                       "digest": "abc"}) + "\n"


def _write(tmp_path, rows):
    path = tmp_path / "destination.jsonl"
    path.write_text(rows, encoding="utf-8")
    return path


def test_valid_ledger_rows_remain_coherent(tmp_path):
    summary = summarize_ledger(_write(tmp_path, _row("a", 100) + _row("b", 50)))
    assert summary.coherent is True
    assert summary.files == 2
    assert summary.bytes == 150


def test_non_numeric_size_is_a_malformed_row(tmp_path):
    summary = summarize_ledger(_write(tmp_path, _row("a", "not-a-number")))
    assert summary.coherent is False
    assert summary.malformed == 1
    assert summary.files == 0
    assert summary.bytes == 0


def test_negative_size_is_a_malformed_row(tmp_path):
    summary = summarize_ledger(_write(tmp_path, _row("a", -100)))
    assert summary.coherent is False
    assert summary.malformed == 1
    assert summary.files == 0
