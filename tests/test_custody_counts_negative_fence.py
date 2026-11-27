"""Negative counts must be diagnosed, not silently accepted.

`Counts.from_dict` accepted negative files/bytes. Only some callers check for
negative inventory counts, so a malformed copy/hash ledgers count could produce
a negative byte total without any `coerced_fields` diagnostic.
"""
from __future__ import annotations

from auto_ingest.custody.evidence import Counts


def test_negative_counts_are_clamped_with_diagnostics():
    coerced = []
    counts = Counts.from_dict({"files": -3, "bytes": -100}, "copy.planned", coerced)
    assert counts.files == 0 and counts.bytes == 0
    assert any("copy.planned.files" in x for x in coerced)
    assert any("copy.planned.bytes" in x for x in coerced)


def test_valid_counts_are_unchanged():
    coerced = []
    counts = Counts.from_dict({"files": 3, "bytes": 100}, "copy.planned", coerced)
    assert counts.files == 3 and counts.bytes == 100
    assert coerced == []
