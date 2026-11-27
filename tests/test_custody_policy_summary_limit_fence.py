"""The bounded-summary limit must be a positive integer.

`CustodyPolicy.from_dict` accepted `max_summary_entries: 0`, negative values and
non-integers. Every evidence/ledger summary list is captured through
`_bounded(values, limit)`, so a zero/negative cap silently truncated diagnostics
to one row while the policy still looked valid.
"""
from __future__ import annotations

import pytest

from auto_ingest.custody.policy import CustodyPolicy


@pytest.mark.parametrize("value", [0, -1, -20, True, 1.5, "20"])
def test_invalid_summary_limit_is_refused(value):
    with pytest.raises(ValueError, match="max_summary_entries"):
        CustodyPolicy.from_dict({"max_summary_entries": value})


def test_valid_summary_limit_and_default_are_unchanged():
    assert CustodyPolicy.from_dict({"max_summary_entries": 20}).max_summary_entries == 20
    assert CustodyPolicy.from_dict({}).max_summary_entries == 20
    assert CustodyPolicy.from_dict({"max_summary_entries": None}).max_summary_entries == 20
