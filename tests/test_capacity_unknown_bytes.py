"""Capacity precondition must fail closed when inventory byte accounting is absent.

`capacity_report` derives the bytes still to copy from the source inventory. When
the inventory discovered files but recorded zero (or negative) bytes, scaling by
a mean object size of zero used to report ``outstanding_bytes == 0`` and
``sufficient=True`` with only the headroom floor -- so a nearly-full destination
passed the precondition and filled up mid-copy. These tests pin the fail-closed
answer and the unchanged behaviour when bytes are known or nothing is left.
"""
from __future__ import annotations

import tempfile
from dataclasses import replace

from custody_helpers import fully_copied_campaign

from auto_ingest.custody.capacity import (
    DEFAULT_HEADROOM_MIN_BYTES,
    capacity_report,
    outstanding_bytes,
)


def _resolved(camp, path=None):
    """Point the campaign's destination at a real, statable path (no live mount)."""
    return replace(
        camp,
        destination=replace(camp.destination, host_path=path or tempfile.gettempdir()),
    )


def test_zero_inventory_bytes_cannot_read_as_sufficient():
    """1000 files with 0 recorded bytes is not a 0-byte copy."""
    camp, ev = fully_copied_campaign(total_files=1000, total_bytes=0, verified_files=0)
    rep = capacity_report(_resolved(camp), ev)
    assert rep.sufficient is False
    assert rep.checked is False
    assert rep.reason == "inventory_bytes_unknown"


def test_negative_inventory_bytes_fail_closed():
    """A negative byte total is malformed evidence, not a small copy."""
    camp, ev = fully_copied_campaign(total_files=1000, total_bytes=-5, verified_files=0)
    rep = capacity_report(_resolved(camp), ev)
    assert rep.sufficient is False
    assert rep.reason == "inventory_bytes_unknown"


def test_known_bytes_still_report_sufficient(tmp_path):
    """Control: with real byte accounting the guard stays out of the way."""
    camp, ev = fully_copied_campaign(total_files=1000, total_bytes=1_000_000,
                                     verified_files=0)
    rep = capacity_report(_resolved(camp, str(tmp_path)), ev)
    assert rep.checked is True
    assert rep.sufficient is True
    assert rep.reason == ""
    assert outstanding_bytes(ev) == 1_000_000


def test_nothing_outstanding_is_unaffected_by_zero_bytes(tmp_path):
    """Control: a fully-copied campaign has no bytes outstanding regardless."""
    camp, ev = fully_copied_campaign(total_files=1000, total_bytes=0, verified_files=1000)
    rep = capacity_report(_resolved(camp, str(tmp_path)), ev)
    assert rep.checked is True
    assert rep.sufficient is True
    assert outstanding_bytes(ev) == 0
    assert rep.total_needed == DEFAULT_HEADROOM_MIN_BYTES
