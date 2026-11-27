"""An empty-but-present hash ledger must not pass the inventory cross-check.

Hermetic: JSONL fixtures in tmp_path only - no card, mount, graph or network.
"""
from __future__ import annotations

import json

import pytest

from auto_ingest.custody.ledger import (
    ReconciliationUnavailable,
    reconcile_ledgers,
)


def _ledger(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_empty_hash_ledger_fails_the_inventory_cross_check(tmp_path):
    """0 of N inventoried objects is a coverage shortfall, not a clean diff."""
    h = _ledger(tmp_path / "h.jsonl", [])
    d = _ledger(tmp_path / "d.jsonl", [])
    r = reconcile_ledgers(h, d, expected_source_objects=7)
    assert r.usable is False
    assert any("covers_0_of_7_inventoried_objects" in reason
               for reason in r.incoherent), r.incoherent
    assert r.complete is False
    assert r.proposal() is None


def test_empty_hash_ledger_cannot_propose_completed_custody(tmp_path):
    """to_evidence() must refuse, not claim verification_complete for 0 files."""
    h = _ledger(tmp_path / "h.jsonl", [])
    d = _ledger(tmp_path / "d.jsonl", [])
    r = reconcile_ledgers(h, d, expected_source_objects=7)
    with pytest.raises(ReconciliationUnavailable):
        r.to_evidence()


def test_empty_hash_ledger_without_an_inventory_stays_usable(tmp_path):
    """Control: no count available differs from a count that says 7, ledger 0."""
    h = _ledger(tmp_path / "h.jsonl", [])
    d = _ledger(tmp_path / "d.jsonl", [])
    r = reconcile_ledgers(h, d)
    assert r.usable is True
    assert r.complete is True
