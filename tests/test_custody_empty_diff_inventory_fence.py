"""A zero-row reconciliation is not coherent when inventory expects objects.

`ReconciliationResult.incoherent` cross-checked the hash ledger against the
recorded inventory only when `source_objects` was truthy. A ledger containing
zero qualifying rows (for example every row has status ``failed``) therefore
produced an empty incoherence list and a `usable` proposal of "nothing is in
custody" even though the campaign inventory said five objects existed.
"""
from __future__ import annotations

import json

from auto_ingest.custody.ledger import reconcile_ledgers


def _ledgers(tmp_path, hash_rows, dest_rows=""):
    hash_path = tmp_path / "hash.jsonl"
    dest_path = tmp_path / "destination.jsonl"
    hash_path.write_text(hash_rows, encoding="utf-8")
    dest_path.write_text(dest_rows, encoding="utf-8")
    return hash_path, dest_path


def test_zero_qualifying_rows_with_expected_inventory_is_incoherent(tmp_path):
    rows = json.dumps({"key": "a", "status": "failed", "digest": "abc"}) + "\n"
    result = reconcile_ledgers(*_ledgers(tmp_path, rows),
                               expected_source_objects=5)
    assert result.source_objects == 0
    assert any("hash_ledger_covers_0_of_5" in reason
               for reason in result.incoherent)
    assert result.usable is False


def test_zero_expected_and_zero_rows_remains_coherent(tmp_path):
    result = reconcile_ledgers(*_ledgers(tmp_path, ""),
                               expected_source_objects=0)
    assert result.incoherent == ()
    assert result.usable is True
