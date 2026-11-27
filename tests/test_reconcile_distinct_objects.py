"""Reconciliation must count distinct objects, not ledger rows.

A ``--recheck`` verification pass re-proves objects it already proved and
appends a fresh row for each one. The destination ledger therefore routinely
holds more rows than objects. Every other reader in this package already
de-duplicates by key (``summarize_ledger``, ``verify.py::verified_bytes`` and
``hashing.py::already_hashed`` all say so in comments); the set-difference
reader must do the same, or a healthy re-verified campaign reports objects
that are "present but unverified".

Hermetic: two tiny JSONL files in ``tmp_path``. No bundles, mounts, DB,
network or live fleet state.
"""
from __future__ import annotations

import json

import pytest

from auto_ingest.custody.ledger import reconcile_ledgers

pytestmark = pytest.mark.usefixtures("hermetic_mounts")


def _write_ledger(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _src(key, digest):
    return {"key": key, "digest": digest, "status": "verified"}


def _dst(key, digest):
    return {"key": key, "digest": digest, "status": "verified_at_destination"}


def test_recheck_rows_do_not_inflate_destination_object_count(tmp_path):
    """Two objects verified twice each = 2 objects, not 4 rows."""
    h = _write_ledger(
        tmp_path / "hash.jsonl", [_src("a", "A1"), _src("b", "B2")]
    )
    # The shape a --recheck pass leaves behind: one fresh row per already-proved
    # key, appended in a later run (see tests/test_custody_verify.py:468-476).
    d = _write_ledger(
        tmp_path / "destination.jsonl",
        [
            _dst("a", "A1"),
            _dst("b", "B2"),
            _dst("a", "A1"),
            _dst("b", "B2"),
        ],
    )

    result = reconcile_ledgers(h, d)

    assert result.verified == 2
    assert result.source_objects == 2
    # Pre-fix: 4 counted rows. These are 2 distinct destination objects.
    assert result.destination_objects == 2


def test_recheck_rows_do_not_create_phantom_unverified_files(tmp_path):
    """A fully verified campaign must not claim unverified objects at the dest."""
    h = _write_ledger(
        tmp_path / "hash.jsonl", [_src("a", "A1"), _src("b", "B2")]
    )
    d = _write_ledger(
        tmp_path / "destination.jsonl",
        [
            _dst("a", "A1"),
            _dst("b", "B2"),
            _dst("a", "A1"),
            _dst("b", "B2"),
        ],
    )

    proposal = reconcile_ledgers(h, d).proposal()

    assert proposal is not None
    # Pre-fix: max(4 rows - 2 verified keys, 0) == 2 phantom unverified files,
    # beside verification_complete=True - a self-contradictory evidence fragment.
    assert proposal["destination"]["verified_files"] == 2
    assert proposal["destination"]["unverified_present_files"] == 0
    assert proposal["destination"]["verification_complete"] is True


def test_one_row_per_key_is_unchanged(tmp_path):
    """Positive control: the ordinary, unre-checked ledger shape still works."""
    h = _write_ledger(
        tmp_path / "hash.jsonl", [_src("a", "A1"), _src("b", "B2")]
    )
    d = _write_ledger(
        tmp_path / "destination.jsonl", [_dst("a", "A1"), _dst("b", "B2")]
    )

    result = reconcile_ledgers(h, d)

    assert result.destination_objects == 2
    assert result.proposal()["destination"]["unverified_present_files"] == 0
