"""Both spellings of the hash block must merge onto the same evidence.

Canonical evidence documents emit ``hash``; older producers used ``hashing``.
`merge_evidence_documents` normalised one direction only, so an incoming
``hashing`` observation was merged beside the recorded ``hash`` block and then
ignored by `CampaignEvidence.from_dict`, which prefers ``hash``. The recorded
verified-file count stayed stale even though import reported success.
"""
from __future__ import annotations

from auto_ingest.custody import CampaignEvidence
from auto_ingest.custody.store import merge_evidence_documents


def _existing():
    return {
        "campaign_id": "alias-test",
        "hash": {"verified_files": 100, "verified_bytes": 1000, "complete": True},
        "inventory": {"discovered_files": 100, "complete": True},
    }


def test_legacy_hashing_alias_updates_the_recorded_hash_block():
    merged = merge_evidence_documents(
        _existing(), {"hashing": {"verified_files": 200, "complete": True}}
    )
    evidence = CampaignEvidence.from_dict(merged)
    assert evidence.hashing.verified_files == 200
    assert evidence.hashing.complete is True
    # Fields the incoming document did not mention are preserved.
    assert evidence.hashing.verified_bytes == 1000
    assert "hashing" not in merged


def test_canonical_hash_alias_still_wins_and_clears_stale_hashing():
    merged = merge_evidence_documents(
        {**_existing(), "hashing": {"verified_files": 999}},
        {"hash": {"verified_files": 300, "complete": True}},
    )
    assert CampaignEvidence.from_dict(merged).hashing.verified_files == 300
    assert "hashing" not in merged


def test_unmentioned_blocks_are_preserved():
    merged = merge_evidence_documents(
        _existing(), {"hashing": {"verified_files": 150}}
    )
    assert CampaignEvidence.from_dict(merged).inventory.discovered_files == 100
