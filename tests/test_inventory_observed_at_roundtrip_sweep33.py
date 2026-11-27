"""Custody inventory provenance must survive an evidence round trip (sweep33).

``InventoryEvidence.from_dict`` reads ``inventory.observed_at`` (the instant the
source card's inventory walk ran), but ``InventoryEvidence.to_dict`` omitted it.
Because ``custody evidence import --apply`` rewrites ``evidence.json`` through
``CampaignEvidence.to_dict()``, that timestamp was silently lost on exactly the
path whose job is to preserve declared evidence fields.

All tests are pure/offline: in-memory evidence dicts plus a tiny synthetic
bundle under pytest's ``tmp_path``. No Neo4j, no mounts, no network, no clock.
"""
from __future__ import annotations

import json

from custody_helpers import campaign

from auto_ingest.custody import CustodyPolicy, InventoryEvidence
from auto_ingest.custody.evidence import CampaignEvidence
from auto_ingest.custody.store import import_evidence

OBSERVED = "2026-11-27T01:02:03Z"


def _raw_inventory() -> dict:
    return {
        "discovered_files": 3,
        "discovered_bytes": 300,
        "complete": True,
        "verified": True,
        "roots": ["card01/DCIM"],
        "observed_at": OBSERVED,
    }


def test_inventory_observed_at_survives_direct_roundtrip():
    """DEFECT: the field is parsed but dropped by InventoryEvidence.to_dict."""
    ev = InventoryEvidence.from_dict(_raw_inventory(), 20)
    assert ev.observed_at == OBSERVED  # parsed correctly on both sides
    assert ev.to_dict()["observed_at"] == OBSERVED  # fails on pristine base


def test_campaign_evidence_roundtrip_preserves_inventory_observed_at():
    """DEFECT: the loss survives the normal CampaignEvidence wrapper."""
    raw = {"inventory": _raw_inventory(), "observed_at": "2026-11-27T02:00:00Z"}
    out = CampaignEvidence.from_dict(raw, CustodyPolicy()).to_dict()
    assert out["inventory"]["observed_at"] == OBSERVED  # fails on pristine base


def test_import_evidence_apply_preserves_inventory_observed_at(tmp_path):
    """DEFECT via the real write path: `evidence import --apply` loses it.

    A minimal synthetic bundle is enough; no live card or mount is involved.
    """
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "campaign.json").write_text(
        json.dumps(campaign().to_dict()), encoding="utf-8")

    result = import_evidence(
        bundle,
        {"inventory": _raw_inventory()},
        CustodyPolicy(),
        apply=True,
    )
    assert result["applied"] is True
    written = json.loads((bundle / "evidence.json").read_text(encoding="utf-8"))
    assert written["inventory"]["observed_at"] == OBSERVED  # fails on pristine base


def test_inventory_other_fields_still_roundtrip():
    """Control: the fix must not disturb the rest of the inventory block."""
    ev = InventoryEvidence.from_dict(_raw_inventory(), 20)
    out = ev.to_dict()
    assert out["discovered_files"] == 3
    assert out["discovered_bytes"] == 300
    assert out["roots"] == ["card01/DCIM"]
    assert out["complete"] is True
    assert out["verified"] is True


def test_campaign_top_level_observed_at_still_roundtrips():
    """Control: the distinct top-level timestamp is already preserved."""
    raw = {"observed_at": "2026-11-27T02:00:00Z"}
    out = CampaignEvidence.from_dict(raw, CustodyPolicy()).to_dict()
    assert out["observed_at"] == "2026-11-27T02:00:00Z"


def test_missing_inventory_observed_at_reads_as_none():
    """Control: absent timestamp stays absent/None on both sides."""
    ev = InventoryEvidence.from_dict({"discovered_files": 1}, 20)
    assert ev.observed_at is None
    assert ev.to_dict().get("observed_at") is None
