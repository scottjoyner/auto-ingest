"""Source-custody campaign bootstrap must never promote metadata to SHA custody."""
from __future__ import annotations

import hashlib
import json

import pytest

from auto_ingest.custody import fleet_campaigns as fc
from auto_ingest.custody.mounts import MountObservation
from auto_ingest.custody.store import load_campaign, load_status


@pytest.fixture()
def snapshot(tmp_path, monkeypatch):
    cards = {}
    for uuid, filename in (
        ("ABCD-1234", "DCIM/Movie/2026_0101_101010_F.MP4"),
        ("EF01-2345", "DCIM/Movie/2026_0101_101010_R.MP4"),
    ):
        root = tmp_path / uuid
        source = root / filename
        source.parent.mkdir(parents=True)
        source.write_bytes(b"old source survives")
        cards[uuid] = root

    ledger = tmp_path / "frozen.jsonl"
    rows = [
        {"card_uuid": uuid, "source_relative_path": filename,
         "logical_bytes": len(b"old source survives"),
         "joyner_local_host_full_sha_verified": True,
         "canonical_nas_verified": False,
         "source_release_authorized": False}
        for uuid, filename in (
            ("ABCD-1234", "DCIM/Movie/2026_0101_101010_F.MP4"),
            ("EF01-2345", "DCIM/Movie/2026_0101_101010_R.MP4"),
        )
    ]
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
    device_to_uuid = {
        "/dev/testA": "ABCD-1234",
        "/dev/testB": "EF01-2345",
    }

    def mounted(path, **_):
        match = next((k for k, v in cards.items() if str(v) == path), None)
        device = {"ABCD-1234": "/dev/testA", "EF01-2345": "/dev/testB"}[match]
        return MountObservation(path, True, device=device, filesystem_type="vfat",
                                options=("rw",))

    monkeypatch.setattr(fc, "observe_mount", mounted)
    monkeypatch.setattr(fc, "uuid_for_device", lambda device: device_to_uuid[device])
    return ledger, [fc.parse_card(f"{uuid}={root}") for uuid, root in cards.items()], cards


def test_metadata_assessment_matches_and_never_promotes_custody(snapshot, tmp_path):
    ledger, cards, roots = snapshot
    sha = hashlib.sha256(ledger.read_bytes()).hexdigest()
    report = fc.plan(ledger, cards, pinned_sha=sha)
    assert report["all_metadata_match"]
    assert report["total_files"] == 2
    assert sum(x["historical_offhost_qualified_matches"] for x in report["cards"]) == 2
    assert not report["source_release_allowed"]
    assert not report["destination_sha_verified"]
    assert not report["custody_inventory_evidence_imported"]
    draft = fc.create(report, tmp_path / "bundles")
    assert not draft["applied"]
    assert not (tmp_path / "bundles").exists()
    assert all(any(root.rglob("*.MP4")) for root in roots.values())


def test_apply_creates_only_closed_uuid_bound_campaigns(snapshot, tmp_path):
    ledger, cards, roots = snapshot
    report = fc.plan(ledger, cards, pinned_sha=hashlib.sha256(ledger.read_bytes()).hexdigest())
    out = tmp_path / "bundles"
    result = fc.create(report, out, apply=True)
    assert result["applied"]
    assert len(result["campaigns"]) == 2
    for card in result["cards"]:
        path = out / card["card_uuid"]
        campaign = load_campaign(path)
        assert campaign.source.card.filesystem_uuid == card["card_uuid"]
        assert campaign.source.card.device is None
        assert campaign.source.read_only is False
        assert campaign.destination.host_path is None
        assert (path / "frozen-source-assessment.json").exists()
        assert not (path / "evidence.json").exists()
        assert not (path / "ledgers").exists()
        assert not load_status(path).source_release_allowed
    replay = fc.create(report, out, apply=False)
    assert not replay["applied"]
    assert [r["campaign_id"] for r in replay["campaigns"]] == [
        r["campaign_id"] for r in result["campaigns"]]
    with pytest.raises(ValueError, match="nonempty"):
        fc.create(report, out, apply=True)
    assert all(any(root.rglob("*.MP4")) for root in roots.values())


def test_changed_file_prevents_any_campaign_creation(snapshot, tmp_path):
    ledger, cards, roots = snapshot
    next(iter(roots.values())).rglob("*.MP4").__next__().write_bytes(b"changed bytes")
    report = fc.plan(ledger, cards, pinned_sha=hashlib.sha256(ledger.read_bytes()).hexdigest())
    assert not report["all_metadata_match"]
    with pytest.raises(ValueError, match="physical card differs"):
        fc.create(report, tmp_path / "bundles", apply=True)
    assert not (tmp_path / "bundles").exists()


def test_bad_digest_or_wrong_observed_uuid_fails(snapshot, tmp_path, monkeypatch):
    ledger, cards, _ = snapshot
    with pytest.raises(ValueError, match="SHA"):
        fc.plan(ledger, cards, pinned_sha="f" * 64)
    monkeypatch.setattr(fc, "uuid_for_device", lambda device: "FFFF-FFFF")
    report = fc.plan(ledger, cards, pinned_sha=hashlib.sha256(ledger.read_bytes()).hexdigest())
    assert not report["all_metadata_match"]
    with pytest.raises(ValueError):
        fc.create(report, tmp_path / "bundles", apply=True)


def test_duplicate_source_key_in_frozen_ledger_is_invalid(snapshot):
    ledger, cards, _ = snapshot
    original = ledger.read_text().splitlines(keepends=True)
    ledger.write_text("".join(original + [original[0]]))
    with pytest.raises(ValueError, match="duplicate source key"):
        fc.plan(ledger, cards)


def test_unsafe_paths_and_duplicate_uuids_are_rejected(snapshot):
    ledger, cards, _ = snapshot
    for key in ("/etc/passwd", "../escape", "A/../B", "a\\b", ""):
        assert not fc.safe_key(key)
    with pytest.raises(ValueError, match="duplicate card UUID"):
        fc.plan(ledger, cards + cards[:1])
    for spec in ("UNTITLED=/a", "ABCD-1234=relative", "ABCD-1234=/"):
        with pytest.raises(ValueError):
            fc.parse_card(spec)
