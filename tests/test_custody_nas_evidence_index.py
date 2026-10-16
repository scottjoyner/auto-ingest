"""Historical NAS evidence index cannot promote old receipts to custody."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from auto_ingest.custody import nas_evidence_index as ix
from auto_ingest.custody.campaign import CardIdentity
from auto_ingest.custody.store import new_campaign


def _csv(path: Path, names, values):
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=names)
        writer.writeheader()
        writer.writerows(values)


@pytest.fixture
def indexed(tmp_path):
    mountmap = dict(ix.SOURCE_MOUNTS)
    dest = tmp_path / "campaigns"
    old = tmp_path / "frozen.jsonl"
    sources = [
        ("5140-6932", "DCIM/Movie/2026_0627_113505_R_YOLOv8n.csv", 3),
        ("68FC-E499", "DCIM/Movie/RO/2025_1205_195317_F.MP4", 9),
        ("69BE-B963", "DCIM/Movie/2026_0907_130818_R.MP4", 4),
        ("5140-6932", "no_indexed_candidate.MP4", 5),
    ]
    old.write_text("".join(json.dumps({
        "card_uuid": uuid, "source_relative_path": path, "logical_bytes": size,
        "x1_two_nvme_full_sha_verified": True, "canonical_nas_verified": False,
    }) + "\n" for uuid, path, size in sources))
    for uuid, mount in mountmap.items():
        new_campaign(dest / uuid, card_id=uuid,
                     observed=CardIdentity(filesystem_uuid=uuid),
                     mount_point=mount, read_only=False, env={}, config={}, apply=True)
    stage = tmp_path / "stage.csv"
    _csv(stage, ["card_uuid", "source_relative_path", "source_expected_bytes",
                 "staged_observed_size", "staged_exact_status"], [{
        "card_uuid": sources[0][0], "source_relative_path": sources[0][1],
        "source_expected_bytes": 3, "staged_observed_size": 3,
        "staged_exact_status": "MATCHING_SIZE_ONLY",
    }])
    alt = tmp_path / "alternate.csv"
    destination = "/nas/archive/dashcam/deathstar/2025_1205_195317_F.MP4"
    _csv(alt, ["card_uuid", "source_relative_path", "expected_bytes",
               "observed_bytes", "candidate_path", "metadata_status"], [{
        "card_uuid": sources[1][0], "source_relative_path": sources[1][1],
        "expected_bytes": 9, "observed_bytes": 9,
        "candidate_path": destination, "metadata_status": "PATH_AND_SIZE_MATCH",
    }])
    stage_w = tmp_path / "staged-witness.json"
    stage_w.write_text(json.dumps({
        "source_card_uuid": "5140-6932", "target_mount": ix.STAGING_ROOT,
        "records": [{"path": sources[0][1], "state": "SHA_MATCH_STABLE",
                     "stable_pre_post": True, "bytes": 3,
                     "source_sha256": "a" * 64, "nas_stage_sha256": "a" * 64}],
    }))
    alt_w = tmp_path / "alternate-witness.json"
    alt_w.write_text(json.dumps({"cases": [{
        "card_uuid": "68FC-E499",
        "source": mountmap["68FC-E499"] + "/" + sources[1][1],
        "alternative_nas_target": destination, "source_bytes": 9,
        "source_sha256_now": "b" * 64, "destination_sha256_now": "b" * 64,
        "equal_source_dest": True, "stable_pre_post": True,
    }]}))
    return old, hashlib.sha256(old.read_bytes()).hexdigest(), dest, stage, alt, stage_w, alt_w


def run(args):
    return ix.build_index(*args)


def test_manifest_join_keeps_all_old_sha_outside_live_custody(indexed):
    report = run(indexed)
    assert report["total_source_files"] == 4
    assert report["counts"]["prior_sha_witness_recheck_required"] == 2
    assert report["counts"]["no_indexed_candidate_not_proven_missing"] == 2
    assert report["current_nas_sha_verified_files"] == 0
    assert report["source_release_allowed"] is False
    assert report["counts"]["video_mp4_prior_sha_witness_recheck_required"] == 1
    assert report["counts"]["video_mp4_no_indexed_candidate_not_proven_missing"] == 2
    assert all(r["current_nas_sha_verified"] is False for r in report["rows"])
    assert all(r["source_release_allowed"] is False for r in report["rows"])
    assert set(report["cards"]) == set(ix.SOURCE_MOUNTS)


def test_corrupt_staged_witness_or_file_size_refused(indexed):
    old, sha, root, stage, alt, stage_w, alt_w = indexed
    doc = json.loads(stage_w.read_text())
    doc["records"][0]["nas_stage_sha256"] = "c" * 64
    stage_w.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="inconsistent historical staged"):
        run(indexed)


def test_wrong_frozen_digest_refused(indexed):
    args = list(indexed)
    args[1] = "f" * 64
    with pytest.raises(ValueError, match="pinned frozen-ledger"):
        run(args)


def test_untrusted_alternate_path_refused(indexed):
    old, sha, root, stage, alt, stage_w, alt_w = indexed
    text = alt.read_text().replace(
        "/nas/archive/dashcam/deathstar/", "/nas/archive/../escape/")
    alt.write_text(text)
    with pytest.raises(ValueError, match="unsafe indexed candidate"):
        run(indexed)


def test_cross_card_or_missing_campaign_blocked(indexed):
    old, sha, root, stage, alt, stage_w, alt_w = indexed
    (root / "69BE-B963" / "campaign.json").unlink()
    with pytest.raises(Exception):
        run(indexed)


def test_prior_source_receipts_are_not_live_nas_verification(indexed):
    report = run(indexed)
    assert sum(r["historical_offhost_source_receipt"]
               for r in report["rows"]) == 4
    assert not any(r["current_nas_sha_verified"] for r in report["rows"])
