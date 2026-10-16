"""Manifest-only NAS anti-join never produces release proofs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from auto_ingest.custody.campaign import CardIdentity
from auto_ingest.custody.nas_evidence_index import build_index
from auto_ingest.custody.store import new_campaign


def write_csv(path: Path, rows: list[dict]) -> None:
    assert rows
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture()
def inputs(tmp_path):
    from auto_ingest.custody.nas_evidence_index import input_sha
    cards = {
        "5140-6932": "/mnt/reader/cardC",
        "69BE-B963": "/mnt/reader/cardA",
        "68FC-E499": "/mnt/reader/cardB",
    }
    bundles = tmp_path / "bundles"
    for uuid, mount in cards.items():
        new_campaign(
            bundles / uuid, card_id=f"SD-{uuid}",
            observed=CardIdentity(filesystem_uuid=uuid),
            mount_point=mount, read_only=False,
            created_at="2026-10-16T14:00:00Z",
            config={}, env={}, apply=True,
        )
    ledger = tmp_path / "source.jsonl"
    sources = [
        ("5140-6932", "DCIM/a.MP4", 3),
        ("5140-6932", "DCIM/b.csv", 4),
        ("69BE-B963", "DCIM/c.MP4", 3),
        ("68FC-E499", "DCIM/d.MP4", 8),
        ("68FC-E499", "DCIM/e.MP4", 2),
    ]
    ledger.write_text("".join(json.dumps({
        "card_uuid": uuid, "source_relative_path": key,
        "logical_bytes": size, "joyner_local_host_full_sha_verified": True,
    }) + "\n" for uuid, key, size in sources))
    stage = tmp_path / "staging.csv"
    staging_rows = [
        {"card_uuid": "5140-6932", "source_relative_path": key,
         "source_expected_bytes": size, "staged_observed_size": size,
         "staged_exact_status": "MATCHING_SIZE_ONLY"}
        for key, size in (("DCIM/a.MP4", 3), ("DCIM/b.csv", 4))
    ]
    write_csv(stage, staging_rows)
    alt = tmp_path / "alternate.csv"
    alternate_rows = [
        {"card_uuid": uuid, "source_relative_path": key,
         "expected_bytes": size, "observed_bytes": size,
         "metadata_status": "PATH_AND_SIZE_MATCH",
         "candidate_path": location}
        for uuid, key, size, location in (
            ("69BE-B963", "DCIM/c.MP4", 3, "/nas/archive/c.MP4"),
            ("68FC-E499", "DCIM/e.MP4", 2, "/nas/archive/e.MP4"),
        )
    ]
    write_csv(alt, alternate_rows)
    root = "/nas/recovery/sd-card-c"
    digest = "a" * 64
    stage_witness = tmp_path / "stage_witness.json"
    stage_witness.write_text(json.dumps({
        "source_card_uuid": "5140-6932", "target_mount": root,
        "records": [{
            "path": "DCIM/a.MP4", "state": "SHA_MATCH_STABLE",
            "stable_pre_post": True, "bytes": 3,
            "source_sha256": digest, "nas_stage_sha256": digest,
        }],
    }))
    alt_witness = tmp_path / "alternate_witness.json"
    alt_witness.write_text(json.dumps({"cases": [{
        "card_uuid": "69BE-B963",
        "source": cards["69BE-B963"] + "/DCIM/c.MP4",
        "alternative_nas_target": "/nas/archive/c.MP4",
        "source_bytes": 3,
        "source_sha256_now": digest, "destination_sha256_now": digest,
        "stable_pre_post": True, "equal_source_dest": True,
    }]}))
    return {
        "ledger": ledger, "sha": input_sha(ledger), "bundles": bundles,
        "stage": stage, "alternate": alt,
        "stage_witness": stage_witness, "alt_witness": alt_witness,
        "root": root, "stage_rows": staging_rows,
        "alternate_rows": alternate_rows, "cards": cards,
    }


def run(x, **overrides):
    return build_index(
        x["ledger"], overrides.get("sha", x["sha"]), x["bundles"],
        x["stage"], x["alternate"], x["stage_witness"], x["alt_witness"],
        staging_root=overrides.get("root", x["root"]),
    )


def test_full_join_is_manifest_only_with_zero_current_nas_verifications(inputs):
    result = run(inputs)
    assert result["total_source_files"] == 5
    assert result["counts"]["prior_sha_witness_recheck_required"] == 2
    assert result["counts"]["metadata_candidate_needs_sha"] == 2
    assert result["counts"]["no_indexed_candidate_not_proven_missing"] == 1
    assert result["current_nas_sha_verified_files"] == 0
    assert result["source_release_allowed"] is False
    assert result["counts"]["mp4_named_files"] == 4
    assert result["counts"]["mp4_prior_sha_witness_recheck_required_files"] == 2
    assert result["counts"]["mp4_metadata_candidate_needs_sha_files"] == 1
    assert result["counts"]["mp4_no_indexed_candidate_not_proven_missing_files"] == 1
    assert len(result["rows"]) == 5
    assert result["cards"]["5140-6932"]["source_mount"] == "/mnt/reader/cardC"
    assert all(not row["current_nas_sha_verified"] and
               not row["source_release_allowed"] for row in result["rows"])


def test_wrong_pinned_ledger_and_staging_root_refused(inputs):
    with pytest.raises(ValueError, match="pinned frozen"):
        run(inputs, sha="0" * 64)
    with pytest.raises(ValueError, match="staging root"):
        run(inputs, root="/mnt/arbitrary")
    with pytest.raises(ValueError, match="staging root"):
        run(inputs, root="/nas/../escape")


def test_staging_witness_does_not_accept_forged_sha_match(inputs):
    obj = json.loads(inputs["stage_witness"].read_text())
    obj["records"][0]["nas_stage_sha256"] = "0" * 64
    inputs["stage_witness"].write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="inconsistent historical staged SHA"):
        run(inputs)


def test_candidate_path_may_not_be_shared_between_sources(inputs):
    rows = inputs["alternate_rows"]
    rows[1]["candidate_path"] = rows[0]["candidate_path"]
    rows[1]["observed_bytes"] = 2
    write_csv(inputs["alternate"], rows)
    with pytest.raises(ValueError, match="multiple source objects"):
        run(inputs)


def test_malformed_or_out_of_scope_candidate_refused(inputs):
    rows = inputs["alternate_rows"]
    rows[0]["candidate_path"] = "/nas/../../outside"
    write_csv(inputs["alternate"], rows)
    with pytest.raises(ValueError, match="unsafe indexed candidate"):
        run(inputs)


def test_wrong_historical_source_uuid_or_path_cannot_join(inputs):
    obj = json.loads(inputs["alt_witness"].read_text())
    obj["cases"][0]["source"] = "/mnt/other/card/DCIM/c.MP4"
    inputs["alt_witness"].write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="source path/UUID mismatch"):
        run(inputs)


def test_missing_campaign_or_wrong_identity_fails_closed(inputs):
    (inputs["bundles"] / "69BE-B963" / "campaign.json").unlink()
    with pytest.raises(Exception):
        run(inputs)
