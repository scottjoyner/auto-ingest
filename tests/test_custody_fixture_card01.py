"""The CARD-01 fixture: the latest observed situation, as data.

Expected state is *derived by the implementation*, then asserted. The fixture is
never connected to the physical card and nothing here opens `/media/scott/UNTITLED`.
"""
from __future__ import annotations

import json

from custody_helpers import CARD_01_BUNDLE, strict_policy

from auto_ingest.custody import (
    CampaignState,
    build_status,
    derive_state,
)
from auto_ingest.custody.store import load_campaign, load_evidence, load_status


def test_fixture_files_exist():
    assert (CARD_01_BUNDLE / "campaign.json").is_file()
    assert (CARD_01_BUNDLE / "evidence.json").is_file()


def test_fixture_encodes_the_reported_card_01_facts():
    campaign = load_campaign(CARD_01_BUNDLE)
    evidence = load_evidence(CARD_01_BUNDLE)
    assert campaign.card_id == "CARD-01"
    assert campaign.source.mount_point == "/media/scott/UNTITLED"
    assert campaign.source.read_only is True
    assert evidence.hashing.verified_files == 67644
    assert evidence.hashing.complete is True
    assert evidence.copy.result_complete is False
    assert evidence.destination.verified_files == 0
    assert evidence.destination.verified_bytes == 0
    assert evidence.worker.status == "stopped"
    assert evidence.errors.unresolved == 0


def test_card_01_is_not_safe_to_release():
    status = load_status(CARD_01_BUNDLE)
    assert status.source_release_allowed is False
    assert status.release.allowed is False
    assert status.state is not CampaignState.SAFE_TO_RELEASE


def test_card_01_derives_reconcile_required():
    campaign = load_campaign(CARD_01_BUNDLE)
    evidence = load_evidence(CARD_01_BUNDLE)
    derivation = derive_state(campaign, evidence, strict_policy())
    assert derivation.state is CampaignState.RECONCILE_REQUIRED
    assert derivation.state is not CampaignState.COPY_PENDING


def test_card_01_plan_prefers_destination_reconciliation():
    status = load_status(CARD_01_BUNDLE)
    plan = status.plan
    assert plan.next_phase == "destination_reconciliation"
    assert plan.next_safe_action == "reconcile_destination"
    assert plan.source_mutation_allowed is False
    assert plan.source_deletion_allowed is False
    assert [a.operation for a in plan.actions][:2] == [
        "verify_existing_destination", "copy_objects",
    ]


def test_card_01_destination_is_unresolved_and_that_is_a_blocker():
    status = load_status(CARD_01_BUNDLE)
    codes = {b.code for b in status.release.blockers}
    assert "destination_unresolved" in codes
    assert "destination_identity_unproven" in codes
    assert "missing_destination" in codes


def test_card_01_destination_sample_ledger_verifies_nothing():
    status = load_status(CARD_01_BUNDLE)
    summary = status.ledgers["destination.jsonl"]
    assert summary.present is True
    assert summary.records == 3
    assert summary.files == 0
    assert summary.by_status.get("pending") == 3


def test_card_01_with_a_destination_configured_is_still_not_releasable():
    """Resolving a destination changes the plan, never the verdict."""
    from dataclasses import replace

    from auto_ingest.custody import StorageIdentity

    campaign = load_campaign(CARD_01_BUNDLE)
    evidence = load_evidence(CARD_01_BUNDLE)
    resolved = replace(
        campaign,
        destination=replace(
            campaign.destination,
            host_path="/mnt/custody",
            resolved_from="test",
            mounted=True,
            identity=StorageIdentity(filesystem_uuid="DEST-4242", device="/dev/nvme0n1p1"),
        ),
    )
    observed = replace(
        evidence,
        destination=replace(
            evidence.destination,
            observed_identity=StorageIdentity(filesystem_uuid="DEST-4242",
                                             device="/dev/nvme0n1p1"),
        ),
    )
    status = build_status(resolved, observed, strict_policy())
    assert status.state is CampaignState.RECONCILE_REQUIRED
    assert status.source_release_allowed is False
    # ...but it is now safe to resume the reconciliation work.
    assert status.plan.safe_to_resume is True


def test_card_01_json_status_contract_keys():
    status = load_status(CARD_01_BUNDLE)
    data = status.to_dict()
    assert data["state"] == "RECONCILE_REQUIRED"
    assert data["source_read_only"] is True
    assert data["hash"]["verified"] == 67644
    assert data["copy"]["complete"] is False
    assert data["destination"]["verified_files"] == 0
    assert data["destination"]["verified_bytes"] == 0
    assert data["source_release_allowed"] is False
    assert data["next_safe_action"] == "reconcile_destination"
    assert json.loads(json.dumps(data)) == data
