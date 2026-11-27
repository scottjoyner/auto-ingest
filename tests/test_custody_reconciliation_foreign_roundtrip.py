"""`reconciliation.foreign_objects` must survive the evidence round trip.

`custody reconcile` offers a proposal carrying ``foreign_objects``: files at the
destination root that this campaign never recorded. `custody import --apply`
re-parses that document into ``CampaignEvidence`` before writing it back and
before deriving SAFE_TO_RELEASE. If the parser drops the field, the strict
destination-scope gate loses the one fact it was added to enforce, and the
operator's own proposal silently clears a blocker it just raised.

Everything here is in-memory: no campaign bundle, mount, ledger or network path.
"""
from __future__ import annotations

from custody_helpers import fully_copied_campaign, strict_policy

from auto_ingest.custody import CampaignEvidence, ReconciliationEvidence, evaluate_release


def test_reconciliation_evidence_parses_and_round_trips_foreign_objects():
    declared = {
        "source_only": 0,
        "destination_only": 0,
        "mismatched": 0,
        "foreign_objects": 7,
        "reconciled_at": "2026-11-26T00:00:00Z",
    }
    rec = ReconciliationEvidence.from_dict(declared)
    assert rec.foreign_objects == 7
    assert rec.to_dict()["foreign_objects"] == 7
    assert ReconciliationEvidence.from_dict(rec.to_dict()).foreign_objects == 7


def test_campaign_evidence_round_trip_preserves_foreign_objects():
    raw = {"reconciliation": {"foreign_objects": 7}}
    evidence = CampaignEvidence.from_dict(raw, strict_policy())
    assert evidence.reconciliation.foreign_objects == 7
    assert evidence.to_dict()["reconciliation"]["foreign_objects"] == 7
    # import --apply writes to_dict() and every later read parses it again.
    assert CampaignEvidence.from_dict(
        evidence.to_dict(), strict_policy()
    ).reconciliation.foreign_objects == 7


def test_imported_foreign_objects_close_the_strict_release_gate():
    camp, ev = fully_copied_campaign(total_files=4, total_bytes=40)
    # The proposal an operator gets from `custody reconcile --json` and feeds to
    # `custody import --apply` declares foreign objects at the destination.
    declared = ev.to_dict()
    declared["reconciliation"]["foreign_objects"] = 7
    rehydrated = CampaignEvidence.from_dict(
        declared, strict_policy(strict_destination_scope=True)
    )

    strict = evaluate_release(camp, rehydrated,
                              strict_policy(strict_destination_scope=True))
    assert strict.allowed is False
    assert "foreign_destination_objects" in {b.code for b in strict.blockers}
    assert any("foreign_objects=7" in b.detail for b in strict.blockers)

    # Even the advisory (non-strict) path must still report the fact.
    lenient = evaluate_release(camp, rehydrated, strict_policy())
    assert lenient.allowed is True
    assert any("foreign_objects=7" in w for w in lenient.warnings)


def test_malformed_foreign_count_is_coerced_and_diagnosed_not_silently_dropped():
    evidence = CampaignEvidence.from_dict(
        {"reconciliation": {"foreign_objects": "not-a-count"}}, strict_policy()
    )
    assert evidence.reconciliation.foreign_objects == 0
    assert "reconciliation.foreign_objects='not-a-count'" in evidence.coerced_fields
