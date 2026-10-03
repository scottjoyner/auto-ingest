"""Idempotency: inspect / plan / inspect / plan must be a no-op and identical.

Covers both halves of the requirement:

* byte-identical output for unchanged evidence (no clock, no randomness, no
  dict-iteration order leaking into the payload);
* zero filesystem mutation while computing status (asserted with a real audit
  hook in a subprocess - see ``test_custody_readonly.py``).
"""
from __future__ import annotations

import json

from custody_helpers import (
    CARD_01_BUNDLE,
    campaign,
    copying,
    destination_evidence,
    evidence,
    fully_copied_campaign,
    hashing,
    inventory,
    reconciliation,
    strict_policy,
    worker,
)

from auto_ingest.custody import (
    CampaignEvidence,
    build_status,
    derive_state,
    evaluate_release,
    plan_resume,
)
from auto_ingest.custody.report import plan_json, plan_text, status_json, status_text
from auto_ingest.custody.store import load_status

SCENARIOS = {}


def _register():
    SCENARIOS["card01"] = load_status(CARD_01_BUNDLE)
    camp, ev = fully_copied_campaign()
    SCENARIOS["released"] = build_status(camp, ev, strict_policy())
    interrupted = evidence(
        inv=inventory(67644, 676_440_000, complete=True, verified=True),
        hsh=hashing(67644, verified_bytes=676_440_000, complete=True),
        cpy=copying(planned_files=67644, planned_bytes=676_440_000, started=True,
                    result_complete=False, interrupted=True),
        dst=destination_evidence(verified_files=12000, verified_bytes=120_000_000,
                                 verification_started=True, unverified_present_files=900),
        rec=reconciliation(source_only=55644),
        wkr=worker(identity="w1", status="stopped"),
    )
    SCENARIOS["reconcile"] = build_status(campaign(), interrupted, strict_policy())
    SCENARIOS["fresh"] = build_status(campaign(), evidence(), strict_policy())


_register()


def test_status_json_is_byte_identical_across_runs():
    for name, status in SCENARIOS.items():
        first = status_json(status)
        for _ in range(3):
            assert status_json(status) == first, name


def test_status_text_is_identical_across_runs():
    for name, status in SCENARIOS.items():
        first = status_text(status)
        for _ in range(3):
            assert status_text(status) == first, name


def test_plan_json_and_text_are_identical_across_runs():
    for name, status in SCENARIOS.items():
        first = (plan_json(status.plan), plan_text(status.plan))
        for _ in range(3):
            assert (plan_json(status.plan), plan_text(status.plan)) == first, name


def test_status_json_keys_are_sorted():
    blob = status_json(SCENARIOS["card01"])
    parsed = json.loads(blob)
    assert list(parsed) == sorted(parsed)
    # round trip is stable
    assert json.dumps(parsed, sort_keys=True, indent=2) == blob


def test_no_duplicate_action_ids_when_planning_repeatedly():
    status = SCENARIOS["reconcile"]
    ids = [a.action_id for a in status.plan.actions]
    assert status.campaign_id
    again = plan_resume(status.campaign, status.evidence, strict_policy())
    assert [a.action_id for a in again.actions] == ids
    assert len(ids) == len(set(ids))
    assert again.plan_fingerprint == status.plan.plan_fingerprint


def test_inspect_plan_inspect_plan_sequence_is_stable():
    bundle = CARD_01_BUNDLE
    outputs = []
    for _ in range(2):
        outputs.append(status_json(load_status(bundle)))
        outputs.append(plan_json(load_status(bundle).plan))
    assert outputs[0] == outputs[2]
    assert outputs[1] == outputs[3]


def test_deriving_twice_produces_the_same_state_and_reasons():
    camp, ev = fully_copied_campaign(total_files=42, total_bytes=4200)
    a = derive_state(camp, ev, strict_policy())
    b = derive_state(camp, ev, strict_policy())
    assert (a.state, a.reasons, a.blockers) == (b.state, b.reasons, b.blockers)


def test_release_decision_is_deterministic():
    camp, ev = fully_copied_campaign(unresolved=2)
    a = evaluate_release(camp, ev, strict_policy()).to_dict()
    b = evaluate_release(camp, ev, strict_policy()).to_dict()
    assert a == b


def test_evidence_dict_round_trip_is_stable():
    camp, ev = fully_copied_campaign()
    raw = ev.to_dict()
    assert CampaignEvidence.from_dict(raw, strict_policy()).to_dict() == raw
    assert json.dumps(raw, sort_keys=True) == json.dumps(ev.to_dict(), sort_keys=True)


def test_campaign_dict_round_trip_is_stable():
    camp, ev = fully_copied_campaign()
    from auto_ingest.custody import Campaign

    raw = camp.to_dict()
    assert Campaign.from_dict(raw).to_dict() == raw


def test_status_has_no_generated_timestamp():
    """No `now()` in the core: two builds seconds apart are identical."""
    a = build_status(campaign(), evidence(), strict_policy())
    b = build_status(campaign(), evidence(), strict_policy())
    assert a.to_dict() == b.to_dict()


def test_planner_output_never_claims_execution():
    for name, status in SCENARIOS.items():
        payload = status.plan.to_dict()
        assert payload["source_mutation_allowed"] is False
        assert payload["source_deletion_allowed"] is False
        assert payload["requires_operator_authorization"] is True
        for action in payload["actions"]:
            assert action["requires_authorization"] is True
            assert action["mutates_source"] is False
        assert name  # keep parametrised failures readable
