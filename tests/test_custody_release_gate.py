"""The fail-closed source-release gate.

SAFE_TO_RELEASE must be unreachable unless every configured custody condition
holds. These tests attack the gate from the "looks finished but is not" angles.
"""
from __future__ import annotations

from dataclasses import replace

from custody_helpers import (
    campaign,
    copying,
    destination,
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
    CampaignState,
    StorageIdentity,
    derive_state,
    evaluate_release,
)
from auto_ingest.custody.release import CONDITION_ORDER


def codes(decision):
    return {b.code for b in decision.blockers}


def test_clean_campaign_releases():
    camp, ev = fully_copied_campaign()
    decision = evaluate_release(camp, ev, strict_policy())
    assert decision.allowed is True
    assert decision.blockers == ()
    assert decision.conditions == CONDITION_ORDER


def test_every_condition_is_evaluated_even_when_one_fails():
    """No short circuit: a broken campaign reports every unmet condition."""
    camp, ev = fully_copied_campaign(total_files=10, total_bytes=100, unresolved=2,
                                     mismatched=1)
    decision = evaluate_release(camp, ev, strict_policy())
    assert decision.allowed is False
    assert {"mismatch_present", "unresolved_errors"} <= codes(decision)


def test_source_must_be_read_only():
    camp, ev = fully_copied_campaign()
    writable = replace(camp, source=replace(camp.source, read_only=False))
    assert "source_not_read_only" in codes(evaluate_release(writable, ev, strict_policy()))
    # ...and the policy can be told not to care (it still must be a decision).
    lax = evaluate_release(writable, ev, strict_policy(require_read_only_source=False))
    assert "source_not_read_only" not in codes(lax)


def test_destination_must_resolve_and_be_mounted():
    camp, ev = fully_copied_campaign()
    unresolved = replace(camp, destination=destination(host_path=None))
    assert "destination_unresolved" in codes(evaluate_release(unresolved, ev, strict_policy()))
    unmounted = replace(camp, destination=destination(mounted=False))
    assert "destination_not_mounted" in codes(evaluate_release(unmounted, ev, strict_policy()))


def test_destination_identity_must_match():
    camp, ev = fully_copied_campaign()
    other = replace(
        camp,
        destination=replace(camp.destination,
                            identity=StorageIdentity(filesystem_uuid="DEST-0001")),
    )
    assert "destination_identity_unproven" in codes(evaluate_release(other, ev, strict_policy()))
    loose = evaluate_release(other, ev, strict_policy(require_destination_identity=False))
    assert "destination_identity_unproven" not in codes(loose)


def test_inventory_hash_and_copy_must_all_be_complete():
    camp = campaign()
    base_incomplete = evidence(inv=inventory(10, 100, complete=False))
    assert "inventory_incomplete" in codes(evaluate_release(camp, base_incomplete, strict_policy()))

    no_hash = evidence(inv=inventory(10, 100, complete=True, verified=True),
                       hsh=hashing(4, verified_bytes=40, complete=False))
    assert "hash_evidence_incomplete" in codes(evaluate_release(camp, no_hash, strict_policy()))

    no_copy = evidence(inv=inventory(10, 100, complete=True, verified=True),
                       hsh=hashing(10, verified_bytes=100, complete=True),
                       cpy=copying(planned_files=10, planned_bytes=100, started=True,
                                   result_complete=False))
    found = codes(evaluate_release(camp, no_copy, strict_policy()))
    assert {"copy_incomplete", "copy_ledger_incomplete"} <= found


def test_undeclared_hash_exemption_is_a_blocker_not_a_pass():
    camp = campaign()
    ev = evidence(inv=inventory(10, 100, complete=True, verified=True),
                  hsh=hashing(9, verified_bytes=90, complete=False,
                              exemptions=("*.tmp",)),
                  cpy=copying(planned_files=10, planned_bytes=100, started=True,
                              result_complete=True, ledger_complete=True,
                              completed_files=10, completed_bytes=100),
                  dst=destination_evidence(verified_files=10, verified_bytes=100,
                                           verification_started=True,
                                           verification_complete=True,
                                           observed_identity=camp.destination.identity),
                  rec=reconciliation(), wkr=worker(status="stopped"))
    found = codes(evaluate_release(camp, ev, strict_policy()))
    assert "hash_exemptions_not_declared" in found
    assert "hash_evidence_incomplete" in found


def test_declared_hash_exemption_is_honoured():
    camp, ev = fully_copied_campaign(total_files=9, total_bytes=90)
    ev = replace(ev, hashing=replace(ev.hashing, verified_files=8, complete=True,
                                     exemptions=("*.tmp",)))
    ev = replace(ev, copy=replace(ev.copy, planned=replace(ev.copy.planned, files=9)))
    decision = evaluate_release(camp, ev, strict_policy(declared_hash_exemptions=("*.tmp",)))
    assert decision.allowed is True


def test_one_missing_destination_object_denies_release():
    camp, ev = fully_copied_campaign(total_files=67644, total_bytes=1, verified_files=67643,
                                     source_only=1)
    assert "missing_destination" in codes(evaluate_release(camp, ev, strict_policy()))


def test_one_mismatch_denies_release():
    camp, ev = fully_copied_campaign(total_files=67644, total_bytes=1, mismatched=1)
    assert "mismatch_present" in codes(evaluate_release(camp, ev, strict_policy()))


def test_destination_only_is_warning_unless_strict():
    camp, ev = fully_copied_campaign(total_files=10, total_bytes=10, destination_only=1)
    lenient = evaluate_release(camp, ev, strict_policy())
    assert lenient.allowed is True
    assert any("destination_only=1" in w for w in lenient.warnings)
    strict = evaluate_release(camp, ev, strict_policy(strict_destination_scope=True))
    assert strict.allowed is False
    assert "destination_only_present" in codes(strict)


def test_plan_scope_must_cover_every_inventoried_object():
    """`all_inventory` is the strong default; `hashed_set` is a weaker opt-in."""
    camp = campaign()
    ev = evidence(
        inv=inventory(100, 100, complete=True, verified=True),
        hsh=hashing(97, verified_bytes=97, complete=True,
                    exemptions=("*.tmp", "*.part", "*.crdownload")),
        cpy=copying(planned_files=97, planned_bytes=97, completed_files=97,
                    completed_bytes=97, started=True, result_complete=True,
                    ledger_complete=True),
        dst=destination_evidence(verified_files=97, verified_bytes=97,
                                 verification_started=True, verification_complete=True,
                                 observed_identity=camp.destination.identity),
        rec=reconciliation(), wkr=worker(status="stopped"),
    )
    policy = strict_policy(declared_hash_exemptions=("*.tmp", "*.part", "*.crdownload"))
    assert "plan_scope_shortfall" in codes(evaluate_release(camp, ev, policy))

    weaker = replace(policy, required_scope="hashed_set")
    assert "plan_scope_shortfall" not in codes(evaluate_release(camp, ev, weaker))
    assert evaluate_release(camp, ev, weaker).allowed is True


def test_a_narrower_scope_is_never_silent():
    """`hashed_set` may release a subset, but must say how much it left out."""
    camp = campaign()
    ev = evidence(
        inv=inventory(100, 100, complete=True, verified=True),
        hsh=hashing(40, verified_bytes=40, complete=False),
        cpy=copying(planned_files=40, planned_bytes=40, completed_files=40,
                    completed_bytes=40, started=True, result_complete=True,
                    ledger_complete=True),
        dst=destination_evidence(verified_files=40, verified_bytes=40,
                                 verification_started=True, verification_complete=True,
                                 observed_identity=camp.destination.identity),
        rec=reconciliation(), wkr=worker(status="stopped"),
    )
    weaker = strict_policy(required_scope="hashed_set")
    decision = evaluate_release(camp, ev, weaker)
    assert decision.allowed is True
    assert any("60 inventoried objects are OUTSIDE the required scope" in w
               for w in decision.warnings)

    # the default scope is unaffected and still refuses
    strict = evaluate_release(camp, ev, strict_policy())
    assert strict.allowed is False
    assert not any("OUTSIDE the required scope" in w for w in strict.warnings)


def test_required_objects_is_the_single_definition_of_scope():
    policy = strict_policy()
    assert policy.required_objects(100, 40) == 100
    assert strict_policy(required_scope="hashed_set").required_objects(100, 40) == 40
    assert strict_policy(required_scope="hashed_set").required_objects(100, 0) == 0


def test_operator_witness_requirement():
    camp, ev = fully_copied_campaign()
    assert "operator_witness_missing" in codes(
        evaluate_release(camp, ev, strict_policy(require_operator_witness=True))
    )
    witnessed = replace(ev, errors=replace(ev.errors, witness="scott@2026-10-02"))
    assert evaluate_release(camp, witnessed,
                            strict_policy(require_operator_witness=True)).allowed is True


def test_unresolved_errors_deny_release():
    camp, ev = fully_copied_campaign(unresolved=3)
    assert "unresolved_errors" in codes(evaluate_release(camp, ev, strict_policy()))


def test_unknown_policy_scope_is_rejected():
    import pytest

    from auto_ingest.custody import CustodyPolicy

    with pytest.raises(ValueError):
        CustodyPolicy.from_dict({"required_scope": "whatever"})


def test_stopped_worker_alone_does_not_open_or_close_the_gate():
    camp, ev = fully_copied_campaign()
    assert ev.worker.status == "stopped"
    assert evaluate_release(camp, ev, strict_policy()).allowed is True

    partial = replace(ev, copy=replace(ev.copy, result_complete=False, ledger_complete=False),
                      destination=replace(ev.destination, verified_files=0, verified_bytes=0,
                                          verification_complete=False),
                      reconciliation=reconciliation(source_only=100))
    assert evaluate_release(camp, partial, strict_policy()).allowed is False


def test_derived_state_never_exceeds_the_gate():
    """SAFE_TO_RELEASE implies the gate is open; VERIFIED implies it is closed."""
    camp, ev = fully_copied_campaign(total_files=10, total_bytes=100, unresolved=1)
    decision = evaluate_release(camp, ev, strict_policy())
    state = derive_state(camp, ev, strict_policy()).state
    if state is CampaignState.SAFE_TO_RELEASE:
        assert decision.allowed is True
    else:
        assert decision.allowed is False


def test_errors_blocker_count_matches_unresolved():
    camp, ev = fully_copied_campaign(unresolved=4)
    decision = evaluate_release(camp, ev, strict_policy())
    assert "unresolved_errors" in codes(decision)
    blocker = next(b for b in decision.blockers if b.code == "unresolved_errors")
    assert blocker.detail == "unresolved=4"
