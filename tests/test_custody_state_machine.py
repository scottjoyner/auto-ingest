"""The derived SD-card campaign state machine.

Each test names a physically meaningful situation and asserts the state the
machine derives from its evidence. None of them declares a state: there is no
API to do so (see ``test_declared_state_is_ignored``).
"""
from __future__ import annotations

import pytest
from custody_helpers import (
    campaign,
    copying,
    destination,
    destination_evidence,
    errors,
    evidence,
    fully_copied_campaign,
    hashing,
    inventory,
    reconciliation,
    strict_policy,
    worker,
)

from auto_ingest.custody import CampaignState, derive_state, find_contradictions
from auto_ingest.custody.machine import hash_coverage_complete


def derive(ev, camp=None, policy=None):
    return derive_state(camp or campaign(), ev, policy or strict_policy())


# ---------------------------------------------------------------------------
# untouched / early states
# ---------------------------------------------------------------------------
def test_new_untouched_card_is_discovered():
    d = derive(evidence())
    assert d.state is CampaignState.DISCOVERED
    assert d.reasons == ("no_inventory_evidence",)


def test_inventory_in_progress_without_work_is_discovered():
    ev = evidence(inv=inventory(1200, 5_000_000, complete=False, verified=False,
                                started=True))
    assert derive(ev).state is CampaignState.DISCOVERED


def test_inventory_complete_and_verified_without_hashing_is_source_verified():
    ev = evidence(inv=inventory(100, 1_000, complete=True, verified=True, started=True),
                  hsh=hashing(0, started=False))
    d = derive(ev)
    assert d.state is CampaignState.SOURCE_VERIFIED


def test_inventory_complete_hashing_incomplete_is_hashing():
    ev = evidence(inv=inventory(67644, 400_000_000, complete=True, verified=True),
                  hsh=hashing(41_000, verified_bytes=250_000_000, complete=False))
    d = derive(ev)
    assert d.state is CampaignState.HASHING
    assert "hash_evidence_incomplete" in d.reasons


def test_inventory_not_complete_but_hashing_started_is_hashing():
    ev = evidence(inv=inventory(500, 1000, complete=False, started=True),
                  hsh=hashing(120, started=True))
    assert derive(ev).state is CampaignState.HASHING


# ---------------------------------------------------------------------------
# hash complete
# ---------------------------------------------------------------------------
def test_hash_complete_no_copy_plan_is_hash_complete():
    ev = evidence(inv=inventory(100, 1_000, complete=True, verified=True),
                  hsh=hashing(100, verified_bytes=1_000, complete=True))
    d = derive(ev)
    assert d.state is CampaignState.HASH_COMPLETE
    assert "no_copy_plan" in d.reasons


def test_hash_complete_copy_planned_not_started_is_copy_pending():
    ev = evidence(inv=inventory(100, 1_000, complete=True, verified=True),
                  hsh=hashing(100, verified_bytes=1_000, complete=True),
                  cpy=copying(planned_files=100, planned_bytes=1_000, started=False))
    d = derive(ev)
    assert d.state is CampaignState.COPY_PENDING
    assert d.state is not CampaignState.SAFE_TO_RELEASE


def test_copy_started_and_worker_alive_is_copying():
    ev = evidence(
        inv=inventory(100, 1_000, complete=True, verified=True),
        hsh=hashing(100, verified_bytes=1_000, complete=True),
        cpy=copying(planned_files=100, planned_bytes=1_000, completed_files=30,
                    started=True),
        wkr=worker(status="running"),
    )
    assert derive(ev).state is CampaignState.COPYING


# ---------------------------------------------------------------------------
# the CARD-01 shape: interrupted copy -> RECONCILE_REQUIRED, never COPY_PENDING
# ---------------------------------------------------------------------------
def test_partial_copy_with_interrupted_worker_is_reconcile_required():
    ev = evidence(
        inv=inventory(67644, 412_885_402_112, complete=True, verified=True),
        hsh=hashing(67644, verified_bytes=412_885_402_112, complete=True),
        cpy=copying(planned_files=67644, planned_bytes=412_885_402_112,
                    started=True, result_complete=False, interrupted=True,
                    ledger_complete=False),
        dst=destination_evidence(verification_started=True, verification_complete=False),
        rec=reconciliation(source_only=67644),
        wkr=worker(identity="legacy-copy", status="stopped"),
    )
    d = derive(ev)
    assert d.state is CampaignState.RECONCILE_REQUIRED
    assert d.state is not CampaignState.COPY_PENDING
    assert d.state is not CampaignState.SAFE_TO_RELEASE
    assert "destination_ledger_incomplete" in d.reasons


def test_copy_started_ledger_incomplete_worker_stopped_is_reconcile_required():
    ev = evidence(
        inv=inventory(10, 100, complete=True, verified=True),
        hsh=hashing(10, verified_bytes=100, complete=True),
        cpy=copying(planned_files=10, planned_bytes=100, started=True,
                    result_complete=True, ledger_complete=False),
        wkr=worker(status="stopped"),
    )
    d = derive(ev)
    assert d.state is CampaignState.RECONCILE_REQUIRED
    assert "destination_ledger_incomplete" in d.reasons


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------
def test_copy_complete_verification_not_started_is_copy_complete():
    ev = evidence(
        inv=inventory(10, 100, complete=True, verified=True),
        hsh=hashing(10, verified_bytes=100, complete=True),
        cpy=copying(planned_files=10, planned_bytes=100, completed_files=10,
                    completed_bytes=100, started=True, result_complete=True,
                    ledger_complete=True),
        wkr=worker(status="stopped"),
    )
    d = derive(ev)
    assert d.state is CampaignState.COPY_COMPLETE


def test_verification_started_incomplete_is_verifying():
    ev = evidence(
        inv=inventory(10, 100, complete=True, verified=True),
        hsh=hashing(10, verified_bytes=100, complete=True),
        cpy=copying(planned_files=10, planned_bytes=100, completed_files=10,
                    completed_bytes=100, started=True, result_complete=True,
                    ledger_complete=True),
        dst=destination_evidence(verified_files=4, verification_started=True,
                                 verification_complete=False),
        wkr=worker(status="running"),
    )
    assert derive(ev).state is CampaignState.VERIFYING


def test_complete_clean_verified_copy_is_safe_to_release():
    camp, ev = fully_copied_campaign()
    d = derive_state(camp, ev, strict_policy())
    assert d.state is CampaignState.SAFE_TO_RELEASE


def test_custody_proven_but_gate_closed_is_verified_not_safe():
    camp, ev = fully_copied_campaign()
    policy = strict_policy(require_operator_witness=True)
    d = derive_state(camp, ev, policy)
    assert d.state is CampaignState.VERIFIED
    assert d.state is not CampaignState.SAFE_TO_RELEASE
    assert any("operator_witness_missing" in b for b in d.blockers)


def test_operator_witness_recorded_opens_the_gate():
    camp, ev = fully_copied_campaign()
    ev = type(ev)(**{**ev.__dict__, "errors": errors(witness="scott@2026-04-12")})
    d = derive_state(camp, ev, strict_policy(require_operator_witness=True))
    assert d.state is CampaignState.SAFE_TO_RELEASE


# ---------------------------------------------------------------------------
# release denial cases
# ---------------------------------------------------------------------------
def test_one_missing_file_denies_release_and_requires_reconciliation():
    camp, ev = fully_copied_campaign(total_files=100, total_bytes=1000,
                                     verified_files=99, source_only=1)
    d = derive_state(camp, ev, strict_policy())
    assert d.state is CampaignState.RECONCILE_REQUIRED
    assert "missing_destination=1" in d.reasons


def test_one_mismatch_denies_release_and_requires_reconciliation():
    camp, ev = fully_copied_campaign(total_files=100, total_bytes=1000,
                                     mismatched=1)
    d = derive_state(camp, ev, strict_policy())
    assert d.state is CampaignState.RECONCILE_REQUIRED
    assert "mismatched=1" in d.reasons


def test_destination_only_object_denies_release_under_strict_scope():
    camp, ev = fully_copied_campaign(total_files=10, total_bytes=100, destination_only=1)
    lenient = derive_state(camp, ev, strict_policy(strict_destination_scope=False))
    assert lenient.state is CampaignState.SAFE_TO_RELEASE
    strict = derive_state(camp, ev, strict_policy(strict_destination_scope=True))
    assert strict.state is CampaignState.RECONCILE_REQUIRED


def test_unresolved_error_denies_release():
    camp, ev = fully_copied_campaign(total_files=10, total_bytes=100, unresolved=1)
    d = derive_state(camp, ev, strict_policy())
    assert d.state is CampaignState.VERIFIED
    assert any("unresolved_errors" in b for b in d.blockers)


def test_writable_source_denies_release():
    camp, ev = fully_copied_campaign()
    camp = type(camp)(**{**camp.__dict__,
                         "source": type(camp.source)(mount_point=camp.source.mount_point,
                                                    read_only=False,
                                                    card=camp.source.card)})
    d = derive_state(camp, ev, strict_policy())
    assert d.state is CampaignState.VERIFIED
    assert any("source_not_read_only" in b for b in d.blockers)


def test_unresolved_destination_denies_release():
    camp, ev = fully_copied_campaign()
    camp = type(camp)(**{**camp.__dict__, "destination": destination(host_path=None)})
    d = derive_state(camp, ev, strict_policy())
    assert d.state is CampaignState.VERIFIED
    assert any("destination_unresolved" in b for b in d.blockers)


def test_unmounted_destination_denies_release():
    camp, ev = fully_copied_campaign()
    camp = type(camp)(**{**camp.__dict__,
                         "destination": destination(mounted=False)})
    d = derive_state(camp, ev, strict_policy())
    assert d.state is CampaignState.VERIFIED
    assert any("destination_not_mounted" in b for b in d.blockers)


# ---------------------------------------------------------------------------
# a stopped worker is never a verdict
# ---------------------------------------------------------------------------
def test_stopped_worker_alone_does_not_imply_failure():
    ev = evidence(
        inv=inventory(67644, 1, complete=True, verified=True),
        hsh=hashing(67644, verified_bytes=1, complete=True),
        cpy=copying(planned_files=67644, planned_bytes=1, started=True,
                    result_complete=False, interrupted=True),
        wkr=worker(status="stopped"),
    )
    d = derive(ev)
    assert d.state is CampaignState.RECONCILE_REQUIRED
    assert d.state is not CampaignState.BLOCKED


def test_stopped_worker_alone_does_not_imply_success():
    """A stopped worker on a card that was never copied is not SAFE_TO_RELEASE."""
    ev = evidence(
        inv=inventory(10, 100, complete=True, verified=True),
        hsh=hashing(10, verified_bytes=100, complete=True),
        cpy=copying(planned_files=10, planned_bytes=100, started=False),
        wkr=worker(status="stopped"),
    )
    assert derive(ev).state is CampaignState.COPY_PENDING


def test_stopped_worker_after_clean_verification_still_releases():
    """A stopped worker is not a reason to withhold a proven release."""
    camp, ev = fully_copied_campaign()
    assert ev.worker.status == "stopped"
    assert derive_state(camp, ev, strict_policy()).state is CampaignState.SAFE_TO_RELEASE


# ---------------------------------------------------------------------------
# contradictions -> BLOCKED
# ---------------------------------------------------------------------------
def test_copy_completed_exceeding_plan_is_blocked():
    ev = evidence(
        inv=inventory(10, 100, complete=True, verified=True),
        hsh=hashing(10, verified_bytes=100, complete=True),
        cpy=copying(planned_files=10, planned_bytes=100, completed_files=11,
                    completed_bytes=110, started=True),
    )
    d = derive(ev)
    assert d.state is CampaignState.BLOCKED
    assert any("copy_completed_exceeds_planned" in c for c in d.contradictions)


def test_destination_exceeding_inventory_is_blocked():
    ev = evidence(
        inv=inventory(10, 100, complete=True, verified=True),
        hsh=hashing(10, verified_bytes=100, complete=True),
        cpy=copying(planned_files=10, planned_bytes=100, completed_files=10,
                    completed_bytes=100, started=True, result_complete=True,
                    ledger_complete=True),
        dst=destination_evidence(verified_files=11, verification_started=True,
                                 verification_complete=True),
    )
    assert derive(ev).state is CampaignState.BLOCKED


def test_destination_identity_conflict_is_blocked():
    camp, ev = fully_copied_campaign()
    from auto_ingest.custody import StorageIdentity

    ev = type(ev)(**{**ev.__dict__,
                    "destination": destination_evidence(
                        verified_files=10, verified_bytes=100,
                        verification_started=True, verification_complete=True,
                        observed_identity=StorageIdentity(filesystem_uuid="OTHER-0001"),
                    )})
    d = derive_state(camp, ev, strict_policy())
    assert d.state is CampaignState.BLOCKED
    assert any("destination_identity_conflict" in c for c in d.contradictions)


def test_fatal_error_is_blocked():
    ev = evidence(inv=inventory(10, 100, complete=True, verified=True),
                  hsh=hashing(10, verified_bytes=100, complete=True),
                  err=errors(fatal=1))
    d = derive(ev)
    assert d.state is CampaignState.BLOCKED
    assert "fatal_errors:1" in d.contradictions


def test_find_contradictions_is_empty_for_a_coherent_campaign():
    camp, ev = fully_copied_campaign()
    assert find_contradictions(camp, ev) == ()


# ---------------------------------------------------------------------------
# hash coverage + policy
# ---------------------------------------------------------------------------
def test_undeclared_exemption_does_not_complete_hash_coverage():
    ev = evidence(inv=inventory(10, 100, complete=True, verified=True),
                  hsh=hashing(9, verified_bytes=90, complete=False,
                              exemptions=("*.tmp",)))
    policy = strict_policy(declared_hash_exemptions=())
    assert hash_coverage_complete(ev, policy) is False
    assert derive(ev, policy=policy).state is CampaignState.HASHING


def test_declared_exemption_completes_hash_coverage():
    ev = evidence(inv=inventory(10, 100, complete=True, verified=True),
                  hsh=hashing(9, verified_bytes=90, complete=False,
                              exemptions=("*.tmp",)),
                  cpy=copying(planned_files=10, planned_bytes=100, started=False))
    policy = strict_policy(declared_hash_exemptions=("*.tmp",))
    assert hash_coverage_complete(ev, policy) is True
    assert derive(ev, policy=policy).state is CampaignState.COPY_PENDING


# ---------------------------------------------------------------------------
# a declared state is ignored, never honoured
# ---------------------------------------------------------------------------
def test_declared_state_is_ignored():
    raw = {
        "campaign_id": "sdcard-evil",
        "state": "SAFE_TO_RELEASE",
        "safe_to_release": True,
        "status": "SAFE_TO_RELEASE",
        "inventory": {"discovered_files": 10, "discovered_bytes": 100, "complete": True},
        "hash": {"verified_files": 10, "verified_bytes": 100, "complete": True},
        "copy": {"planned": {"files": 10, "bytes": 100}, "started": True},
        "destination": {"verified_files": 0},
        "worker": {"status": "stopped"},
    }
    from auto_ingest.custody import CampaignEvidence

    ev = CampaignEvidence.from_dict(raw, strict_policy())
    assert ev.ignored_declared_fields == ("state", "status", "safe_to_release")
    assert ev.destination.verified_files == 0
    d = derive_state(campaign(card_id="CARD-X"), ev, strict_policy())
    assert d.state is CampaignState.RECONCILE_REQUIRED
    assert d.state is not CampaignState.SAFE_TO_RELEASE


def test_derive_state_signature_rejects_a_state_argument():
    import inspect

    params = inspect.signature(derive_state).parameters
    assert "state" not in params
    assert list(params) == ["campaign", "evidence", "policy"]


@pytest.mark.parametrize("state", list(CampaignState))
def test_every_state_is_representable_and_serialisable(state):
    assert state.value == state.name
    assert state.value.isupper()
