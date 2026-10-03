"""The read-only resume planner.

Two guarantees are tested here: the planner never grants source mutation or
deletion, and after an interrupted copy it prefers *verifying* what already
landed at the destination over copying it again.
"""
from __future__ import annotations

import inspect
import json

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
    SOURCE_DELETION_ALLOWED,
    SOURCE_MUTATION_ALLOWED,
    CampaignState,
    derive_state,
    plan_resume,
)
from auto_ingest.custody import planner as planner_mod


def plan_for(ev, camp=None, policy=None):
    camp = camp or campaign()
    policy = policy or strict_policy()
    return plan_resume(camp, ev, policy)


# ---------------------------------------------------------------------------
# capability ceiling
# ---------------------------------------------------------------------------
def test_planner_never_allows_source_mutation_or_deletion():
    assert SOURCE_MUTATION_ALLOWED is False
    assert SOURCE_DELETION_ALLOWED is False
    ev = fully_copied_campaign()[1]
    plan = plan_resume(campaign(), ev, strict_policy())
    assert plan.source_mutation_allowed is False
    assert plan.source_deletion_allowed is False
    assert plan.requires_operator_authorization is True


def test_no_action_ever_mutates_the_source():
    evs = [
        evidence(),
        evidence(inv=inventory(10, 100, complete=True, verified=True),
                 hsh=hashing(5, started=True)),
        fully_copied_campaign()[1],
    ]
    for ev in evs:
        for action in plan_for(ev).actions:
            assert action.mutates_source is False


def test_planner_module_has_no_execution_primitives():
    """The planner is a description function: no copy/remove/subprocess at all."""
    import ast

    tree = ast.parse(inspect.getsource(planner_mod))
    banned_modules = {"subprocess", "shutil", "socket", "requests", "neo4j", "sqlite3",
                      "urllib", "http"}
    banned_calls = {"remove", "unlink", "rmtree", "copyfile", "copy2", "copytree",
                    "move", "system", "popen", "check_call", "check_output",
                    "mkdir", "makedirs", "rmdir", "removedirs", "rename",
                    "write_text", "write_bytes", "open", "urlopen", "connect",
                    "touch", "truncate", "chmod", "chown"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in banned_modules
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in banned_modules
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            assert name not in banned_calls, f"planner must not call {name}()"


def test_plan_json_shape_matches_the_contract():
    camp, ev = fully_copied_campaign()
    payload = plan_resume(camp, ev, strict_policy()).to_dict()
    for key in ("safe_to_resume", "current_state", "next_phase", "source_mutation_allowed",
                "source_deletion_allowed", "actions", "plan_fingerprint"):
        assert key in payload
    assert payload["source_mutation_allowed"] is False
    assert payload["source_deletion_allowed"] is False


# ---------------------------------------------------------------------------
# per-state next action
# ---------------------------------------------------------------------------
def test_discovered_plan_inventories_the_source():
    plan = plan_for(evidence())
    assert plan.current_state is CampaignState.DISCOVERED
    assert plan.next_safe_action == "inventory_source"
    assert [a.operation for a in plan.actions] == ["inventory_source"]


def test_hashing_plan_estimates_the_remainder():
    ev = evidence(inv=inventory(67644, 676_440_000, complete=True, verified=True),
                  hsh=hashing(41_000, verified_bytes=410_000_000, complete=False))
    plan = plan_for(ev)
    assert plan.next_phase == "hashing"
    assert plan.actions[0].operation == "hash_source"
    assert plan.actions[0].estimated_files == 67644 - 41_000


def test_copy_pending_plan_copies_then_verifies_excluding_verified():
    ev = evidence(inv=inventory(100, 1000, complete=True, verified=True),
                  hsh=hashing(100, verified_bytes=1000, complete=True),
                  cpy=copying(planned_files=100, planned_bytes=1000, started=False),
                  dst=destination_evidence(verified_files=30))
    plan = plan_for(ev)
    assert plan.current_state is CampaignState.COPY_PENDING
    copy_action, verify_action = plan.actions
    assert copy_action.operation == "copy_objects"
    assert copy_action.estimated_files == 70
    assert copy_action.excludes_verified is True
    assert copy_action.mutates_destination is True
    assert verify_action.gated_by == (copy_action.action_id,)


def test_copying_plan_observes_and_does_not_restart():
    ev = evidence(inv=inventory(100, 1000, complete=True, verified=True),
                  hsh=hashing(100, verified_bytes=1000, complete=True),
                  cpy=copying(planned_files=100, planned_bytes=1000, completed_files=25,
                              started=True),
                  wkr=worker(status="running"))
    plan = plan_for(ev)
    assert plan.current_state is CampaignState.COPYING
    assert [a.operation for a in plan.actions] == ["await_copy"]
    assert plan.mutating_actions == ()


def test_verify_plan_for_copy_complete():
    ev = evidence(inv=inventory(100, 1000, complete=True, verified=True),
                  hsh=hashing(100, verified_bytes=1000, complete=True),
                  cpy=copying(planned_files=100, planned_bytes=1000, completed_files=100,
                              completed_bytes=1000, started=True, result_complete=True,
                              ledger_complete=True),
                  wkr=worker(status="stopped"))
    plan = plan_for(ev)
    assert plan.current_state is CampaignState.COPY_COMPLETE
    assert plan.actions[0].operation == "verify_destination"
    assert plan.actions[0].estimated_files == 100


# ---------------------------------------------------------------------------
# reconciliation preference (spec case)
# ---------------------------------------------------------------------------
def test_reconciliation_verifies_destination_before_copying_again():
    ev = evidence(
        inv=inventory(67644, 676_440_000, complete=True, verified=True),
        hsh=hashing(67644, verified_bytes=676_440_000, complete=True),
        cpy=copying(planned_files=67644, planned_bytes=676_440_000, started=True,
                    result_complete=False, interrupted=True, ledger_complete=False),
        dst=destination_evidence(verified_files=0, verification_started=True,
                                 verification_complete=False,
                                 unverified_present_files=50_000),
        rec=reconciliation(source_only=67644),
        wkr=worker(identity="legacy", status="stopped"),
    )
    plan = plan_for(ev)
    assert plan.current_state is CampaignState.RECONCILE_REQUIRED
    ops = [a.operation for a in plan.actions]
    assert ops == ["verify_existing_destination", "copy_objects", "verify_destination"]

    verify_first, copy_after, verify_after = plan.actions
    assert verify_first.estimated_files == 50_000
    assert verify_first.mutates_destination is False
    assert copy_after.gated_by == (verify_first.action_id,)
    assert copy_after.excludes_verified is True
    assert verify_after.gated_by == (copy_after.action_id,)


def test_resume_plan_skips_content_already_verified_at_destination():
    """A verified subset must not be scheduled for copying again."""
    verified = 60_000
    ev = evidence(
        inv=inventory(67644, 676_440_000, complete=True, verified=True),
        hsh=hashing(67644, verified_bytes=676_440_000, complete=True),
        cpy=copying(planned_files=67644, planned_bytes=676_440_000, started=True,
                    result_complete=False, interrupted=True, ledger_complete=False),
        dst=destination_evidence(verified_files=verified, verified_bytes=600_000_000,
                                 verification_started=True, verification_complete=False,
                                 unverified_present_files=7_000),
        rec=reconciliation(source_only=67644 - verified),
        wkr=worker(status="stopped"),
    )
    plan = plan_for(ev)
    copy_action = next(a for a in plan.actions if a.operation == "copy_objects")
    assert copy_action.estimated_files == 67644 - verified
    assert copy_action.excludes_verified is True
    assert verified not in (copy_action.estimated_files,)
    # nothing anywhere claims to copy the verified 60k again
    assert all(a.estimated_files <= 67644 - verified
               for a in plan.actions if a.operation in {"copy_objects",
                                                        "verify_destination"})


def test_reconciled_campaign_with_verified_subset_keeps_verified_content():
    """A partially reconciled campaign still excludes the verified objects."""
    camp, ev = fully_copied_campaign(total_files=1000, total_bytes=10_000_000,
                                     verified_files=900, source_only=100)
    plan = plan_for(ev, camp=camp)
    assert plan.current_state is CampaignState.RECONCILE_REQUIRED
    copy_action = next(a for a in plan.actions if a.operation == "copy_objects")
    assert copy_action.estimated_files == 100
    assert copy_action.excludes_verified is True


# ---------------------------------------------------------------------------
# safe_to_resume
# ---------------------------------------------------------------------------
def test_reconcile_is_safe_to_resume_when_the_destination_resolves():
    ev = evidence(
        inv=inventory(100, 1000, complete=True, verified=True),
        hsh=hashing(100, verified_bytes=1000, complete=True),
        cpy=copying(planned_files=100, planned_bytes=1000, started=True,
                    result_complete=False, interrupted=True),
        dst=destination_evidence(verification_started=True),
        rec=reconciliation(source_only=100),
        wkr=worker(status="stopped"),
    )
    plan = plan_for(ev)
    assert plan.current_state is CampaignState.RECONCILE_REQUIRED
    assert plan.safe_to_resume is True
    assert plan.next_phase == "destination_reconciliation"


def test_unresolved_destination_is_not_safe_to_resume():
    ev = evidence(inv=inventory(100, 1000, complete=True, verified=True),
                  hsh=hashing(100, verified_bytes=1000, complete=True),
                  cpy=copying(planned_files=100, planned_bytes=1000, started=False))
    plan = plan_resume(campaign(dest=destination(host_path=None)), ev, strict_policy())
    assert plan.safe_to_resume is False
    assert any(b.code == "destination_unresolved" for b in plan.blockers)


def test_blocked_campaign_is_not_safe_to_resume():
    ev = evidence(inv=inventory(10, 100, complete=True, verified=True),
                  hsh=hashing(10, verified_bytes=100, complete=True),
                  cpy=copying(planned_files=10, planned_bytes=10, completed_files=11,
                              started=True))
    plan = plan_for(ev)
    assert plan.current_state is CampaignState.BLOCKED
    assert plan.safe_to_resume is False
    assert [a.operation for a in plan.actions] == ["resolve_contradiction"]


def test_safe_to_release_campaign_has_no_actions():
    camp, ev = fully_copied_campaign()
    plan = plan_for(ev, camp=camp)
    assert plan.current_state is CampaignState.SAFE_TO_RELEASE
    assert plan.actions == ()
    assert plan.safe_to_resume is False
    assert plan.next_safe_action == "none"


# ---------------------------------------------------------------------------
# idempotency
# ---------------------------------------------------------------------------
def test_repeated_planning_is_identical_and_generates_no_duplicate_actions():
    camp, ev = fully_copied_campaign(total_files=100, total_bytes=1000)
    first = plan_resume(camp, ev, strict_policy()).to_dict()
    for _ in range(4):
        again = plan_resume(camp, ev, strict_policy()).to_dict()
        assert again == first
    ids = [a["action_id"] for a in first["actions"]]
    assert len(ids) == len(set(ids))


def test_action_ids_are_content_derived_not_random():
    ev_a = evidence(inv=inventory(100, 1000, complete=True, verified=True),
                    hsh=hashing(100, verified_bytes=1000, complete=True),
                    cpy=copying(planned_files=100, planned_bytes=1000, started=False))
    ev_b = evidence(inv=inventory(100, 1000, complete=True, verified=True),
                    hsh=hashing(100, verified_bytes=1000, complete=True),
                    cpy=copying(planned_files=100, planned_bytes=1000, started=False))
    assert plan_for(ev_a).plan_fingerprint == plan_for(ev_b).plan_fingerprint
    ev_c = replace_plan_input(ev_a, planned=200)
    assert plan_for(ev_a).plan_fingerprint != plan_for(ev_c).plan_fingerprint


def replace_plan_input(ev, planned):
    from dataclasses import replace

    return replace(ev, copy=replace(ev.copy, planned=replace(ev.copy.planned, files=planned)))


def test_plan_serialisation_is_stable_json():
    camp, ev = fully_copied_campaign()
    plan = plan_resume(camp, ev, strict_policy())
    blob = json.dumps(plan.to_dict(), sort_keys=True, indent=2, default=str)
    for _ in range(3):
        assert json.dumps(plan.to_dict(), sort_keys=True, indent=2, default=str) == blob


def test_derivation_is_passed_through_not_recomputed_sideways():
    """Passing a precomputed derivation must not change the plan."""
    camp, ev = fully_copied_campaign()
    policy = strict_policy()
    d = derive_state(camp, ev, policy)
    assert plan_resume(camp, ev, policy, derivation=d).to_dict() == \
        plan_resume(camp, ev, policy).to_dict()
