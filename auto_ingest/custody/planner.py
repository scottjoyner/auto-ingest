"""auto_ingest.custody.planner - the read-only resume planner.

The planner answers "what would the next safe operation be?" and nothing else.
It has no filesystem, network, database or subprocess access at all: given a
campaign, evidence and policy it returns a description. It never copies, never
deletes, never truncates, never mutates the source, and never schedules
anything. Execution is a separate, operator-authorized step performed by an
executor outside this package.

Two properties matter operationally:

* **Preference for verification over re-copy.** After an interrupted copy the
  destination may already hold most of the payload. The plan therefore emits
  ``verify_existing_destination`` *first*, and every later copy action carries
  ``excludes_verified=True`` with an estimate that already subtracts the
  verified objects. Re-copying a verified object would waste hours of I/O and
  rewrite bytes that are already proven.
* **Idempotent task identity.** Each action's ``action_id`` is a digest of the
  action's own content, and ``plan_fingerprint`` digests the whole action set.
  Re-planning unchanged evidence yields byte-identical output and identical ids,
  so nothing downstream can generate duplicate work.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .campaign import Campaign
from .evidence import CampaignEvidence
from .machine import Derivation, derive_state
from .policy import CustodyPolicy
from .release import Blocker, evaluate_release
from .states import NEXT_SAFE_ACTION, STATE_PHASE, CampaignState

#: The planner's only capability ceiling. Nothing here may ever raise these.
SOURCE_MUTATION_ALLOWED = False
SOURCE_DELETION_ALLOWED = False

_OPERATION_TARGET = {
    "inventory_source": "source",
    "hash_source": "source",
    "plan_copy": "plan",
    "copy_objects": "destination",
    "verify_existing_destination": "destination",
    "verify_destination": "destination",
    "await_copy": "none",
    "review_for_release": "none",
    "resolve_contradiction": "none",
}

_MUTATES_DESTINATION = frozenset({"copy_objects"})


@dataclass(frozen=True)
class PlannedAction:
    """A described operation. A description, never an invocation."""

    action_id: str
    phase: str
    operation: str
    target: str
    estimated_files: int = 0
    estimated_bytes: int = 0
    excludes_verified: bool = False
    mutates_source: bool = False
    mutates_destination: bool = False
    requires_authorization: bool = True
    gated_by: Tuple[str, ...] = ()
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "estimated_bytes": self.estimated_bytes,
            "estimated_files": self.estimated_files,
            "excludes_verified": self.excludes_verified,
            "gated_by": list(self.gated_by),
            "mutates_destination": self.mutates_destination,
            "mutates_source": self.mutates_source,
            "operation": self.operation,
            "phase": self.phase,
            "rationale": self.rationale,
            "requires_authorization": self.requires_authorization,
            "target": self.target,
        }


@dataclass(frozen=True)
class ResumePlan:
    """The read-only description of the next safe operation."""

    campaign_id: str
    safe_to_resume: bool
    current_state: CampaignState
    next_phase: str
    next_safe_action: str
    source_mutation_allowed: bool = SOURCE_MUTATION_ALLOWED
    source_deletion_allowed: bool = SOURCE_DELETION_ALLOWED
    requires_operator_authorization: bool = True
    actions: Tuple[PlannedAction, ...] = ()
    blockers: Tuple[Blocker, ...] = ()
    warnings: Tuple[str, ...] = ()
    reasons: Tuple[str, ...] = ()
    plan_fingerprint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actions": [a.to_dict() for a in self.actions],
            "blockers": [b.to_dict() for b in self.blockers],
            "campaign_id": self.campaign_id,
            "current_state": self.current_state.value,
            "next_phase": self.next_phase,
            "next_safe_action": self.next_safe_action,
            "plan_fingerprint": self.plan_fingerprint,
            "reasons": list(self.reasons),
            "requires_operator_authorization": self.requires_operator_authorization,
            "safe_to_resume": self.safe_to_resume,
            "source_deletion_allowed": self.source_deletion_allowed,
            "source_mutation_allowed": self.source_mutation_allowed,
            "warnings": list(self.warnings),
        }

    @property
    def mutating_actions(self) -> Tuple[PlannedAction, ...]:
        return tuple(a for a in self.actions if a.mutates_destination or a.mutates_source)


def _action_id(phase: str, operation: str, target: str, files: int, nbytes: int) -> str:
    blob = f"{phase}|{operation}|{target}|{files}|{nbytes}"
    return "act-" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def _make(
    phase: str,
    operation: str,
    *,
    files: int = 0,
    nbytes: int = 0,
    excludes_verified: bool = False,
    gated_by: Tuple[str, ...] = (),
    rationale: str = "",
) -> PlannedAction:
    return PlannedAction(
        action_id=_action_id(phase, operation, _OPERATION_TARGET[operation], files, nbytes),
        phase=phase,
        operation=operation,
        target=_OPERATION_TARGET[operation],
        estimated_files=max(int(files), 0),
        estimated_bytes=max(int(nbytes), 0),
        excludes_verified=excludes_verified,
        mutates_source=False,
        mutates_destination=operation in _MUTATES_DESTINATION,
        gated_by=gated_by,
        rationale=rationale,
    )


def _bytes_per_file(evidence: CampaignEvidence) -> float:
    files = evidence.inventory.discovered_files
    if files <= 0:
        return 0.0
    return evidence.inventory.discovered_bytes / files


def plan_resume(
    campaign: Campaign,
    evidence: CampaignEvidence,
    policy: CustodyPolicy | None = None,
    derivation: Optional[Derivation] = None,
) -> ResumePlan:
    """Describe the next safe operation for a campaign. Read-only and pure."""
    policy = policy or CustodyPolicy()
    derivation = derivation or derive_state(campaign, evidence, policy)
    state = derivation.state

    inv = evidence.inventory
    hsh = evidence.hashing
    cpy = evidence.copy
    dst = evidence.destination

    honoured_exemptions = sum(
        1 for p in hsh.exemptions if policy.honour_exemption(p)
    )
    hash_remaining = max(inv.discovered_files - hsh.verified_files - honoured_exemptions, 0)
    verified = dst.verified_files
    remaining_files = max(inv.discovered_files - verified, 0)
    unverified_present = dst.unverified_present_files
    copy_remaining_files = max(cpy.planned.files - verified, 0)
    copy_remaining_bytes = int(round(_bytes_per_file(evidence) * copy_remaining_files))

    actions: List[PlannedAction] = []
    phase = STATE_PHASE[state]

    if state is CampaignState.DISCOVERED:
        actions.append(_make(
            phase, "inventory_source",
            rationale="no source inventory evidence has been recorded yet",
        ))
    elif state in (CampaignState.SOURCE_VERIFIED, CampaignState.HASHING):
        actions.append(_make(
            phase, "hash_source",
            files=hash_remaining,
            nbytes=int(round(_bytes_per_file(evidence) * hash_remaining)),
            rationale="hashing coverage is short of the source inventory",
        ))
    elif state is CampaignState.HASH_COMPLETE:
        actions.append(_make(
            phase, "plan_copy",
            files=inv.discovered_files,
            nbytes=inv.discovered_bytes,
            rationale="hash evidence is complete; a copy plan must be recorded",
        ))
    elif state is CampaignState.COPY_PENDING:
        actions.append(_make(
            phase, "copy_objects",
            files=copy_remaining_files,
            nbytes=copy_remaining_bytes,
            excludes_verified=True,
            rationale="copy is planned but has not started; verified objects are excluded",
        ))
        actions.append(_make(
            phase, "verify_destination",
            files=copy_remaining_files,
            nbytes=copy_remaining_bytes,
            excludes_verified=True,
            gated_by=(actions[0].action_id,),
            rationale="verify the destination copy once the copy completes",
        ))
    elif state is CampaignState.COPYING:
        actions.append(_make(
            phase, "await_copy",
            files=max(cpy.planned.files - cpy.completed.files, 0),
            rationale="a worker is still copying; observe, do not restart",
        ))
    elif state is CampaignState.COPY_COMPLETE:
        actions.append(_make(
            phase, "verify_destination",
            files=max(cpy.completed.files - verified, 0),
            rationale="copy result is attested; destination has no verification yet",
        ))
    elif state is CampaignState.VERIFYING:
        actions.append(_make(
            phase, "verify_destination",
            files=max(cpy.completed.files - verified, 0),
            rationale="destination verification is underway and incomplete",
        ))
    elif state is CampaignState.RECONCILE_REQUIRED:
        # Objects already sitting at the destination are NOT copy work: they are
        # verification work. If the copy action counted them too, the plan would
        # tell the operator to verify N objects and then re-copy those same N -
        # the exact waste reconciliation exists to avoid. So the copy estimate is
        # what remains *after* setting aside what is present-but-unattested.
        present_unverified = min(unverified_present, remaining_files)
        genuinely_absent = max(remaining_files - present_unverified, 0)
        verify_first = _make(
            phase, "verify_existing_destination",
            files=present_unverified,
            excludes_verified=True,
            rationale=(
                "destination objects are present but unattested; verify them before "
                "copying anything again"
            ),
        )
        actions.append(verify_first)
        actions.append(_make(
            phase, "copy_objects",
            files=genuinely_absent,
            nbytes=int(round(_bytes_per_file(evidence) * genuinely_absent)),
            excludes_verified=True,
            gated_by=(verify_first.action_id,),
            rationale=(
                "objects with neither a verified nor an unverified destination "
                f"presence ({genuinely_absent}); "
                f"{verified} already verified, {present_unverified} present but "
                "unattested and therefore excluded from the copy"
            ),
        ))
        actions.append(_make(
            phase, "verify_destination",
            files=genuinely_absent,
            nbytes=int(round(_bytes_per_file(evidence) * genuinely_absent)),
            excludes_verified=True,
            gated_by=(actions[1].action_id,),
            rationale="verify what was copied during reconciliation",
        ))
    elif state is CampaignState.VERIFIED:
        actions.append(_make(
            phase, "review_for_release",
            files=verified,
            rationale="custody is proven; the release gate is closed and needs a human",
        ))
    elif state is CampaignState.BLOCKED:
        actions.append(_make(
            phase, "resolve_contradiction",
            files=inv.discovered_files,
            rationale="evidence contradicts itself; no automatic operation is safe",
        ))
    # SAFE_TO_RELEASE: nothing left to do.

    decision = evaluate_release(campaign, evidence, policy)
    blockers: Tuple[Blocker, ...] = ()
    warnings = tuple(decision.warnings)
    safe = True
    reasons = list(derivation.reasons)

    if state is CampaignState.BLOCKED:
        safe = False
        blockers = tuple(
            Blocker("evidence_contradiction", item, "reconcile the evidence before any operation")
            for item in derivation.contradictions
        )
        reasons.append("state_is_blocked")
    elif state in (CampaignState.SAFE_TO_RELEASE, CampaignState.VERIFIED):
        safe = False
        reasons.append("no_ingest_work_outstanding")
        blockers = decision.blockers
    elif not campaign.destination.resolved:
        safe = False
        blockers = decision.blockers
        reasons.append("destination_unresolved")

    fingerprint = hashlib.sha256(
        json.dumps(
            [a.to_dict() for a in actions], sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()[:16]

    return ResumePlan(
        campaign_id=campaign.campaign_id,
        safe_to_resume=safe,
        current_state=state,
        next_phase=phase,
        next_safe_action=NEXT_SAFE_ACTION[state],
        actions=tuple(actions),
        blockers=blockers,
        warnings=warnings,
        reasons=tuple(reasons),
        plan_fingerprint=fingerprint,
    )


__all__ = [
    "PlannedAction",
    "ResumePlan",
    "SOURCE_DELETION_ALLOWED",
    "SOURCE_MUTATION_ALLOWED",
    "plan_resume",
]
