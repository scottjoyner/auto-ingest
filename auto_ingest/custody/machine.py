"""auto_ingest.custody.machine - derive campaign state from evidence.

One pure function: ``derive_state(campaign, evidence, policy) -> Derivation``.

There is no setter, no override, no "operator says so". The state is a total
function of the evidence, which is what makes ``status`` idempotent: running it
twice against unchanged evidence cannot produce a different answer, and it
cannot mutate anything because it reads nothing but its arguments.

Precedence (first matching rule wins, evaluated in this fixed order):

    0. arithmetic contradictions, fatal errors, unproven destination identity
       -> BLOCKED
    1. nothing observed at all                          -> DISCOVERED
    2. inventory incomplete but work underway            -> HASHING
    3. inventory incomplete, nothing underway            -> DISCOVERED
    4. inventory complete, no hashing underway           -> SOURCE_VERIFIED
    5. hashing coverage short of the inventory          -> HASHING
    6. hashing complete, no copy plan                   -> HASH_COMPLETE
    7. hashing complete, copy planned, not started      -> COPY_PENDING
    8. copy started, ambiguous ledger, worker idle      -> RECONCILE_REQUIRED
    9. copy started, worker still alive                 -> COPYING
  10. copy complete, verification not started          -> COPY_COMPLETE
  11. verification started, incomplete                 -> VERIFYING
  12. set difference non-empty                          -> RECONCILE_REQUIRED
  13. custody proven, release gate closed               -> VERIFIED
  14. custody proven, release gate open                 -> SAFE_TO_RELEASE
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .campaign import Campaign
from .destination import match_identity
from .evidence import CampaignEvidence
from .policy import CustodyPolicy
from .release import evaluate_release
from .states import CampaignState

#: Worker statuses that mean "no worker is driving this campaign right now".
_IDLE_STATUSES = frozenset({"stopped", "absent", "exited", "dead", "inactive", "not_started"})


@dataclass(frozen=True)
class Derivation:
    """The derived verdict plus the reasons that produced it."""

    state: CampaignState
    reasons: Tuple[str, ...] = ()
    blockers: Tuple[str, ...] = ()
    contradictions: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "blockers": list(self.blockers),
            "contradictions": list(self.contradictions),
            "reasons": list(self.reasons),
            "state": self.state.value,
        }


def find_contradictions(campaign: Campaign, evidence: CampaignEvidence) -> Tuple[str, ...]:
    """Evidence that cannot all be true. Any entry forces ``BLOCKED``."""
    out = []
    inv = evidence.inventory
    hsh = evidence.hashing
    cpy = evidence.copy
    dst = evidence.destination
    rec = evidence.reconciliation

    if inv.discovered_files < 0 or inv.discovered_bytes < 0:
        out.append("negative_source_inventory_counts")
    if hsh.verified_files < 0 or hsh.failed < 0:
        out.append("negative_hash_counts")
    if cpy.planned.files < 0 or cpy.completed.files < 0:
        out.append("negative_copy_counts")
    if dst.verified_files < 0:
        out.append("negative_destination_counts")
    if inv.complete and hsh.verified_files > inv.discovered_files:
        out.append(
            f"hashed_exceeds_inventory:{hsh.verified_files}>{inv.discovered_files}"
        )
    if cpy.completed.files > cpy.planned.files:
        out.append(f"copy_completed_exceeds_planned:{cpy.completed.files}>{cpy.planned.files}")
    if cpy.completed.files > max(inv.discovered_files, 0):
        out.append(
            f"copy_completed_exceeds_inventory:{cpy.completed.files}>{inv.discovered_files}"
        )
    if inv.complete and dst.verified_files > inv.discovered_files:
        out.append(
            f"destination_exceeds_inventory:{dst.verified_files}>{inv.discovered_files}"
        )
    if rec.source_only and dst.verified_files + rec.source_only > max(inv.discovered_files, 0):
        out.append("reconciliation_counts_exceed_inventory")
    if (
        campaign.destination.identity is not None
        and evidence.destination.observed_identity is not None
    ):
        match = match_identity(campaign.destination.identity,
                               evidence.destination.observed_identity)
        if not match.matched:
            out.append(f"destination_identity_conflict:{match.reason}")
    if evidence.errors.fatal > 0:
        out.append(f"fatal_errors:{evidence.errors.fatal}")
    return tuple(out)


def hash_coverage_complete(evidence: CampaignEvidence, policy: CustodyPolicy) -> bool:
    """Hash evidence covers every object in scope (or an honoured exemption)."""
    undeclared = policy.undeclared_exemptions(evidence.hashing.exemptions)
    if undeclared:
        return False
    return evidence.hash_coverage >= evidence.inventory_files


def derive_state(
    campaign: Campaign,
    evidence: CampaignEvidence,
    policy: CustodyPolicy | None = None,
) -> Derivation:
    """Derive the campaign state from evidence. Pure; no I/O, no clock, no writes."""
    policy = policy or CustodyPolicy()
    inv = evidence.inventory
    hsh = evidence.hashing
    cpy = evidence.copy
    dst = evidence.destination
    rec = evidence.reconciliation
    worker_idle = evidence.worker.status in _IDLE_STATUSES or evidence.worker.idle

    contradictions = find_contradictions(campaign, evidence)
    if contradictions:
        return Derivation(
            CampaignState.BLOCKED,
            reasons=("evidence_contradicts_itself",),
            blockers=contradictions,
            contradictions=contradictions,
        )

    # 1. nothing observed yet
    if (
        inv.discovered_files == 0
        and not inv.complete
        and hsh.verified_files == 0
        and cpy.completed.files == 0
        and dst.verified_files == 0
    ):
        return Derivation(CampaignState.DISCOVERED, reasons=("no_inventory_evidence",))

    # 2/3. inventory still in flight
    if not inv.complete:
        if hsh.started or hsh.verified_files or cpy.started:
            return Derivation(
                CampaignState.HASHING,
                reasons=(
                    "source_inventory_not_complete",
                    "hashing_underway",
                ),
            )
        return Derivation(
            CampaignState.DISCOVERED,
            reasons=("source_inventory_in_progress", "no_hashing_yet"),
        )

    # 4. inventory complete, hashing not begun
    if not hsh.started and inv.verified:
        return Derivation(
            CampaignState.SOURCE_VERIFIED,
            reasons=("source_inventory_complete_and_verified", "hashing_not_started"),
        )

    # 5. hashing coverage short of the inventory
    if not hash_coverage_complete(evidence, policy):
        return Derivation(
            CampaignState.HASHING,
            reasons=(
                "hash_evidence_incomplete",
                f"hashed={hsh.verified_files}+exempt={len(hsh.exemptions)}"
                f" of discovered={inv.discovered_files}",
            ),
        )

    # 6. hashing complete, nothing planned yet
    if cpy.planned.files == 0 and cpy.completed.files == 0 and not cpy.started:
        return Derivation(
            CampaignState.HASH_COMPLETE,
            reasons=("hash_evidence_complete", "no_copy_plan"),
        )

    # 7. copy planned, worker has not started
    if not cpy.started:
        return Derivation(
            CampaignState.COPY_PENDING,
            reasons=("hash_evidence_complete", f"copy_planned={cpy.planned.files}"),
        )

    # 8/9. copy in flight or ambiguous
    if not (cpy.result_complete and cpy.ledger_complete):
        if evidence.has_copy_ambiguity or worker_idle or cpy.interrupted:
            reasons = ["copy_started", "copy_result_not_attestable"]
            if cpy.interrupted:
                reasons.append("copy_interrupted")
            if worker_idle:
                reasons.append(f"worker_{evidence.worker.status}")
            if not cpy.ledger_complete:
                reasons.append("destination_ledger_incomplete")
            return Derivation(CampaignState.RECONCILE_REQUIRED, reasons=tuple(reasons))
        return Derivation(
            CampaignState.COPYING,
            reasons=("copy_started", "copy_in_progress", f"completed={cpy.completed.files}"),
        )

    # 10. copy attested complete, verification not started
    if not dst.verification_started:
        return Derivation(
            CampaignState.COPY_COMPLETE,
            reasons=("copy_result_complete", "destination_verification_not_started"),
        )

    # 11. verification underway
    if not dst.verification_complete:
        return Derivation(
            CampaignState.VERIFYING,
            reasons=("destination_verification_started", "destination_verification_incomplete"),
        )

    # 12. verification complete but the sets disagree
    if rec.source_only or rec.mismatched or (
        rec.destination_only and policy.strict_destination_scope
    ):
        reasons = ["destination_verification_complete", "source_destination_disagree"]
        if rec.source_only:
            reasons.append(f"missing_destination={rec.source_only}")
        if rec.destination_only:
            reasons.append(f"destination_only={rec.destination_only}")
        if rec.mismatched:
            reasons.append(f"mismatched={rec.mismatched}")
        return Derivation(CampaignState.RECONCILE_REQUIRED, reasons=tuple(reasons))

    # 13/14. custody proven; the release gate decides the last word
    decision = evaluate_release(campaign, evidence, policy)
    if decision.allowed:
        return Derivation(
            CampaignState.SAFE_TO_RELEASE,
            reasons=("destination_custody_proven", "all_release_conditions_satisfied"),
            blockers=(),
        )
    return Derivation(
        CampaignState.VERIFIED,
        reasons=("destination_custody_proven", "release_gate_closed"),
        blockers=tuple(f"{b.code}:{b.detail}" for b in decision.blockers),
    )


__all__ = ["Derivation", "derive_state", "find_contradictions", "hash_coverage_complete"]
