"""auto_ingest.custody.release - the fail-closed source-release gate.

``SAFE_TO_RELEASE`` is not something an operator types and not something a
stopped worker implies. It is the single boolean produced here, and it is only
``True`` when every condition in the configured custody policy holds:

* the source mount is read-only;
* the destination resolved, is mounted, and its storage identity is proven;
* the source inventory is complete;
* hash evidence is complete, or the only gaps are exemptions this policy itself
  declares;
* the copy result is attested complete with a complete ledger;
* destination verification is complete;
* mismatches = 0, missing destination = 0;
* no unresolved errors;
* the copy plan covered the required scope (nothing silently dropped).

Every condition is evaluated on every call - the gate never short-circuits -
so the returned blocker list is a complete, deterministic explanation.

How the release *record* enters the gate
----------------------------------------

``evidence.source_release`` records what happened to the objects that used to be
on the card. It is read here, and it is read in one direction only:

* **It can never open the gate.** Not one of its fields is a custody condition.
  ``destination.verified_files`` still means "proven present and correct at the
  destination"; ``source_release.released.files`` means "no longer on the card".
  Those are different facts about different objects in opposite directions, and a
  release that could stand in for verification would let deleting the card
  manufacture a pass - the exact inversion this subsystem exists to prevent.
* **It can only close it.** Two conditions are added, both fail-closed:

  ``source_release_exceeds_inventory``
      more objects destroyed than the card was ever recorded to hold. Impossible
      arithmetic, so the evidence is wrong somewhere.

  ``source_release_incomplete``
      a release attempt failed or was refused on a specific object. Bytes are
      gone, the pass did not finish, and a human has to look at the audit.
* **It is never silent.** A recorded release is reported as a warning, so a
  campaign whose source is already gone says so instead of looking untouched.

So a campaign carrying release evidence and no verification is exactly as
blocked as it was before the evidence existed - which is the property that makes
recording the release safe at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .campaign import Campaign
from .destination import match_identity
from .evidence import CampaignEvidence
from .policy import CustodyPolicy


@dataclass(frozen=True)
class Blocker:
    """One unmet release condition."""

    code: str
    detail: str
    remedy: str

    def to_dict(self) -> dict:
        return {"code": self.code, "detail": self.detail, "remedy": self.remedy}


@dataclass(frozen=True)
class ReleaseDecision:
    """The gate verdict, with the full list of conditions and why they failed."""

    allowed: bool
    blockers: Tuple[Blocker, ...] = ()
    warnings: Tuple[str, ...] = ()
    conditions: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "blockers": [b.to_dict() for b in self.blockers],
            "conditions": list(self.conditions),
            "warnings": list(self.warnings),
        }


#: Deterministic order in which conditions are evaluated and reported.
CONDITION_ORDER: Tuple[str, ...] = (
    "source_read_only",
    "destination_resolved",
    "destination_mounted",
    "destination_identity_proven",
    "source_inventory_complete",
    "hash_evidence_complete",
    "hash_exemptions_declared",
    "copy_result_complete",
    "copy_ledger_complete",
    "destination_verification_complete",
    "destination_verification_covers_required",
    "mismatch_count_zero",
    "missing_destination_zero",
    "unresolved_errors_zero",
    "plan_scope_covers_required_objects",
    "operator_witness_recorded",
    "destination_scope_strict",
    "source_release_consistent",
    "source_release_clean",
)


def evaluate_release(
    campaign: Campaign,
    evidence: CampaignEvidence,
    policy: CustodyPolicy | None = None,
) -> ReleaseDecision:
    """Decide whether the source may ever be released. Fails closed."""
    policy = policy or CustodyPolicy()
    dest = campaign.destination
    inv = evidence.inventory
    hsh = evidence.hashing
    cpy = evidence.copy
    dst = evidence.destination
    rec = evidence.reconciliation
    rel = evidence.source_release

    blockers = []
    warnings = []

    if policy.require_read_only_source and not campaign.source.read_only:
        blockers.append(Blocker(
            "source_not_read_only",
            f"source mount {campaign.source.mount_point or '?'} is not read-only",
            "remount the source read-only and re-observe the campaign",
        ))

    if not dest.resolved:
        blockers.append(Blocker(
            "destination_unresolved",
            f"no destination configured (logical {dest.logical.canonical})",
            "set CUSTODY_DESTINATION_ROOT or custody.destination_root",
        ))

    if policy.require_mounted_destination and dest.resolved and dest.mounted is False:
        blockers.append(Blocker(
            "destination_not_mounted",
            f"{dest.host_path} is not a mount point",
            "mount the canonical destination before release",
        ))

    if policy.require_destination_identity:
        match = match_identity(dest.identity, dst.observed_identity)
        if not match.matched:
            blockers.append(Blocker(
                "destination_identity_unproven",
                match.reason,
                "record the destination filesystem identity and re-verify",
            ))

    if not inv.complete:
        blockers.append(Blocker(
            "inventory_incomplete",
            f"discovered={inv.discovered_files} complete={inv.complete}",
            "finish the source inventory before considering release",
        ))

    undeclared = policy.undeclared_exemptions(hsh.exemptions)
    if undeclared:
        blockers.append(Blocker(
            "hash_exemptions_not_declared",
            "exemptions not declared by policy: " + ",".join(sorted(undeclared)),
            "declare the exemption patterns in custody.policy.declared_hash_exemptions",
        ))
    if evidence.hash_coverage_under(policy) < policy.required_objects(
        inv.discovered_files, hsh.verified_files
    ):
        blockers.append(Blocker(
            "hash_evidence_incomplete",
            f"covered={evidence.hash_coverage_under(policy)} "
            f"of required={policy.required_objects(inv.discovered_files, hsh.verified_files)}",
            "finish hashing the remaining objects (or declare the exemption)",
        ))

    if not cpy.result_complete:
        blockers.append(Blocker(
            "copy_incomplete",
            f"completed={cpy.completed.files}/{cpy.planned.files}"
            f" result_complete={cpy.result_complete}",
            "finish the copy and attest its result",
        ))
    if not cpy.ledger_complete:
        blockers.append(Blocker(
            "copy_ledger_incomplete",
            f"ledger_complete={cpy.ledger_complete}",
            "rebuild the destination ledger before release",
        ))

    if not dst.verification_complete:
        blockers.append(Blocker(
            "destination_verification_incomplete",
            f"verified={dst.verified_files} failures={dst.failures}"
            f" complete={dst.verification_complete}",
            "run full destination verification",
        ))

    # `verification_complete` is a claim ABOUT the verification having finished;
    # it is not evidence that anything was proven. An imported document can assert
    # the flag while carrying verified_files=0, and that combination passed this
    # gate - so a campaign could be declared releasable with custody proven for
    # nothing at all. Cross-check the count against the same required-scope
    # definition the machine uses, so a bare flag cannot stand in for the work.
    required_objects = policy.required_objects(inv.discovered_files, hsh.verified_files)
    if dst.verification_complete and dst.verified_files < required_objects:
        blockers.append(Blocker(
            "destination_verification_short",
            f"verified={dst.verified_files} required={required_objects}"
            f" scope={policy.required_scope}",
            "verification claims complete but proves fewer objects than this "
            "policy requires; re-run verification over the full scope",
        ))

    if rec.mismatched:
        blockers.append(Blocker(
            "mismatch_present",
            f"mismatched={rec.mismatched}",
            "re-copy the mismatched objects and re-verify",
        ))
    if rec.source_only:
        blockers.append(Blocker(
            "missing_destination",
            f"source_only={rec.source_only}",
            "copy the missing objects to the destination",
        ))
    if rec.destination_only:
        if policy.strict_destination_scope:
            blockers.append(Blocker(
                "destination_only_present",
                f"destination_only={rec.destination_only}",
                "remove destination objects outside the campaign scope",
            ))
        else:
            warnings.append(f"destination_only={rec.destination_only} (advisory)")

    # The source-release record is observational: nothing above reads it to allow
    # anything. These two checks can only close the gate, never open it.
    if rel.released.files > max(inv.discovered_files, 0):
        blockers.append(Blocker(
            "source_release_exceeds_inventory",
            f"released={rel.released.files} inventory={inv.discovered_files}",
            "the release record destroys more objects than the card was ever "
            "recorded to hold; the evidence is inconsistent, so rebuild it",
        ))
    if not rel.clean:
        blockers.append(Blocker(
            "source_release_incomplete",
            f"failed={rel.failed} refused={rel.refused} released={rel.released.files}",
            "a release pass did not remove everything it proposed; read "
            "ledgers/release.jsonl before trusting this campaign",
        ))

    if evidence.errors.unresolved:
        blockers.append(Blocker(
            "unresolved_errors",
            f"unresolved={evidence.errors.unresolved}",
            "resolve every campaign error before release",
        ))

    required = policy.required_objects(inv.discovered_files, hsh.verified_files)
    if cpy.planned.files != required:
        blockers.append(Blocker(
            "plan_scope_shortfall",
            f"planned={cpy.planned.files} required={required}"
            f" (scope={policy.required_scope})",
            "re-plan the copy over the full required scope",
        ))

    if policy.require_operator_witness and not evidence.errors.witness:
        blockers.append(Blocker(
            "operator_witness_missing",
            "policy requires an operator witness and none is recorded",
            "record errors.witness in the campaign evidence",
        ))

    if dst.unverified_present_files:
        warnings.append(f"unverified_objects_present_at_destination={dst.unverified_present_files}")

    if rel.observed:
        # Named, never silent. A campaign whose source is already gone must not
        # look untouched, and this must never read as extra destination
        # verification - so it says what was released, from where, and nothing
        # about the destination.
        warnings.append(
            f"source_release_recorded: released={rel.released.files} "
            f"released_bytes={rel.released.bytes} absent={rel.absent} "
            f"failed={rel.failed} refused={rel.refused} "
            f"audit_records={rel.audit_records} "
            f"(a source-side record; it is NOT destination verification)"
        )

    out_of_scope = inv.discovered_files - required
    if out_of_scope > 0:
        # The operator opted into a narrower scope, so this is not a blocker -
        # but it must never be silent, or a later reader will assume the whole
        # card was verified when only part of it was.
        warnings.append(
            f"{out_of_scope} inventoried objects are OUTSIDE the required scope "
            f"({policy.required_scope}): they are not covered by this release"
        )

    return ReleaseDecision(
        allowed=not blockers,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        conditions=CONDITION_ORDER,
    )


__all__ = ["CONDITION_ORDER", "Blocker", "ReleaseDecision", "evaluate_release"]
