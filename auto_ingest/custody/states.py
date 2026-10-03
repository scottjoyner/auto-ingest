"""auto_ingest.custody.states - the SD-card ingest campaign state vocabulary.

There is exactly one source of truth for "what happened to this card": the
derived state produced by :mod:`auto_ingest.custody.machine`. Nothing in this
package accepts a caller-declared state, and no caller can hand a card a
``SAFE_TO_RELEASE`` verdict. State is a *consequence* of evidence.

The vocabulary is intentionally small and total. Every campaign, at every
observation, is in exactly one of these states.

    DISCOVERED          card seen, nothing trustworthy observed yet
    SOURCE_VERIFIED     source inventoried + inventory verified, hashing not begun
    HASHING             hashing started, coverage below the inventory
    HASH_COMPLETE       hashing complete, no copy plan exists yet
    COPY_PENDING        copy plan exists, copy has not started
    COPYING             copy started and still in flight
    COPY_COMPLETE       copy result complete, destination verification not started
    VERIFYING           destination verification started, not complete
    VERIFIED            destination custody proven; release gate still closed
    RECONCILE_REQUIRED  evidence is ambiguous or inconsistent with the destination
    BLOCKED             evidence contradicts itself; a human must intervene
    SAFE_TO_RELEASE      custody proven AND every release condition satisfied
"""

from __future__ import annotations

from enum import Enum
from typing import Dict


class CampaignState(str, Enum):
    """The total set of derived SD-card campaign states."""

    DISCOVERED = "DISCOVERED"
    SOURCE_VERIFIED = "SOURCE_VERIFIED"
    HASHING = "HASHING"
    HASH_COMPLETE = "HASH_COMPLETE"
    COPY_PENDING = "COPY_PENDING"
    COPYING = "COPYING"
    COPY_COMPLETE = "COPY_COMPLETE"
    VERIFYING = "VERIFYING"
    VERIFIED = "VERIFIED"
    RECONCILE_REQUIRED = "RECONCILE_REQUIRED"
    BLOCKED = "BLOCKED"
    SAFE_TO_RELEASE = "SAFE_TO_RELEASE"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


#: States in which destination custody has been proven byte-for-byte against
#: the source inventory. ``VERIFIED`` means "custody holds"; ``SAFE_TO_RELEASE``
#: additionally means "the release gate is satisfied".
CUSTODY_PROVEN: frozenset = frozenset({CampaignState.VERIFIED, CampaignState.SAFE_TO_RELEASE})

#: States that require a human decision before anything else happens.
NEEDS_HUMAN: frozenset = frozenset({CampaignState.BLOCKED})

#: States where resuming ingest work is meaningful.
RESUMABLE: frozenset = frozenset(
    {
        CampaignState.HASHING,
        CampaignState.COPY_PENDING,
        CampaignState.COPYING,
        CampaignState.COPY_COMPLETE,
        CampaignState.VERIFYING,
        CampaignState.RECONCILE_REQUIRED,
        CampaignState.VERIFIED,
    }
)

#: Coarse phase each state belongs to. Used by the planner to name the next
#: operation without hardcoding a phase transition table in the CLI.
STATE_PHASE: Dict[CampaignState, str] = {
    CampaignState.DISCOVERED: "source_inventory",
    CampaignState.SOURCE_VERIFIED: "hashing",
    CampaignState.HASHING: "hashing",
    CampaignState.HASH_COMPLETE: "copy_planning",
    CampaignState.COPY_PENDING: "copy",
    CampaignState.COPYING: "copy",
    CampaignState.COPY_COMPLETE: "destination_verification",
    CampaignState.VERIFYING: "destination_verification",
    CampaignState.VERIFIED: "operator_release_review",
    CampaignState.RECONCILE_REQUIRED: "destination_reconciliation",
    CampaignState.BLOCKED: "operator_intervention",
    CampaignState.SAFE_TO_RELEASE: "complete",
}

#: The single concrete next operation the operator (or Hermes) should perform
#: for a state. Kept here so the CLI, the docs and the tests cannot drift.
NEXT_SAFE_ACTION: Dict[CampaignState, str] = {
    CampaignState.DISCOVERED: "inventory_source",
    CampaignState.SOURCE_VERIFIED: "hash_source",
    CampaignState.HASHING: "hash_source",
    CampaignState.HASH_COMPLETE: "plan_copy",
    CampaignState.COPY_PENDING: "copy_objects",
    CampaignState.COPYING: "await_copy",
    CampaignState.COPY_COMPLETE: "verify_destination",
    CampaignState.VERIFYING: "verify_destination",
    CampaignState.VERIFIED: "review_for_release",
    CampaignState.RECONCILE_REQUIRED: "reconcile_destination",
    CampaignState.BLOCKED: "resolve_contradiction",
    CampaignState.SAFE_TO_RELEASE: "none",
}

__all__ = [
    "CampaignState",
    "CUSTODY_PROVEN",
    "NEEDS_HUMAN",
    "RESUMABLE",
    "STATE_PHASE",
    "NEXT_SAFE_ACTION",
]
