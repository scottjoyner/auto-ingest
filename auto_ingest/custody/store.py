"""auto_ingest.custody.store - read campaign bundles and assemble status.

A campaign bundle is a directory:

    <bundle>/campaign.json        identity: campaign_id, card_id, source, destination
    <bundle>/evidence.json        bounded evidence counters (never per-file rows)
    <bundle>/ledgers/*.jsonl      optional per-file detail (read-only)

Everything here is read-only except :func:`import_evidence`, which is the single
explicit write in the package and refuses to do anything without ``apply=True``.
Loading never creates, repairs or "fixes" a bundle: a bundle that is missing
fields yields a campaign whose derived state is *BLOCKED* or *DISCOVERED*,
which is the honest answer.

Status assembly is a pure fold over (campaign, evidence, policy): running
``status`` twice against an unchanged bundle produces identical output.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from .campaign import Campaign, CampaignResolution, CardIdentity, resolve_campaign
from .destination import (
    DestinationRef,
    load_custody_config,
    load_policy,
    resolve_destination,
)
from .evidence import CampaignEvidence
from .ledger import LedgerSummary, ledger_disagreements, summarize_bundle_ledgers
from .machine import Derivation, derive_state
from .planner import ResumePlan, plan_resume
from .policy import CustodyPolicy
from .release import ReleaseDecision, evaluate_release
from .states import NEXT_SAFE_ACTION, STATE_PHASE, CampaignState

STATUS_SCHEMA = "auto_ingest.custody.status.v1"
PLAN_SCHEMA = "auto_ingest.custody.plan.v1"

CAMPAIGN_FILE = "campaign.json"
EVIDENCE_FILE = "evidence.json"


class BundleError(RuntimeError):
    """Raised when a bundle cannot be read at all (not merely incomplete)."""


@dataclass(frozen=True)
class CampaignStatus:
    """Everything ``custody status`` reports, in one deterministic structure."""

    campaign: Campaign
    evidence: CampaignEvidence
    derivation: Derivation
    release: ReleaseDecision
    plan: ResumePlan
    ledgers: Dict[str, LedgerSummary]
    policy: CustodyPolicy = CustodyPolicy()
    ledger_disagreements: Tuple[str, ...] = ()
    observed_card_matches: Optional[bool] = None

    # -- projections -----------------------------------------------------
    @property
    def campaign_id(self) -> str:
        return self.campaign.campaign_id

    @property
    def state(self) -> CampaignState:
        return self.derivation.state

    @property
    def next_phase(self) -> str:
        return STATE_PHASE[self.derivation.state]

    @property
    def next_safe_action(self) -> str:
        return NEXT_SAFE_ACTION[self.derivation.state]

    @property
    def source_release_allowed(self) -> bool:
        return self.release.allowed and self.state is CampaignState.SAFE_TO_RELEASE

    def to_dict(self) -> Dict[str, Any]:
        campaign = self.campaign
        ev = self.evidence
        return {
            "schema": STATUS_SCHEMA,
            "blockers": [b.to_dict() for b in self.release.blockers],
            "card": campaign.source.card.to_dict(),
            "card_id": campaign.card_id,
            "campaign_id": campaign.campaign_id,
            "copy": {
                "completed_files": ev.copy.completed.files,
                "complete": ev.copy.result_complete,
                "interrupted": ev.copy.interrupted,
                "in_flight_files": ev.copy.in_flight,
                "ledger_complete": ev.copy.ledger_complete,
                "planned_files": ev.copy.planned.files,
                "planned_bytes": ev.copy.planned.bytes,
                "started": ev.copy.started,
            },
            "created_at": campaign.created_at,
            "destination": {
                "canonical": campaign.destination.logical.canonical,
                "host_path": campaign.destination.host_path,
                "identity": campaign.destination.identity.to_dict()
                if campaign.destination.identity else None,
                "mounted": campaign.destination.mounted,
                "resolved": campaign.destination.resolved,
                "resolved_from": campaign.destination.resolved_from,
                "verification_complete": ev.destination.verification_complete,
                "verified_bytes": ev.destination.verified_bytes,
                "verified_files": ev.destination.verified_files,
            },
            "errors": {
                "fatal": ev.errors.fatal,
                "summaries": list(ev.errors.summaries),
                "unresolved": ev.errors.unresolved,
                "witness": ev.errors.witness,
            },
            "hash": {
                "algorithm": ev.hashing.algorithm,
                "complete": ev.hashing.complete,
                "errors": ev.hashing.failed,
                "exemptions": list(ev.hashing.exemptions),
                "verified": ev.hashing.verified_files,
                "verified_bytes": ev.hashing.verified_bytes,
            },
            "inventory": {
                "complete": ev.inventory.complete,
                "discovered_bytes": ev.inventory.discovered_bytes,
                "discovered_files": ev.inventory.discovered_files,
                "verified": ev.inventory.verified,
            },
            "ledgers": {name: summary.to_dict() for name, summary in sorted(self.ledgers.items())},
            "ledger_disagreements": list(self.ledger_disagreements),
            "next_phase": self.next_phase,
            "next_safe_action": self.next_safe_action,
            "observed_at": ev.observed_at or campaign.last_observed_at,
            "observed_card_matches_campaign": self.observed_card_matches,
            "policy": self.policy_dict,
            "reconciliation": ev.reconciliation.to_dict(),
            "reasons": list(self.derivation.reasons),
            "reconcile_required": self.state is CampaignState.RECONCILE_REQUIRED,
            "source": {
                "card_identity_proven": bool(campaign.source.card.key),
                "device": campaign.source.card.device,
                "filesystem_uuid": campaign.source.card.filesystem_uuid,
                "label": campaign.source.card.label,
                "mount_point": campaign.source.mount_point,
                "read_only": campaign.source.read_only,
            },
            "source_deletion_allowed": False,
            "source_mutation_allowed": False,
            "source_read_only": bool(campaign.source.read_only),
            "source_release_allowed": self.source_release_allowed,
            "state": self.derivation.state.value,
            "warnings": list(self.release.warnings),
            "worker": ev.worker.to_dict(),
        }

    @property
    def policy_dict(self) -> Dict[str, Any]:
        return _policy_dict(self.policy)

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent, default=str)


def _policy_dict(policy: CustodyPolicy) -> Dict[str, Any]:
    return dict(sorted(policy.to_dict().items()))


def load_policy_and_destination(
    repo_root: Optional[str | Path] = None,
    *,
    env: Optional[Mapping[str, str]] = None,
) -> Tuple[CustodyPolicy, DestinationRef, Dict[str, Any]]:
    """Load the custody config block, policy and resolved destination."""
    config = load_custody_config(repo_root)
    policy = load_policy(config, env)
    destination = resolve_destination(config, env=env)
    return policy, destination, config


def load_campaign(bundle: str | Path) -> Campaign:
    """Read ``campaign.json`` (read-only)."""
    path = Path(bundle) / CAMPAIGN_FILE
    if not path.is_file():
        raise BundleError(f"no campaign record at {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise BundleError(f"campaign record at {path} is not an object")
    return Campaign.from_dict(raw)


def load_evidence(bundle: str | Path, policy: CustodyPolicy | None = None) -> CampaignEvidence:
    """Read ``evidence.json`` (read-only). A missing file means "no evidence"."""
    path = Path(bundle) / EVIDENCE_FILE
    if not path.is_file():
        return CampaignEvidence()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise BundleError(f"evidence record at {path} is not an object")
    return CampaignEvidence.from_dict(raw, policy)


def build_status(
    campaign: Campaign,
    evidence: CampaignEvidence,
    policy: CustodyPolicy | None = None,
    *,
    observed_card: Optional[CardIdentity] = None,
    ledgers: Optional[Dict[str, LedgerSummary]] = None,
) -> CampaignStatus:
    """Fold campaign + evidence + policy into a status. Pure."""
    policy = policy or CustodyPolicy()
    derivation = derive_state(campaign, evidence, policy)
    release = evaluate_release(campaign, evidence, policy)
    plan = plan_resume(campaign, evidence, policy, derivation=derivation)
    ledgers = ledgers if ledgers is not None else {}
    raw_dest = evidence.destination.to_dict()
    matches = None
    if observed_card is not None:
        matches = not observed_card.differences(campaign.source.card)
    status = CampaignStatus(
        campaign=campaign,
        evidence=evidence,
        derivation=derivation,
        release=release,
        plan=plan,
        ledgers=ledgers,
        policy=policy,
        ledger_disagreements=ledger_disagreements(raw_dest, ledgers),
        observed_card_matches=matches,
    )
    return status


def load_status(
    bundle: str | Path,
    policy: CustodyPolicy | None = None,
    *,
    observed_card: Optional[CardIdentity] = None,
) -> CampaignStatus:
    """Load a bundle and compute its status. Read-only, no mutation."""
    root = Path(bundle)
    campaign = load_campaign(root)
    if policy is None:
        policy = load_policy(load_custody_config())
    evidence = load_evidence(root, policy)
    ledgers = summarize_bundle_ledgers(root, max_error_samples=policy.max_summary_entries)
    return build_status(campaign, evidence, policy, observed_card=observed_card,
                        ledgers=ledgers)


def import_evidence(
    bundle: str | Path,
    raw: Mapping[str, Any],
    policy: CustodyPolicy | None = None,
    *,
    apply: bool = False,
) -> Dict[str, Any]:
    """Validate (and optionally write) an evidence document for a campaign.

    Read-only unless ``apply=True``. Validation parses the evidence, reports what
    state it would derive, and lists any declared-state keys that were ignored -
    a caller cannot smuggle a verdict in through this path.
    """
    policy = policy or CustodyPolicy()
    root = Path(bundle)
    campaign = load_campaign(root)
    evidence = CampaignEvidence.from_dict(raw, policy)
    status = build_status(campaign, evidence, policy)
    declared = list(evidence.ignored_declared_fields)
    result: Dict[str, Any] = {
        "applied": False,
        "campaign_id": campaign.campaign_id,
        "declared_state_keys_ignored": declared,
        "derived_state": status.derivation.state.value,
        "source_release_allowed": status.source_release_allowed,
        "would_write": str(root / EVIDENCE_FILE),
    }
    if not apply:
        result["mode"] = "validate_only"
        return result

    if isinstance(raw.get("campaign_id"), str) and raw["campaign_id"] != campaign.campaign_id:
        result["applied"] = False
        result["error"] = (
            f"evidence campaign_id {raw['campaign_id']!r} does not match "
            f"campaign {campaign.campaign_id!r}"
        )
        result["mode"] = "rejected"
        return result

    root.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(root / EVIDENCE_FILE, evidence.to_dict())
    result["applied"] = True
    result["mode"] = "applied"
    return result


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, default=str)
        handle.write("\n")
    os.replace(tmp, path)


def select_campaign(
    observed: CardIdentity,
    bundle: str | Path,
    *,
    mount_point: Optional[str] = None,
) -> CampaignResolution:
    """Decide whether the hardware now at ``bundle`` owns an existing campaign."""
    root = Path(bundle)
    campaigns: list[Campaign] = []
    if root.is_file():
        campaigns.append(load_campaign(root.parent))
    elif (root / CAMPAIGN_FILE).is_file():
        campaigns.append(load_campaign(root))
    return resolve_campaign(observed, campaigns, mount_point=mount_point)


__all__ = [
    "BundleError",
    "CAMPAIGN_FILE",
    "CampaignStatus",
    "EVIDENCE_FILE",
    "PLAN_SCHEMA",
    "STATUS_SCHEMA",
    "build_status",
    "import_evidence",
    "load_campaign",
    "load_custody_config",
    "load_evidence",
    "load_policy",
    "load_policy_and_destination",
    "load_status",
    "select_campaign",
]
