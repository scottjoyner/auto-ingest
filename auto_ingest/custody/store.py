"""auto_ingest.custody.store - read campaign bundles and assemble status.

A campaign bundle is a directory:

    <bundle>/campaign.json        identity: campaign_id, card_id, source, destination
    <bundle>/evidence.json        bounded evidence counters (never per-file rows)
    <bundle>/ledgers/*.jsonl      optional per-file detail (read-only)

Everything here is read-only except :func:`import_evidence` and
:func:`new_campaign`, which are the only two explicit writes in the package.
``import_evidence`` refuses to do anything without ``apply=True`` and
``new_campaign`` refuses to overwrite an existing campaign. Loading never
creates, repairs or "fixes" a bundle: a bundle that is missing fields yields a
campaign whose derived state is *BLOCKED* or *DISCOVERED*, which is the honest
answer.

Status assembly is a pure fold over (campaign, evidence, policy): running
``status`` twice against an unchanged bundle produces identical output.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from .campaign import (
    Campaign,
    CampaignResolution,
    CardIdentity,
    SourceRef,
    resolve_campaign,
)
from .destination import (
    DestinationRef,
    load_custody_config,
    load_policy,
    resolve_destination,
)
from .evidence import DECLARED_STATE_KEYS, CampaignEvidence
from .ledger import (
    LedgerSummary,
    ReconciliationResult,
    ledger_disagreements,
    reconcile_bundle,
    summarize_bundle_ledgers,
)
from .machine import Derivation, derive_state
from .planner import ResumePlan, plan_resume
from .policy import CustodyPolicy
from .release import ReleaseDecision, evaluate_release
from .states import NEXT_SAFE_ACTION, STATE_PHASE, CampaignState

STATUS_SCHEMA = "auto_ingest.custody.status.v1"
PLAN_SCHEMA = "auto_ingest.custody.plan.v1"

CAMPAIGN_FILE = "campaign.json"
EVIDENCE_FILE = "evidence.json"

#: Evidence blocks an imported document may supply. Anything else in the
#: document (including a declared state) is ignored. Import merges these onto
#: the existing evidence block-by-block: a document that only reports
#: reconciliation must not erase the hash evidence already recorded.
EVIDENCE_BLOCKS = (
    "inventory",
    "hash",
    "hashing",
    "copy",
    "destination",
    "reconciliation",
    "worker",
    "errors",
)
EVIDENCE_SCALARS = ("campaign_id", "observed_at")


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


def merge_evidence_documents(
    existing: Mapping[str, Any],
    incoming: Mapping[str, Any],
) -> Dict[str, Any]:
    """Overlay ``incoming`` onto ``existing``, field by field.

    An evidence document describes *some* observations. Treating it as a whole
    replacement would let a narrow document silently erase evidence already on
    record - the precise failure this package exists to prevent. Even a
    block-level replacement is too coarse: a reconciliation diff speaks about
    ``verified_files`` and the verification flags, and says nothing about
    ``observed_identity`` or ``failures``; replacing the block would drop those
    recorded facts.

    So the rule is per field: what the incoming document states wins (including a
    declared ``0``), and what it does not mention is preserved. Unknown keys -
    including a declared ``state`` - are dropped.
    """
    incoming = incoming or {}
    merged: Dict[str, Any] = {}
    for key in EVIDENCE_BLOCKS:
        old, new = existing.get(key), incoming.get(key)
        if isinstance(old, Mapping) and isinstance(new, Mapping):
            merged[key] = {**old, **new}
        elif new is not None:
            merged[key] = new
        elif old is not None:
            merged[key] = old
    if "hash" in incoming and "hashing" not in incoming:
        merged.pop("hashing", None)
    for key in EVIDENCE_SCALARS:
        value = incoming.get(key)
        if value is not None:
            merged[key] = value
        elif existing.get(key) is not None:
            merged[key] = existing[key]
    return merged


def _preserved_fields(existing: Mapping[str, Any],
                      incoming: Mapping[str, Any]) -> list:
    """Recorded ``block.field`` pairs the incoming document did not mention."""
    incoming = incoming or {}
    out = []
    for key in EVIDENCE_BLOCKS:
        old = existing.get(key)
        new = incoming.get(key)
        if not isinstance(old, Mapping):
            continue
        for field, value in old.items():
            if isinstance(new, Mapping) and field in new:
                continue
            if value in (None, 0, 0.0, "", [], (), False):
                continue
            out.append(f"{key}.{field}")
    return sorted(out)


def import_evidence(
    bundle: str | Path,
    raw: Mapping[str, Any],
    policy: CustodyPolicy | None = None,
    *,
    apply: bool = False,
) -> Dict[str, Any]:
    """Validate (and optionally write) an evidence document for a campaign.

    Read-only unless ``apply=True``. Validation merges the document onto the
    evidence already on record, reports what state the merged evidence derives,
    and lists any declared-state keys that were ignored - a caller cannot smuggle
    a verdict in through this path.
    """
    policy = policy or CustodyPolicy()
    root = Path(bundle)
    campaign = load_campaign(root)
    existing_raw: Dict[str, Any] = {}
    if (root / EVIDENCE_FILE).is_file():
        loaded = json.loads((root / EVIDENCE_FILE).read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            existing_raw = loaded
    merged_raw = merge_evidence_documents(existing_raw, raw or {})
    evidence = CampaignEvidence.from_dict(merged_raw, policy)
    status = build_status(campaign, evidence, policy)
    # Report the ignored declared-state keys from the *incoming* document: the
    # merge already dropped them, so reading them back off the merged evidence
    # would silently lose the fact that someone tried to declare a verdict.
    declared = sorted(
        set(evidence.ignored_declared_fields)
        | {k for k in (raw or {}) if k in DECLARED_STATE_KEYS}
    )
    result: Dict[str, Any] = {
        "applied": False,
        "campaign_id": campaign.campaign_id,
        "declared_state_keys_ignored": declared,
        "derived_state": status.derivation.state.value,
        "merged_with_existing": bool(existing_raw),
        "preserved_fields": _preserved_fields(existing_raw, raw or {}),
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


class CampaignCreationError(RuntimeError):
    """Raised when a new campaign must not be created as requested."""


def new_campaign(
    bundle: str | Path,
    *,
    card_id: str,
    observed: CardIdentity,
    mount_point: Optional[str] = None,
    read_only: bool = False,
    created_at: Optional[str] = None,
    destination: Optional[DestinationRef] = None,
    config: Optional[Mapping[str, Any]] = None,
    env: Optional[Mapping[str, str]] = None,
    apply: bool = False,
) -> Dict[str, Any]:
    """Create a campaign record for physically observed card hardware.

    The entry point of the operator loop, and the only place a campaign id is
    minted. Three refusals, all fail-closed:

    * **no provable identity** - a label alone (``UNTITLED``) identifies nothing,
      so the campaign is not created;
    * **an existing campaign** - creation never overwrites. A card that reappears
      keeps its evidence; a *different* card gets a new bundle;
    * **unresolved destination** is recorded as unresolved (``host_path=None``)
      rather than guessed; it blocks the release gate, it does not block
      onboarding.

    Read-only unless ``apply=True``.
    """
    root = Path(bundle)
    target = root / CAMPAIGN_FILE
    result: Dict[str, Any] = {
        "applied": False,
        "bundle": str(root),
        "would_write": str(target),
    }

    if not observed.known:
        raise CampaignCreationError(
            "card identity is unprovable: provide at least one of device, "
            "filesystem_uuid or serial (a label alone is not identity)"
        )
    if target.exists():
        raise CampaignCreationError(
            f"campaign already exists at {target}; refusing to overwrite evidence"
        )

    if destination is None:
        destination = resolve_destination(config or {}, env=env)

    source = SourceRef(mount_point=mount_point, read_only=bool(read_only), card=observed)
    campaign = Campaign.create(
        card_id=card_id,
        source=source,
        destination=destination,
        created_at=created_at,
    )

    # Provenance: if another campaign already exists elsewhere we cannot see it
    # from here, but within this bundle a prior record would have been refused
    # above, so a fresh id is always correct.
    resolution = resolve_campaign(observed, [], mount_point=mount_point)
    result.update({
        "campaign_id": campaign.campaign_id,
        "card_id": card_id,
        "card_key": campaign.card_key,
        "created_at": campaign.created_at,
        "destination_resolved": campaign.destination.resolved,
        "identity_reason": resolution.reason,
        "source_read_only": campaign.source.read_only,
        "state": CampaignState.DISCOVERED.value,
    })
    if not apply:
        result["mode"] = "validate_only"
        return result

    root.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(target, campaign.to_dict())
    result["applied"] = True
    result["mode"] = "applied"
    return result


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


def reconcile_preview(
    bundle: str | Path,
    policy: CustodyPolicy | None = None,
    *,
    max_samples: Optional[int] = None,
) -> Tuple[ReconciliationResult, CampaignStatus]:
    """Reconcile a bundle's ledgers and show what the result *would* mean.

    Read-only in the strict sense: the ledgers are read, the set difference is
    computed in memory, and the status is built from the **hypothetical**
    evidence without touching ``evidence.json``. Applying the proposal is a
    separate, explicit ``custody import --apply``.
    """
    root = Path(bundle)
    if policy is None:
        policy = load_policy(load_custody_config())
    samples = max_samples if max_samples is not None else policy.max_summary_entries
    result = reconcile_bundle(root, max_samples=samples)
    campaign = load_campaign(root)
    current = load_evidence(root, policy)
    proposal = result.proposal()
    # Preview and import MUST agree, so both go through the same merge: an
    # unusable proposal (absent ledger) leaves the current evidence alone rather
    # than proposing zeroed counts over real ones, and a usable one is layered
    # with exactly the semantics `custody import --apply` will use.
    merged_raw = merge_evidence_documents(current.to_dict(), proposal or {})
    merged = CampaignEvidence.from_dict(merged_raw, policy)
    status = build_status(campaign, merged, policy,
                          ledgers=summarize_bundle_ledgers(root, max_error_samples=samples))
    return result, status


__all__ = [
    "BundleError",
    "CAMPAIGN_FILE",
    "CampaignCreationError",
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
    "EVIDENCE_BLOCKS",
    "load_status",
    "merge_evidence_documents",
    "new_campaign",
    "reconcile_preview",
    "select_campaign",
]
