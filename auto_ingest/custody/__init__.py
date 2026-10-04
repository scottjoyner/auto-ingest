"""auto_ingest.custody - deterministic custody state for SD-card ingest.

The problem this package solves: Hermes (and a human operator) should never have
to reconstruct "what happened to this card?" by reading logs and ad-hoc shell.

It provides:

``Campaign`` / ``CardIdentity``
    Hardware-anchored campaign identity. Two different cards at the same mount
    point never share a campaign.

``CampaignEvidence``
    The bounded evidence contract: inventory, hash, copy, destination,
    reconciliation, worker and error counters. Per-file detail stays in the
    external ledgers (``auto_ingest.custody.ledger``).

``derive_state``
    The pure state machine. ``CampaignState`` is *derived*; there is no API that
    accepts a declared state, and a declared ``state`` key in an evidence file
    is ignored (and reported in ``ignored_declared_fields``).

``evaluate_release``
    The fail-closed release gate. ``SAFE_TO_RELEASE`` requires every configured
    custody condition to hold; a stopped worker never implies either success or
    failure on its own.

``plan_resume``
    A read-only planner. It describes the next safe operation and can never
    copy, delete or execute anything.

``cli``
    The custody CLI exposes read-only observation/planning plus explicitly
    authorized campaign creation/evidence/copy operations. ``inspect-device``
    discovers source hardware without changing mount state.

The package never mounts, unmounts, remounts, formats, or deletes source media.
Copy execution remains an explicit authorized custody operation; read commands
do not trigger legacy autonomous watchers.
"""

from .campaign import (
    Campaign,
    CampaignResolution,
    CardIdentity,
    SourceRef,
    derive_campaign_id,
    resolve_campaign,
)
from .device import DeviceInspection, DeviceInspectionError, inspect_device
from .destination import (
    DestinationMatch,
    DestinationRef,
    LogicalDestination,
    StorageIdentity,
    load_custody_config,
    load_policy,
    match_identity,
    resolve_destination,
)
from .evidence import (
    CampaignEvidence,
    CopyEvidence,
    Counts,
    DestinationEvidence,
    ErrorEvidence,
    HashEvidence,
    InventoryEvidence,
    ReconciliationEvidence,
    WorkerEvidence,
)
from .machine import Derivation, derive_state, find_contradictions
from .planner import (
    SOURCE_DELETION_ALLOWED,
    SOURCE_MUTATION_ALLOWED,
    PlannedAction,
    ResumePlan,
    plan_resume,
)
from .policy import CustodyPolicy
from .release import Blocker, ReleaseDecision, evaluate_release
from .states import CampaignState
from .store import (
    BundleError,
    CampaignStatus,
    build_status,
    import_evidence,
    load_campaign,
    load_evidence,
    load_status,
)

__all__ = [
    "Blocker",
    "BundleError",
    "Campaign",
    "CampaignEvidence",
    "CampaignResolution",
    "CampaignState",
    "CampaignStatus",
    "CardIdentity",
    "CopyEvidence",
    "Counts",
    "CustodyPolicy",
    "Derivation",
    "DestinationEvidence",
    "DeviceInspection",
    "DeviceInspectionError",
    "DestinationMatch",
    "DestinationRef",
    "ErrorEvidence",
    "HashEvidence",
    "InventoryEvidence",
    "LogicalDestination",
    "PlannedAction",
    "ReconciliationEvidence",
    "ReleaseDecision",
    "ResumePlan",
    "SOURCE_DELETION_ALLOWED",
    "SOURCE_MUTATION_ALLOWED",
    "SourceRef",
    "StorageIdentity",
    "WorkerEvidence",
    "build_status",
    "derive_campaign_id",
    "derive_state",
    "evaluate_release",
    "find_contradictions",
    "import_evidence",
    "inspect_device",
    "load_campaign",
    "load_custody_config",
    "load_evidence",
    "load_policy",
    "load_status",
    "match_identity",
    "plan_resume",
    "resolve_campaign",
    "resolve_destination",
]
