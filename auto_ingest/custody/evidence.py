"""auto_ingest.custody.evidence - the bounded evidence contract.

A campaign summary answers "what is known" in aggregate. It never carries
per-file rows: those live in the external ledgers (``ledgers/*.jsonl``) which
the ledger reader can aggregate on demand. Every list in this module is capped
by ``policy.max_summary_entries`` so a campaign bundle cannot grow without
bound and cannot smuggle a caller-declared verdict:

``from_dict`` *ignores* a ``state`` key if one is present and records it under
``ignored_declared_fields``. State is derived in
:mod:`auto_ingest.custody.machine`; nobody gets to declare ``SAFE_TO_RELEASE``.

Counters are named so their direction is unambiguous:
``source_only`` = source objects with no verified destination copy;
``destination_only`` = destination objects with no source counterpart in scope;
``mismatched`` = objects present on both sides whose digests disagree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from .destination import StorageIdentity
from .policy import MAX_SUMMARY_ENTRIES, CustodyPolicy

#: Keys a bundle may declare that the machine owns exclusively.
DECLARED_STATE_KEYS = ("state", "campaign_state", "status", "safe_to_release")


def _int(value: Any) -> int:
    try:
        if value is None or value is False:
            return 0
        if value is True:
            return 1
        return int(value)
    except (TypeError, ValueError):
        return 0


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _opt_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _bounded(values: Any, limit: int) -> Tuple[str, ...]:
    if not values:
        return ()
    if isinstance(values, str):
        values = [values]
    out: list[str] = []
    for value in values:
        text = str(value).strip()
        if text:
            out.append(text)
        if len(out) >= limit:
            break
    return tuple(out)


@dataclass(frozen=True)
class Counts:
    """A (files, bytes) pair."""

    files: int = 0
    bytes: int = 0

    @classmethod
    def from_dict(cls, raw: Any) -> "Counts":
        if not isinstance(raw, Mapping):
            return cls(_int(raw), 0)
        return cls(_int(raw.get("files")), _int(raw.get("bytes")))

    def to_dict(self) -> Dict[str, int]:
        return {"bytes": self.bytes, "files": self.files}

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(self.files + other.files, self.bytes + other.bytes)


@dataclass(frozen=True)
class InventoryEvidence:
    """What is physically present on the source."""

    discovered_files: int = 0
    discovered_bytes: int = 0
    complete: bool = False
    verified: bool = False
    started: bool = False
    roots: Tuple[str, ...] = ()
    observed_at: Optional[str] = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], limit: int) -> "InventoryEvidence":
        raw = raw or {}
        return cls(
            discovered_files=_int(raw.get("discovered_files", raw.get("files"))),
            discovered_bytes=_int(raw.get("discovered_bytes", raw.get("bytes"))),
            complete=_bool(raw.get("complete")),
            verified=_bool(raw.get("verified")),
            started=_bool(raw.get("started", raw.get("complete") or raw.get("verified"))),
            roots=_bounded(raw.get("roots"), limit),
            observed_at=_opt_str(raw.get("observed_at")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "complete": self.complete,
            "discovered_bytes": self.discovered_bytes,
            "discovered_files": self.discovered_files,
            "roots": list(self.roots),
            "started": self.started,
            "verified": self.verified,
        }


@dataclass(frozen=True)
class HashEvidence:
    """Per-object digest evidence for the source."""

    verified_files: int = 0
    verified_bytes: int = 0
    failed: int = 0
    complete: bool = False
    started: bool = False
    algorithm: Optional[str] = None
    exemptions: Tuple[str, ...] = ()
    error_summary: Tuple[str, ...] = ()
    last_checkpoint: Optional[str] = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], limit: int) -> "HashEvidence":
        raw = raw or {}
        return cls(
            verified_files=_int(raw.get("verified_files", raw.get("hashed_files"))),
            verified_bytes=_int(raw.get("verified_bytes", raw.get("hashed_bytes"))),
            failed=_int(raw.get("errors", raw.get("failed"))),
            complete=_bool(raw.get("complete")),
            started=_bool(raw.get("started", raw.get("complete") or
                               _int(raw.get("verified_files", raw.get("hashed_files"))) > 0)),
            algorithm=_opt_str(raw.get("algorithm")),
            exemptions=_bounded(raw.get("exemptions"), limit),
            error_summary=_bounded(raw.get("error_summary") or raw.get("errors_summary"), limit),
            last_checkpoint=_opt_str(raw.get("last_checkpoint")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "complete": self.complete,
            "error_summary": list(self.error_summary),
            "exemptions": list(self.exemptions),
            "failed": self.failed,
            "last_checkpoint": self.last_checkpoint,
            "started": self.started,
            "verified_bytes": self.verified_bytes,
            "verified_files": self.verified_files,
        }


@dataclass(frozen=True)
class CopyEvidence:
    """Source -> destination transfer evidence."""

    planned: Counts = field(default_factory=Counts)
    completed: Counts = field(default_factory=Counts)
    started: bool = False
    result_complete: bool = False
    interrupted: bool = False
    ledger_complete: bool = False
    error_summary: Tuple[str, ...] = ()
    last_checkpoint: Optional[str] = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], limit: int) -> "CopyEvidence":
        raw = raw or {}
        return cls(
            planned=Counts.from_dict(raw.get("planned")),
            completed=Counts.from_dict(raw.get("completed")),
            started=_bool(raw.get("started")),
            result_complete=_bool(raw.get("result_complete", raw.get("complete"))),
            interrupted=_bool(raw.get("interrupted")),
            ledger_complete=_bool(raw.get("ledger_complete")),
            error_summary=_bounded(raw.get("error_summary"), limit),
            last_checkpoint=_opt_str(raw.get("last_checkpoint")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "completed": self.completed.to_dict(),
            "error_summary": list(self.error_summary),
            "interrupted": self.interrupted,
            "last_checkpoint": self.last_checkpoint,
            "ledger_complete": self.ledger_complete,
            "planned": self.planned.to_dict(),
            "result_complete": self.result_complete,
            "started": self.started,
        }

    @property
    def in_flight(self) -> int:
        return max(self.planned.files - self.completed.files, 0)


@dataclass(frozen=True)
class DestinationEvidence:
    """What has been *proven* to exist at the destination."""

    verified_files: int = 0
    verified_bytes: int = 0
    failures: int = 0
    verification_started: bool = False
    verification_complete: bool = False
    unverified_present_files: int = 0
    observed_identity: Optional[StorageIdentity] = None
    error_summary: Tuple[str, ...] = ()
    last_checkpoint: Optional[str] = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], limit: int) -> "DestinationEvidence":
        raw = raw or {}
        identity_raw = raw.get("observed_identity")
        return cls(
            verified_files=_int(raw.get("verified_files")),
            verified_bytes=_int(raw.get("verified_bytes")),
            failures=_int(raw.get("failures")),
            verification_started=_bool(raw.get("verification_started", raw.get("started"))),
            verification_complete=_bool(raw.get("verification_complete", raw.get("complete"))),
            unverified_present_files=_int(raw.get("unverified_present_files")),
            observed_identity=StorageIdentity.from_dict(identity_raw)
            if isinstance(identity_raw, Mapping) else None,
            error_summary=_bounded(raw.get("error_summary"), limit),
            last_checkpoint=_opt_str(raw.get("last_checkpoint")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_summary": list(self.error_summary),
            "failures": self.failures,
            "last_checkpoint": self.last_checkpoint,
            "observed_identity": (self.observed_identity.to_dict()
                                  if self.observed_identity else None),
            "unverified_present_files": self.unverified_present_files,
            "verification_complete": self.verification_complete,
            "verification_started": self.verification_started,
            "verified_bytes": self.verified_bytes,
            "verified_files": self.verified_files,
        }


@dataclass(frozen=True)
class ReconciliationEvidence:
    """The three-way set difference between source and destination."""

    source_only: int = 0
    destination_only: int = 0
    mismatched: int = 0
    reconciled_at: Optional[str] = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ReconciliationEvidence":
        raw = raw or {}
        return cls(
            source_only=_int(raw.get("source_only")),
            destination_only=_int(raw.get("destination_only")),
            mismatched=_int(raw.get("mismatched", raw.get("mismatch"))),
            reconciled_at=_opt_str(raw.get("reconciled_at")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "destination_only": self.destination_only,
            "mismatched": self.mismatched,
            "reconciled_at": self.reconciled_at,
            "source_only": self.source_only,
        }

    @property
    def clean(self) -> bool:
        return not (self.source_only or self.destination_only or self.mismatched)


@dataclass(frozen=True)
class WorkerEvidence:
    """Which worker touched the card, and what it was doing when it stopped.

    A stopped worker is a *fact about the process*, never a verdict about the
    data: the state machine only reads ``status`` to distinguish "copy still in
    flight" from "copy was interrupted".
    """

    identity: Optional[str] = None
    status: str = "unknown"
    started_at: Optional[str] = None
    stopped_at: Optional[str] = None
    last_checkpoint: Optional[str] = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "WorkerEvidence":
        raw = raw or {}
        return cls(
            identity=_opt_str(raw.get("identity") or raw.get("worker_id")),
            status=str(raw.get("status") or "unknown").strip().lower() or "unknown",
            started_at=_opt_str(raw.get("started_at")),
            stopped_at=_opt_str(raw.get("stopped_at")),
            last_checkpoint=_opt_str(raw.get("last_checkpoint")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identity": self.identity,
            "last_checkpoint": self.last_checkpoint,
            "started_at": self.started_at,
            "status": self.status,
            "stopped_at": self.stopped_at,
        }

    @property
    def active(self) -> bool:
        return self.status in {"running", "copying", "hashing", "verifying", "active"}

    @property
    def idle(self) -> bool:
        return self.status in {"stopped", "absent", "exited", "dead", "inactive", "not_started"}


@dataclass(frozen=True)
class ErrorEvidence:
    """Unresolved failures. Any unresolved error denies release."""

    unresolved: int = 0
    fatal: int = 0
    summaries: Tuple[str, ...] = ()
    witness: Optional[str] = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], limit: int) -> "ErrorEvidence":
        raw = raw or {}
        witness = _opt_str(raw.get("witness") or raw.get("operator_witness"))
        return cls(
            unresolved=_int(raw.get("unresolved")),
            fatal=_int(raw.get("fatal")),
            summaries=_bounded(raw.get("summaries") or raw.get("error_summary"), limit),
            witness=witness,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fatal": self.fatal,
            "summaries": list(self.summaries),
            "unresolved": self.unresolved,
            "witness": self.witness,
        }


@dataclass(frozen=True)
class CampaignEvidence:
    """Bounded evidence for one campaign. No verdict, no state, no clocks."""

    campaign_id: str = ""
    inventory: InventoryEvidence = field(default_factory=InventoryEvidence)
    hashing: HashEvidence = field(default_factory=HashEvidence)
    copy: CopyEvidence = field(default_factory=CopyEvidence)
    destination: DestinationEvidence = field(default_factory=DestinationEvidence)
    reconciliation: ReconciliationEvidence = field(default_factory=ReconciliationEvidence)
    worker: WorkerEvidence = field(default_factory=WorkerEvidence)
    errors: ErrorEvidence = field(default_factory=ErrorEvidence)
    observed_at: Optional[str] = None
    ignored_declared_fields: Tuple[str, ...] = ()

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, Any] | None,
        policy: Optional[CustodyPolicy] = None,
    ) -> "CampaignEvidence":
        raw = raw or {}
        limit = (policy or CustodyPolicy()).max_summary_entries or MAX_SUMMARY_ENTRIES
        declared = tuple(k for k in DECLARED_STATE_KEYS if k in raw)
        return cls(
            campaign_id=str(raw.get("campaign_id") or ""),
            inventory=InventoryEvidence.from_dict(raw.get("inventory") or {}, limit),
            hashing=HashEvidence.from_dict(raw.get("hash") or raw.get("hashing") or {}, limit),
            copy=CopyEvidence.from_dict(raw.get("copy") or {}, limit),
            destination=DestinationEvidence.from_dict(raw.get("destination") or {}, limit),
            reconciliation=ReconciliationEvidence.from_dict(raw.get("reconciliation") or {}),
            worker=WorkerEvidence.from_dict(raw.get("worker") or {}),
            errors=ErrorEvidence.from_dict(raw.get("errors") or {}, limit),
            observed_at=_opt_str(raw.get("observed_at")),
            ignored_declared_fields=declared,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "copy": self.copy.to_dict(),
            "destination": self.destination.to_dict(),
            "errors": self.errors.to_dict(),
            "hash": self.hashing.to_dict(),
            "ignored_declared_fields": list(self.ignored_declared_fields),
            "inventory": self.inventory.to_dict(),
            "observed_at": self.observed_at,
            "reconciliation": self.reconciliation.to_dict(),
            "worker": self.worker.to_dict(),
        }

    # -- derived arithmetic (pure) --------------------------------------
    @property
    def inventory_files(self) -> int:
        return self.inventory.discovered_files

    @property
    def verified_at_destination(self) -> int:
        return self.destination.verified_files

    @property
    def missing_at_destination(self) -> int:
        """Source objects with no verified destination copy (fail-closed)."""
        return max(self.inventory_files - self.destination.verified_files, 0)

    @property
    def remaining_objects(self) -> int:
        """Objects still needing a verified destination copy."""
        return self.missing_at_destination

    @property
    def hash_coverage(self) -> int:
        return self.hashing.verified_files + len(self.hashing.exemptions)

    def hash_coverage_under(self, policy: "CustodyPolicy") -> int:
        """Coverage counting ONLY exemptions the policy actually honours."""
        honoured = sum(1 for p in self.hashing.exemptions if policy.honour_exemption(p))
        return self.hashing.verified_files + honoured

    @property
    def has_copy_ambiguity(self) -> bool:
        """True when the copy ledger cannot say what actually landed.

        This is the CARD-01 shape: hashing finished, a copy was attempted, the
        worker stopped, and the destination ledger is incomplete. We do not know
        which bytes landed, so the campaign must be reconciled - not assumed
        pending.
        """
        if not self.copy.started:
            return False
        if self.copy.result_complete and self.copy.ledger_complete:
            return False
        return bool(self.copy.interrupted or self.worker.idle)


__all__ = [
    "CampaignEvidence",
    "CopyEvidence",
    "Counts",
    "DECLARED_STATE_KEYS",
    "DestinationEvidence",
    "ErrorEvidence",
    "HashEvidence",
    "InventoryEvidence",
    "ReconciliationEvidence",
    "WorkerEvidence",
]
