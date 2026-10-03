"""auto_ingest.custody.ledger - read-only access to external per-file ledgers.

The campaign summary stays bounded; per-file detail lives in newline-delimited
JSON ledgers next to it (``ledgers/hash.jsonl``, ``copy.jsonl``,
``destination.jsonl``). This module is the only reader. It opens files ``r``
and never writes, never truncates, never appends, and never creates a ledger -
a missing ledger is reported as absent, not created.

Two read-only operations live here:

* :func:`summarize_ledger` - aggregate counts plus a capped error sample, never
  the record list. That keeps a 67k-object campaign cheap to reason about.
* :func:`reconcile_ledgers` - the set difference between what the *source* says
  it hashed and what the *destination* says it verified. This is the answer to
  "how much of this card is actually in custody?", computed instead of shelled.
  It is a pure read: it emits a proposal, and the only way that proposal becomes
  campaign evidence is an explicit ``custody import --apply``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .policy import MAX_SUMMARY_ENTRIES

LEDGER_DIRNAME = "ledgers"
HASH_LEDGER = "hash.jsonl"
COPY_LEDGER = "copy.jsonl"
DESTINATION_LEDGER = "destination.jsonl"

#: Statuses that count as "proven present" on the respective side of the diff.
SOURCE_VERIFIED_STATUSES = ("verified", "hashed")
DESTINATION_VERIFIED_STATUSES = ("verified", "verified_at_destination", "copied")

_STATUSES = (
    "verified",
    "copied",
    "verified_at_destination",
    "failed",
    "mismatch",
    "missing",
    "skipped",
    "pending",
)


@dataclass(frozen=True)
class LedgerRecord:
    """One ledger row. Per-file detail stays here, not in the campaign summary."""

    key: str
    path: Optional[str] = None
    size: int = 0
    digest: Optional[str] = None
    status: str = "pending"
    phase: str = ""
    detail: Optional[str] = None

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "LedgerRecord":
        try:
            size = int(raw.get("size", raw.get("bytes", 0)) or 0)
        except (TypeError, ValueError):
            size = 0
        return cls(
            key=str(raw.get("key") or raw.get("path") or ""),
            path=raw.get("path"),
            size=size,
            digest=raw.get("digest") or raw.get("sha256"),
            status=str(raw.get("status") or "pending").strip().lower(),
            phase=str(raw.get("phase") or ""),
            detail=raw.get("detail"),
        )


@dataclass(frozen=True)
class LedgerSummary:
    """Bounded aggregate view of a ledger."""

    present: bool
    path: Optional[str]
    records: int = 0
    files: int = 0
    bytes: int = 0
    by_status: Dict[str, int] = field(default_factory=dict)
    error_samples: Tuple[str, ...] = ()
    truncated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bytes": self.bytes,
            "by_status": dict(sorted(self.by_status.items())),
            "error_samples": list(self.error_samples),
            "files": self.files,
            "path": self.path,
            "present": self.present,
            "records": self.records,
            "truncated": self.truncated,
        }


def summarize_ledger(
    path: str | Path,
    *,
    max_error_samples: int = MAX_SUMMARY_ENTRIES,
    verified_statuses: Iterable[str] = ("verified", "verified_at_destination"),
) -> LedgerSummary:
    """Aggregate a ledger into bounded counts. Never returns the record list."""
    p = Path(path)
    if not p.is_file():
        return LedgerSummary(present=False, path=str(p), by_status={})
    by_status: Dict[str, int] = {}
    records = 0
    files = 0
    nbytes = 0
    errors: List[str] = []
    verified = set(verified_statuses)
    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            record = LedgerRecord.from_dict(row)
            records += 1
            by_status[record.status] = by_status.get(record.status, 0) + 1
            if record.status in verified:
                files += 1
                nbytes += record.size
            elif record.status in {"failed", "mismatch", "missing"}:
                if len(errors) < max_error_samples:
                    errors.append(f"{record.key or record.path}:{record.status}")
    return LedgerSummary(
        present=True,
        path=str(p),
        records=records,
        files=files,
        bytes=nbytes,
        by_status=by_status,
        error_samples=tuple(errors),
        truncated=bool(errors) and len(errors) >= max_error_samples,
    )


def ledger_dir(bundle: str | Path) -> Path:
    """Where a campaign bundle keeps its ledgers."""
    return Path(bundle) / LEDGER_DIRNAME


def summarize_bundle_ledgers(
    bundle: str | Path, *, max_error_samples: int = MAX_SUMMARY_ENTRIES
) -> Dict[str, LedgerSummary]:
    """Summarise every standard ledger in a campaign bundle (absent is fine)."""
    root = ledger_dir(bundle)
    return {
        name: summarize_ledger(root / name, max_error_samples=max_error_samples)
        for name in (HASH_LEDGER, COPY_LEDGER, DESTINATION_LEDGER)
    }


def ledger_disagreements(
    evidence_summary: Dict[str, Any], ledger_summaries: Dict[str, LedgerSummary]
) -> Tuple[str, ...]:
    """Compare declared evidence against the ledgers, when both exist.

    Returns human-readable disagreements only. A missing ledger produces no
    disagreement: the evidence file is authoritative for campaigns that were
    instrumented elsewhere.
    """
    problems: List[str] = []
    destination = ledger_summaries.get("destination.jsonl")
    if destination is not None and destination.present:
        declared = int(evidence_summary.get("verified_files") or 0)
        if destination.files != declared:
            problems.append(
                f"destination ledger verifies {destination.files} objects but evidence "
                f"declares {declared}"
            )
    hashing = ledger_summaries.get("hash.jsonl")
    if hashing is not None and hashing.present:
        declared_hash = int(evidence_summary.get("hashed_files") or 0)
        if declared_hash and hashing.files != declared_hash:
            problems.append(
                f"hash ledger verifies {hashing.files} objects but evidence declares "
                f"{declared_hash}"
            )
    return tuple(problems)


# ---------------------------------------------------------------------------
# set-difference reconciliation
# ---------------------------------------------------------------------------
class ReconciliationUnavailable(RuntimeError):
    """Raised when a reconciliation proposal cannot be justified by ledgers."""


@dataclass(frozen=True)
class ReconciliationResult:
    """Bounded source-vs-destination set difference."""

    hash_ledger_present: bool
    destination_ledger_present: bool
    source_objects: int = 0
    destination_objects: int = 0
    verified: int = 0
    source_only: int = 0
    destination_only: int = 0
    mismatched: int = 0
    unverifiable: int = 0
    source_only_samples: Tuple[str, ...] = ()
    destination_only_samples: Tuple[str, ...] = ()
    mismatched_samples: Tuple[str, ...] = ()
    unverifiable_samples: Tuple[str, ...] = ()

    @property
    def usable(self) -> bool:
        """True only when both ledgers were actually read.

        Without this an absent ledger would yield an all-zero proposal, and
        applying that would silently *overwrite* real counts with zeros - the
        exact opposite of fail-closed. So an unusable result proposes nothing.
        """
        return self.hash_ledger_present and self.destination_ledger_present

    @property
    def complete(self) -> bool:
        """True only when both sides were present and agree exactly.

        An absent ledger is never "complete": we cannot prove what we did not
        read.
        """
        return (self.usable and self.source_only == 0 and self.destination_only == 0
                and self.mismatched == 0 and self.unverifiable == 0)

    def proposal(self) -> Optional[Dict[str, Any]]:
        """The evidence fragment to import, or ``None`` if unusable.

        Not a claim of verification - a claim of *coverage*. The ledgers are the
        verification evidence, so a readable destination ledger means a
        verification pass ran, and a diff in which every source object is proven
        at the destination means that pass covered the whole campaign.

        Deliberately conservative:

        * objects present on both sides but lacking a digest on either one are
          folded into ``source_only``, because they do not have *proven* custody
          (the ``unverifiable`` count stays visible in the report);
        * ``copy`` is never touched - copy attestation is a separate fact that a
          diff cannot establish.
        """
        if not self.usable:
            return None
        complete = (self.source_only == 0 and self.destination_only == 0
                    and self.mismatched == 0 and self.unverifiable == 0)
        return {
            "reconciliation": {
                "source_only": self.source_only + self.unverifiable,
                "destination_only": self.destination_only,
                "mismatched": self.mismatched,
            },
            "destination": {
                "verified_files": self.verified,
                "unverified_present_files": max(self.destination_objects - self.verified, 0),
                "verification_started": True,
                "verification_complete": complete,
            },
        }

    def to_evidence(self) -> Dict[str, Any]:
        """Like :meth:`proposal`, but refuses loudly instead of returning None."""
        if not self.usable:
            missing = []
            if not self.hash_ledger_present:
                missing.append(HASH_LEDGER)
            if not self.destination_ledger_present:
                missing.append(DESTINATION_LEDGER)
            raise ReconciliationUnavailable(
                "cannot reconcile: missing ledger(s) " + ", ".join(missing)
                + " (refusing to propose zeroed counts over real evidence)"
            )
        return self.proposal()  # type: ignore[return-value]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "complete": self.complete,
            "destination_ledger_present": self.destination_ledger_present,
            "destination_only": self.destination_only,
            "destination_only_samples": list(self.destination_only_samples),
            "destination_objects": self.destination_objects,
            "hash_ledger_present": self.hash_ledger_present,
            "mismatched": self.mismatched,
            "mismatched_samples": list(self.mismatched_samples),
            "source_objects": self.source_objects,
            "source_only": self.source_only,
            "source_only_samples": list(self.source_only_samples),
            "unverifiable": self.unverifiable,
            "unverifiable_samples": list(self.unverifiable_samples),
            "verified": self.verified,
        }


def _digest_index(
    path: Path,
    statuses: Tuple[str, ...],
    max_samples: int,
) -> Tuple[Dict[str, Optional[str]], bool, int, List[str]]:
    """Build ``key -> digest`` for records in a verified status.

    Returns the index, whether the file existed, the object count, and capped
    samples of keys whose record carries no digest. Read-only.
    """
    if not path.is_file():
        return {}, False, 0, []
    wanted = set(statuses)
    index: Dict[str, Optional[str]] = {}
    undigested: List[str] = []
    present = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            record = LedgerRecord.from_dict(row)
            if record.status not in wanted or not record.key:
                continue
            present += 1
            digest = record.digest.strip().lower() if record.digest else None
            index[record.key] = digest
            if digest is None and len(undigested) < max_samples:
                undigested.append(record.key)
    return index, True, present, undigested


def reconcile_ledgers(
    hash_ledger: str | Path,
    destination_ledger: str | Path,
    *,
    max_samples: int = MAX_SUMMARY_ENTRIES,
    source_statuses: Tuple[str, ...] = SOURCE_VERIFIED_STATUSES,
    destination_statuses: Tuple[str, ...] = DESTINATION_VERIFIED_STATUSES,
) -> ReconciliationResult:
    """Compute the source-vs-destination set difference from two ledgers.

    Pure and read-only. Classification per source key:

    * present in both, digests equal -> ``verified``
    * present in both, digests differ -> ``mismatched``
    * source only -> ``source_only`` (no verified destination copy)
    * either side's digest absent -> ``unverifiable`` (never counted as verified)

    Destination keys with no source counterpart -> ``destination_only``.

    Fail-closed throughout: an absent ledger, a record without a digest, or an
    empty digest never counts as custody.
    """
    src, src_present, src_count, _src_undigested = _digest_index(
        Path(hash_ledger), source_statuses, max_samples
    )
    dst, dst_present, dst_count, _ = _digest_index(
        Path(destination_ledger), destination_statuses, max_samples
    )

    verified = 0
    mismatched = 0
    unverifiable = 0
    source_only: List[str] = []
    mismatched_samples: List[str] = []
    unverifiable_samples: List[str] = []

    for key, src_digest in src.items():
        if key not in dst:
            source_only.append(key)
            continue
        dst_digest = dst[key]
        if src_digest is None or dst_digest is None:
            unverifiable += 1
            if len(unverifiable_samples) < max_samples:
                unverifiable_samples.append(key)
            continue
        if src_digest == dst_digest:
            verified += 1
            continue
        mismatched += 1
        if len(mismatched_samples) < max_samples:
            mismatched_samples.append(key)

    dst_keys: Set[str] = set(dst)
    destination_only = sorted(dst_keys - set(src))

    return ReconciliationResult(
        hash_ledger_present=src_present,
        destination_ledger_present=dst_present,
        source_objects=src_count,
        destination_objects=dst_count,
        verified=verified,
        source_only=len(source_only),
        destination_only=len(destination_only),
        mismatched=mismatched,
        unverifiable=unverifiable,
        source_only_samples=tuple(sorted(source_only)[:max_samples]),
        destination_only_samples=tuple(destination_only[:max_samples]),
        mismatched_samples=tuple(mismatched_samples),
        unverifiable_samples=tuple(unverifiable_samples),
    )


def reconcile_bundle(
    bundle: str | Path,
    *,
    max_samples: int = MAX_SUMMARY_ENTRIES,
) -> ReconciliationResult:
    """Reconcile the ledgers of a campaign bundle. Absent ledgers stay absent."""
    root = ledger_dir(bundle)
    return reconcile_ledgers(
        root / HASH_LEDGER,
        root / DESTINATION_LEDGER,
        max_samples=max_samples,
    )


__all__ = [
    "COPY_LEDGER",
    "DESTINATION_LEDGER",
    "DESTINATION_VERIFIED_STATUSES",
    "HASH_LEDGER",
    "LEDGER_DIRNAME",
    "LedgerRecord",
    "LedgerSummary",
    "ReconciliationResult",
    "ReconciliationUnavailable",
    "SOURCE_VERIFIED_STATUSES",
    "ledger_dir",
    "ledger_disagreements",
    "reconcile_bundle",
    "reconcile_ledgers",
    "summarize_bundle_ledgers",
    "summarize_ledger",
]
