"""auto_ingest.custody.ledger - read-only access to external per-file ledgers.

The campaign summary stays bounded; per-file detail lives in newline-delimited
JSON ledgers next to it (``ledgers/hash.jsonl``, ``copy.jsonl``,
``destination.jsonl``). This module is the only reader. It opens files ``r``
and never writes, never truncates, never appends, and never creates a ledger -
a missing ledger is reported as absent, not created.

Summarisation is aggregate-only: :func:`summarize_ledger` returns counts and a
capped sample of error strings, never the record list. That keeps a
67k-object campaign cheap to reason about.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .policy import MAX_SUMMARY_ENTRIES

LEDGER_DIRNAME = "ledgers"
HASH_LEDGER = "hash.jsonl"
COPY_LEDGER = "copy.jsonl"
DESTINATION_LEDGER = "destination.jsonl"

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


def read_records(path: str | Path, *, limit: Optional[int] = None) -> List[LedgerRecord]:
    """Read up to ``limit`` ledger records. Read-only; missing file -> ``[]``."""
    p = Path(path)
    if not p.is_file():
        return []
    out: List[LedgerRecord] = []
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
            out.append(LedgerRecord.from_dict(row))
            if limit is not None and len(out) >= limit:
                break
    return out


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


__all__ = [
    "COPY_LEDGER",
    "DESTINATION_LEDGER",
    "HASH_LEDGER",
    "LEDGER_DIRNAME",
    "LedgerRecord",
    "LedgerSummary",
    "ledger_dir",
    "ledger_disagreements",
    "read_records",
    "summarize_bundle_ledgers",
    "summarize_ledger",
]
