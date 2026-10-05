"""auto_ingest.custody.ledger - read-only access to external per-file ledgers.

The campaign summary stays bounded; per-file detail lives in newline-delimited
JSON ledgers next to it (``ledgers/hash.jsonl``, ``copy.jsonl``,
``destination.jsonl``, ``collisions.jsonl``). This module is the only reader. It
opens files ``r`` and never writes, never truncates, never appends, and never
creates a ledger - a missing ledger is reported as absent, not created.

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
#: Per-name detail for objects the destination filesystem cannot tell apart or
#: cannot represent: two source keys that case-fold onto one name, a source key
#: that folds onto something already at the destination, and a name the
#: destination's filesystem cannot store. Per-file rows live here; the campaign
#: summary keeps only bounded counters and samples.
COLLISION_LEDGER = "collisions.jsonl"
#: The staged layout, written by `custody stage` and read by `custody execute`.
#:
#: It is a file rather than something both commands recompute, on purpose. If
#: `execute` re-derived the layout from the source, then editing the exclusion
#: policy between the two commands would relocate files with no record of the
#: decision - and a custody ledger that cannot say where a byte was told to go
#: cannot answer for it afterwards. Recorded once, consumed once.
STAGED_LEDGER = "staged.jsonl"

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
    malformed: int = 0
    coherent: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bytes": self.bytes,
            "by_status": dict(sorted(self.by_status.items())),
            "coherent": self.coherent,
            "error_samples": list(self.error_samples),
            "files": self.files,
            "malformed_lines": self.malformed,
            "path": self.path,
            "present": self.present,
            "records": self.records,
            "truncated": self.truncated,
        }


def read_records(path: str | Path, *, limit: Optional[int] = None) -> List[LedgerRecord]:
    """Read up to ``limit`` ledger records. Read-only; missing file -> ``[]``.

    This is the row-level entry point, for a caller that genuinely needs
    per-object detail - :mod:`auto_ingest.custody.verify` does, to compare each
    source digest against its destination counterpart. Aggregation-only callers
    should prefer :func:`summarize_ledger` or :func:`reconcile_ledgers`, which
    never materialise the record list.

    Unparseable lines are skipped rather than raising: a producer may have been
    killed mid-append, and the caller is deciding what that means.
    """
    p = Path(path)
    if not p.is_file():
        return []
    out: List[LedgerRecord] = []
    with p.open("r", encoding="utf-8", errors="replace") as handle:
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
    """Aggregate a ledger into bounded counts. Never returns the record list.

    Corrupt ledgers are reported, not silently under-counted: ``malformed`` and
    ``truncated`` make ``coherent`` False, which is what stops a half-written
    ledger from being read as "everything present is verified".
    """
    p = Path(path)
    if not p.is_file():
        return LedgerSummary(present=False, path=str(p), by_status={})
    by_status: Dict[str, int] = {}
    records = 0
    files = 0
    nbytes = 0
    errors: List[str] = []
    malformed = 0
    truncated = False
    verified = set(verified_statuses)
    with p.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            if not raw_line.endswith("\n"):
                truncated = True
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if not isinstance(row, dict):
                malformed += 1
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
        truncated=truncated,
        malformed=malformed,
        coherent=(malformed == 0 and not truncated),
    )


def ledger_dir(bundle: str | Path) -> Path:
    """Where a campaign bundle keeps its ledgers."""
    return Path(bundle) / LEDGER_DIRNAME


def summarize_bundle_ledgers(
    bundle: str | Path, *, max_error_samples: int = MAX_SUMMARY_ENTRIES
) -> Dict[str, LedgerSummary]:
    """Summarise every standard ledger in a campaign bundle (absent is fine).

    ``collisions.jsonl`` is included so a campaign carrying name problems reports
    them as a bounded ``by_status`` count instead of leaving the operator to diff
    the card by hand. Its rows carry their own kinds as statuses, so they can
    never be mistaken for verified custody: only ``verified`` /
    ``verified_at_destination`` count as ``files``.
    """
    root = ledger_dir(bundle)
    return {
        name: summarize_ledger(root / name, max_error_samples=max_error_samples)
        for name in (HASH_LEDGER, COPY_LEDGER, DESTINATION_LEDGER, COLLISION_LEDGER)
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
    for name, summary in sorted(ledger_summaries.items()):
        if not summary.present:
            continue
        if not summary.coherent:
            problems.append(
                f"{name} ledger is not coherent "
                f"(malformed_lines={summary.malformed}, truncated={summary.truncated}); "
                "its counts cannot be trusted"
            )
    destination = ledger_summaries.get("destination.jsonl")
    if destination is not None and destination.present and destination.coherent:
        declared = int(evidence_summary.get("verified_files") or 0)
        if destination.files != declared:
            problems.append(
                f"destination ledger verifies {destination.files} objects but evidence "
                f"declares {declared}"
            )
    hashing = ledger_summaries.get("hash.jsonl")
    if hashing is not None and hashing.present and hashing.coherent:
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
    source_read: Optional[LedgerRead] = None
    destination_read: Optional[LedgerRead] = None
    expected_source_objects: Optional[int] = None

    @property
    def incoherent(self) -> Tuple[str, ...]:
        """Reasons this diff must not be trusted, in report order."""
        reasons: List[str] = []
        for name, read in (("hash", self.source_read), ("destination", self.destination_read)):
            if read is None or not read.present:
                reasons.append(f"{name}_ledger_absent")
                continue
            if read.malformed:
                reasons.append(f"{name}_ledger_malformed_lines={read.malformed}")
            if read.truncated:
                reasons.append(f"{name}_ledger_truncated")
        if self.expected_source_objects is not None and self.source_objects:
            if self.source_objects != self.expected_source_objects:
                reasons.append(
                    f"hash_ledger_covers_{self.source_objects}_of_"
                    f"{self.expected_source_objects}_inventoried_objects"
                )
        return tuple(reasons)

    @property
    def usable(self) -> bool:
        """True only when both ledgers were read *coherently*.

        Coherent, not merely present. A ledger truncated by an interrupted
        producer reads as healthy for every row it does contain, so a
        presence-only gate would propose "these N objects are in custody" and
        say nothing about the M it never reached - which would then read as
        proven custody for the whole card. Corruption must void the proposal,
        not shrink it.
        """
        return not self.incoherent

    @property
    def complete(self) -> bool:
        """True only when both sides were read coherently and agree exactly."""
        return (self.usable and self.source_only == 0 and self.destination_only == 0
                and self.mismatched == 0 and self.unverifiable == 0)

    def proposal(self) -> Optional[Dict[str, Any]]:
        """The evidence fragment to import, or ``None`` if not usable.

        Not a claim of verification - a claim of *coverage*. The ledgers are the
        verification evidence, so a coherently-read destination ledger means a
        verification pass ran, and a diff in which every source object is proven
        at the destination means that pass covered the whole campaign.

        Deliberately conservative:

        * objects present on both sides but lacking a digest on either one are
          folded into ``source_only``, because they do not have *proven* custody
          (the ``unverifiable`` count stays visible in the report);
        * a malformed or truncated ledger yields **no** proposal at all, rather
          than a proposal covering only the rows that survived;
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
            raise ReconciliationUnavailable(
                "cannot reconcile: "
                + "; ".join(self.incoherent)
                + " (refusing to propose a partial or untrusted diff)"
            )
        return self.proposal()  # type: ignore[return-value]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "complete": self.complete,
            "destination_ledger_present": self.destination_ledger_present,
            "destination_only": self.destination_only,
            "destination_only_samples": list(self.destination_only_samples),
            "destination_objects": self.destination_objects,
            "expected_source_objects": self.expected_source_objects,
            "hash_ledger_present": self.hash_ledger_present,
            "incoherent": list(self.incoherent),
            "mismatched": self.mismatched,
            "mismatched_samples": list(self.mismatched_samples),
            "source_objects": self.source_objects,
            "source_only": self.source_only,
            "source_only_samples": list(self.source_only_samples),
            "unverifiable": self.unverifiable,
            "unverifiable_samples": list(self.unverifiable_samples),
            "usable": self.usable,
            "verified": self.verified,
        }


@dataclass(frozen=True)
class LedgerRead:
    """Result of reading one ledger, including how badly it was read."""

    present: bool = False
    key_digests: Dict[str, Optional[str]] = field(default_factory=dict)
    objects: int = 0
    total_lines: int = 0
    malformed: int = 0
    truncated: bool = False
    undigested_samples: Tuple[str, ...] = ()

    @property
    def coherent(self) -> bool:
        """True when every line parsed and the file ends where a writer ended.

        A ledger truncated by an interrupted producer is the single most likely
        corruption here, and it is the most dangerous: the surviving rows look
        perfectly healthy, so a naive diff happily reports "all these objects are
        in custody" for the rows it did read and stays silent about the rest.
        """
        return self.present and self.malformed == 0 and not self.truncated

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coherent": self.coherent,
            "malformed_lines": self.malformed,
            "objects": self.objects,
            "present": self.present,
            "total_lines": self.total_lines,
            "truncated": self.truncated,
            "undigested_samples": list(self.undigested_samples),
        }


def _read_ledger(path: Path, statuses: Tuple[str, ...], max_samples: int) -> LedgerRead:
    """Read one ledger into ``key -> digest``. Read-only; absent is reported."""
    if not path.is_file():
        return LedgerRead(present=False)
    wanted = set(statuses)
    index: Dict[str, Optional[str]] = {}
    undigested: List[str] = []
    objects = 0
    total = 0
    malformed = 0
    truncated = False

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            total += 1
            line = raw_line.strip()
            if not line:
                continue
            # A final line without a terminating newline means the writer was
            # interrupted mid-append: that row cannot be trusted.
            if not raw_line.endswith("\n"):
                truncated = True
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if not isinstance(row, dict):
                malformed += 1
                continue
            record = LedgerRecord.from_dict(row)
            if record.status not in wanted:
                continue
            if not record.key:
                malformed += 1
                continue
            objects += 1
            digest = record.digest.strip().lower() if record.digest else None
            index[record.key] = digest
            if digest is None and len(undigested) < max_samples:
                undigested.append(record.key)
    return LedgerRead(
        present=True,
        key_digests=index,
        objects=objects,
        total_lines=total,
        malformed=malformed,
        truncated=truncated,
        undigested_samples=tuple(undigested),
    )


def _digest_index(
    path: Path,
    statuses: Tuple[str, ...],
    max_samples: int,
) -> Tuple[Dict[str, Optional[str]], bool, int, List[str]]:
    """Legacy tuple shim over :func:`_read_ledger`. Kept for the aggregate API."""
    read = _read_ledger(path, statuses, max_samples)
    return read.key_digests, read.present, read.objects, list(read.undigested_samples)


def reconcile_ledgers(
    hash_ledger: str | Path,
    destination_ledger: str | Path,
    *,
    max_samples: int = MAX_SUMMARY_ENTRIES,
    source_statuses: Tuple[str, ...] = SOURCE_VERIFIED_STATUSES,
    destination_statuses: Tuple[str, ...] = DESTINATION_VERIFIED_STATUSES,
    expected_source_objects: Optional[int] = None,
) -> ReconciliationResult:
    """Compute the source-vs-destination set difference from two ledgers.

    Pure and read-only. Classification per source key:

    * present in both, digests equal -> ``verified``
    * present in both, digests differ -> ``mismatched``
    * source only -> ``source_only`` (no verified destination copy)
    * either side's digest absent -> ``unverifiable`` (never counted as verified)

    Destination keys with no source counterpart -> ``destination_only``.

    Fail-closed throughout: an absent, malformed or truncated ledger, or a record
    without a digest, never counts as custody. ``expected_source_objects`` (the
    campaign's recorded inventory) is cross-checked so a partially-written ledger
    cannot masquerade as a complete one.

    ``errors="replace"`` on the read keeps undecodable bytes from raising - they
    are counted as malformed instead.
    """
    src_read = _read_ledger(Path(hash_ledger), source_statuses, max_samples)
    dst_read = _read_ledger(Path(destination_ledger), destination_statuses, max_samples)
    src, dst = src_read.key_digests, dst_read.key_digests

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
        hash_ledger_present=src_read.present,
        destination_ledger_present=dst_read.present,
        source_objects=src_read.objects,
        destination_objects=dst_read.objects,
        verified=verified,
        source_only=len(source_only),
        destination_only=len(destination_only),
        mismatched=mismatched,
        unverifiable=unverifiable,
        source_only_samples=tuple(sorted(source_only)[:max_samples]),
        destination_only_samples=tuple(destination_only[:max_samples]),
        mismatched_samples=tuple(mismatched_samples),
        unverifiable_samples=tuple(unverifiable_samples),
        source_read=src_read,
        destination_read=dst_read,
        expected_source_objects=expected_source_objects,
    )


def reconcile_bundle(
    bundle: str | Path,
    *,
    max_samples: int = MAX_SUMMARY_ENTRIES,
    expected_source_objects: Optional[int] = None,
) -> ReconciliationResult:
    """Reconcile the ledgers of a campaign bundle. Absent ledgers stay absent."""
    root = ledger_dir(bundle)
    return reconcile_ledgers(
        root / HASH_LEDGER,
        root / DESTINATION_LEDGER,
        max_samples=max_samples,
        expected_source_objects=expected_source_objects,
    )


__all__ = [
    "COLLISION_LEDGER",
    "COPY_LEDGER",
    "DESTINATION_LEDGER",
    "DESTINATION_VERIFIED_STATUSES",
    "HASH_LEDGER",
    "LEDGER_DIRNAME",
    "LedgerRead",
    "LedgerRecord",
    "LedgerSummary",
    "ReconciliationResult",
    "ReconciliationUnavailable",
    "SOURCE_VERIFIED_STATUSES",
    "ledger_dir",
    "ledger_disagreements",
    "read_records",
    "reconcile_bundle",
    "reconcile_ledgers",
    "summarize_bundle_ledgers",
    "summarize_ledger",
]


def read_staged_ledger(bundle: str | Path) -> Optional[Dict[str, str]]:
    """The recorded ``source_key -> destination_key`` map, or None if absent.

    None means "no layout was decided", which is different from "an empty
    layout". The caller must not treat them alike: the first means copy source
    keys verbatim, the second means copy nothing.
    """
    path = Path(bundle) / LEDGER_DIRNAME / STAGED_LEDGER
    if not path.exists():
        return None
    mapping: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        mapping[row["source_key"]] = row["destination_key"]
    return mapping
