"""auto_ingest.custody.verify - the destination verification producer.

The hash ledger says what the *source* contains. This produces the other half:
what is actually present and correct at the destination. It is the evidence that
turns ``VERIFIED`` into ``SAFE_TO_RELEASE``, and the only thing that can.

It reads both ledgers and the destination tree, and writes ``destination.jsonl``.
It never copies, never deletes, and never writes to the source - a verification
pass that moved bytes would be worse than no verification at all, because the
result would look authoritative.

Classification per source object, deliberately conservative:

======================  ==========================================================
digest matches          ``verified_at_destination`` - custody proven
destination absent      ``missing`` - nothing there to prove
digest differs          ``mismatch`` - bytes there, and they are wrong
destination unreadable  ``failed`` - present but unreadable; never custody
======================  ==========================================================

Only ``verified_at_destination`` counts toward custody. A digest is never inferred,
never carried over from the source ledger, and never assumed because a file
exists - which is exactly the ``--ignore-existing`` defect that lets a truncated
copy count as done forever.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from .hashing import DEFAULT_ALGORITHM, digest_file
from .ledger import DESTINATION_LEDGER, HASH_LEDGER, ledger_dir, read_records
from .policy import MAX_SUMMARY_ENTRIES

#: Destination-side statuses. Only VERIFIED counts as custody.
VERIFIED = "verified_at_destination"
MISSING = "missing"
MISMATCH = "mismatch"
FAILED = "failed"


@dataclass(frozen=True)
class VerifyProgress:
    """What one verification pass did. Counts exact, samples capped."""

    ledger_path: str
    verified: int = 0
    verified_bytes: int = 0
    missing: int = 0
    mismatched: int = 0
    failed: int = 0
    skipped_existing: int = 0
    errors: Tuple[str, ...] = ()
    complete: bool = False
    limit: Optional[int] = None

    @property
    def checked(self) -> int:
        return self.verified + self.missing + self.mismatched + self.failed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checked": self.checked,
            "complete": self.complete,
            "error_samples": list(self.errors),
            "failed": self.failed,
            "ledger_path": self.ledger_path,
            "limit": self.limit,
            "mismatched": self.mismatched,
            "missing": self.missing,
            "skipped_existing": self.skipped_existing,
            "verified": self.verified,
            "verified_bytes": self.verified_bytes,
        }


def _record(key: str, status: str, *, digest: Optional[str] = None,
            size: int = 0, path: Optional[str] = None,
            detail: Optional[str] = None) -> str:
    import json

    row: Dict[str, Any] = {"key": key, "status": status}
    if digest:
        row["digest"] = digest
    if size:
        row["size"] = size
    if path:
        row["path"] = path
    if detail:
        row["detail"] = detail
    # Always newline-terminated: see auto_ingest.custody.hashing for why this is
    # load-bearing rather than cosmetic.
    return json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"


def source_digests(bundle: str | Path) -> Dict[str, Tuple[str, int]]:
    """``key -> (digest, size)`` from the hash ledger.

    Read-only. Objects the source never hashed are absent here, and therefore
    cannot be verified - the pass reports them as missing rather than trusting
    the destination's own claim about itself.
    """
    ledger = ledger_dir(bundle) / HASH_LEDGER
    if not ledger.is_file():
        return {}
    out: Dict[str, Tuple[str, int]] = {}
    for record in read_records(ledger):
        if record.status not in ("verified", "hashed"):
            continue
        if not record.key or not record.digest:
            continue
        out[record.key] = (record.digest.strip().lower(), record.size)
    return out


def already_verified(ledger: Path) -> Dict[str, str]:
    """Keys already proven at the destination, and their digests."""
    import json

    done: Dict[str, str] = {}
    if not ledger.is_file():
        return done
    try:
        text = ledger.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return done
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or row.get("status") != VERIFIED:
            continue
        key, digest = row.get("key"), row.get("digest")
        if key and digest:
            done[str(key)] = str(digest).strip().lower()
    return done


def verified_bytes_in_ledger(ledger: Path) -> int:
    """Bytes already proven at the destination, summed from the ledger.

    ``already_verified`` returns digests because that is what a resume needs to
    decide what to skip. Evidence additionally needs the *size* of what was
    already proven, or the campaign records a file count with no bytes beside it.
    """
    import json

    if not ledger.is_file():
        return 0
    try:
        text = ledger.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    # Per key, not per row. A `--recheck` pass re-proves objects it already
    # proved and appends fresh rows for them, so summing rows counts the same
    # bytes twice - which is how a verified_bytes of 3,835,406,336 appeared
    # beside a verified_files of 3.
    sizes: Dict[str, int] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or row.get("status") != VERIFIED:
            continue
        key, size = row.get("key"), row.get("size")
        if key and isinstance(size, int) and size > 0:
            sizes[str(key)] = size
    return sum(sizes.values())


def ends_unterminated(ledger: Path) -> bool:
    if not ledger.is_file():
        return False
    try:
        if ledger.stat().st_size == 0:
            return False
        with ledger.open("rb") as handle:
            handle.seek(-1, os.SEEK_END)
            return handle.read(1) != b"\n"
    except OSError:  # pragma: no cover
        return False


def verify_destination(
    bundle: str | Path,
    destination_root: str | Path,
    *,
    algorithm: str = DEFAULT_ALGORITHM,
    limit: Optional[int] = None,
    recheck: bool = False,
    max_errors: int = MAX_SUMMARY_ENTRIES,
    progress: Optional[Callable[[VerifyProgress], None]] = None,
    staged_destinations: Optional[Mapping[str, str]] = None,
) -> VerifyProgress:
    """Verify every source object against the destination tree.

    Read-only. For each key in the hash ledger the expected digest is taken from
    the ledger and the actual digest is computed from the destination file; the
    two are compared. A destination file that merely *exists* proves nothing -
    the bytes are read and hashed.

    ``recheck`` forces re-verification of keys already proven; by default they are
    skipped, so the pass is resumable.

    ``staged_destinations`` maps a source key to the relative path it was copied
    to. Without it this looks for the file at ``destination_root / key``, which is
    right for a flat copy and wrong for every staged one - and it fails in the
    worst direction, reporting a completed copy as MISSING. Omitted, the behaviour
    is unchanged.
    """
    expected = source_digests(bundle)
    root = ledger_dir(bundle)
    root.mkdir(parents=True, exist_ok=True)
    ledger = root / DESTINATION_LEDGER

    done = {} if recheck else already_verified(ledger)
    pending = sorted(k for k in expected if k not in done)

    verified = missing = mismatched = failed = 0
    verified_bytes = 0
    errors: List[str] = []
    base = Path(destination_root)
    staged = staged_destinations or {}

    with ledger.open("a", encoding="utf-8") as handle:
        if pending and ends_unterminated(ledger):
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        for key in pending:
            if limit is not None and (verified + missing + mismatched + failed) >= limit:
                break
            want_digest, size = expected[key]
            target = base / staged.get(key, key)
            if not target.exists():
                missing += 1
                handle.write(_record(key, MISSING, size=size,
                                     path=target.as_posix()))
                handle.flush()
                os.fsync(handle.fileno())
                continue
            try:
                actual, read = digest_file(target, algorithm=algorithm)
            except OSError as exc:
                failed += 1
                if len(errors) < max_errors:
                    errors.append(f"{key}:{exc.strerror or exc}")
                handle.write(_record(key, FAILED, size=size, path=target.as_posix(),
                                     detail=str(exc)))
                handle.flush()
                os.fsync(handle.fileno())
                continue
            if actual == want_digest:
                verified += 1
                verified_bytes += read
                handle.write(_record(key, VERIFIED, digest=actual, size=read,
                                     path=target.as_posix()))
            else:
                mismatched += 1
                handle.write(_record(key, MISMATCH, digest=actual, size=read,
                                     path=target.as_posix()))
            handle.flush()
            os.fsync(handle.fileno())
            if progress is not None:
                progress(VerifyProgress(
                    ledger_path=str(ledger), verified=verified,
                    verified_bytes=verified_bytes, missing=missing,
                    mismatched=mismatched, failed=failed,
                    skipped_existing=len(done), errors=tuple(errors), limit=limit,
                ))

    result = VerifyProgress(
        ledger_path=str(ledger),
        verified=verified,
        verified_bytes=verified_bytes,
        missing=missing,
        mismatched=mismatched,
        failed=failed,
        skipped_existing=len(done),
        errors=tuple(errors),
        complete=(verified + missing + mismatched + failed) >= len(pending),
        limit=limit,
    )
    if progress is not None:
        progress(result)
    return result


def _proven_bytes(result: "VerifyProgress") -> int:
    """Bytes the destination ledger proves, falling back to the pass's own count.

    The fallback exists only for a pass whose ledger was never written - an
    unwritable bundle directory, say. A zero that is real stays zero.
    """
    path = Path(result.ledger_path)
    if path.is_file():
        return verified_bytes_in_ledger(path)
    return result.verified_bytes


def to_evidence(result: VerifyProgress, *,
                reconciled_at: Optional[str] = None) -> Dict[str, Any]:
    """The evidence fragment a verification pass contributes.

    ``verified_files`` is **cumulative**: this pass's findings plus everything the
    ledger already proved. Reporting only this pass's delta would write 0 over a
    real count on every resume and regress the campaign out of custody.

    ``source_only`` is derived from *this pass's* findings rather than asserted:
    an object that is missing, mismatched or unreadable has no proven custody, so
    it counts against release. ``verification_complete`` is only true when the
    pass finished examining every outstanding key.
    """
    outstanding = result.missing + result.mismatched + result.failed
    return {
        "destination": {
            "verified_files": result.verified + result.skipped_existing,
            # Cumulative for the same reason as the count directly above. Leaving
            # this as the pass's own delta wrote `verified_files: 3` beside
            # `verified_bytes: 0` on every resume - a campaign claiming three
            # proven objects and no bytes at all, which the state machine then
            # read as custody.
            #
            # Read from the ledger, not summed from the result: the ledger already
            # contains this pass's own records, so adding the two counts the same
            # bytes twice. The ledger is the one place that survives a resume, so
            # it is the one place the total comes from.
            "verified_bytes": _proven_bytes(result),
            "failures": result.failed,
            "verification_started": True,
            "verification_complete": result.complete,
            "error_summary": list(result.errors),
        },
        "reconciliation": {
            "source_only": outstanding,
            "destination_only": 0,
            "mismatched": result.mismatched,
            "reconciled_at": reconciled_at,
        },
    }


__all__ = [
    "FAILED",
    "MISMATCH",
    "MISSING",
    "VERIFIED",
    "VerifyProgress",
    "already_verified",
    "ends_unterminated",
    "source_digests",
    "to_evidence",
    "verify_destination",
]
