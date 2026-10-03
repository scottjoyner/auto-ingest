"""auto_ingest.custody.hashing - the first producer of a custody ledger.

Nothing else in this repository writes one. ``hash.jsonl`` had a reader and a
documented contract but no writer, which is why ``67,644`` exists only in tests
and in a hand-written fixture: it was an assertion, not a measurement.

This module produces it. It reads the source and appends newline-terminated
records, nothing more. The source is opened ``rb`` and only ever read; the only
file created is the ledger inside the campaign bundle.

The contract it must honour is already strict, because
:mod:`auto_ingest.custody.ledger` was written to fail closed:

* **Every line is newline-terminated.** An unterminated final line reads as an
  interrupted append, and :func:`~auto_ingest.custody.ledger.reconcile_ledgers`
  voids the *entire* diff on it. So a crash mid-line costs the whole
  reconciliation rather than silently shrinking it - correct, and the reason a
  partial line must never be written.
* **Records carry ``key``, ``digest`` and ``status``** with a source-side status
  of ``verified``.
* **Append-only, resumable.** Objects already present as ``verified`` are
  skipped, so re-running resumes rather than restarting.

Ordering is by sorted key so two runs over an unchanged source produce identical
ledgers. That is what lets a hash ledger be compared for equality across runs.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .ledger import HASH_LEDGER, ledger_dir
from .policy import MAX_SUMMARY_ENTRIES

#: Read size for digest streaming. Large enough to keep syscall overhead low on a
#: slow SD card, small enough that memory stays flat regardless of file size.
CHUNK_BYTES = 1 << 20  # 1 MiB

DEFAULT_ALGORITHM = "sha256"

#: Source-side status. Matches the reader's SOURCE_VERIFIED_STATUSES.
HASHED_STATUS = "verified"


@dataclass(frozen=True)
class HashProgress:
    """What one hashing pass did. Counts are exact; samples are capped."""

    ledger_path: str
    hashed: int = 0
    bytes_read: int = 0
    skipped_existing: int = 0
    failed: int = 0
    errors: Tuple[str, ...] = ()
    complete: bool = False
    interrupted: bool = False
    repaired_partial_line: bool = False
    limit: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bytes_read": self.bytes_read,
            "complete": self.complete,
            "error_samples": list(self.errors),
            "failed": self.failed,
            "hashed": self.hashed,
            "interrupted": self.interrupted,
            "ledger_path": self.ledger_path,
            "repaired_partial_line": self.repaired_partial_line,
            "skipped_existing": self.skipped_existing,
        }


def digest_file(
    path: str | Path,
    *,
    algorithm: str = DEFAULT_ALGORITHM,
    chunk_bytes: int = CHUNK_BYTES,
) -> Tuple[str, int]:
    """Stream one file through a hash. Returns ``(hexdigest, bytes_read)``.

    Read-only. The file is opened ``rb`` and never written, renamed or removed,
    and nothing about the source filesystem is modified by reading it.
    """
    hasher = hashlib.new(algorithm)
    total = 0
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            total += len(chunk)
            hasher.update(chunk)
    return hasher.hexdigest(), total


def already_hashed(ledger: Path) -> Dict[str, int]:
    """Keys already recorded as verified, with their sizes.

    Read-only. A malformed or truncated line is skipped rather than raising: the
    caller is resuming, and the point is to know what is safely done. A truncated
    final line is exactly the state a crashed producer leaves behind, so it must
    not prevent resuming the other 67,643 objects.
    """
    import json

    done: Dict[str, int] = {}
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
        if not isinstance(row, dict):
            continue
        if row.get("status") != HASHED_STATUS:
            continue
        key = row.get("key")
        if not key:
            continue
        try:
            done[str(key)] = int(row.get("size") or 0)
        except (TypeError, ValueError):
            done[str(key)] = 0
    return done


def _record(key: str, size: int, digest: str, algorithm: str) -> str:
    import json

    # Compact, sorted keys, and ALWAYS newline-terminated. The trailing newline
    # is load-bearing: an unterminated last line reads as an interrupted producer
    # and voids the whole reconciliation.
    return json.dumps({
        "key": key,
        "size": size,
        "digest": digest,
        "status": HASHED_STATUS,
        "algorithm": algorithm,
    }, sort_keys=True, separators=(",", ":")) + "\n"


def ends_unterminated(ledger: Path) -> bool:
    """True when the ledger's last line was never newline-terminated.

    That is the fingerprint of a killed producer. :func:`already_hashed` skips
    the damaged line and resumes, so without this check the next appended record
    would be *glued onto* the partial one and be lost with it - one crash would
    then cost two objects instead of one.
    """
    if not ledger.is_file():
        return False
    try:
        if ledger.stat().st_size == 0:
            return False
        with ledger.open("rb") as handle:
            handle.seek(-1, os.SEEK_END)
            return handle.read(1) != b"\n"
    except OSError:  # pragma: no cover - unreadable ledger is handled elsewhere
        return False


def _terminate_partial_line(handle: Any) -> bool:
    """Close off a damaged trailing line so the next record stands alone.

    The partial line stays malformed - it is not repaired, because guessing where
    a truncated JSON object ended would be inventing evidence. Terminating it
    keeps the invariant that matters: **every complete record stays readable**.
    The reconciler still sees one malformed line and still voids the diff, so
    nothing is quietly laundered.
    """
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
    return True


def hash_source(
    bundle: str | Path,
    keys: Dict[str, Path],
    *,
    algorithm: str = DEFAULT_ALGORITHM,
    limit: Optional[int] = None,
    max_errors: int = MAX_SUMMARY_ENTRIES,
    progress: Optional[Callable[[HashProgress], None]] = None,
) -> HashProgress:
    """Hash every key in ``keys`` and append the results to ``hash.jsonl``.

    ``keys`` maps the custody key (the join key used by reconciliation) to the
    source path to read. Sorted by key so two runs over an unchanged source
    produce identical ledgers.

    Resumable: keys already recorded as ``verified`` are skipped, so an
    interrupted pass continues rather than restarting. Appends are flushed and
    fsynced per record, so a kill loses at most the record in flight - never a
    half-written line.

    ``limit`` stops after that many *new* hashes, which is how a bounded probe
    run is expressed without corrupting the ledger.
    """
    root = ledger_dir(bundle)
    root.mkdir(parents=True, exist_ok=True)
    ledger = root / HASH_LEDGER

    done = already_hashed(ledger)
    pending = [(k, p) for k, p in sorted(keys.items()) if k not in done]
    skipped = len(keys) - len(pending)

    hashed = 0
    failed = 0
    bytes_read = 0
    errors: List[str] = []
    interrupted = False

    with ledger.open("a", encoding="utf-8") as handle:
        repaired = False
        if pending and ends_unterminated(ledger):
            # A previous producer died mid-line. Terminate that line first, or the
            # next record below would be appended to it and lost with it.
            repaired = _terminate_partial_line(handle)
        for key, path in pending:
            if limit is not None and hashed + failed >= limit:
                interrupted = True
                break
            try:
                size = os.path.getsize(path)
                digest, read = digest_file(path, algorithm=algorithm)
            except OSError as exc:
                failed += 1
                if len(errors) < max_errors:
                    errors.append(f"{key}:{exc.strerror or exc}")
                continue
            except Exception as exc:  # a digest impl refusing the algorithm
                failed += 1
                if len(errors) < max_errors:
                    errors.append(f"{key}:{exc}")
                continue
            handle.write(_record(key, size, digest, algorithm))
            handle.flush()
            os.fsync(handle.fileno())
            hashed += 1
            bytes_read += read
            if progress is not None:
                progress(HashProgress(
                    ledger_path=str(ledger), hashed=hashed, bytes_read=bytes_read,
                    skipped_existing=skipped, failed=failed, errors=tuple(errors),
                    repaired_partial_line=repaired, limit=limit,
                ))

    result = HashProgress(
        ledger_path=str(ledger),
        hashed=hashed,
        bytes_read=bytes_read,
        skipped_existing=skipped,
        failed=failed,
        errors=tuple(errors),
        complete=not interrupted and failed == 0,
        interrupted=interrupted,
        repaired_partial_line=repaired,
        limit=limit,
    )
    if progress is not None:
        progress(result)
    return result


def to_evidence(result: HashProgress, *, algorithm: str = DEFAULT_ALGORITHM,
                last_checkpoint: Optional[str] = None) -> Dict[str, Any]:
    """The evidence fragment a completed hash pass contributes.

    Only a *complete* pass sets ``complete``. A partial or failing pass records
    its real counts, so the state machine reports HASHING rather than claiming
    the card is hashed.
    """
    return {
        "hash": {
            "algorithm": algorithm,
            "verified_files": result.hashed,
            "verified_bytes": result.bytes_read,
            "complete": result.complete,
            "started": True,
            "failed": result.failed,
            "error_summary": list(result.errors),
            "last_checkpoint": last_checkpoint,
        }
    }


__all__ = [
    "CHUNK_BYTES",
    "DEFAULT_ALGORITHM",
    "HASHED_STATUS",
    "HashProgress",
    "already_hashed",
    "digest_file",
    "ends_unterminated",
    "hash_source",
    "to_evidence",
]
