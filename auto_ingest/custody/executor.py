"""auto_ingest.custody.executor - the authorized copy step.

This is the only module in the package that writes bytes to the destination. It
is deliberately small and deliberately narrow, because it is the only part that
can lose data.

What it is allowed to do
------------------------
* read every source object (``rb``, streaming);
* create **new** files under the destination root, by writing a temporary file
  and atomically renaming it into place;
* append records to ``copy.jsonl`` and update campaign evidence;
* refresh the mtime of an **already present** campaign-active marker as the copy
  progresses, so the marker's TTL cannot expire a healthy campaign.

What it is forbidden to do
--------------------------
* write, rename, delete or truncate anything on the **source**;
* delete or overwrite anything already at the destination;
* copy an object that is not in the plan it was given;
* create the campaign-active marker - taking the lock is ``cli``'s job, and a
  refresh must never be able to start a campaign that nobody authorized.

Atomicity is the load-bearing property. A copy goes to
``<dest>/.custody-tmp/<key>``, is fsynced, and is then ``os.replace``d into its
final name. A killed executor therefore leaves a temp file and **never a
half-written object under a real name** - which is what makes a later
``custody verify`` meaningful, and what prevents a truncated copy from being
mistaken for a good one the way ``--ignore-existing`` does.

Existing destination objects are never overwritten. An object already proven
present is skipped; an object present but unproven is *verified* in place by
``custody verify`` rather than re-copied, so the 7,000-objects scenario from the
planner cannot turn into 7,000 wasted copies.

The digest of what was written is computed during the copy, from the bytes
actually streamed, so "copied" means "these bytes were produced and hashed", not
"cp returned 0".
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from .hashing import CHUNK_BYTES, DEFAULT_ALGORITHM
from .ledger import (
    COPY_LEDGER,
    DESTINATION_LEDGER,
    HASH_LEDGER,
    STAGED_LEDGER,
    STAGED_META,
    ledger_dir,
    read_records,
)
from .lock import lock_root, touch_active
from .policy import MAX_SUMMARY_ENTRIES

#: Where in-flight bytes live, relative to the destination root. A dot-directory
#: so it is never confused for campaign content, and so ``custody verify`` can
#: tell a temp file from a real object.
TEMP_DIRNAME = ".custody-tmp"

#: Seconds between marker refreshes while copying. The marker's TTL is what stops
#: a SIGKILLed campaign from standing writers down forever, and ingest_claim - the
#: convention this marker follows - has no heartbeat primitive. Without a refresh
#: here a multi-hour campaign would be reaped mid-copy and let a second writer in.
#: Time-based rather than per-object because a single 4GB video can take longer
#: than the whole interval.
MARKER_REFRESH_SEC = 30.0

COPIED = "copied"
SKIPPED = "skipped"
FAILED = "failed"

#: Destination-side statuses that mean "do not copy this again".
_PROVEN = {"verified_at_destination"}


class ExecutionRefused(RuntimeError):
    """A precondition for copying is not met. Nothing was written."""


@dataclass
class CopyProgress:
    """What one execution pass did."""

    ledger_path: str
    copied: int = 0
    copied_bytes: int = 0
    skipped_verified: int = 0
    skipped_present: int = 0
    failed: int = 0
    errors: Tuple[str, ...] = ()
    planned: int = 0
    complete: bool = False
    interrupted: bool = False
    limit: Optional[int] = None
    temp_files_remaining: Tuple[str, ...] = ()

    @property
    def handled(self) -> int:
        return self.copied + self.skipped_verified + self.skipped_present + self.failed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "complete": self.complete,
            "copied": self.copied,
            "copied_bytes": self.copied_bytes,
            "errors": list(self.errors),
            "failed": self.failed,
            "interrupted": self.interrupted,
            "ledger_path": self.ledger_path,
            "limit": self.limit,
            "planned": self.planned,
            "skipped_present": self.skipped_present,
            "skipped_verified": self.skipped_verified,
            "temp_files_remaining": list(self.temp_files_remaining),
        }


@dataclass(frozen=True)
class CopyPlan:
    """Exactly which objects to copy, derived from the two ledgers.

    This is the concrete form of the planner's ``copy_objects`` action: every
    source object, minus those already proven at the destination, minus those
    already present at the destination (which are verification work, not copy
    work). Nothing else is eligible.
    """

    keys: Tuple[str, ...] = ()
    already_verified: Tuple[str, ...] = ()
    present_unverified: Tuple[str, ...] = ()
    absent: Tuple[str, ...] = ()

    @property
    def total_objects(self) -> int:
        return (len(self.already_verified) + len(self.present_unverified)
                + len(self.absent))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "absent": len(self.absent),
            "already_verified": len(self.already_verified),
            "present_unverified": len(self.present_unverified),
            "to_copy": len(self.keys),
            "total_objects": self.total_objects,
        }


def _keys_by_status(path: Path, statuses: Tuple[str, ...]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.is_file():
        return out
    for record in read_records(path):
        if record.status in statuses and record.key:
            out[record.key] = (record.digest or "").strip().lower()
    return out


def plan_copy(bundle: str | Path, destination_root: str | Path,
              staged_destinations: Optional[Mapping[str, str]] = None) -> CopyPlan:
    """Derive the copy set from the hash and destination ledgers. Read-only.

    Three buckets, and the copy set is exactly the third:

    * ``already_verified`` - proven at the destination, never copied again;
    * ``present_unverified`` - bytes exist but are unattested. These are
      ``custody verify``'s job. Copying them would be the waste the reconciliation
      design exists to prevent.
    * ``absent`` - genuinely missing, and the only things to copy.
    """
    ledgers = ledger_dir(bundle)
    source = _keys_by_status(ledgers / HASH_LEDGER, ("verified", "hashed"))
    dest = _keys_by_status(ledgers / DESTINATION_LEDGER, tuple(sorted(_PROVEN)))

    already_verified: List[str] = []
    present_unverified: List[str] = []
    absent: List[str] = []
    root = Path(destination_root)
    # Staging may place an object at a different relative path than its source
    # key, so presence is judged against where the bytes would actually LAND.
    # Absent the map this is the identity, and every existing behaviour is
    # unchanged.
    def dest_key(key: str) -> str:
        return (staged_destinations or {}).get(key, key)

    for key in sorted(source):
        if key in dest:
            already_verified.append(key)
        elif (root / dest_key(key)).exists():
            present_unverified.append(key)
        else:
            absent.append(key)

    return CopyPlan(
        keys=tuple(absent),
        already_verified=tuple(already_verified),
        present_unverified=tuple(present_unverified),
        absent=tuple(absent),
    )


def _safe_join(root: Path, key: str) -> Optional[Path]:
    """Join ``key`` under ``root``, refusing anything that escapes the root.

    Keys come from a ledger, which is a file on disk. A key containing ``..`` or
    an absolute path must never be able to write outside the destination, so this
    is checked rather than trusted.
    """
    candidate = (root / key).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return root / key


def stream_copy(
    source: Path,
    target: Path,
    *,
    algorithm: str = DEFAULT_ALGORITHM,
    chunk_bytes: int = CHUNK_BYTES,
) -> Tuple[str, int]:
    """Copy one file atomically, returning ``(digest_of_bytes_written, size)``.

    Writes to a temp file in the destination, fsyncs, then ``os.replace``. The
    source is opened ``rb`` and only ever read.
    """
    import hashlib

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = target.parent / TEMP_DIRNAME
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir / (target.name + ".partial")

    hasher = hashlib.new(algorithm)
    written = 0
    try:
        with source.open("rb") as src, tmp.open("xb") as dst:
            while True:
                chunk = src.read(chunk_bytes)
                if not chunk:
                    break
                dst.write(chunk)
                hasher.update(chunk)
                written += len(chunk)
            dst.flush()
            os.fsync(dst.fileno())
        if not target.exists():
            os.replace(tmp, target)
        else:
            # Someone else got there first. Never overwrite: remove our temp and
            # let the caller treat the existing object as present-unverified.
            tmp.unlink()
            written = -1
    except BaseException:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise
    return hasher.hexdigest(), written


def leftover_temp_files(destination_root: str | Path) -> Tuple[str, ...]:
    """Temp files an interrupted executor left behind. Read-only."""
    root = Path(destination_root)
    tmp = root / TEMP_DIRNAME
    if not tmp.is_dir():
        return ()
    return tuple(sorted(str(p.relative_to(root)) for p in tmp.rglob("*") if p.is_file()))


def execute_copy(
    bundle: str | Path,
    source_root: str | Path,
    destination_root: str | Path,
    keys: Tuple[str, ...],
    *,
    algorithm: str = DEFAULT_ALGORITHM,
    limit: Optional[int] = None,
    max_errors: int = MAX_SUMMARY_ENTRIES,
    progress: Optional[Callable[[CopyProgress], None]] = None,
    staged_destinations: Optional[Mapping[str, str]] = None,
) -> CopyProgress:
    """Copy exactly ``keys`` to the destination and record every outcome.

    Never overwrites an existing destination object. Never touches the source
    beyond reading it. Appends one ``copy.jsonl`` record per object, so an
    interrupted pass is resumable and auditable rather than invisible.

    ``staged_destinations`` maps a source key to the relative path it should
    occupy at the destination - what ``auto_ingest.custody.staging`` computes to
    put media where the ingest pipeline's suffix-based discovery will find it.
    Omitted, the destination path equals the source key, which is the pre-staging
    behaviour. Each ledger record keeps the source ``key`` AND adds
    ``destination_key``, because reconciliation joins on the source key and an
    operator reading the ledger needs to know where the bytes went.
    """
    ledgers = ledger_dir(bundle)
    ledgers.mkdir(parents=True, exist_ok=True)
    ledger = ledgers / COPY_LEDGER
    src_root = Path(source_root)
    dest_root = Path(destination_root)

    copied = skipped = failed = 0
    copied_bytes = 0
    errors: List[str] = []

    def emit(record: Dict[str, Any], handle) -> None:
        import json

        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    lock_home = lock_root()
    last_refresh = time.monotonic()

    def refresh_marker() -> None:
        nonlocal last_refresh
        now = time.monotonic()
        if now - last_refresh < MARKER_REFRESH_SEC:
            return
        touch_active(lock_home)
        last_refresh = now

    with ledger.open("a", encoding="utf-8") as handle:
        if keys and _ends_unterminated(ledger):
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        staged = staged_destinations or {}
        for key in keys:
            refresh_marker()
            if limit is not None and copied + skipped + failed >= limit:
                break
            landed = staged.get(key, key)
            target = _safe_join(dest_root, landed)
            if target is None:
                failed += 1
                if len(errors) < max_errors:
                    errors.append(f"{key}:key_escapes_destination")
                emit({"key": key, "status": FAILED,
                      "destination_key": landed,
                      "detail": "key_escapes_destination"}, handle)
                continue
            source = src_root / key
            try:
                digest, written = stream_copy(source, target, algorithm=algorithm)
            except OSError as exc:
                failed += 1
                if len(errors) < max_errors:
                    errors.append(f"{key}:{exc.strerror or exc}")
                emit({"key": key, "status": FAILED,
                      "destination_key": landed, "detail": str(exc)}, handle)
                continue
            if written < 0:
                # Already present: never overwrite. Verification decides.
                skipped += 1
                emit({"key": key, "status": SKIPPED, "path": str(target),
                      "destination_key": landed}, handle)
                continue
            copied += 1
            copied_bytes += written
            emit({"key": key, "status": COPIED, "digest": digest,
                  "size": written, "path": str(target),
                  "destination_key": landed}, handle)
            if progress is not None:
                progress(CopyProgress(
                    ledger_path=str(ledger), copied=copied, copied_bytes=copied_bytes,
                    skipped_verified=skipped, failed=failed, errors=tuple(errors),
                    planned=len(keys), limit=limit,
                ))

    result = CopyProgress(
        ledger_path=str(ledger),
        copied=copied,
        copied_bytes=copied_bytes,
        skipped_present=skipped,
        failed=failed,
        errors=tuple(errors),
        planned=len(keys),
        complete=(copied + skipped + failed) >= len(keys) and failed == 0,
        temp_files_remaining=leftover_temp_files(dest_root),
        limit=limit,
    )
    if progress is not None:
        progress(result)
    return result


def _ends_unterminated(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with path.open("rb") as handle:
            handle.seek(-1, os.SEEK_END)
            return handle.read(1) != b"\n"
    except OSError:  # pragma: no cover
        return False


def accounted_keys_in_ledger(ledger: Path) -> int:
    """Distinct keys this ledger records as accounted for at the destination.

    Copied, or already present and therefore deliberately skipped. Failed rows do
    not count: a pass that tried three objects and failed all three has accounted
    for nothing.
    """
    if not ledger.is_file():
        return 0
    import json

    keys = set()
    try:
        text = ledger.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("status") in (COPIED, SKIPPED):
            key = row.get("key")
            if key:
                keys.add(str(key))
    return len(keys)


def to_evidence(result: CopyProgress, plan: Optional[CopyPlan] = None,
                *, planned: Optional[int] = None) -> Dict[str, Any]:
    """The evidence fragment an execution pass contributes.

    This is the piece no other command can supply. ``custody verify`` can prove
    bytes are correct, but only the executor can attest that a *copy happened* -
    which is why ``copy.result_complete`` and ``copy.ledger_complete`` exist and
    why release stays shut without them.

    ``result_complete`` is cumulative: the pass is complete when every planned
    object was handled, not merely when this run's subset was.
    """
    total_planned = planned if planned is not None else (plan.total_objects if plan else result.planned)
    handled_this_pass = (result.copied + result.skipped_present + result.skipped_verified
                         + result.failed)
    # Cumulative, and read from the ledger rather than summed from this pass.
    #
    # A re-run over an already-copied set has nothing to do: plan_copy returns no
    # `absent` keys because the destination ledger already proves them, so
    # handled_this_pass is 0. Writing that over a real count regressed a finished
    # campaign from VERIFIED back to COPYING on every re-run - the same
    # delta-vs-cumulative bug that `verified_bytes` had.
    #
    # max(), not a sum: the ledger already contains this pass's own rows.
    handled_total = max(handled_this_pass, accounted_keys_in_ledger(
        Path(result.ledger_path)))
    return {
        "copy": {
            "planned": {"files": total_planned},
            "completed": {"files": handled_total, "bytes": result.copied_bytes},
            "started": True,
            "result_complete": result.complete and handled_total >= total_planned,
            "interrupted": result.interrupted,
            "ledger_complete": True,
            "error_summary": list(result.errors),
        },
    }


__all__ = [
    "COPIED",
    "FAILED",
    "MARKER_REFRESH_SEC",
    "SKIPPED",
    "TEMP_DIRNAME",
    "CopyPlan",
    "CopyProgress",
    "ExecutionRefused",
    "execute_copy",
    "leftover_temp_files",
    "plan_copy",
    "stream_copy",
    "to_evidence",
]


def write_staged_ledger(bundle: str | Path, plan: Any, *,
                       source_roots: Optional[Sequence[str]] = None) -> Path:
    """Record the layout in the campaign bundle, atomically.

    ``custody execute`` reads this rather than re-deriving the layout, so the
    recorded decision and the copied bytes cannot drift apart. Written whole then
    replaced: a torn file here would leave ``execute`` copying to paths that were
    never proposed.

    ``plan`` is a ``custody.staging.StagingPlan``, passed loosely so this module
    keeps no dependency on the planner that produced it.
    """
    ledgers = ledger_dir(bundle)
    ledgers.mkdir(parents=True, exist_ok=True)
    path = ledgers / STAGED_LEDGER
    tmp = path.with_suffix(path.suffix + ".tmp")
    lines = []
    for obj in sorted(plan.staged, key=lambda o: o.source_key):
        lines.append(json.dumps({
            "source_key": obj.source_key,
            "destination_key": obj.destination_key,
            "role": obj.role,
            "key": obj.key,
            # Provenance of the date, not just its result. A staged path whose
            # date came from the filesystem is a weaker claim, and the record
            # that outlives this run has to be able to say so.
            "key_source": obj.key_source,
            "camera": obj.camera,
        }, sort_keys=True, separators=(",", ":")))
    with open(tmp, "w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)

    # The roots, recorded next to the layout they produced.
    meta_path = ledgers / STAGED_META
    meta_tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    with open(meta_tmp, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "source_roots": sorted({str(r) for r in (source_roots or ())}),
            "staged_objects": len(plan.staged),
            "keyed_from_mtime": plan.by_key_source().get("mtime", 0),
            # Detections preserved but not paired with a clip on this card. Counted
            # here so `preflight` can tell an operator what a copy will contain
            # before the copy runs, rather than after.
            "orphaned_detections": len(plan.unpaired_detections()),
        }, sort_keys=True, indent=2))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(meta_tmp, meta_path)
    return path


