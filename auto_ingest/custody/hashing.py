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

Names on the real filesystems
-----------------------------

The card this package exists for is **vfat**, and the destination is an SMB2
share on **exFAT**. Both are case-insensitive and both cap a path component, so
``A.mp4`` and ``a.mp4`` are *one file* at either end, and a name that cannot be
represented cannot be copied at all. Treating those keys as two distinct objects
is not a cosmetic bug: the planner would schedule ``a.mp4`` believing it is
absent, and the executor's ``if not target.exists()`` check is an *exact-name*
check, so the copy either silently clobbers the existing object or fails on the
server. ``executor``'s "never overwrite" invariant does not protect against this.

So this module **detects** the problem and reports it. What it must never do:

* **never rename a key** - the stored key stays byte-accurate to the filename on
  the card, because that is what re-opens the object later;
* **never pick a winner** between two colliding keys, never skip one, and never
  deduplicate - both files exist and the operator decides which spelling the
  campaign keeps;
* **never rewrite the stored key into a normalised form** - normalisation exists
  for *comparison* only, so existing fixture campaigns and existing ledgers,
  which use exact-case keys, keep reading unchanged.

Normalisation uses :meth:`str.casefold`, never :meth:`str.lower`. That is not
stylistic: ``"Straße".lower()`` is ``"straße"``, so a ``.lower()`` comparison
misses a real ``straße.mp4`` / ``STRASSE.mp4`` collision on a case-insensitive
filesystem, while ``casefold`` maps both to ``"strasse"`` and finds it.
``casefold`` is also the operation that scales: it folds ligatures and the
dotted capital I, both of which a ``.lower()``-based guess would miss.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from .ledger import COLLISION_LEDGER, DESTINATION_LEDGER, HASH_LEDGER, ledger_dir, read_records
from .policy import MAX_SUMMARY_ENTRIES

#: Read size for digest streaming. Large enough to keep syscall overhead low on a
#: slow SD card, small enough that memory stays flat regardless of file size.
CHUNK_BYTES = 1 << 20  # 1 MiB

DEFAULT_ALGORITHM = "sha256"

#: Source-side status. Matches the reader's SOURCE_VERIFIED_STATUSES.
HASHED_STATUS = "verified"

#: Name-problem kinds. These double as the ``status`` of their ledger rows, and
#: none of them is a source-side verified status, so a name problem can never be
#: read as custody.
CASE_COLLISION = "case_collision"
DESTINATION_CASE_COLLISION = "destination_case_collision"
UNREPRESENTABLE_NAME = "unrepresentable_name"

#: Prefix of each bounded summary line, so a reader can tell the three problems
#: apart without parsing prose. :func:`auto_ingest.custody.planner.plan_resume`
#: reads these back to build one dedicated blocker per problem: an operator told
#: only "unresolved=2" would have to diff the card by hand to learn what collided.
CASE_COLLISION_PREFIX = f"{CASE_COLLISION}: "
DESTINATION_COLLISION_PREFIX = f"{DESTINATION_CASE_COLLISION}: "
UNREPRESENTABLE_NAME_PREFIX = f"{UNREPRESENTABLE_NAME}: "

#: Longest single path component, in UTF-8 bytes, that every filesystem in play
#: can store. vfat and exFAT count UTF-16 code units (255); ext4 and APFS count
#: UTF-8 bytes (255). Bytes are the stricter of the two, so a component that
#: passes here passes everywhere - which is the point, since the question is
#: "will the copy fail at the destination", not "is the name tasteful".
NAME_COMPONENT_LIMIT_BYTES = 255

#: Longest whole relative path. ext4's PATH_MAX (4096) is the strictest total in
#: play; the executor joins the key under the destination root, so the absolute
#: path is longer than the key by however deep that root is.
PATH_LIMIT_BYTES = 4096

#: Characters neither vfat nor exFAT can store in a filename. Both allocate name
#: entries in UCS-2 and both reserve the Win32 set ``" * / : < > ? \ |`` plus the
#: control range U+0001-U+001F; NUL is checked separately because it terminates a
#: C string and so can never appear in a filename at all. Deliberately NOT in the
#: set, with reasons, because inventing problems is its own failure:
#:
#: * Win32 reserved device names (CON, NUL, COM1, ...) are an API-namespace
#:   reservation, not a filesystem one - exFAT stores them and Linux, curl and SMB
#:   write them without complaint, so flagging them would be a false alarm.
#: * Normalisation-equivalent spellings (NFC vs NFD) are a real hazard for a
#:   macOS destination but are out of scope here; they need a normalisation pass
#:   of their own, not a guess inside a case-fold comparison.
VFAT_ILLEGAL_CHARS = frozenset('"*:/<>?\\|\x00') | {chr(c) for c in range(1, 0x20)}


@dataclass(frozen=True)
class NameProblem:
    """One key that cannot be told apart from, or stored by, the destination.

    A record of a problem, never a resolution of one: no winner is chosen and no
    key is rewritten. ``key`` is the byte-accurate custody key, ``others`` names
    the keys it collides with, and ``detail`` is the one-line form that appears
    both in the collision ledger and (bounded) in the campaign summary.
    """

    kind: str
    key: str
    detail: str
    folded: str = ""
    others: Tuple[str, ...] = ()

    @property
    def sample(self) -> str:
        """The bounded-summary line: a stable prefix plus the human detail.

        The prefix is ``f"{kind}: "``, which is exactly what the three public
        ``*_PREFIX`` constants hold - a reader gets the kind without parsing prose.
        """
        return f"{self.kind}: {self.detail}"


def fold_key(key: str) -> str:
    """The comparison form of a custody key: case-folded, per path component.

    Case folding is applied to each component separately because it is each
    component the filesystem compares, and the separators are already structural.
    :meth:`str.casefold` rather than :meth:`str.lower`: only casefold treats
    ``ß`` as ``ss`` (and folds ligatures, and the Turkish dotted capital I), which
    is precisely the class of name a ``.lower()`` comparison silently misses.

    The result is for COMPARISON ONLY. Stored keys - in the ledgers and in
    evidence - stay byte-accurate to the filename, because a custody key is what
    re-opens the object on the card.
    """
    return "/".join(part.casefold() for part in key.split("/"))


def filename_problem(key: str) -> Optional[str]:
    """Why the destination filesystem cannot represent ``key``, else ``None``.

    **This is detection, never renaming and never rejection.** The caller still
    hashes the object, still records it under its exact key, and still counts it
    as verified custody on the source side. A problem recorded here becomes an
    *unresolved* campaign error, which refuses release and tells the operator to
    resolve it - it never fails the pass, and it never drops a file. That is the
    whole reason the rule below can be strict without breaking a campaign that is
    merely on a case-sensitive filesystem.

    The rule, precisely:

    * every path component must be at most :data:`NAME_COMPONENT_LIMIT_BYTES`
      UTF-8 bytes, and the whole key at most :data:`PATH_LIMIT_BYTES`;
    * no component may contain a character from :data:`VFAT_ILLEGAL_CHARS`;
    * no component may end in ``.`` or a space: Windows strips both, so such a
      name cannot round-trip through the destination and the key would stop
      re-opening what it named.

    A name is therefore never rejected for being *merely long on a
    case-sensitive path* - the object is hashed and recorded either way, and only
    the release gate objects. Every fixture campaign in this repository uses short
    ASCII names, so none of them is affected.
    """
    if len(key.encode("utf-8")) > PATH_LIMIT_BYTES:
        return (f"the whole key is {len(key.encode('utf-8'))} bytes of UTF-8; the "
                f"strictest path limit in play is {PATH_LIMIT_BYTES}")
    for part in key.split("/"):
        size = len(part.encode("utf-8"))
        if size > NAME_COMPONENT_LIMIT_BYTES:
            return (f"path component {part!r} is {size} bytes of UTF-8; vfat and exFAT "
                    f"allow {NAME_COMPONENT_LIMIT_BYTES} per component")
        illegal = sorted({ch for ch in part if ch in VFAT_ILLEGAL_CHARS})
        if illegal:
            listed = ", ".join(repr(ch) for ch in illegal)
            return (f"path component {part!r} contains {listed}; vfat and exFAT cannot "
                    f"store {'it' if len(illegal) == 1 else 'them'}")
        if part.endswith((".", " ")):
            return (f"path component {part!r} ends in {part[-1]!r}; that is stripped on "
                    f"the way to vfat/exFAT, so the name would not round-trip")
    return None


def detect_name_problems(
    keys: Iterable[str],
    *,
    destination_keys: Iterable[str] = (),
) -> List[NameProblem]:
    """Every way these keys cannot be told apart at the destination.

    Two independent checks, both over the FULL key set rather than over the
    objects this pass happens to hash - a collision is a property of the names
    present on the card, not of one file, so a ``--limit`` probe still reports it.

    * Two **source** keys whose :func:`fold_key` matches: on the card they are
      one file. Every member of the group is reported, because naming only one of
      them would be choosing which object silently disappears.
    * One **source** key folding onto a name already at the **destination** -
      but only when the spelling *differs*. This is the dangerous one: the copy
      would land on a name that already exists under a different spelling, which
      ``os.replace`` resolves by clobbering or by failing, depending on the
      server. An **exact** match is the ordinary, healthy case - the object is
      already in custody under its own name - so it is not a collision and must
      never be reported as one.

    Deterministic in the sorted key order, so two runs over an unchanged source
    produce identical ledgers and identical reports.
    """
    ordered = sorted(set(keys))
    destination_index: Dict[str, List[str]] = {}
    for key in sorted(set(destination_keys)):
        destination_index.setdefault(fold_key(key), []).append(key)

    by_fold: Dict[str, List[str]] = {}
    for key in ordered:
        by_fold.setdefault(fold_key(key), []).append(key)

    problems: List[NameProblem] = []
    for folded in sorted(by_fold):
        group = by_fold[folded]
        if len(group) > 1:
            for key in group:
                others = tuple(other for other in group if other != key)
                listed = ", ".join(repr(other) for other in others)
                problems.append(NameProblem(
                    kind=CASE_COLLISION,
                    key=key,
                    folded=folded,
                    others=others,
                    detail=(f"{key!r} is the same file as {listed} on a "
                            f"case-insensitive filesystem"),
                ))
        for key in destination_index.get(folded, ()):
            for source in group:
                if source == key:
                    # Already in custody under its own name. Not a collision.
                    continue
                problems.append(NameProblem(
                    kind=DESTINATION_CASE_COLLISION,
                    key=source,
                    folded=folded,
                    others=(key,),
                    detail=(f"{source!r} is already present at the destination as "
                            f"{key!r}; copying it would hit the existing file"),
                ))
    for key in ordered:
        reason = filename_problem(key)
        if reason is not None:
            problems.append(NameProblem(
                kind=UNREPRESENTABLE_NAME, key=key, folded=fold_key(key),
                detail=reason,
            ))
    problems.sort(key=lambda problem: (problem.key, problem.kind))
    return problems


def destination_keys_in_bundle(bundle: str | Path) -> List[str]:
    """Names the campaign's own destination ledger already records.

    Read-only, and deliberately the *recorded* destination rather than a live
    directory listing: the ledger is the campaign's own account of what is at the
    destination, so it is available to a read-only pass with no extra plumbing and
    it cannot be widened by whatever happens to be mounted right now. An absent
    ledger simply yields no names. A caller that knows the live destination listing
    can pass those names explicitly as well.

    Both the join ``key`` and the recorded ``path`` are taken: they are two names
    the destination really holds (CARD-01's sample ledger records the key
    ``2026_0412_100205_F`` beside the path ``2026/04/12/2026_0412_100205_F.MP4``),
    and a source key is only safe from a collision against every one of them.
    """
    ledger = ledger_dir(bundle) / DESTINATION_LEDGER
    seen: List[str] = []
    known = set()
    for record in read_records(ledger):
        for name in (record.key, record.path):
            if name and name not in known:
                known.add(name)
                seen.append(name)
    return seen


def recorded_problem_keys(ledger: Path) -> Set[str]:
    """Keys ``collisions.jsonl`` already records a problem for.

    Read-only, and used for the same reason :func:`already_hashed` exists: the
    collision ledger is append-only, so a resume must not re-append a problem it has
    already recorded, or a second pass over an unchanged card would double every
    row and break the byte-identical-rerun property.

    Keyed by custody key, not by (key, kind): one key can carry two problems - a
    name can both fold onto its twin and contain an illegal character - and both
    belong in the ledger. What matters for idempotency is only "has this key been
    recorded", which is why the whole pass's findings are appended or skipped
    together.
    """
    import json

    done: Set[str] = set()
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
        key = row.get("key")
        if key:
            done.add(str(key))
    return done


@dataclass(frozen=True)
class HashProgress:
    """What one hashing pass did. Counts are exact; samples are capped."""

    ledger_path: str
    hashed: int = 0
    bytes_read: int = 0
    #: Bytes the ledger already accounted for before this pass; tracked so
    #: verified_bytes can stay cumulative across a resume like verified_files.
    bytes_skipped: int = 0
    skipped_existing: int = 0
    failed: int = 0
    errors: Tuple[str, ...] = ()
    complete: bool = False
    interrupted: bool = False
    repaired_partial_line: bool = False
    limit: Optional[int] = None
    #: Source keys that fold onto another source key. CUMULATIVE over the key set
    #: this pass was given, not a delta: detection is a property of the names, so
    #: a resume must not report fewer collisions than the pass before it.
    collisions: int = 0
    #: Source keys that fold onto a name already recorded at the destination.
    destination_collisions: int = 0
    #: Keys the destination filesystem cannot represent (character or length).
    unrepresentable_names: int = 0
    #: Bounded, deterministic sample lines - one per problem, capped at
    #: ``max_errors``. Per-file detail belongs in ``collisions.jsonl``; this
    #: exists so a human is told *which* names, not only how many.
    name_problems: Tuple[str, ...] = ()

    @property
    def name_problem_count(self) -> int:
        """Every name problem, in one number. Non-zero denies release."""
        return self.collisions + self.destination_collisions + self.unrepresentable_names

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bytes_read": self.bytes_read,
            "bytes_skipped": self.bytes_skipped,
            "case_collisions": self.collisions,
            "complete": self.complete,
            "collision_samples": list(self.name_problems),
            "destination_case_collisions": self.destination_collisions,
            "error_samples": list(self.errors),
            "failed": self.failed,
            "hashed": self.hashed,
            "interrupted": self.interrupted,
            "ledger_path": self.ledger_path,
            "name_problem_count": self.name_problem_count,
            "repaired_partial_line": self.repaired_partial_line,
            "skipped_existing": self.skipped_existing,
            "unrepresentable_names": self.unrepresentable_names,
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


def _problem_record(problem: NameProblem) -> str:
    """One collision-ledger row. Newline-terminated for the same reason as above.

    ``key`` is the byte-accurate custody key and ``status`` is the problem's kind -
    never ``verified``, so no reader can mistake a name problem for custody.
    """
    import json

    return json.dumps({
        "key": problem.key,
        "status": problem.kind,
        "folded": problem.folded,
        "detail": problem.detail,
        "also_collides_with": list(problem.others),
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
    destination_keys: Optional[Iterable[str]] = None,
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

    It also records what the **names** cannot survive, in ``collisions.jsonl``:
    source keys that fold onto each other, source keys that fold onto something
    already recorded at the destination, and keys the destination filesystem
    cannot represent. Detection never changes what is hashed - every key is still
    read and still recorded under its exact spelling - and it never resolves a
    problem: no key is renamed, no winner is picked and none is skipped.

    ``destination_keys`` are the names already at the destination. When omitted,
    the campaign's own ``destination.jsonl`` is used, so the cross-boundary check
    works with no extra plumbing; pass the live listing explicitly to widen it.
    """
    root = ledger_dir(bundle)
    root.mkdir(parents=True, exist_ok=True)
    ledger = root / HASH_LEDGER
    collisions_ledger = root / COLLISION_LEDGER

    done = already_hashed(ledger)
    pending = [(k, p) for k, p in sorted(keys.items()) if k not in done]
    skipped = len(keys) - len(pending)
    skipped_bytes = sum(done.values())

    hashed = 0
    failed = 0
    bytes_read = 0
    errors: List[str] = []
    interrupted = False

    # Over the whole key set, not over `pending`: a collision is a property of the
    # names on the card, so a --limit probe still reports it, and a resume reports
    # the same count the first pass did rather than fewer.
    at_destination = list(destination_keys_in_bundle(bundle))
    if destination_keys is not None:
        at_destination.extend(destination_keys)
    problems = detect_name_problems(keys, destination_keys=at_destination)
    already = recorded_problem_keys(collisions_ledger)
    # Computed once: re-deriving it per record would be quadratic on a card with
    # many badly-named objects.
    name_fields = _name_counters(problems, max_errors)

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
                    bytes_skipped=skipped_bytes, skipped_existing=skipped,
                    failed=failed, errors=tuple(errors),
                    repaired_partial_line=repaired, limit=limit, **name_fields,
                ))

        # Appended after the hashes, and only for problems this bundle has not
        # recorded yet, so a second pass over an unchanged card leaves the ledger
        # byte-identical instead of doubling every row.
        _append_name_problems(collisions_ledger, problems, already)

    result = HashProgress(
        ledger_path=str(ledger),
        hashed=hashed,
        bytes_read=bytes_read,
        bytes_skipped=skipped_bytes,
        skipped_existing=skipped,
        failed=failed,
        errors=tuple(errors),
        complete=not interrupted and failed == 0,
        interrupted=interrupted,
        repaired_partial_line=repaired,
        limit=limit,
        **name_fields,
    )
    if progress is not None:
        progress(result)
    return result


def _append_name_problems(
    ledger: Path,
    problems: Sequence[NameProblem],
    already: Set[str],
) -> int:
    """Append the not-yet-recorded problems to ``collisions.jsonl``. Returns how many.

    Append-only, one fsynced record per row, and idempotent across passes: a key
    already in ``already`` is skipped, because a second pass over an unchanged card
    must leave the ledger byte-identical rather than double every row.
    """
    fresh = [problem for problem in problems if problem.key not in already]
    if not fresh:
        return 0
    with ledger.open("a", encoding="utf-8") as handle:
        if ends_unterminated(ledger):
            # Same rule as the hash ledger: a killed producer's partial line is
            # closed off, never glued onto.
            _terminate_partial_line(handle)
        for problem in fresh:
            handle.write(_problem_record(problem))
            handle.flush()
            os.fsync(handle.fileno())
    return len(fresh)


def _name_counters(problems: Sequence[NameProblem], max_errors: int) -> Dict[str, Any]:
    """The name-problem fields of a :class:`HashProgress`, from the findings.

    Counts are exact; the sample list is capped, because the campaign summary must
    stay bounded no matter how badly a card is named. Per-file detail is in
    ``collisions.jsonl`` - which is why a 10,000-name problem produces 10,000 rows
    there and at most ``max_errors`` lines in the summary.
    """
    counts = {CASE_COLLISION: 0, DESTINATION_CASE_COLLISION: 0, UNREPRESENTABLE_NAME: 0}
    samples: List[str] = []
    for problem in problems:
        counts[problem.kind] = counts.get(problem.kind, 0) + 1
        if len(samples) < max_errors:
            samples.append(problem.sample)
    return {
        "collisions": counts[CASE_COLLISION],
        "destination_collisions": counts[DESTINATION_CASE_COLLISION],
        "unrepresentable_names": counts[UNREPRESENTABLE_NAME],
        "name_problems": tuple(samples),
    }


def to_evidence(result: HashProgress, *, algorithm: str = DEFAULT_ALGORITHM,
                last_checkpoint: Optional[str] = None,
                discovered: Optional[int] = None) -> Dict[str, Any]:
    """The evidence fragments a hash pass contributes.

    Only a *complete* pass sets ``complete``. A partial or failing pass records
    its real counts, so the state machine reports HASHING rather than claiming
    the card is hashed.

    It also records the **inventory it just walked**. That is a measured fact,
    not an assertion, and omitting it left the bundle self-contradictory -
    ``hash.verified_files = 4`` beside ``inventory.discovered_files = 0`` - which
    the state machine quite correctly refused to accept as BLOCKED. A campaign
    could never leave HASHING while its own producer knew the count.

    Name problems ride the **existing** bounded counters rather than a new block:
    they are unresolved campaign errors, and ``errors.unresolved`` is what the
    release gate already refuses on - so an unrepresentable or colliding name can
    never be a pass, without the gate having to learn a new vocabulary. The
    ``errors`` block is written **only when there is something to say**: a clean
    pass must not write ``unresolved: 0`` over a count some other producer
    recorded. Note the consequence, which is the package's ordinary merge rule -
    what this document states wins, so a later hash pass restates the count as the
    hash pass measured it rather than adding to it.
    """
    fragment: Dict[str, Any] = {
        "hash": {
            "algorithm": algorithm,
            # CUMULATIVE, not this pass's delta. On a resume the ledger already
            # proves `skipped_existing` objects; reporting only `hashed` would
            # write 0 over a real count and regress the campaign from HASH
            # COMPLETE back to HASHING on every re-run.
            "verified_files": result.hashed + result.skipped_existing,
            "verified_bytes": result.bytes_read + result.bytes_skipped,
            "complete": result.complete,
            "started": True,
            "failed": result.failed,
            "error_summary": list(result.errors),
            "last_checkpoint": last_checkpoint,
        }
    }
    if discovered is not None:
        known = result.hashed + result.skipped_existing
        fragment["inventory"] = {
            "discovered_files": known,
            # Only claim a complete inventory when the pass actually covered
            # everything it walked; a --limit probe leaves it open.
            "complete": result.complete and known == discovered,
            "verified": result.complete and known == discovered,
            "started": True,
        }
    if result.name_problem_count:
        fragment["errors"] = {
            "unresolved": result.name_problem_count,
            # Capped here as well as in the pass: this is the function that writes
            # the campaign summary, so the bound belongs at the boundary that
            # writes it. The reader caps it a third time, which means a
            # hand-edited or machine-written fragment cannot smuggle one row per
            # file into a bundle.
            "summaries": list(result.name_problems[:MAX_SUMMARY_ENTRIES]),
        }
    return fragment


__all__ = [
    "CASE_COLLISION",
    "CASE_COLLISION_PREFIX",
    "CHUNK_BYTES",
    "DEFAULT_ALGORITHM",
    "DESTINATION_CASE_COLLISION",
    "DESTINATION_COLLISION_PREFIX",
    "HASHED_STATUS",
    "NAME_COMPONENT_LIMIT_BYTES",
    "PATH_LIMIT_BYTES",
    "UNREPRESENTABLE_NAME",
    "UNREPRESENTABLE_NAME_PREFIX",
    "VFAT_ILLEGAL_CHARS",
    "HashProgress",
    "NameProblem",
    "already_hashed",
    "detect_name_problems",
    "destination_keys_in_bundle",
    "digest_file",
    "ends_unterminated",
    "filename_problem",
    "fold_key",
    "hash_source",
    "recorded_problem_keys",
    "to_evidence",
]
