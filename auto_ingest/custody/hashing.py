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

Which fold, and why the old one was wrong
-----------------------------------------

An earlier version of this module normalised with :meth:`str.casefold` and argued
for it on the strength of ``ß``. That argument was half right and it picked the
wrong fold, because *case-insensitive* is not the same operation as *case-folded*.

**The destination compares names with a per-character uppercase (upcase) table**:
a fixed one-to-one lookup from code point to code point, exactly what NTFS keeps
in ``$UpCase`` and what the case-insensitive drivers use. Case *folding* is a
different operation with blind spots in the **opposite** direction, so neither
fold alone is sufficient and this module keeps both (:func:`fold_forms`):

* :meth:`str.casefold` **misses** what the upcase table equates. ``'I'.upper()`` and
  ``'ı'.upper()`` are both ``U+0049``, so on exFAT/NTFS ``ILKAY.mp4`` and
  ``ılkay.mp4`` (U+0131, dotless i) are **one filename** - one overwrites the
  other - while ``'I'.casefold() == 'i'`` and ``'ı'.casefold() == 'ı'``. A
  casefold-only check called two different objects distinct, hashed both, and
  queued a copy onto the name already on the card. :func:`simple_upcase` is the
  primary fold because it models that table.
* :meth:`str.casefold` **catches** what the upcase table does not. ``'ß'.upper()``
  is ``'SS'``, but an upcase table maps the single character ``ß`` to ``ß``; the
  filesystem compares per character, so ``ß`` and ``ss`` are two different files
  there. casefold maps both to ``ss`` and reports a collision vfat would not.

So a collision is reported when the two keys match under **either** fold. The
second fold's extra reports are false positives, and that is the intended
direction, because the two errors are not the same size:

    **over-reporting costs an operator five minutes of renaming;
    under-reporting costs them the card.**

This code decides whether a human may erase one, so it errs toward reporting.

What the model still gets wrong
-------------------------------

*It is a model.* No upcase table is available here, so the comparisons are
Unicode's simple uppercase mappings plus the shipped full-uppercase for every
character where the two agree. Stated precisely:

* **Turkish locale collation is deliberately NOT modelled, because it is not a
  collision.** Windows maps ``i`` to ``U+0130 İ`` under a Turkish locale, so
  Explorer *sorts and displays* ``i.mp4`` beside ``İ.mp4``. But that is display
  collation; the **on-disk** upcase table is locale-invariant, and it maps both
  ``i`` (U+0069) and ``ı`` (U+0131) to ``U+0049 I`` while leaving ``İ``
  (U+0130) mapped to itself. So on the actual destination ``i.mp4`` and
  ``İ.mp4`` are two genuinely different files, and reporting them as a
  collision would be a false positive - a blocker on a real corpus for no
  on-disk ambiguity. The locale-equivalent pair, ``ILKAY``/``ılkay``, *is*
  reported, because there the upcase table really does equate them.
  Measured on this repository's corpus: 67,644 card filenames and 332
  destination filenames, zero non-ASCII, so the Turkish range does not arise
  here at all.
* **No NFC/NFD normalisation, deliberately.** NFC and NFD spellings of one name
  are distinct code-point sequences and are reported as distinct. That is
  **correct for the actual destination** - exFAT and SMB2 store the bytes and
  compare them through the upcase table, so the two spellings are two files -
  and wrong only for APFS, which normalises to NFD on write. Folding them here
  would block every campaign containing a decomposed accented filename against a
  destination that has no such problem. Same measurement: zero non-ASCII
  filenames on the card or at the destination. If this host ever targets APFS,
  that is a separate, deliberate pass - not a guess inside a case fold.
* **One code point in, one out.** Every character folds to exactly one character,
  verified across all of Unicode, because that is what a one-to-one table does.
* **Unknown tables are unknowable.** A future exFAT revision, or a filesystem
  whose table is not Unicode-derived, is not covered by anything written here.
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


def _build_simple_upcase_expansion() -> Dict[str, str]:
    """The 27 characters whose SIMPLE uppercase is a single *other* character.

    :meth:`str.upper` is Unicode's *full* uppercase mapping, which applies
    ``SpecialCasing.txt`` on top of the simple mapping and therefore expands 102
    characters. An upcase table has one slot per code point, so it can only ever do
    the simple mapping, and for the 75 of those 102 with an empty simple-uppercase
    field the simple mapping is the identity - the character maps to itself. Only
    the 27 below have a different one-character answer.

    All 27 are the polytonic Greek small letters with ypogegrammeni whose
    precomposed capital-with-prosgegrammeni exists, and they come in regular
    blocks of eight at a fixed ``+0x08`` offset plus three singles: ``1FB3 -> 1FBC``,
    ``1FC3 -> 1FCC``, ``1FF3 -> 1FFC``. Generated rather than written out so the
    regularity is visible and a block cannot drift.

    Derived from ``UnicodeData.txt`` field 12 (the *simple* uppercase mapping) and
    checked against it character for character. Note what it deliberately does NOT
    do: it does not map ``1F80`` to plain ``Α``. It maps it to ``1F88``, because
    that is what the Unicode table says and therefore what a Unicode-derived upcase
    table does - so ``ᾀ`` and ``Α`` really are two names on the card.
    """
    table: Dict[str, str] = {}
    for start in (0x1F80, 0x1F90, 0x1FA0):
        for offset in range(8):
            table[chr(start + offset)] = chr(start + 0x08 + offset)
    for small, capital in ((0x1FB3, 0x1FBC), (0x1FC3, 0x1FCC), (0x1FF3, 0x1FFC)):
        table[chr(small)] = chr(capital)
    return table


#: Simple-uppercase overrides for characters whose full uppercase expands. Read-only.
#: See :func:`_build_simple_upcase_expansion` and :func:`simple_upcase`.
SIMPLE_UPPERCASE_EXPANSION: Dict[str, str] = _build_simple_upcase_expansion()


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


def simple_upcase(text: str) -> str:
    """Uppercase ``text`` the way a one-to-one upcase table does: no expansion.

    One character in, one character out, always. :meth:`str.upper` is the *full*
    mapping and it expands - ``ß`` becomes ``SS``, ``ﬁ`` becomes ``FI``, the
    polytonic Greek small letters with ypogegrammeni become a capital plus
    ``U+0399`` - none of which a per-character table can do, because a table has
    exactly one slot per code point. Full expansion here would invent collisions
    the destination does not have.

    Only 102 characters expand at all. For 75 of them Unicode's *simple* uppercase
    is the character unchanged - ``ß``, ``ﬁ``, ``ﬄ``, ``ŉ``, the Armenian
    ligatures, ``ǰ``, ``ẖ`` - which is what
    :data:`SIMPLE_UPPERCASE_EXPANSION`'s absence of them means. The other 27 are
    the polytonic Greek letters whose precomposed capital exists, and they are
    listed there. Every one of the 102 is covered.

    The fast path is not a guess: no character's uppercase is *shorter* than itself,
    so a whole string whose uppercased length equals its own length had every
    character mapped one-to-one, and the expansion table cannot apply to it.
    """
    upcased = text.upper()
    if len(upcased) == len(text):
        return upcased
    return "".join(_simple_upcase_char(ch) for ch in text)


def _simple_upcase_char(ch: str) -> str:
    """The one-character simple uppercase of ``ch``. Private; see :func:`simple_upcase`."""
    upcased = ch.upper()
    if len(upcased) == 1:
        return upcased
    return SIMPLE_UPPERCASE_EXPANSION.get(ch, ch)


def fold_key(key: str) -> str:
    """The comparison form of a custody key: simple upcased, per path component.

    Uppercasing is applied to each component separately because it is each
    component the destination compares, and the separators are already structural.

    :func:`simple_upcase` rather than :meth:`str.casefold`: the destination resolves
    a name through a one-to-one upcase table, and that table sends ``I`` and ``ı``
    (U+0131) to the same code point. casefold keeps them apart, so a casefold-only
    comparison called one filename two objects. :meth:`str.lower` is wrong in the
    other direction again and is never used.

    This is the *primary* comparison form and the one recorded as ``folded``. It is
    not the whole test - :func:`fold_forms` adds :meth:`str.casefold`, which catches
    ``ß``/``ss`` and ``ﬁ``/``fi`` that no upcase table equates.

    The result is for COMPARISON ONLY. Stored keys - in the ledgers and in
    evidence - stay byte-accurate to the filename, because a custody key is what
    re-opens the object on the card.
    """
    return "/".join(simple_upcase(part) for part in key.split("/"))


def fold_forms(key: str) -> Tuple[str, ...]:
    """Every form under which two keys are called the same name. One, or two.

    The upcase form first (:func:`fold_key`), then :meth:`str.casefold` - omitted
    when the two already agree, which is every ASCII key, so the common case costs
    one form and one string.

    Two folds rather than one because neither is the destination's behaviour on its
    own, and a collision is declared when **either** form matches. That is
    deliberately one-sided: the extra reports are false positives a human resolves
    in five minutes by renaming, while a missed report is a file overwritten on the
    card that no one ever sees again. See the module docstring for which pairs each
    fold catches and which it misses.
    """
    upcased = fold_key(key)
    folded = "/".join(part.casefold() for part in key.split("/"))
    return (upcased,) if upcased == folded else (upcased, folded)


def _fold_index(keys: Iterable[str]) -> Dict[str, List[str]]:
    """Group ``keys`` by every fold form they share. Form -> the keys holding it.

    The index both comparison checks need, and the reason they cannot simply look
    one form up: a key appears once per form it has, so a collision under the
    second fold is found by an entry under the second form. A key whose two forms
    coincide appears once.
    """
    index: Dict[str, List[str]] = {}
    for key in sorted(set(keys)):
        for form in fold_forms(key):
            index.setdefault(form, []).append(key)
    return index


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

    * Two **source** keys that share a :func:`fold_forms` form: at the destination
      they are one file. Every member of every group is reported, because naming
      only one of them would be choosing which object silently disappears.
    * One **source** key folding onto a name already at the **destination** -
      but only when the spelling *differs*. This is the dangerous one: the copy
      would land on a name that already exists under a different spelling, which
      ``os.replace`` resolves by clobbering or by failing, depending on the
      server. An **exact** match is the ordinary, healthy case - the object is
      already in custody under its own name - so it is not a collision and must
      never be reported as one.

    A collision is a match under **either** fold form (:func:`fold_forms`), so
    ``I``/``ı`` - one filename to the destination's upcase table - is found even
    though casefold keeps them apart, and ``ß``/``ss`` is reported even though the
    destination's table keeps them apart. The second of those is a false positive
    and is meant to be: see the module docstring on why over-reporting is the safe
    direction. Because a key holds one row per kind regardless of how many groups
    it sits in, one key found by two folds still gets one row listing both partners.

    ``others`` names the keys that share a form with this one and is NOT made
    transitive. The join of two equivalence relations need not be transitive - if
    ``b`` shares the upcase form with ``a`` and the casefold form with ``c``, then
    ``a`` and ``c`` are two genuinely different files at the destination, and
    calling them a collision would invent a problem out of a name the operator never
    has to touch. ``b``'s row already names both.

    Deterministic in the sorted key order, so two runs over an unchanged source
    produce identical ledgers and identical reports.
    """
    ordered = sorted(set(keys))
    destination_index = _fold_index(destination_keys)
    source_index = _fold_index(ordered)

    # Every key's direct partners under either form, gathered before any row is
    # written so one key in two groups still yields a single row naming both.
    partners: Dict[str, Set[str]] = {key: set() for key in ordered}
    for group in source_index.values():
        if len(group) > 1:
            for key in group:
                partners[key].update(other for other in group if other != key)

    problems: List[NameProblem] = []
    for key in ordered:
        # Computed once and reused: it is both the `folded` field and the lookup
        # key for the destination check, and a full card is 67,000 of these.
        forms = fold_forms(key)
        others = tuple(sorted(partners[key]))
        if others:
            listed = ", ".join(repr(other) for other in others)
            problems.append(NameProblem(
                kind=CASE_COLLISION,
                key=key,
                folded=forms[0],
                others=others,
                detail=(f"{key!r} is the same file as {listed} on a "
                        f"case-insensitive filesystem"),
            ))
        at_destination: Set[str] = set()
        for form in forms:
            at_destination.update(destination_index.get(form, ()))
        for existing in sorted(at_destination):
            if existing == key:
                # Already in custody under its own name. Not a collision.
                continue
            problems.append(NameProblem(
                kind=DESTINATION_CASE_COLLISION,
                key=key,
                folded=forms[0],
                others=(existing,),
                detail=(f"{key!r} is already present at the destination as "
                        f"{existing!r}; copying it would hit the existing file"),
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
                discovered: Optional[int] = None,
                excluded: Tuple[str, ...] = ()) -> Dict[str, Any]:
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
            # Cumulative for the same reason as hash.verified_files above: a
            # resumed pass measures the skipped bytes from the ledger it resumed
            # from, so this is the total across every pass, not this pass's delta.
            # It was previously absent entirely, which left a resumed campaign
            # claiming 67,644 files and ZERO bytes - and capacity.py:85 scales
            # its requirement by discovered_bytes, so the whole copy looked free.
            "discovered_bytes": result.bytes_read + result.bytes_skipped,
            # Only claim a complete inventory when the pass actually covered
            # everything it walked; a --limit probe leaves it open.
            "complete": result.complete and known == discovered,
            "verified": result.complete and known == discovered,
            "started": True,
        }
    if excluded:
        # Recorded, bounded, and carrying the patterns as well as the keys - so an
        # exclusion is auditable from the campaign alone, and the release gate can
        # compare it against what the policy actually declares (an exclusion the
        # policy does not honour is a blocker, same rule as hash exemptions).
        fragment["errors"] = {
            "declared_exclusions": sorted({pattern.split("*")[0].rstrip("/")
                                           for pattern in excluded}),
            "excluded_objects": len(excluded),
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
    "SIMPLE_UPPERCASE_EXPANSION",
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
    "fold_forms",
    "fold_key",
    "hash_source",
    "recorded_problem_keys",
    "simple_upcase",
    "to_evidence",
]
