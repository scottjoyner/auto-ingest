"""auto_ingest.custody.release_source - the ONE place a source object may be unlinked.

Until this module existed, the package could ingest a card, verify it byte-for-byte
at a destination and still had no way to reclaim the card: nothing in
``auto_ingest.custody`` could delete anything. That was deliberate, it was
enforced by an AST test, and it was correct as long as nobody had built a
deletion path at all. This module is that path - and the only one.

What makes it narrow
--------------------

* **It is the only module in the package that may unlink anything.** Every other
  custody module is banned from ``shutil``, ``os.remove``, ``os.unlink``,
  ``os.rmdir``, ``os.removedirs``, ``rmtree``, ``os.chmod`` and ``os.truncate``,
  and ``executor.py`` - the only module that writes bytes - is still held to the
  original ban unchanged. ``tests/test_custody_legacy_watchers.py`` pins that:
  exactly one module in the directory may call an unlink-shaped function, and
  inside this module exactly one call site is permitted, and it must be
  ``Path(...).unlink()`` - never ``os.unlink``, never a recursive delete. A
  tree-delete cannot be introduced here without that test failing.
* **Default is a proposal, never a deletion.** Without ``execute=True`` nothing is
  unlinked and no file is created; the caller gets the exact set of paths that
  *would* go. A malformed or incoherent ledger yields **no** proposal at all
  rather than a partial one, following ``ledger.ReconciliationResult.proposal()``:
  a proposal covering only the rows that survived would read as custody for the
  whole card.
* **It calls the real release gate.** ``evaluate_release`` and ``derive_state``
  are invoked, never re-implemented or re-derived. If the campaign is not
  ``SAFE_TO_RELEASE`` the answer is a refusal carrying the gate's own
  ``Blocker`` objects, so a green light here can never be weaker than the gate.
* **The deletable set is derived, never supplied.** It comes from the
  intersection of the source hash ledger and the *verification records* in
  ``ledgers/destination.jsonl`` - never from the card's own directory listing,
  never from a path handed in by the caller. Presence at the destination is
  necessary but not sufficient: a ``copied`` record proves bytes were written, not
  that they are correct, so ``copied`` is deliberately **excluded** here (this is
  the ``--ignore-existing`` defect the subsystem exists to prevent), and a
  per-key digest *and* size agreement is required on top of it.
* **The destination copy is re-verified immediately before each unlink.** The
  ledger record is from an earlier pass; the destination bytes can rot since. The
  object is streamed and re-hashed at delete time, and a mismatch refuses the
  deletion. Deleting a sound source while the destination is corrupt is the one
  catastrophe this subsystem exists to prevent, so the last thing before the
  unlink is a fresh read of the bytes that are supposed to replace it.
* **Every path is resolved defensively.** ``_resolve_under_root`` refuses an
  empty key, an absolute key, any ``..`` component (checked lexically, before
  anything touches the disk) and any symlink lying between the root and the
  object. A key that cannot be trusted refuses the *whole* run, because a ledger
  containing ``../../etc/passwd`` is evidence of tampering and continuing to
  delete other rows from a tampered ledger is not conservative.
* **The audit ledger is append-only, and it is written BEFORE the unlink.** Every
  deletion, every already-absent object, every failure and every refusal is
  appended to ``ledgers/release.jsonl``, newline-terminated, ``flush()`` +
  ``os.fsync()`` per record, exactly as ``hashing.py`` does. It is never rewritten
  and never truncated - the audit is the only record that bytes are gone, so
  erasing it would erase the fact. A partial final line means an interrupted run,
  so the next run terminates that line rather than gluing onto it, and
  ``read_audit()`` reports ``coherent=False`` until it is rebuilt.
* **The record of an intent to delete is durable before the delete happens.**
  This is the load-bearing ordering in the module. Unlink-then-record loses the
  record whenever the append fails - a full volume, a read-only remount, a quota -
  and then the card holds fewer files than any record admits while
  ``read_audit()`` still reports ``coherent=True``. So the ``deleted`` row is
  appended and fsynced *first*, and the unlink only runs once that append
  succeeded. A pass whose append fails stops there: the object it was about to
  delete is still on the card, and nothing after it is attempted. The bias this
  buys is deliberate and one-directional - if the unlink then fails, the ledger
  holds a ``deleted`` **and** a ``failed`` row for that key, so the audit can
  over-report a deletion that did not happen, but it can never under-report one
  that did. For an irreversible ledger, over-reporting is the survivable error.
* **A release that cannot be recorded is not performed.** Before the loop the
  pass asks the audit's filesystem whether it plausibly has room for every row
  the pass could write, and refuses the whole run if it does not; each row is
  checked again immediately before it is written; and an audit that cannot be
  opened at all is a refusal rather than an exception. Nothing is deleted on the
  strength of a record that could not be written.
* **Evidence is bounded.** Deletion goes into counters and capped samples, never
  one row per file: a 10,000-key card produces the same small report as a
  3-object one. The *ledger* is per-object on purpose - it is the record, not the
  report - so only ``to_dict()`` and the samples are capped.

What it is forbidden to do
--------------------------

* delete anything except the exact keys its own plan derived;
* delete anything when the release gate is closed;
* delete a directory, a symlink, or anything outside the source root;
* delete anything from the source root it was also given as the destination root;
* rewrite, truncate or reorder the audit ledger;
* unlink before the deletion has been recorded in the audit;
* derive the release verdict itself, or set ``source_release_allowed``;
* write a campaign evidence document - see ``LIMITS`` below.

The evidence it contributes
---------------------------

:func:`to_evidence` is the fragment this pass contributes to the campaign's
bounded evidence, and it follows the convention ``hashing.to_evidence``,
``verify.to_evidence`` and ``executor.to_evidence`` established: a module-level
function, returning a plain dict of blocks, pure in its argument.

Its counters are read from the **append-only audit ledger**, not from this pass.
That is the same cumulative-not-delta rule the hash and verify producers follow,
and for the same reason - reporting this pass's delta would write ``0`` over a
real count on every resume and quietly lose the record that bytes are gone.
Applying the fragment twice therefore restates the same numbers instead of
doubling them.

The block it writes is ``source_release``, and its direction is load-bearing:
releasing the source is a fact about the objects that used to be on the card, not
about the destination. It never touches ``destination`` or ``reconciliation``,
because a release is not verification - folding it in would let deleting the card
read as proof that the destination holds the bytes.

Symlinks
--------

A key must not resolve *through* a symlink, in either direction. The test is
``resolved == lexically_joined``: because the join already contains no ``..``, any
mismatch means a link is on the path, and where a link points is not something to
discover by following it immediately before a delete. So a symlinked object, and
any key that would traverse a link, is refused with the key named - the operator
sees which entry to fix. (A symlink *pointing* elsewhere is likewise refused:
unlinking it would remove the link, not the data, and a link whose target is
outside the card is not the card's byte at all.)

LIMITS - what this module deliberately does not do
--------------------------------------------------

**It never writes the evidence document.** :func:`to_evidence` returns the
fragment; applying it is the command layer's job, exactly as it is for
``custody hash``, ``custody verify`` and ``custody execute``. Nothing here opens
``evidence.json``, so a library caller that never applies the fragment simply does
not get the record - which is the honest outcome, not a silent one.

The nearest thing to a vocabulary for "the source was released" used to be
zeroing ``inventory``, and that would make the bundle self-contradictory
(``discovered_files = 0`` beside ``hash.verified_files = 5``), which the state
machine quite correctly reports as ``BLOCKED``. ``inventory`` is what the
campaign *walked*, a historical measurement; it does not shrink because files
were later unlinked. Hence a separate block instead.

That block is observational only. ``release.py`` reads it to refuse and never to
allow, so recording a release cannot turn a ``BLOCKED`` campaign into a pass.
Re-running a release stays idempotent (every key is now accounted for and nothing
is unlinked again) and the derived state is unchanged: custody proven is still
custody proven after the source is gone. **A new derived state is deliberately
not added for this** - see ``states.CampaignState``, which stays at twelve
members, and the note in ``tests/test_custody_release_evidence.py`` that pins
the reason.

ABSENT - released, or lost?
---------------------------

The audit used to declare an ``absent`` event and never write one, which left a
hole in the only durable record of a card. For an irreversible deletion ledger
that hole matters: a key that is in the hash ledger, proven present at the
destination, and *no longer on the card* has two possible histories - an earlier
pass released it, or it was removed by something else and never existed again.
A reader of ``release.jsonl`` could not tell them apart, because the only event
that would have said so had no producer.

So ``absent`` has one, and its scope is exactly the ambiguous case. A key earns
an ``absent`` row when it is

* proven in the source hash ledger (``verified``/``hashed``), **and**
* proven at the destination (``verified``/``verified_at_destination``), **and**
* gone from the source root, **and**
* not already accounted for by a ``deleted`` or ``absent`` row.

That last clause is what keeps this honest rather than noisy. A key this module
deleted already has a ``deleted`` row, so its absence is explained; writing an
``absent`` row beside it would grow the ledger on every no-op re-run and inflate
a counter without recording anything new. A key with neither row is the one an
auditor needs to see, and the ``absent`` row is the difference between "released
by an earlier pass" and "gone, and nobody here did it".

``already_absent`` on the plan is therefore a **plan-time figure** and ``absent``
on :class:`AuditRead` is an **audit counter**, and they are allowed to differ:
the plan counts every already-gone key it classified, the audit counts the ones
whose absence was not already on the record. ``to_evidence`` reports the audit's
figure, never the plan's. An ``absent`` finding does not close the release gate
(``SourceReleaseEvidence.clean`` ignores it) - the bytes are verified at the
destination, so nothing is at risk; it is a completeness note about the card,
not a failure.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from .campaign import Campaign
from .evidence import CampaignEvidence
from .hashing import digest_file
from .ledger import (
    DESTINATION_LEDGER,
    HASH_LEDGER,
    SOURCE_VERIFIED_STATUSES,
    LedgerRecord,
    ledger_dir,
    read_records,
    summarize_ledger,
)
from .machine import derive_state
from .policy import MAX_SUMMARY_ENTRIES, CustodyPolicy
from .release import Blocker, evaluate_release
from .states import CampaignState

#: Append-only audit ledger, under the campaign bundle's ``ledgers/``. Never
#: rewritten: it is the only record that a source object was destroyed.
RELEASE_LEDGER = "release.jsonl"

#: The only mode this module may ever open a file for writing. Named so a call
#: site cannot introduce a truncating write by accident, and so the audit test in
#: tests/test_custody_legacy_watchers.py can assert one mode rather than grep.
AUDIT_MODE = "a"

#: Destination-side statuses that count as *proof*. A ``copied`` record is
#: deliberately absent: it proves the executor wrote bytes, not that they are the
#: right bytes. Requiring a verification record is the whole point of intersecting
#: the ledgers rather than walking the destination tree.
DESTINATION_PROVEN_STATUSES = ("verified", "verified_at_destination")

#: Audit events. One record per decision, never a batch summary standing in for
#: per-object ones. Every event listed here has a producer in this module;
#: :data:`AUDIT_EVENTS` is the closed set the reader trusts.
DELETED = "deleted"
REFUSED = "refused"
ABSENT = "absent"
FAILED = "failed"
RUN = "run"

#: The closed set of events this module writes. A row naming anything else was put
#: there by something else, so :func:`_read_audit_path` counts it once as
#: ``unknown`` instead of letting it become a new key in ``by_event`` - that dict
#: is part of a report that promises a fixed shape, and an event name is attacker
#: controlled once a ledger is a file on disk.
AUDIT_EVENTS = (ABSENT, DELETED, FAILED, REFUSED, RUN)

#: Events that settle a key's fate: once one of these is on the ledger, the object
#: is accounted for and a later pass has nothing new to record about it. This is
#: what keeps ``absent`` from re-reporting a key an earlier pass already deleted.
SETTLING_EVENTS = (ABSENT, DELETED)

#: Bytes held back in the pre-flight for a record whose ``detail`` is not yet
#: known. Every ``detail`` this module writes is a bounded reason string.
_DETAIL_RESERVE = 64

#: Bytes assumed for one keyed ``refused`` row, whose key and path come from a
#: ledger rather than from a derived object, so their length is not known up front.
_REFUSAL_RESERVE_BYTES = 512

#: Slack on top of the computed append: the run summary, whatever the filesystem
#: rounds a write up to, and the record a torn tail may need. The pre-flight is a
#: guard, not a proof - the per-record room check and the stop-on-append-failure
#: below are what actually bound the damage.
_AUDIT_SLACK_BYTES = 4096

#: Report modes, mirroring how ``custody execute`` labels a dry run.
MODE_PROPOSAL = "proposal"
MODE_REFUSED = "refused"
MODE_EXECUTED = "executed"

#: Exit codes, mirroring auto_ingest.custody.cli exactly. Duplicated rather than
#: imported because cli.py imports this module, and
#: tests/test_custody_release_source.py asserts the two sets stay identical.
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_GATE_CLOSED = 3


@dataclass(frozen=True)
class SourceObject:
    """One source object cleared for removal, already resolved inside the root."""

    key: str
    path: str
    size: int
    digest: str

    def to_dict(self) -> Dict[str, Any]:
        return {"digest": self.digest, "key": self.key, "path": self.path, "size": self.size}


@dataclass(frozen=True)
class ReleasePlan:
    """The exact set of paths that *would* be removed. A description, never an act.

    ``objects`` is the full set, because a proposal that truncated itself would be
    a lie. ``to_dict()`` is what stays bounded: it reports counts plus a capped
    sample, so a 67k-object card produces the same small report as a 3-object one.
    """

    campaign_id: str
    state: str
    source_root: str
    release_allowed: bool
    considered: int = 0
    deletable: int = 0
    deletable_bytes: int = 0
    #: Plan-time figure: keys the ledgers classify and that are already gone from
    #: the source. Deliberately *not* the same number as
    #: :attr:`AuditRead.absent`, which counts only the already-gone keys whose
    #: absence was not already explained by a ``deleted`` or ``absent`` row.
    already_absent: int = 0
    objects: Tuple[SourceObject, ...] = ()
    #: The same keys as :attr:`absent_objects`, resolved but not present. They are
    #: not offered as deletion candidates - there is nothing there to unlink - but
    #: the executed pass still records each one that the audit has not already
    #: accounted for, so a later reader can tell "released" from "lost".
    absent_objects: Tuple[SourceObject, ...] = ()
    #: Keys a whole-run *blocker* named, so a refusal says which objects were
    #: near-missed rather than only that something was refused. Small by
    #: construction: the classifier stops at the first key it cannot trust.
    near_missed: Tuple[str, ...] = ()
    ledger_notes: Tuple[str, ...] = ()
    blockers: Tuple[Blocker, ...] = ()
    #: Every per-key refusal, uncapped: the audit writes one ``refused`` row each,
    #: because a report that names 20 of 5000 refusals is a report that says the
    #: release was clean. ``to_dict()`` and the samples cap it; the ledger does not.
    refusals: Tuple[Blocker, ...] = ()
    #: The key each entry of :attr:`refusals` is about, index for index. Carried
    #: separately rather than parsed back out of the blocker's prose: the key is a
    #: fact, and the audit needs it in the ``key`` field, not in a sentence.
    refusal_keys: Tuple[str, ...] = ()
    mode: str = MODE_PROPOSAL

    @property
    def usable(self) -> bool:
        """True when nothing blocks and an exact set is offered.

        A blocked plan is never usable, and its ``objects`` is empty: a partial
        proposal is worse than none, because the rows it omits would read as
        proven custody.
        """
        return self.release_allowed and not self.blockers

    @property
    def keys(self) -> Tuple[str, ...]:
        return tuple(o.key for o in self.objects)

    @property
    def paths(self) -> Tuple[str, ...]:
        return tuple(o.path for o in self.objects)

    @property
    def absent_keys(self) -> Tuple[str, ...]:
        return tuple(o.key for o in self.absent_objects)

    @property
    def refusal_entries(self) -> Tuple[Tuple[str, Blocker], ...]:
        """``(key, blocker)`` for every per-key refusal, index-aligned.

        A key the caller did not supply pairs with the empty string, which the
        reader counts as an unattributed refusal rather than dropping.
        """
        keys = self.refusal_keys + ("",) * max(len(self.refusals) - len(self.refusal_keys), 0)
        return tuple(zip(keys, self.refusals))

    def to_dict(self, *, max_samples: int = MAX_SUMMARY_ENTRIES) -> Dict[str, Any]:
        return {
            "absent_key_samples": list(self.absent_keys)[:max_samples],
            "already_absent": self.already_absent,
            "blockers": [b.to_dict() for b in self.blockers],
            "campaign_id": self.campaign_id,
            "considered": self.considered,
            "deletable": self.deletable,
            "deletable_bytes": self.deletable_bytes,
            "ledger_notes": list(self.ledger_notes),
            "mode": self.mode,
            "near_missed_keys": list(self.near_missed)[:max_samples],
            "refusal_codes": [b.code for b in self.refusals][:max_samples],
            "refused_count": len(self.refusals),
            "release_allowed": self.release_allowed,
            "sample_paths": list(self.paths)[:max_samples],
            "source_root": self.source_root,
            "state": self.state,
            "usable": self.usable,
        }


@dataclass(frozen=True)
class AuditRead:
    """Bounded read of the append-only audit ledger. Fails closed on a torn tail."""

    present: bool = False
    path: str = ""
    records: int = 0
    deleted: int = 0
    #: Cumulative bytes destroyed. A single integer for the whole card: the
    #: per-object sizes stay in the ledger, which is where the detail belongs.
    deleted_bytes: int = 0
    refused: int = 0
    absent: int = 0
    failed: int = 0
    malformed: int = 0
    truncated: bool = False
    #: Refusal records that named no object - a whole-run refusal, where the gate
    #: was closed and not a single candidate was considered. Counted apart from
    #: :attr:`refused` because "the gate said no" and "this object could not be
    #: released" are different facts, and only the second one means bytes are
    #: still on the card for a reason.
    unattributed: int = 0
    #: Rows naming an event outside :data:`AUDIT_EVENTS`. Counted rather than
    #: indexed, so a hand-written ledger cannot widen ``by_event`` - which is part
    #: of a report that promises a fixed shape.
    unknown: int = 0
    #: Whole-run summary rows seen, and the figures from the most recent one. A pass
    #: that proposed more than it handled did not finish, and the ledger says so
    #: even when every row it managed to write is a clean deletion.
    runs: int = 0
    run_proposed: int = 0
    run_handled: int = 0
    #: Keys the ledger has already accounted for: a ``deleted`` or ``absent`` row
    #: exists for each. Deliberately never serialised - it is one entry per card,
    #: and this is a bounded report - and never compared, so two reads of the same
    #: ledger still compare equal. The executed pass uses it to avoid recording a
    #: fact the ledger already holds.
    settled_keys: FrozenSet[str] = field(
        default_factory=frozenset, compare=False, repr=False,
    )
    by_event: Dict[str, int] = field(default_factory=dict)

    @property
    def coherent(self) -> bool:
        """False when a producer died mid-append.

        The rows that *did* land look perfectly healthy, so a reader that ignored
        this would report a confident partial history of an interrupted release.
        """
        return self.present and self.malformed == 0 and not self.truncated

    @property
    def object_refusals(self) -> int:
        """Refusals that named an object, i.e. objects left on the card on purpose."""
        return max(self.refused - self.unattributed, 0)

    @property
    def finished(self) -> bool:
        """False when the last pass proposed more objects than it reached a verdict on.

        The audit cannot see how many objects the pass *would* have deleted, only
        how many the pass itself claimed to propose - and that claim is on the
        record, which is the point. A truncated pass therefore stays visible as an
        unfinished release even though every row it managed to write is a clean
        deletion.
        """
        return self.run_proposed <= self.run_handled

    def to_dict(self) -> Dict[str, Any]:
        return {
            "absent": self.absent,
            "by_event": dict(sorted(self.by_event.items())),
            "coherent": self.coherent,
            "deleted": self.deleted,
            "deleted_bytes": self.deleted_bytes,
            "failed": self.failed,
            "finished": self.finished,
            "malformed_lines": self.malformed,
            "object_refusals": self.object_refusals,
            "path": self.path,
            "present": self.present,
            "records": self.records,
            "refused": self.refused,
            "runs": self.runs,
            "truncated": self.truncated,
            "unattributed": self.unattributed,
            "unknown_events": self.unknown,
        }


@dataclass(frozen=True)
class ReleaseResult:
    """What one release pass did, or what it refused to do.

    All counters, plus capped samples. A card with 10,000 deletable objects and a
    card with 3 produce the same report size.
    """

    campaign_id: str
    mode: str
    audit_path: str
    gate_open: bool
    state: str
    considered: int = 0
    proposed: int = 0
    proposed_bytes: int = 0
    deleted: int = 0
    deleted_bytes: int = 0
    already_absent: int = 0
    failed: int = 0
    refused: int = 0
    audit_unterminated_repaired: bool = False
    #: True when the audit refused a row this pass needed. Nothing is deleted on the
    #: strength of a record that could not be written, so this means the pass
    #: stopped short - and it forces ``complete`` False whatever else it managed,
    #: because a deletion the ledger could not record is not a finished release.
    audit_append_failed: bool = False
    blockers: Tuple[Blocker, ...] = ()
    ledger_notes: Tuple[str, ...] = ()
    error_samples: Tuple[str, ...] = ()
    key_samples: Tuple[str, ...] = ()
    #: The append-only audit re-read *after* this pass. :func:`to_evidence`
    #: publishes from here rather than from the per-pass counters above, so the
    #: fragment is cumulative and re-importing it cannot double-count. A proposal
    #: carries the previous pass's view, which is what makes applying a fragment
    #: from a dry run idempotent too.
    audit_summary: AuditRead = field(default_factory=AuditRead)

    @property
    def handled(self) -> int:
        """Proposed objects this pass reached a decision on."""
        return self.deleted + self.failed + self.refused

    @property
    def complete(self) -> bool:
        """Nothing left dangling.

        A proposal is complete when it is not blocked: it names exactly the set
        that would go. An executed pass is complete when every proposed object
        was either removed or refused/failed on with nothing silently skipped -
        and when the audit took every row the pass tried to give it.
        """
        if self.mode == MODE_PROPOSAL:
            return self.gate_open and not self.blockers
        if self.audit_append_failed:
            return False
        return self.failed == 0 and self.refused == 0 and self.handled >= self.proposed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "already_absent": self.already_absent,
            # The cumulative audit view. Bounded: every event name comes from the
            # fixed set this module emits, so `by_event` cannot grow with the card.
            "audit": self.audit_summary.to_dict(),
            "audit_append_failed": self.audit_append_failed,
            "audit_path": self.audit_path,
            "audit_unterminated_repaired": self.audit_unterminated_repaired,
            "blockers": [b.to_dict() for b in self.blockers],
            "campaign_id": self.campaign_id,
            "complete": self.complete,
            "considered": self.considered,
            "deleted": self.deleted,
            "deleted_bytes": self.deleted_bytes,
            "error_samples": list(self.error_samples),
            "failed": self.failed,
            "gate_open": self.gate_open,
            "handled": self.handled,
            "key_samples": list(self.key_samples),
            "ledger_notes": list(self.ledger_notes),
            "mode": self.mode,
            "proposed": self.proposed,
            "proposed_bytes": self.proposed_bytes,
            "refused": self.refused,
            "state": self.state,
        }


# ---------------------------------------------------------------------------
# defensive path resolution
# ---------------------------------------------------------------------------
def _resolve_under_root(root: str | Path, key: str) -> Optional[Path]:
    """Resolve ``key`` under ``root``, or ``None`` when it cannot be trusted.

    Keys come out of a ledger, which is a file on disk. Five things are refused:

    * an empty or whitespace-only key;
    * an absolute key (a leading ``/``);
    * any ``..`` component - checked lexically, before anything touches the disk,
      because the safe reading is to reject the spelling rather than to normalise
      a key that was trying to escape;
    * a symlink anywhere between the root and the object;
    * anything whose real path leaves the root.

    **Symlinks are refused, not followed.** The test is ``resolved ==
    lexically_joined``: the join already contains no ``..``, so any mismatch means
    a link is on the path. Where a link points is not something to learn by
    following it in the instant before a delete, and a link whose target is
    outside the card is not the card's byte at all.
    """
    if not isinstance(key, str) or not key.strip():
        return None
    pure = PurePosixPath(key)
    if not pure.parts or pure.is_absolute() or pure.root or pure.drive:
        return None
    if any(part == ".." for part in pure.parts):
        return None
    base = Path(root).resolve()
    joined = base.joinpath(*pure.parts)
    try:
        real = joined.resolve()
    except OSError:
        return None
    if real != joined:
        return None
    try:
        real.relative_to(base)
    except ValueError:
        return None
    return joined


def _same_place(left: Path, right: Path) -> bool:
    """True when two roots are the same directory once symlinks are resolved.

    Used to refuse a "destination" that is the card. ``resolve()`` on a path that
    does not exist still normalises it, so an operator who has not created the
    destination yet gets a plain ``False`` rather than an exception.
    """
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return False


def _proven_index(path: Path, statuses: Tuple[str, ...]) -> Dict[str, LedgerRecord]:
    """``key -> the last record written for it``, keeping only proven statuses.

    Last-write-wins on purpose. ``custody verify`` appends a *fresh* record for a
    key on every ``--recheck``, so a key that verified on pass 1 and mismatched on
    pass 2 has two rows. Keeping the first would treat a retracted proof as
    current, and a retraction must always win.
    """
    latest: Dict[str, LedgerRecord] = {}
    for record in read_records(path):
        if record.key:
            latest[record.key] = record
    return {k: v for k, v in latest.items() if v.status in statuses}


def _blocker_summary(blockers: Tuple[Blocker, ...], limit: int) -> str:
    """Comma-joined codes, capped. Used as an audit record's ``detail``."""
    return ",".join(b.code for b in blockers[:limit])


# ---------------------------------------------------------------------------
# the plan (pure: reads, decides, proposes; never writes)
# ---------------------------------------------------------------------------
def plan_release(
    bundle: str | Path,
    source_root: str | Path,
    destination_root: str | Path,
    *,
    campaign: Campaign,
    evidence: CampaignEvidence,
    policy: CustodyPolicy | None = None,
    recheck: bool = False,
    max_samples: int = MAX_SUMMARY_ENTRIES,
) -> ReleasePlan:
    """Derive the exact set of source objects that may be removed. Read-only.

Refuses - offering **no** partial set - when the release gate is closed, when
either ledger is absent or incoherent, when the hash ledger does not cover the
recorded inventory, when any candidate key cannot be resolved defensively, or
when the source root and the destination root are the same place.

``recheck`` additionally re-hashes each destination object now instead of
trusting the recorded verification. It is off by default because a dry run
over a full card would re-read the whole destination; the executed pass always
re-verifies per object immediately before unlinking, whether or not this flag
was set.

``absent_objects`` and ``near_missed`` are findings, not proposals: they name
what is *not* there to delete, and the executed pass turns them into audit
rows so a later reader can tell "released by an earlier pass" from "gone, and
nobody here did it".
    """
    policy = policy or CustodyPolicy()
    root = Path(source_root)
    dest_root = Path(destination_root)
    blockers: List[Blocker] = []
    notes: List[str] = []

    # The real gate and the real state machine, invoked - never re-derived. The
    # conjunction below is exactly what store.CampaignStatus.source_release_allowed
    # reports, so a green light here cannot be weaker than `custody status`.
    decision = evaluate_release(campaign, evidence, policy)
    derivation = derive_state(campaign, evidence, policy)
    state = derivation.state
    allowed = decision.allowed and state is CampaignState.SAFE_TO_RELEASE
    if not allowed:
        blockers.extend(decision.blockers)
        if state is not CampaignState.SAFE_TO_RELEASE:
            blockers.append(Blocker(
                "state_is_not_safe_to_release",
                f"derived state is {state.value}",
                "reach SAFE_TO_RELEASE through the normal pipeline before releasing",
            ))

    ledgers = ledger_dir(bundle)
    hash_path = ledgers / HASH_LEDGER
    dest_path = ledgers / DESTINATION_LEDGER

    # The one configuration that makes every other guarantee hollow. If the card is
    # also the destination then "proven present at the destination" is proved by
    # the very bytes about to be unlinked, the live re-verification hashes the
    # source against itself, and the gate below passes on the card's own contents.
    # Nothing in the ledgers can catch it: `custody verify` would have compared
    # the card to the card. So it is refused here, by path equality after
    # resolution, which also catches a destination symlinked onto the card.
    if _same_place(root, dest_root):
        blockers.append(Blocker(
            "source_and_destination_are_the_same_root",
            f"the source root and the destination root are both {root.resolve()}",
            "point --destination at the custody copy; releasing the card would "
            "delete the only copy of every byte",
        ))

    for label, summary in (
        ("hash", summarize_ledger(hash_path, max_error_samples=max_samples)),
        ("destination", summarize_ledger(dest_path, max_error_samples=max_samples)),
    ):
        if not summary.present:
            blockers.append(Blocker(
                f"{label}_ledger_absent",
                f"the {label} ledger is absent ({summary.path}); where every byte of "
                "this card is, is unknown",
                f"run the producer that writes {label}.jsonl before releasing the source",
            ))
            continue
        notes.append(f"{label}_ledger_records={summary.records}")
        if not summary.coherent:
            blockers.append(Blocker(
                f"{label}_ledger_incoherent",
                f"malformed_lines={summary.malformed} truncated={summary.truncated}",
                "rebuild the ledger; an interrupted producer's rows cannot prove custody",
            ))

    if blockers:
        return _blocked_plan(campaign, state, root, allowed, blockers, notes)

    source_index = _proven_index(hash_path, SOURCE_VERIFIED_STATUSES)
    dest_index = _proven_index(dest_path, DESTINATION_PROVEN_STATUSES)
    notes.append(f"source_proven_objects={len(source_index)}")
    notes.append(f"destination_proven_objects={len(dest_index)}")

    # Classification pass. A key that cannot be resolved, or that names something
    # other than a file, refuses the WHOLE run rather than being skipped: a ledger
    # holding `../../etc/passwd` is evidence of tampering, and deleting the
    # remaining rows of a tampered ledger is not a conservative reading.
    #
    # This runs BEFORE the inventory cross-check below, because an unsafe key is a
    # safety finding and a coverage shortfall is a completeness one; reporting the
    # escape is what an operator needs first.
    refusals: List[Blocker] = []
    refusal_keys: List[str] = []
    near_missed: List[str] = []
    candidates: List[Tuple[str, Path, LedgerRecord]] = []
    considered = 0
    for key in sorted(source_index):
        want = source_index[key]
        proof = dest_index.get(key)
        if proof is None:
            # Not proven at the destination. Absence of proof is not a refusal to
            # report: it simply is not in the set, and saying so per key would
            # scale with the card.
            continue
        considered += 1
        if not want.digest or not proof.digest:
            refusal_keys.append(key)
            refusals.append(Blocker(
                "destination_proof_carries_no_digest",
                f"{key}: a record without a digest is never custody",
                "re-run custody verify so every object carries a digest",
            ))
            continue
        if proof.digest.strip().lower() != want.digest.strip().lower():
            refusal_keys.append(key)
            refusals.append(Blocker(
                "destination_digest_disagrees_with_the_source",
                f"{key}: source {want.digest[:12]} vs destination "
                f"{proof.digest[:12]}",
                "re-copy and re-verify this object; the source copy is the only good one",
            ))
            continue
        if want.size and proof.size and want.size != proof.size:
            refusal_keys.append(key)
            refusals.append(Blocker(
                "destination_size_disagrees_with_the_source",
                f"{key}: source {want.size}B vs destination {proof.size}B",
                "re-copy and re-verify this object",
            ))
            continue
        target = _resolve_under_root(root, key)
        if target is None:
            near_missed.append(key)
            blockers.append(Blocker(
                "source_key_is_not_defensibly_resolvable",
                f"{key}: the key does not resolve to an object inside the source root "
                "(absolute, containing '..', or reached through a symlink)",
                "fix or remove the ledger key; nothing is deleted while a key can escape",
            ))
            break
        if target.is_dir():
            near_missed.append(key)
            blockers.append(Blocker(
                "source_object_is_a_directory",
                f"{key}: a directory can only be removed recursively, and this module "
                "has no recursive delete",
                "release the files individually; the directory itself stays",
            ))
            break
        candidates.append((key, target, want))

    inventoried = evidence.inventory.discovered_files
    if inventoried and len(source_index) != inventoried:
        blockers.append(Blocker(
            "hash_ledger_does_not_cover_the_inventory",
            f"the hash ledger proves {len(source_index)} objects but the inventory "
            f"records {inventoried}",
            "finish custody hash; a partially written ledger is not a whole card",
        ))

    if blockers:
        return _blocked_plan(campaign, state, root, allowed, blockers, notes,
                             near_missed=near_missed, refusals=refusals,
                             refusal_keys=refusal_keys)

    objects: List[SourceObject] = []
    absent: List[SourceObject] = []
    for key, target, want in candidates:
        if not target.exists():
            # Already gone. Idempotency says an earlier pass released it; the audit
            # may or may not say so, and that is precisely the gap `absent` closes.
            # Kept as a finding rather than a candidate - there is nothing there to
            # unlink - so the executed pass can put it on the record.
            absent.append(SourceObject(
                key=key,
                path=str(target),
                size=want.size,
                digest=want.digest.strip().lower(),
            ))
            continue
        if recheck:
            ok, detail = _destination_proves(dest_root, key, want.digest)
            if not ok:
                refusal_keys.append(key)
                refusals.append(Blocker(
                    f"destination_live_check_failed:{detail}",
                    f"{key}: the destination object no longer matches the source digest",
                    "re-copy and re-verify before releasing the source",
                ))
                continue
        objects.append(SourceObject(
            key=key,
            path=str(target),
            size=want.size,
            digest=want.digest.strip().lower(),
        ))

    return ReleasePlan(
        campaign_id=campaign.campaign_id,
        state=state.value,
        source_root=str(root),
        release_allowed=allowed,
        considered=considered,
        deletable=len(objects),
        deletable_bytes=sum(o.size for o in objects),
        already_absent=len(absent),
        objects=tuple(objects),
        absent_objects=tuple(absent),
        ledger_notes=tuple(notes),
        blockers=(),
        # Uncapped on purpose: every one of these becomes a keyed `refused` row in
        # the audit, and a ledger that names 20 of 5000 refusals reads as a clean
        # release. The report caps it instead - see ReleasePlan.to_dict().
        refusals=tuple(refusals),
        refusal_keys=tuple(refusal_keys),
        mode=MODE_PROPOSAL,
    )


def _blocked_plan(
    campaign: Campaign,
    state: CampaignState,
    root: Path,
    allowed: bool,
    blockers: List[Blocker],
    notes: List[str],
    *,
    near_missed: Optional[List[str]] = None,
    refusals: Optional[List[Blocker]] = None,
    refusal_keys: Optional[List[str]] = None,
) -> ReleasePlan:
    """A refusal. ``objects`` is empty because a partial proposal is not offered.

    ``near_missed`` and ``refusals`` are carried through anyway: the run is
    refused, but the objects it choked on are still named, and the executed pass
    puts each of them on the record so an auditor can see which objects a whole-run
    refusal was about rather than only that something was refused.
    """
    return ReleasePlan(
        campaign_id=campaign.campaign_id,
        state=state.value,
        source_root=str(root),
        release_allowed=allowed,
        considered=0,
        objects=(),
        near_missed=tuple(near_missed or ()),
        ledger_notes=tuple(notes),
        blockers=tuple(blockers),
        refusals=tuple(refusals or ()),
        refusal_keys=tuple(refusal_keys or ()),
        mode=MODE_PROPOSAL,
    )


def _destination_proves(dest_root: Path, key: str, want_digest: str) -> Tuple[bool, str]:
    """Re-read the destination object and compare it to the source digest.

    Returns ``(ok, detail)``. This is the last thing that happens before an
    unlink: the verification record was written by an earlier pass, and bytes at a
    destination can rot, be truncated, or be replaced. Deleting a sound source
    while the destination is corrupt is the exact catastrophe this subsystem
    exists to prevent, so the substitute is proven *now*, not remembered.
    """
    target = _resolve_under_root(dest_root, key)
    if target is None:
        return False, "destination_key_does_not_resolve"
    if not target.is_file():
        return False, "destination_object_absent"
    try:
        actual, read = digest_file(target)
    except OSError as exc:
        return False, f"destination_object_unreadable:{exc.strerror or exc}"
    if actual != want_digest:
        return False, "destination_digest_differs"
    return True, f"reverified_bytes={read}"


# ---------------------------------------------------------------------------
# the append-only audit ledger
# ---------------------------------------------------------------------------
def audit_path(bundle: str | Path) -> Path:
    """Where the release audit lives. Under the bundle, like every other ledger."""
    return ledger_dir(bundle) / RELEASE_LEDGER


def audit_ends_unterminated(path: str | Path) -> bool:
    """True when the audit's last record was never newline-terminated.

    That is the fingerprint of a killed release, and it is the most dangerous
    kind of damage here: the records that did land read as a clean history. It is
    also why the next run closes the line off instead of appending onto it - one
    interrupted record would otherwise swallow the next one too.
    """
    p = Path(path)
    if not p.is_file():
        return False
    try:
        if p.stat().st_size == 0:
            return False
        with p.open("rb") as handle:
            handle.seek(-1, os.SEEK_END)
            return handle.read(1) != b"\n"
    except OSError:  # pragma: no cover - unreadable audit is reported elsewhere
        return False


def _dumps(row: Dict[str, Any]) -> str:
    """One audit row as it is written: compact, sorted keys, no spaces."""
    return json.dumps(row, sort_keys=True, separators=(",", ":"))


def _audit(handle, row: Dict[str, Any]) -> None:
    """Append one record: compact, sorted keys, newline-terminated, fsynced.

    The trailing newline is load-bearing, not cosmetic: an unterminated final
    line reads as an interrupted producer and makes :func:`read_audit` incoherent.
    """
    handle.write(_dumps(row) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _append(handle, row: Dict[str, Any]) -> bool:
    """Append one record, reporting whether the ledger actually took it.

    Never raises. An unwritable audit has to stop the pass, not abort it with an
    exception that names no key and leaves no ``ReleaseResult`` for the caller to
    print - which is exactly what the ``raise`` this replaced did.

    A partial append is left where it fell. The torn tail is the honest fingerprint
    of a producer that died mid-write, and the next run closes the line off and
    ``read_audit()`` keeps reporting ``coherent=False`` until it is rebuilt.
    """
    try:
        _audit(handle, row)
    except OSError:
        return False
    return True


def _record_bytes(row: Dict[str, Any]) -> int:
    """Bytes :func:`_audit` would spend on ``row``, including the newline."""
    return len(_dumps(row).encode("utf-8")) + 1


def _audit_room(path: Path, needed: int) -> bool:
    """True when the audit's filesystem plausibly has ``needed`` bytes free.

    Asked about the *directory*, not the audit file, because free space is a
    property of the filesystem and the file may not exist yet.

    Returns True when the question cannot be answered. A filesystem that will not
    report its free space is not evidence that it has none, and refusing a release
    on a guess would be a worse failure than the one this guards against - so the
    unknown case proceeds, and the per-record check plus the stop-on-append-failure
    in :func:`execute_release` are what actually bound the damage.
    """
    try:
        stat = os.statvfs(path)
    except OSError:
        return True
    return stat.f_bavail * stat.f_frsize >= needed


def _reserve_bytes(plan: ReleasePlan, decided_at: Optional[str]) -> int:
    """Upper bound on the bytes this pass could append, before it appends any.

    Measured per candidate object rather than guessed per card, because the row
    size is dominated by the key and path, which come from a ledger and are not
    known until classification is done. ``_DETAIL_RESERVE`` covers the one field
    whose value is not yet known; ``_AUDIT_SLACK_BYTES`` covers the run summary and
    whatever the filesystem rounds a write up to.

    A lower bound here would be the dangerous direction, so this errs high - and it
    is still only a guard. The per-row check in :func:`execute_release` is the one
    that runs immediately before each write.
    """
    total = _AUDIT_SLACK_BYTES
    for obj in tuple(plan.objects) + tuple(plan.absent_objects):
        total += _record_bytes(_audit_record(
            campaign_id=plan.campaign_id,
            event=DELETED,
            decided_at=decided_at,
            key=obj.key,
            path=obj.path,
            size=obj.size,
            digest=obj.digest,
            detail="x" * _DETAIL_RESERVE,
        ))
    return total + len(plan.refusals) * _REFUSAL_RESERVE_BYTES


def _audit_record(
    *,
    campaign_id: str,
    event: str,
    decided_at: Optional[str],
    key: str = "",
    path: str = "",
    size: int = 0,
    digest: str = "",
    detail: str = "",
) -> Dict[str, Any]:
    """One audit record. ``at`` is supplied by the caller, never invented here.

    Passing ``None`` records a null instant rather than reading a clock, so two
    runs over identical inputs produce byte-identical audit records and a test can
    assert that. The instant belongs to the command layer, exactly as
    ``custody new --created-at`` already works.
    """
    return {
        "at": decided_at,
        "campaign_id": campaign_id,
        "detail": detail,
        "digest": digest,
        "event": event,
        "key": key,
        "path": path,
        "size": size,
    }


def read_audit(bundle: str | Path) -> AuditRead:
    """Read the release audit as bounded counts. Read-only; absent is reported."""
    return _read_audit_path(audit_path(bundle))


def _read_audit_path(path: str | Path) -> AuditRead:
    """Aggregate an audit ledger into counts. The bounded counterpart of the record.

    Kept separate from :func:`read_audit` so a caller that already knows the path
    - the executed pass, immediately after appending its own records - does not
    have to reconstruct it from the bundle. It is a reader and nothing else: no
    mode other than ``r``, no repair, no truncation.

    An event name is data, not structure: a ledger is a file on disk and anyone can
    append to one. So ``by_event`` counts only :data:`AUDIT_EVENTS` and everything
    else lands in ``unknown`` as a single integer. Without that, a ledger padded
    with 67k invented event names would grow a dict that this report promises is a
    fixed shape - and the report is the thing an operator reads.
    """
    p = Path(path)
    by_event: Dict[str, int] = {}
    settled: Set[str] = set()
    records = deleted = deleted_bytes = 0
    refused = absent = failed = malformed = unattributed = unknown = 0
    runs = run_proposed = run_handled = 0
    truncated = False
    if p.is_file():
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
                records += 1
                event = str(row.get("event") or "")
                key = str(row.get("key") or "").strip()
                if event in AUDIT_EVENTS:
                    by_event[event] = by_event.get(event, 0) + 1
                else:
                    unknown += 1
                if event == DELETED:
                    deleted += 1
                    deleted_bytes += _count_bytes(row.get("size"))
                elif event == REFUSED:
                    refused += 1
                    if not key:
                        unattributed += 1
                elif event == ABSENT:
                    absent += 1
                elif event == FAILED:
                    failed += 1
                elif event == RUN:
                    # Last write wins, like every other ledger read in this module:
                    # a re-run's summary describes the card as it is now.
                    runs += 1
                    run_proposed = _count_bytes(row.get("proposed"))
                    run_handled = sum(
                        _count_bytes(row.get(field))
                        for field in ("absent", "deleted", "failed", "refused")
                    )
                if key and event in SETTLING_EVENTS:
                    settled.add(key)
    return AuditRead(
        present=p.is_file(),
        path=str(p),
        records=records,
        deleted=deleted,
        deleted_bytes=deleted_bytes,
        refused=refused,
        absent=absent,
        failed=failed,
        malformed=malformed,
        truncated=truncated,
        unattributed=unattributed,
        unknown=unknown,
        runs=runs,
        run_proposed=run_proposed,
        run_handled=run_handled,
        settled_keys=frozenset(settled),
        by_event=by_event,
    )


def _count_bytes(value: Any) -> int:
    """An untrusted number from a ledger row, as a non-negative int.

    A ledger is a file on disk, so its numbers are untrusted like its keys: a
    missing, negative or non-numeric value must not make a cumulative total go
    backwards, must not raise, and - for the ``run`` summary - must not be able to
    claim a pass handled more than it proposed.
    """
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return number if number > 0 else 0


# ---------------------------------------------------------------------------
# the executed pass
# ---------------------------------------------------------------------------
def execute_release(
    bundle: str | Path,
    source_root: str | Path,
    destination_root: str | Path,
    *,
    campaign: Campaign,
    evidence: CampaignEvidence,
    execute: bool = False,
    policy: CustodyPolicy | None = None,
    limit: Optional[int] = None,
    decided_at: Optional[str] = None,
    max_errors: int = MAX_SUMMARY_ENTRIES,
) -> ReleaseResult:
    """Remove the planned source objects, or describe what would be removed.

    ``execute=False`` (the default) writes nothing: no object is unlinked, no file
    is created, and the returned result is the proposal. ``execute=True`` is the
    authorized pass, and it is the only path that calls ``Path.unlink``.

    **The audit row is written and fsynced before the unlink, never after.** Order
    inside the pass, per object: confirm the gate said yes, resolve the key
    defensively, re-verify the destination bytes live, append the ``deleted`` row,
    then unlink. A refusal at the destination leaves the source object in place and
    records why. An append that fails leaves the source object in place too and
    stops the pass - which is the whole point: unlink-then-record loses the record
    exactly when the ledger is unwritable, and then the card holds fewer files than
    any row admits while ``read_audit()`` still calls the ledger coherent.

    The bias that buys is one-directional. An unlink that fails *after* its row
    landed leaves a ``deleted`` and a ``failed`` row for the same key, so the audit
    can over-report a deletion that did not happen. It can never under-report one
    that did.

    Three guards, in order, all of which refuse before touching the card: the audit
    must be openable, its filesystem must plausibly have room for every row this
    pass could write, and each row is checked again immediately before it is
    written. An audit that cannot take a row is a refusal with a reason, not an
    exception - see :func:`_append`.

    Idempotent. A second run finds every key already accounted for, unlinks nothing,
    and appends a zeroed run summary. It does **not** re-append an ``absent`` row
    for a key an earlier pass already recorded: see :data:`SETTLING_EVENTS` and the
    ABSENT section of the module docstring. An ``absent`` row is written only for a
    key that is proven in both ledgers, gone from the source, and *not* already
    explained by the audit - which is the case an auditor cannot otherwise resolve.

    ``limit`` bounds the destructive decisions per pass, not the audit rows: an
    ``absent`` finding destroys nothing and is recorded whatever the limit, and it
    is charged to nothing.

    Every return path carries ``audit_summary``: the append-only ledger re-read
    after the pass, so :func:`to_evidence` publishes cumulative counts rather
    than this pass's delta.
    """
    plan = plan_release(
        bundle,
        source_root,
        destination_root,
        campaign=campaign,
        evidence=evidence,
        policy=policy,
        max_samples=max_errors,
    )
    audit = audit_path(bundle)
    result = ReleaseResult(
        campaign_id=plan.campaign_id,
        mode=MODE_PROPOSAL,
        audit_path=str(audit),
        gate_open=plan.release_allowed,
        state=plan.state,
        considered=plan.considered,
        proposed=plan.deletable,
        proposed_bytes=plan.deletable_bytes,
        already_absent=plan.already_absent,
        refused=len(plan.refusals),
        blockers=plan.blockers,
        ledger_notes=plan.ledger_notes,
        error_samples=tuple(f"{b.code}:{b.detail}" for b in plan.refusals[:max_errors]),
        key_samples=plan.keys[:max_errors],
        # A proposal writes nothing, so this is the previous pass's view - which
        # is exactly right: applying a dry run's fragment restates the record.
        audit_summary=_read_audit_path(audit),
    )
    if not execute:
        return result

    dest_root = Path(destination_root)
    audit.parent.mkdir(parents=True, exist_ok=True)

    # Guard 1: is there room for everything this pass could write? Asked before the
    # card is touched, so a nearly-full volume is refused rather than emptied first.
    reserve = _reserve_bytes(plan, decided_at)
    if not _audit_room(audit.parent, reserve):
        return replace(
            result,
            mode=MODE_REFUSED,
            blockers=plan.blockers + (Blocker(
                "audit_ledger_lacks_room",
                f"the pass could append up to {reserve}B and the filesystem holding "
                f"{audit.parent} cannot promise it",
                "free space on the campaign volume, or move the bundle, then re-run; "
                "nothing is deleted on the strength of a record that cannot be written",
            ),),
            audit_summary=_read_audit_path(audit),
        )

    try:
        handle = audit.open(AUDIT_MODE, encoding="utf-8")
    except OSError as exc:
        return replace(
            result,
            mode=MODE_REFUSED,
            blockers=plan.blockers + (Blocker(
                "audit_ledger_is_unwritable",
                f"{exc.strerror or exc} ({audit})",
                "make the bundle's ledgers/ writable and re-run; a release that "
                "cannot be recorded is not performed",
            ),),
            audit_summary=_read_audit_path(audit),
        )

    with handle:
        repaired = audit_ends_unterminated(audit)
        if repaired:
            # Close off the torn line so the record below stands alone. It is not
            # repaired - guessing where truncated JSON ended would be inventing
            # evidence - and read_audit() still reports the ledger incoherent.
            try:
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            except OSError:
                return replace(
                    result,
                    mode=MODE_REFUSED,
                    blockers=plan.blockers + (Blocker(
                        "audit_ledger_is_unwritable",
                        f"the torn tail of {audit} could not be closed off",
                        "free space on the campaign volume, or move the bundle, "
                        "then re-run; nothing is deleted",
                    ),),
                    audit_summary=_read_audit_path(audit),
                )

        if plan.blockers:
            recorded = _append(handle, _audit_record(
                campaign_id=plan.campaign_id,
                event=REFUSED,
                decided_at=decided_at,
                detail=_blocker_summary(plan.blockers, max_errors),
            ))
            # A refusal that names no key tells an auditor nothing about which
            # objects were near-missed, so each one the plan did identify gets its
            # own keyed row. `unattributed` is what tells the two apart afterwards.
            for key in plan.near_missed:
                if not _append(handle, _audit_record(
                    campaign_id=plan.campaign_id,
                    event=REFUSED,
                    decided_at=decided_at,
                    key=key,
                    detail=_blocker_summary(plan.blockers, max_errors),
                )):
                    recorded = False
            return replace(result, mode=MODE_REFUSED,
                           audit_unterminated_repaired=repaired,
                           audit_append_failed=not recorded,
                           audit_summary=_read_audit_path(audit))

        errors: List[str] = []
        notes: List[str] = list(result.ledger_notes)
        broke = False

        # A plan-time refusal is a decision about one object, so it gets a row like
        # any other. It used to reach only stdout: the audit held no trace of the
        # key, `to_evidence` reported `refused=0, complete=True`, and
        # `release.py`'s `source_release_incomplete` never fired - a campaign read
        # as a clean release while the object sat on the card unexplained.
        refused = 0
        for key, blocker in plan.refusal_entries:
            refused += 1
            if len(errors) < max_errors:
                errors.append(f"{blocker.code}:{blocker.detail}")
            if not _append(handle, _audit_record(
                campaign_id=plan.campaign_id,
                event=REFUSED,
                decided_at=decided_at,
                key=key,
                detail=blocker.code,
            )):
                broke = True
                notes.append("audit_append_failed_on_a_plan_refusal")
                break

        # An already-gone key with no `deleted` and no `absent` row: the one case
        # the audit could not previously resolve. Only unsettled keys are written,
        # so a re-run adds nothing and no counter drifts.
        recorded_absent = 0
        if not broke:
            for obj in plan.absent_objects:
                if obj.key in result.audit_summary.settled_keys:
                    continue
                recorded_absent += 1
                if not _append(handle, _audit_record(
                    campaign_id=plan.campaign_id,
                    event=ABSENT,
                    decided_at=decided_at,
                    key=obj.key,
                    path=obj.path,
                    size=obj.size,
                    digest=obj.digest,
                    detail="in_both_ledgers_but_not_on_the_source",
                )):
                    broke = True
                    notes.append("audit_append_failed_on_an_absent_finding")
                    break

        deleted = deleted_bytes = failed = 0
        if not broke:
            for obj in plan.objects:
                if limit is not None and (deleted + failed + refused) >= limit:
                    break
                ok, detail = _destination_proves(dest_root, obj.key, obj.digest)
                if not ok:
                    refused += 1
                    if len(errors) < max_errors:
                        errors.append(f"{obj.key}:{detail}")
                    if not _append(handle, _audit_record(
                        campaign_id=plan.campaign_id,
                        event=REFUSED,
                        decided_at=decided_at,
                        key=obj.key,
                        path=obj.path,
                        size=obj.size,
                        digest=obj.digest,
                        detail=detail,
                    )):
                        broke = True
                        notes.append(f"audit_append_failed_at={obj.key}")
                        break
                    continue
                row = _audit_record(
                    campaign_id=plan.campaign_id,
                    event=DELETED,
                    decided_at=decided_at,
                    key=obj.key,
                    path=obj.path,
                    size=obj.size,
                    digest=obj.digest,
                    detail=detail,
                )
                # Guard 2: the row has to fit before the byte it describes is gone.
                if not _audit_room(audit.parent, _record_bytes(row)):
                    broke = True
                    notes.append(f"audit_append_has_no_room_for={obj.key}")
                    break
                # Guard 3, and the load-bearing one: the record lands first. If it
                # does not, this object is still on the card and nothing after it is
                # attempted.
                if not _append(handle, row):
                    broke = True
                    notes.append(f"audit_append_failed_at={obj.key}")
                    break
                try:
                    Path(obj.path).unlink()
                except OSError as exc:
                    failed += 1
                    if len(errors) < max_errors:
                        errors.append(f"{obj.key}:{exc.strerror or exc}")
                    _append(handle, _audit_record(
                        campaign_id=plan.campaign_id,
                        event=FAILED,
                        decided_at=decided_at,
                        key=obj.key,
                        path=obj.path,
                        size=obj.size,
                        detail=str(exc),
                    ))
                    continue
                deleted += 1
                deleted_bytes += obj.size

        # Bounded run summary: counters only, never a row per object. Its `proposed`
        # is every object this pass reached a verdict on - the proposed deletions
        # plus the plan-time refusals - so a reader can tell a finished pass from a
        # truncated one by comparing it against what was actually handled.
        if not _append(handle, {
            "absent": recorded_absent,
            "already_absent": plan.already_absent,
            "at": decided_at,
            "campaign_id": plan.campaign_id,
            "considered": plan.considered,
            "deleted": deleted,
            "deleted_bytes": deleted_bytes,
            "event": RUN,
            "failed": failed,
            "mode": MODE_EXECUTED,
            "proposed": plan.deletable + len(plan.refusals),
            "refused": refused,
        }):
            broke = True
            notes.append("audit_append_failed_on_the_run_summary")
        return replace(
            result,
            mode=MODE_EXECUTED,
            deleted=deleted,
            deleted_bytes=deleted_bytes,
            failed=failed,
            refused=refused,
            audit_unterminated_repaired=repaired,
            audit_append_failed=broke,
            ledger_notes=tuple(notes),
            error_samples=tuple(errors),
            # Re-read through a fresh handle: the append handle above is still
            # open, and the counters must include the records this pass wrote.
            audit_summary=_read_audit_path(audit),
        )


def to_evidence(result: ReleaseResult, *,
                last_checkpoint: Optional[str] = None,
                max_samples: int = MAX_SUMMARY_ENTRIES) -> Dict[str, Any]:
    """The evidence fragments a source-release pass contributes.

    Module-level and pure in its argument, following ``hashing.to_evidence``,
    ``verify.to_evidence`` and ``executor.to_evidence``: it returns a fragment for
    the caller to merge, and never writes ``evidence.json`` itself. Applying it is
    the command layer's job (``custody release-source ... --apply``).

    **Every counter is cumulative, read from the append-only audit ledger.** That
    is the cumulative-not-delta rule the hash and verify producers learned the hard
    way - reporting one pass's work writes ``0`` over a real count on every resume.
    It is also what makes this fragment idempotent: applying it twice restates
    ``released.files = 3`` rather than inflating it to 6.

    **Direction is load-bearing.** The block is ``source_release``, and nothing
    here touches ``destination`` or ``reconciliation``. A release is a fact about
    objects that used to be on the card; recording it as destination verification
    would mean deleting the card manufactured a pass.

    ``complete`` means "a release finished here and nothing was left behind" -
    never "the card is empty", and never "the gate said no": a pass that removed 2
    of 3 objects is ``complete=False`` with ``released.files = 2``, and a
    whole-run refusal attempted nothing so it is not a finished release either.
    The gate reads exactly that as a blocker.

    ``complete`` additionally asks the audit's own last run summary whether it
    proposed more than it handled (:attr:`AuditRead.finished`). Without that, a
    pass cut short by a ``--limit`` or by an append failure would still read as a
    clean release, because every row it managed to write is a clean deletion.

    ``last_checkpoint`` is supplied by the caller, never read from a clock, so two
    runs over identical inputs produce identical evidence.
    """
    audit = result.audit_summary
    refused = audit.object_refusals
    attempted = audit.deleted + audit.absent + audit.failed + refused
    return {
        "source_release": {
            "absent": audit.absent,
            "audit_records": audit.records,
            # "a release finished and nothing was left behind" - not "the card is
            # empty", and not "the gate said no": a whole-run refusal attempted
            # nothing, so it is not a finished release either.
            "complete": audit.coherent and audit.finished and attempted > 0
                        and not (audit.failed or refused),
            "error_summary": list(result.error_samples[:max_samples]),
            "failed": audit.failed,
            "last_checkpoint": last_checkpoint,
            "refused": refused,
            "released": {"bytes": audit.deleted_bytes, "files": audit.deleted},
            "started": bool(attempted),
        },
    }


def exit_code(result: ReleaseResult) -> int:
    """0 for a usable proposal or a clean executed pass; 3 for a refusal.

    Mirrors ``custody execute``: the absence of ``--execute`` is not an error, it
    is the default, so a dry run exits 0. A refusal - or an executed pass that
    failed to remove everything it proposed - exits 3, the same code the release
    gate uses, so a caller needs no second vocabulary.
    """
    if result.mode == MODE_REFUSED:
        return EXIT_GATE_CLOSED
    if result.mode == MODE_EXECUTED and not result.complete:
        return EXIT_GATE_CLOSED
    return EXIT_OK


__all__ = [
    "ABSENT",
    "AUDIT_EVENTS",
    "AUDIT_MODE",
    "DELETED",
    "DESTINATION_PROVEN_STATUSES",
    "EXIT_GATE_CLOSED",
    "EXIT_OK",
    "EXIT_USAGE",
    "FAILED",
    "MODE_EXECUTED",
    "MODE_PROPOSAL",
    "MODE_REFUSED",
    "REFUSED",
    "RELEASE_LEDGER",
    "RUN",
    "SETTLING_EVENTS",
    "AuditRead",
    "ReleasePlan",
    "ReleaseResult",
    "SourceObject",
    "audit_ends_unterminated",
    "audit_path",
    "execute_release",
    "exit_code",
    "plan_release",
    "read_audit",
    "to_evidence",
]
