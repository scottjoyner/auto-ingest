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
* **The audit ledger is append-only.** Every deletion, every already-absent
  object, every failure and every refusal is appended to
  ``ledgers/release.jsonl``, newline-terminated, ``flush()`` + ``os.fsync()`` per
  record, exactly as ``hashing.py`` does. It is never rewritten and never
  truncated - the audit is the only record that bytes are gone, so erasing it
  would erase the fact. A partial final line means an interrupted run, so the
  next run terminates that line rather than gluing onto it, and
  ``read_audit()`` reports ``coherent=False`` until it is rebuilt.
* **Evidence is bounded.** Deletion goes into counters and capped samples, never
  one row per file: a 10,000-key card produces the same small report as a
  3-object one.

What it is forbidden to do
--------------------------

* delete anything except the exact keys its own plan derived;
* delete anything when the release gate is closed;
* delete a directory, a symlink, or anything outside the source root;
* rewrite, truncate or reorder the audit ledger;
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
Re-running a release stays idempotent (every key is now ``absent`` and nothing is
unlinked again) and the derived state is unchanged: custody proven is still
custody proven after the source is gone. **A new derived state is deliberately
not added for this** - see ``states.CampaignState``, which stays at twelve
members, and the note in ``tests/test_custody_release_evidence.py`` that pins the
reason.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

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
#: per-object ones.
DELETED = "deleted"
REFUSED = "refused"
ABSENT = "absent"
FAILED = "failed"
RUN = "run"

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
    already_absent: int = 0
    objects: Tuple[SourceObject, ...] = ()
    ledger_notes: Tuple[str, ...] = ()
    blockers: Tuple[Blocker, ...] = ()
    refusals: Tuple[Blocker, ...] = ()
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

    def to_dict(self, *, max_samples: int = MAX_SUMMARY_ENTRIES) -> Dict[str, Any]:
        return {
            "already_absent": self.already_absent,
            "blockers": [b.to_dict() for b in self.blockers],
            "campaign_id": self.campaign_id,
            "considered": self.considered,
            "deletable": self.deletable,
            "deletable_bytes": self.deletable_bytes,
            "ledger_notes": list(self.ledger_notes),
            "mode": self.mode,
            "refusal_codes": [b.code for b in self.refusals][:max_samples],
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "absent": self.absent,
            "by_event": dict(sorted(self.by_event.items())),
            "coherent": self.coherent,
            "deleted": self.deleted,
            "deleted_bytes": self.deleted_bytes,
            "failed": self.failed,
            "malformed_lines": self.malformed,
            "object_refusals": self.object_refusals,
            "path": self.path,
            "present": self.present,
            "records": self.records,
            "refused": self.refused,
            "truncated": self.truncated,
            "unattributed": self.unattributed,
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
        was either removed or refused/failed on with nothing silently skipped.
        """
        if self.mode == MODE_PROPOSAL:
            return self.gate_open and not self.blockers
        return self.failed == 0 and self.refused == 0 and self.handled >= self.proposed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "already_absent": self.already_absent,
            # The cumulative audit view. Bounded: every event name comes from the
            # fixed set this module emits, so `by_event` cannot grow with the card.
            "audit": self.audit_summary.to_dict(),
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
    recorded inventory, or when any candidate key cannot be resolved defensively.

    ``recheck`` additionally re-hashes each destination object now instead of
    trusting the recorded verification. It is off by default because a dry run
    over a full card would re-read the whole destination; the executed pass always
    re-verifies per object immediately before unlinking, whether or not this flag
    was set.
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
            refusals.append(Blocker(
                "destination_proof_carries_no_digest",
                f"{key}: a record without a digest is never custody",
                "re-run custody verify so every object carries a digest",
            ))
            continue
        if proof.digest.strip().lower() != want.digest.strip().lower():
            refusals.append(Blocker(
                "destination_digest_disagrees_with_the_source",
                f"{key}: source {want.digest[:12]} vs destination "
                f"{proof.digest[:12]}",
                "re-copy and re-verify this object; the source copy is the only good one",
            ))
            continue
        if want.size and proof.size and want.size != proof.size:
            refusals.append(Blocker(
                "destination_size_disagrees_with_the_source",
                f"{key}: source {want.size}B vs destination {proof.size}B",
                "re-copy and re-verify this object",
            ))
            continue
        target = _resolve_under_root(root, key)
        if target is None:
            blockers.append(Blocker(
                "source_key_is_not_defensibly_resolvable",
                f"{key}: the key does not resolve to an object inside the source root "
                "(absolute, containing '..', or reached through a symlink)",
                "fix or remove the ledger key; nothing is deleted while a key can escape",
            ))
            break
        if target.is_dir():
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
        return _blocked_plan(campaign, state, root, allowed, blockers, notes)

    objects: List[SourceObject] = []
    already_absent = 0
    for key, target, want in candidates:
        if not target.exists():
            # Idempotency: a key proven at the destination whose source object is
            # already gone was released by an earlier pass.
            already_absent += 1
            continue
        if recheck:
            ok, detail = _destination_proves(dest_root, key, want.digest)
            if not ok:
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
        already_absent=already_absent,
        objects=tuple(objects),
        ledger_notes=tuple(notes),
        blockers=(),
        refusals=tuple(refusals[:max_samples]),
        mode=MODE_PROPOSAL,
    )


def _blocked_plan(
    campaign: Campaign,
    state: CampaignState,
    root: Path,
    allowed: bool,
    blockers: List[Blocker],
    notes: List[str],
) -> ReleasePlan:
    """A refusal. ``objects`` is empty because a partial proposal is not offered."""
    return ReleasePlan(
        campaign_id=campaign.campaign_id,
        state=state.value,
        source_root=str(root),
        release_allowed=allowed,
        considered=0,
        objects=(),
        ledger_notes=tuple(notes),
        blockers=tuple(blockers),
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


def _audit(handle, row: Dict[str, Any]) -> None:
    """Append one record: compact, sorted keys, newline-terminated, fsynced.

    The trailing newline is load-bearing, not cosmetic: an unterminated final
    line reads as an interrupted producer and makes :func:`read_audit` incoherent.
    """
    handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


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
    """
    p = Path(path)
    by_event: Dict[str, int] = {}
    records = deleted = deleted_bytes = 0
    refused = absent = failed = malformed = unattributed = 0
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
                by_event[event] = by_event.get(event, 0) + 1
                keyed = bool(str(row.get("key") or "").strip())
                if event == DELETED:
                    deleted += 1
                    deleted_bytes += _count_bytes(row.get("size"))
                elif event == REFUSED:
                    refused += 1
                    if not keyed:
                        unattributed += 1
                elif event == ABSENT:
                    absent += 1
                elif event == FAILED:
                    failed += 1
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
        by_event=by_event,
    )


def _count_bytes(value: Any) -> int:
    """A record's ``size`` as a non-negative int. Anything else contributes zero.

    A ledger is a file on disk, so its numbers are untrusted like its keys: a
    missing, negative or non-numeric ``size`` must not make the cumulative byte
    total go backwards, and must not raise.
    """
    try:
        size = int(value)
    except (TypeError, ValueError):
        return 0
    return size if size > 0 else 0


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

    Order inside the pass, per object: confirm the gate said yes, resolve the key
    defensively, re-verify the destination bytes live, unlink, then fsync the audit
    record. A failure at any step leaves the source object in place and records why.

    Idempotent: a second run finds every key already absent, unlinks nothing, and
    appends a zeroed run summary. It does **not** append a per-object ``absent``
    record, because ``plan_release`` counts an already-gone key as ``already_absent``
    and never offers it as a candidate - so ``ABSENT`` remains a defined event with
    no producer. That is a gap in the audit's completeness, not in its safety: the
    destructive counters it reports are unaffected.

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
        error_samples=tuple(f"{b.code}:{b.detail}" for b in plan.refusals),
        key_samples=plan.keys[:max_errors],
        # A proposal writes nothing, so this is the previous pass's view - which
        # is exactly right: applying a dry run's fragment restates the record.
        audit_summary=_read_audit_path(audit),
    )
    if not execute:
        return result

    dest_root = Path(destination_root)
    audit.parent.mkdir(parents=True, exist_ok=True)
    with audit.open(AUDIT_MODE, encoding="utf-8") as handle:
        repaired = audit_ends_unterminated(audit)
        if repaired:
            # Close off the torn line so the record below stands alone. It is not
            # repaired - guessing where truncated JSON ended would be inventing
            # evidence - and read_audit() still reports the ledger incoherent.
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        if plan.blockers:
            _audit(handle, _audit_record(
                campaign_id=plan.campaign_id,
                event=REFUSED,
                decided_at=decided_at,
                detail=_blocker_summary(plan.blockers, max_errors),
            ))
            return replace(result, mode=MODE_REFUSED, audit_unterminated_repaired=repaired,
                           audit_summary=_read_audit_path(audit))

        deleted = deleted_bytes = failed = refused = 0
        errors: List[str] = []
        for obj in plan.objects:
            if limit is not None and (deleted + failed + refused) >= limit:
                break
            ok, detail = _destination_proves(dest_root, obj.key, obj.digest)
            if not ok:
                refused += 1
                if len(errors) < max_errors:
                    errors.append(f"{obj.key}:{detail}")
                _audit(handle, _audit_record(
                    campaign_id=plan.campaign_id,
                    event=REFUSED,
                    decided_at=decided_at,
                    key=obj.key,
                    path=obj.path,
                    size=obj.size,
                    digest=obj.digest,
                    detail=detail,
                ))
                continue
            try:
                Path(obj.path).unlink()
            except OSError as exc:
                failed += 1
                if len(errors) < max_errors:
                    errors.append(f"{obj.key}:{exc.strerror or exc}")
                _audit(handle, _audit_record(
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
            _audit(handle, _audit_record(
                campaign_id=plan.campaign_id,
                event=DELETED,
                decided_at=decided_at,
                key=obj.key,
                path=obj.path,
                size=obj.size,
                digest=obj.digest,
                detail=detail,
            ))

        # Bounded run summary: counters only, never a row per object.
        _audit(handle, {
            "already_absent": plan.already_absent,
            "at": decided_at,
            "campaign_id": plan.campaign_id,
            "considered": plan.considered,
            "deleted": deleted,
            "deleted_bytes": deleted_bytes,
            "event": RUN,
            "failed": failed,
            "mode": MODE_EXECUTED,
            "proposed": plan.deletable,
            "refused": refused,
        })
        return replace(
            result,
            mode=MODE_EXECUTED,
            deleted=deleted,
            deleted_bytes=deleted_bytes,
            failed=failed,
            refused=refused + len(plan.refusals),
            audit_unterminated_repaired=repaired,
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
            "complete": audit.coherent and attempted > 0
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
