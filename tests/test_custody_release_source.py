"""Source release - the one irreversible capability in the custody package.

Almost every assertion here is about a *refusal*, because that is what this
module is for. Until ``auto_ingest.custody.release_source`` existed the package
could ingest a card, verify it byte-for-byte at a destination and still not
reclaim it; this module adds that ability while keeping it default-off, gated on
the real release gate, and impossible outside one file.

The tests therefore pin three things in equal measure:

* the default deletes nothing and names the exact set that would go;
* the refusals - closed gate, missing/damaged ledger, disagreeing digest,
  traversal, absolute key, symlink escape - and that each yields *no* set rather
  than a smaller one;
* the success - real files gone, real audit records, and a re-run that changes
  nothing.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest
from custody_helpers import campaign as make_campaign
from custody_helpers import (
    copying,
    destination_evidence,
    errors,
    hashing,
    inventory,
    reconciliation,
    worker,
    write_bundle,
)
from custody_helpers import destination as make_destination
from custody_helpers import evidence as build_evidence

from auto_ingest.custody import CampaignState, StorageIdentity
from auto_ingest.custody.cli import EXIT_GATE_CLOSED as CLI_GATE_CLOSED
from auto_ingest.custody.cli import EXIT_OK as CLI_OK
from auto_ingest.custody.cli import EXIT_USAGE as CLI_USAGE
from auto_ingest.custody.cli import main
from auto_ingest.custody.executor import TEMP_DIRNAME
from auto_ingest.custody.ledger import DESTINATION_LEDGER, HASH_LEDGER, ledger_dir
from auto_ingest.custody.release import evaluate_release
from auto_ingest.custody.release_source import (
    EXIT_GATE_CLOSED,
    EXIT_OK,
    EXIT_USAGE,
    MODE_EXECUTED,
    MODE_PROPOSAL,
    MODE_REFUSED,
    RELEASE_LEDGER,
    audit_ends_unterminated,
    audit_path,
    execute_release,
    exit_code,
    plan_release,
    read_audit,
    to_evidence,
)
from auto_ingest.custody.store import import_evidence, load_campaign, load_evidence, load_status

pytestmark = pytest.mark.usefixtures("hermetic_mounts")

COUNT = 3
CARDS = [f"c{i}.mp4" for i in range(COUNT)]


@pytest.fixture(autouse=True)
def isolated_locks(tmp_path, monkeypatch):
    monkeypatch.setenv("CUSTODY_LOCK_ROOT", str(tmp_path / "locks"))


# ---------------------------------------------------------------------------
# a real campaign, driven through the real pipeline
# ---------------------------------------------------------------------------
def build_released(root, count=COUNT, size=256):
    """hash -> execute -> verify -> SAFE_TO_RELEASE, all through the real CLI.

    Every step is a command an operator runs. The only ingredient injected by hand
    is the destination's storage identity, because that has to be observed on the
    host and a tmpdir is not a mount point.
    """
    root = Path(root)
    src = root / "card"
    dest = root / "dest"
    src.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (src / f"c{i}.mp4").write_bytes(bytes([65 + i]) * size)
    identity = StorageIdentity(filesystem_uuid="DEST-RELEASE", device="/dev/fake0",
                               filesystem_type="ext4")
    bundle = write_bundle(root / "b",
                          make_campaign(dest=make_destination(host_path=str(dest),
                                                              mounted=True,
                                                              identity=identity)),
                          build_evidence())
    main(["hash", "--bundle", str(bundle), "--root", str(src), "--apply", "--json"])
    main(["execute", "--bundle", str(bundle), "--source-root", str(src),
          "--execute", "--apply", "--json"])
    main(["verify", "--bundle", str(bundle), "--destination", str(dest),
          "--apply", "--json"])
    import_evidence(bundle, {"destination": {"observed_identity": identity.to_dict()}},
                    apply=True)
    return bundle, src, dest


def release(bundle, src, dest, **kwargs):
    """Call the module the way the command layer will."""
    return execute_release(bundle, src, dest, campaign=load_campaign(bundle),
                           evidence=load_evidence(bundle), **kwargs)


def plan(bundle, src, dest, **kwargs):
    return plan_release(bundle, src, dest, campaign=load_campaign(bundle),
                        evidence=load_evidence(bundle), **kwargs)


def src_files(src):
    return sorted(p.name for p in Path(src).iterdir() if p.is_file())


def dest_files(dest):
    return sorted(
        p.relative_to(dest).as_posix()
        for p in Path(dest).rglob("*")
        if p.is_file() and TEMP_DIRNAME not in p.relative_to(dest).parts
    )


def audit_rows(bundle):
    return [json.loads(line) for line in
            audit_path(bundle).read_text(encoding="utf-8").splitlines()]


def codes(blockers):
    return [b.code for b in blockers]


def append_key(bundle, key, digest, *, size=256, status="verified_at_destination"):
    """Append a hand-written ledger pair, so a hostile key can be tested at all."""
    for name, row_status in ((HASH_LEDGER, "verified"), (DESTINATION_LEDGER, status)):
        with (ledger_dir(bundle) / name).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(
                {"key": key, "digest": digest, "size": size, "status": row_status},
                sort_keys=True, separators=(",", ":")) + "\n")


def rewrite_destination_rows(bundle, mutate):
    path = ledger_dir(bundle) / DESTINATION_LEDGER
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows = mutate(rows)
    path.write_text(
        "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n"
                for r in rows),
        encoding="utf-8",
    )
    return rows


# ---------------------------------------------------------------------------
# precondition: the fixture really is releasable, or nothing below means anything
# ---------------------------------------------------------------------------
def test_the_fixture_really_is_safe_to_release(tmp_path):
    bundle, src, dest = build_released(tmp_path)
    status = load_status(bundle)
    assert status.state is CampaignState.SAFE_TO_RELEASE
    assert status.source_release_allowed is True
    assert status.release.blockers == ()


def test_the_audit_ledger_lives_under_the_bundle_ledgers(tmp_path):
    bundle, src, dest = build_released(tmp_path)
    assert audit_path(bundle) == ledger_dir(bundle) / RELEASE_LEDGER


# ---------------------------------------------------------------------------
# 1. the default is a proposal, never a deletion
# ---------------------------------------------------------------------------
def test_without_execute_nothing_is_deleted_and_the_set_is_exact(tmp_path):
    bundle, src, dest = build_released(tmp_path)

    result = release(bundle, src, dest)          # no execute=True

    assert result.mode == MODE_PROPOSAL
    assert result.gate_open is True
    assert result.proposed == COUNT
    assert result.proposed_bytes == COUNT * 256
    assert result.deleted == 0
    assert result.handled == 0
    assert sorted(result.key_samples) == CARDS
    # nothing on the card moved
    assert src_files(src) == CARDS
    # and no file was created at all - not even an empty audit ledger
    assert not audit_path(bundle).exists()
    assert exit_code(result) == EXIT_OK


def test_the_proposal_comes_from_the_ledgers_not_the_directory_listing(tmp_path):
    bundle, src, dest = build_released(tmp_path)
    # an unhashed file on the card is not in the ledger, so it is not proposed
    (src / "unhashed.mp4").write_bytes(b"x" * 32)

    proposal = plan(bundle, src, dest)

    assert sorted(proposal.keys) == CARDS
    assert sorted(proposal.paths) == sorted(str(src / name) for name in CARDS)
    assert "unhashed.mp4" not in proposal.paths
    assert src_files(src) == sorted(CARDS + ["unhashed.mp4"])


def test_the_plan_is_pure_deterministic_and_writes_nothing(tmp_path):
    bundle, src, dest = build_released(tmp_path)
    snapshot = {p.name: p.read_bytes() for p in sorted(src.iterdir())}
    evidence_before = (Path(bundle) / "evidence.json").read_bytes()

    first = plan(bundle, src, dest)
    second = plan(bundle, src, dest)

    assert first.keys == second.keys
    assert first.to_dict() == second.to_dict()
    assert {p.name: p.read_bytes() for p in sorted(src.iterdir())} == snapshot
    assert (Path(bundle) / "evidence.json").read_bytes() == evidence_before
    assert not audit_path(bundle).exists()


def test_a_nested_key_is_resolved_under_the_source_root(tmp_path):
    """Keys are POSIX-relative paths, and only the leaf is ever unlinked."""
    root = tmp_path / "nested"
    src = root / "card"
    dest = root / "dest"
    (src / "2026" / "04").mkdir(parents=True)
    dest.mkdir(parents=True)
    (src / "2026" / "04" / "clip.mp4").write_bytes(b"N" * 64)
    identity = StorageIdentity(filesystem_uuid="DEST-NEST", device="/dev/fake0")
    bundle = write_bundle(root / "b",
                          make_campaign(dest=make_destination(host_path=str(dest),
                                                              mounted=True,
                                                              identity=identity)),
                          build_evidence())
    main(["hash", "--bundle", str(bundle), "--root", str(src), "--apply", "--json"])
    main(["execute", "--bundle", str(bundle), "--source-root", str(src),
          "--execute", "--apply", "--json"])
    main(["verify", "--bundle", str(bundle), "--destination", str(dest),
          "--apply", "--json"])
    import_evidence(bundle, {"destination": {"observed_identity": identity.to_dict()}},
                    apply=True)

    proposal = plan(bundle, src, dest)
    assert proposal.keys == ("2026/04/clip.mp4",)

    result = release(bundle, src, dest, execute=True)

    assert result.deleted == 1
    assert not (src / "2026" / "04" / "clip.mp4").exists()
    # the directory itself is never removed - this module has no recursive delete
    assert (src / "2026" / "04").is_dir()
    assert dest_files(dest) == ["2026/04/clip.mp4"]


# ---------------------------------------------------------------------------
# 2. gated on the real release gate
# ---------------------------------------------------------------------------
def test_a_closed_gate_refuses_and_names_the_gate_own_blockers(tmp_path):
    bundle, src, dest = build_released(tmp_path)
    # one object is no longer proven at the destination
    import_evidence(bundle, {"reconciliation": {"source_only": 1}}, apply=True)

    result = release(bundle, src, dest, execute=True)

    assert result.mode == MODE_REFUSED
    assert result.deleted == 0
    assert src_files(src) == CARDS
    assert "missing_destination" in codes(result.blockers)
    assert exit_code(result) == EXIT_GATE_CLOSED
    # the refusal is on the record, not just on stdout
    assert read_audit(bundle).refused == 1
    assert "missing_destination" in audit_path(bundle).read_text()


def test_the_refusal_repeats_the_real_gate_rather_than_re_deriving_it(tmp_path):
    """A green light here must never be weaker than the gate itself."""
    bundle, src, dest = build_released(tmp_path)
    import_evidence(bundle, {"reconciliation": {"source_only": 1}}, apply=True)

    camp, ev = load_campaign(bundle), load_evidence(bundle)
    gate = evaluate_release(camp, ev)
    result = release(bundle, src, dest, execute=True)

    assert gate.allowed is False
    assert gate.blockers, "the fixture must trip at least one gate condition"
    # every gate blocker appears, first and unaltered
    assert [b.to_dict() for b in result.blockers[:len(gate.blockers)]] == [
        b.to_dict() for b in gate.blockers
    ]
    # ...followed only by this module's own findings, never instead of the gate's
    assert codes(result.blockers)[len(gate.blockers):] == [
        "state_is_not_safe_to_release"
    ]


def test_every_blocker_is_reported_not_just_the_first(tmp_path):
    """The gate never short-circuits; neither may this."""
    bundle, src, dest = build_released(tmp_path)
    import_evidence(bundle, {"reconciliation": {"source_only": 2, "mismatched": 1},
                             "errors": {"unresolved": 1}}, apply=True)

    result = release(bundle, src, dest, execute=True)

    assert {"missing_destination", "mismatch_present", "unresolved_errors"} <= set(
        codes(result.blockers)
    )


def test_a_plan_only_offers_a_set_when_the_derived_state_is_safe_to_release(tmp_path):
    bundle, src, dest = build_released(tmp_path, count=1)
    (Path(bundle) / "evidence.json").write_text("{}", encoding="utf-8")

    proposal = plan(bundle, src, dest)

    assert proposal.usable is False
    assert proposal.objects == ()
    assert proposal.deletable == 0
    assert codes(proposal.blockers), "a refusal must name its reasons"
    assert "state_is_not_safe_to_release" in codes(proposal.blockers)


# ---------------------------------------------------------------------------
# 3. only what is proven present at the destination
# ---------------------------------------------------------------------------
def test_a_missing_destination_ledger_proposes_nothing(tmp_path):
    bundle, src, dest = build_released(tmp_path)
    (ledger_dir(bundle) / DESTINATION_LEDGER).unlink()

    proposal = plan(bundle, src, dest)

    assert proposal.objects == ()
    assert "destination_ledger_absent" in codes(proposal.blockers)
    assert src_files(src) == CARDS


def test_a_missing_hash_ledger_proposes_nothing(tmp_path):
    bundle, src, dest = build_released(tmp_path)
    (ledger_dir(bundle) / HASH_LEDGER).unlink()

    proposal = plan(bundle, src, dest)

    assert proposal.objects == ()
    assert "hash_ledger_absent" in codes(proposal.blockers)


@pytest.mark.parametrize("damage", ["truncated", "unterminated", "malformed"])
def test_a_damaged_ledger_yields_no_partial_proposal(tmp_path, damage):
    """A proposal covering only the surviving rows would read as whole-card custody."""
    bundle, src, dest = build_released(tmp_path)
    path = ledger_dir(bundle) / DESTINATION_LEDGER
    text = path.read_text(encoding="utf-8")
    if damage == "truncated":
        path.write_text(text[:-40], encoding="utf-8")
    elif damage == "unterminated":
        path.write_text(text.rstrip("\n"), encoding="utf-8")
    else:
        path.write_text(text + "{not json}\n", encoding="utf-8")

    proposal = plan(bundle, src, dest)

    assert proposal.objects == ()
    assert proposal.deletable == 0
    assert "destination_ledger_incoherent" in codes(proposal.blockers)
    # even with --execute, nothing is proposed and nothing is deleted
    result = release(bundle, src, dest, execute=True)
    assert result.mode == MODE_REFUSED
    assert result.deleted == 0
    assert src_files(src) == CARDS


def test_a_copied_record_alone_is_not_proof(tmp_path):
    """`copied` says bytes were written; only a verification record says they are right."""
    bundle, src, dest = build_released(tmp_path)
    rewrite_destination_rows(bundle, lambda rows: [
        {**r, "status": "copied"} for r in rows
    ])

    proposal = plan(bundle, src, dest)

    assert proposal.objects == ()
    assert proposal.deletable == 0


def test_a_retracted_verification_is_not_still_proof(tmp_path):
    """A key that verified then mismatched has two rows; the second one wins."""
    bundle, src, dest = build_released(tmp_path)
    rows = rewrite_destination_rows(bundle, lambda rows: rows + [
        {"key": "c0.mp4", "status": "mismatch", "digest": "0" * 64, "size": 256},
    ])

    proposal = plan(bundle, src, dest)

    assert rows[-1]["key"] not in proposal.keys
    # the other keys are unaffected - a retraction is per object
    assert len(proposal.keys) == COUNT - 1


def test_a_key_whose_destination_digest_disagrees_is_refused(tmp_path):
    bundle, src, dest = build_released(tmp_path)
    rows = rewrite_destination_rows(bundle, lambda rows: [
        {**rows[0], "digest": "f" * 64}, *rows[1:]
    ])

    proposal = plan(bundle, src, dest)

    assert len(proposal.keys) == COUNT - 1
    assert rows[0]["key"] not in proposal.keys
    assert "destination_digest_disagrees_with_the_source" in codes(proposal.refusals)
    # and its source copy is still on the card
    assert (src / rows[0]["key"]).exists()


def test_a_key_whose_destination_size_disagrees_is_refused(tmp_path):
    bundle, src, dest = build_released(tmp_path)
    rows = rewrite_destination_rows(bundle, lambda rows: [
        {**rows[0], "size": 1}, *rows[1:]
    ])

    proposal = plan(bundle, src, dest)

    assert rows[0]["key"] not in proposal.keys
    assert "destination_size_disagrees_with_the_source" in codes(proposal.refusals)


def test_a_destination_record_without_a_digest_is_never_custody(tmp_path):
    bundle, src, dest = build_released(tmp_path)
    rows = rewrite_destination_rows(bundle, lambda rows: [
        {k: v for k, v in rows[0].items() if k != "digest"}, *rows[1:]
    ])

    proposal = plan(bundle, src, dest)

    assert rows[0]["key"] not in proposal.keys
    assert "destination_proof_carries_no_digest" in codes(proposal.refusals)


def test_a_hash_ledger_that_does_not_cover_the_inventory_proposes_nothing(tmp_path):
    """A partially written hash ledger must not masquerade as a whole card."""
    bundle, src, dest = build_released(tmp_path)
    path = ledger_dir(bundle) / HASH_LEDGER
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text("".join(lines[:-1]), encoding="utf-8")

    proposal = plan(bundle, src, dest)

    assert proposal.objects == ()
    assert "hash_ledger_does_not_cover_the_inventory" in codes(proposal.blockers)


# ---------------------------------------------------------------------------
# 4. defensive path resolution
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("hostile", ["../escape.mp4", "../../etc/passwd", "/etc/passwd"])
def test_a_traversing_or_absolute_key_refuses_the_whole_run(tmp_path, hostile):
    """Not skipped, not narrowed to the rest: refused outright.

    A ledger holding `../../etc/passwd` is evidence of tampering, and deleting the
    other rows of a tampered ledger is not a conservative reading of it.
    """
    bundle, src, dest = build_released(tmp_path)
    append_key(bundle, hostile, "a" * 64)

    proposal = plan(bundle, src, dest)

    assert proposal.objects == ()
    assert "source_key_is_not_defensibly_resolvable" in codes(proposal.blockers)

    result = release(bundle, src, dest, execute=True)
    assert result.mode == MODE_REFUSED
    assert result.deleted == 0
    assert src_files(src) == CARDS


def test_a_symlinked_object_is_refused_not_followed(tmp_path):
    """Unlinking a link removes the link, not the data - and where a link points is
    not something to learn by following it in the instant before a delete."""
    bundle, src, dest = build_released(tmp_path)
    target = src / "c0.mp4"
    link = src / "link.mp4"
    link.symlink_to(target.name)
    append_key(bundle, "link.mp4", "a" * 64)

    proposal = plan(bundle, src, dest)

    assert proposal.objects == ()
    assert "source_key_is_not_defensibly_resolvable" in codes(proposal.blockers)
    # the real object is untouched, and so is the link
    assert target.exists() and link.is_symlink()
    assert release(bundle, src, dest, execute=True).deleted == 0


def test_a_symlink_escaping_the_source_root_is_refused(tmp_path):
    bundle, src, dest = build_released(tmp_path)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"not the card" * 32)
    (src / "escape.mp4").symlink_to(outside)
    append_key(bundle, "escape.mp4", "a" * 64)

    proposal = plan(bundle, src, dest)

    assert proposal.objects == ()
    assert "source_key_is_not_defensibly_resolvable" in codes(proposal.blockers)
    assert outside.exists()


def test_a_directory_is_never_removed(tmp_path):
    """There is no recursive delete here, so a directory key is a refusal."""
    bundle, src, dest = build_released(tmp_path)
    (src / "album").mkdir()
    (src / "album" / "a.mp4").write_bytes(b"D" * 256)
    append_key(bundle, "album", "a" * 64)

    proposal = plan(bundle, src, dest)

    assert "source_object_is_a_directory" in codes(proposal.blockers)
    assert proposal.objects == ()
    assert release(bundle, src, dest, execute=True).deleted == 0
    assert (src / "album" / "a.mp4").exists()


# ---------------------------------------------------------------------------
# 5. the destination must still verify, live, immediately before the unlink
# ---------------------------------------------------------------------------
def test_the_destination_is_rehashed_before_anything_is_unlinked(tmp_path):
    bundle, src, dest = build_released(tmp_path)
    (dest / "c1.mp4").write_bytes(b"rotted" * 16)      # rots after verification

    result = release(bundle, src, dest, execute=True)

    assert result.refused == 1
    assert result.deleted == COUNT - 1
    assert "c1.mp4:destination_digest_differs" in list(result.error_samples)
    # the sound source copy of the corrupt destination object is still on the card
    assert (src / "c1.mp4").exists()
    read = read_audit(bundle)
    assert read.refused == 1
    assert "c1.mp4" in audit_path(bundle).read_text()


def test_a_destination_object_removed_after_verify_refuses_the_source_delete(tmp_path):
    bundle, src, dest = build_released(tmp_path)
    (dest / "c2.mp4").unlink()

    result = release(bundle, src, dest, execute=True)

    assert result.deleted == COUNT - 1
    assert "c2.mp4:destination_object_absent" in list(result.error_samples)
    assert (src / "c2.mp4").exists()
    assert result.complete is False
    assert exit_code(result) == EXIT_GATE_CLOSED


def test_a_live_recheck_in_the_plan_can_refuse_before_any_execute(tmp_path):
    """`recheck=True` is the same check, run before the operator says go."""
    bundle, src, dest = build_released(tmp_path)
    (dest / "c0.mp4").write_bytes(b"rotted" * 16)

    proposal = plan(bundle, src, dest, recheck=True)

    assert len(proposal.keys) == COUNT - 1
    assert any(c.startswith("destination_live_check_failed") for c in codes(proposal.refusals))


def test_a_limit_bounds_the_pass_and_reports_incomplete(tmp_path):
    bundle, src, dest = build_released(tmp_path)

    result = release(bundle, src, dest, execute=True, limit=1)

    assert result.deleted == 1
    assert result.complete is False
    assert exit_code(result) == EXIT_GATE_CLOSED
    # the rest is still on the card and still releasable
    assert len(src_files(src)) == COUNT - 1


# ---------------------------------------------------------------------------
# 6. the append-only audit ledger
# ---------------------------------------------------------------------------
def test_a_successful_deletion_removes_the_files_and_appends_real_records(tmp_path):
    bundle, src, dest = build_released(tmp_path)

    result = release(bundle, src, dest, execute=True)

    assert result.mode == MODE_EXECUTED
    assert result.deleted == COUNT
    assert result.deleted_bytes == COUNT * 256
    assert result.complete is True
    assert exit_code(result) == EXIT_OK
    assert src_files(src) == []
    assert dest_files(dest) == CARDS

    text = audit_path(bundle).read_text(encoding="utf-8")
    assert text.endswith("\n"), "every audit record is newline-terminated"
    rows = audit_rows(bundle)
    deleted = [r for r in rows if r["event"] == "deleted"]
    assert len(deleted) == COUNT
    assert {r["key"] for r in deleted} == set(CARDS)
    assert all(r["digest"] and r["size"] == 256 for r in deleted)
    summaries = [r for r in rows if r["event"] == "run"]
    assert len(summaries) == 1
    assert summaries[0]["deleted"] == COUNT


def test_the_audit_ledger_is_never_rewritten(tmp_path):
    bundle, src, dest = build_released(tmp_path)
    release(bundle, src, dest, execute=True)
    first = audit_path(bundle).read_text(encoding="utf-8")

    release(bundle, src, dest, execute=True)

    text = audit_path(bundle).read_text(encoding="utf-8")
    assert text.startswith(first), "earlier bytes are still there, verbatim and in order"
    assert len(text) > len(first)


def test_a_re_run_is_idempotent(tmp_path):
    bundle, src, dest = build_released(tmp_path)
    release(bundle, src, dest, execute=True)
    first = audit_path(bundle).read_text(encoding="utf-8")

    again = release(bundle, src, dest, execute=True)

    assert again.deleted == 0
    assert again.already_absent == COUNT
    assert again.complete is True
    assert exit_code(again) == EXIT_OK
    assert src_files(src) == []
    assert audit_path(bundle).read_text(encoding="utf-8").startswith(first)
    assert read_audit(bundle).absent == 0, "absent keys never reached the delete loop"


def test_a_torn_audit_line_is_closed_off_and_the_damage_stays_visible(tmp_path):
    """One interrupted record must not swallow the next one."""
    bundle, src, dest = build_released(tmp_path)
    audit = audit_path(bundle)
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text('{"event":"deleted","key":"c0.mp4"', encoding="utf-8")

    assert audit_ends_unterminated(audit) is True
    assert read_audit(bundle).coherent is False

    result = release(bundle, src, dest, execute=True)

    assert result.audit_unterminated_repaired is True
    assert result.deleted == COUNT
    # the damaged line is closed off, not repaired - guessing where truncated JSON
    # ended would be inventing evidence
    assert audit.read_text(encoding="utf-8").splitlines()[0] == \
        '{"event":"deleted","key":"c0.mp4"'
    assert audit_ends_unterminated(audit) is False
    read = read_audit(bundle)
    assert read.deleted == COUNT          # every complete record survives
    assert read.truncated is False
    assert read.malformed == 1
    assert read.coherent is False          # and the ledger stays honest about it


def test_read_audit_reports_an_absent_ledger_rather_than_inventing_one(tmp_path):
    bundle, src, dest = build_released(tmp_path)

    read = read_audit(bundle)

    assert read.present is False
    assert read.records == 0
    assert read.coherent is False
    assert not audit_path(bundle).exists()


def test_the_instant_is_supplied_not_invented(tmp_path):
    """Two identical campaigns produce byte-identical audits, so nothing is invented."""
    at = "2026-04-12T09:15:00Z"
    bundle, src, dest = build_released(tmp_path / "one")
    release(bundle, src, dest, execute=True, decided_at=at)
    bundle2, src2, dest2 = build_released(tmp_path / "two")
    release(bundle2, src2, dest2, execute=True, decided_at=at)

    assert {row["at"] for row in audit_rows(bundle)} == {at}
    # the only difference between the two cards is where they live
    first = audit_path(bundle).read_text(encoding="utf-8").replace(str(src), "<src>")
    second = audit_path(bundle2).read_text(encoding="utf-8").replace(str(src2), "<src>")
    assert first == second


# ---------------------------------------------------------------------------
# 7. bounded evidence
# ---------------------------------------------------------------------------
def synthetic_campaign(root, count):
    """A released campaign with `count` empty objects, built without copying bytes."""
    src = root / "card"
    dest = root / "dest"
    src.mkdir(parents=True)
    dest.mkdir(parents=True)
    keys = []
    for i in range(count):
        key = f"c{i:05d}.mp4"
        (src / key).touch()
        (dest / key).touch()
        keys.append(key)
    identity = StorageIdentity(filesystem_uuid="DEST-BIG", device="/dev/fake0")
    bundle = write_bundle(
        root / "b",
        make_campaign(dest=make_destination(host_path=str(dest), mounted=True,
                                            identity=identity)),
        build_evidence(
            inv=inventory(count, 0, complete=True, verified=True),
            hsh=hashing(count, verified_bytes=0, complete=True),
            cpy=copying(planned_files=count, completed_files=count, started=True,
                        result_complete=True, ledger_complete=True),
            dst=destination_evidence(verified_files=count, verification_started=True,
                                     verification_complete=True,
                                     observed_identity=identity),
            rec=reconciliation(),
            wkr=worker(status="stopped"),
            err=errors(),
        ),
    )
    ledger_dir(bundle).mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(b"").hexdigest()
    for name, status in ((HASH_LEDGER, "verified"),
                         (DESTINATION_LEDGER, "verified_at_destination")):
        with (ledger_dir(bundle) / name).open("a", encoding="utf-8") as handle:
            for key in keys:
                handle.write(json.dumps(
                    {"key": key, "digest": digest, "size": 0, "status": status},
                    sort_keys=True, separators=(",", ":")) + "\n")
    return bundle, src, dest


def test_ten_thousand_empty_keys_stay_bounded(tmp_path):
    """Counts, not rows: the report must not grow with the card."""
    count = 10_000
    bundle, src, dest = synthetic_campaign(tmp_path, count)

    proposal = plan(bundle, src, dest)
    result = release(bundle, src, dest, execute=True)

    assert proposal.deletable == count
    assert result.deleted == count
    payload = json.dumps(result.to_dict())
    assert len(payload) < 8_000, len(payload)
    assert len(result.key_samples) <= 20
    assert len(result.error_samples) <= 20
    assert len(result.ledger_notes) <= 8
    assert src_files(src) == []
    # the audit ledger is per-object on purpose - it is the record, not the report
    assert read_audit(bundle).deleted == count


# ---------------------------------------------------------------------------
# 8. evidence.json stays truthful, and no verdict is forged
# ---------------------------------------------------------------------------
def test_execution_rewrites_no_evidence_and_forges_no_verdict(tmp_path):
    """`source_release_allowed` is release.py's to derive, and this never touches it."""
    bundle, src, dest = build_released(tmp_path)
    before = (Path(bundle) / "evidence.json").read_bytes()

    release(bundle, src, dest, execute=True)

    assert (Path(bundle) / "evidence.json").read_bytes() == before
    status = load_status(bundle)
    # still derived exactly as before: the derived verdict never depended on the
    # source being physically present
    assert status.source_release_allowed is True
    assert status.state is CampaignState.SAFE_TO_RELEASE
    assert status.evidence.inventory.discovered_files == COUNT
    # `to_evidence()` exists and is a pure function of the result: it returns a
    # fragment for the caller to merge and opens no file, so nothing here can
    # write a verdict. That applying it is the command layer's job is the whole
    # of the separation - see tests/test_custody_release_evidence.py.
    import auto_ingest.custody.release_source as module

    assert callable(module.to_evidence)
    assert "source_release_allowed" not in audit_path(bundle).read_text()


def test_to_evidence_alone_writes_nothing(tmp_path):
    """The fragment is returned, never persisted: the library side effect is nil."""
    bundle, src, dest = build_released(tmp_path)
    result = release(bundle, src, dest, execute=True)
    before = (Path(bundle) / "evidence.json").read_bytes()

    fragment = to_evidence(result)

    assert fragment["source_release"]["released"]["files"] == COUNT
    assert (Path(bundle) / "evidence.json").read_bytes() == before


def test_zeroing_the_inventory_after_a_release_would_block_the_campaign(tmp_path):
    """Why this module writes no evidence: the obvious field is a contradiction."""
    bundle, src, dest = build_released(tmp_path)
    release(bundle, src, dest, execute=True)

    result = import_evidence(bundle, {"inventory": {"discovered_files": 0,
                                                    "discovered_bytes": 0}}, apply=True)

    assert result["derived_state"] == "BLOCKED"


# ---------------------------------------------------------------------------
# exit codes and CLI parity
# ---------------------------------------------------------------------------
def test_the_exit_codes_mirror_the_cli_exactly():
    assert (EXIT_OK, EXIT_USAGE, EXIT_GATE_CLOSED) == (
        CLI_OK, CLI_USAGE, CLI_GATE_CLOSED,
    )


def test_a_proposal_is_not_an_error(tmp_path):
    """The absence of --execute is the default, so a dry run exits 0."""
    bundle, src, dest = build_released(tmp_path)
    assert exit_code(release(bundle, src, dest)) == EXIT_OK
    assert exit_code(release(bundle, src, dest, execute=True)) == EXIT_OK


# ---------------------------------------------------------------------------
# the structural guard is a real detector, not a tautology
# ---------------------------------------------------------------------------
def test_the_ast_guard_detects_tree_deletes_it_has_never_seen():
    """Proven by trying it: the guard fires on every spelling, in every position."""
    from test_custody_legacy_watchers import TREE_DELETE_LITERALS, _deletion_calls

    for bad in (
        "import shutil\nshutil.rmtree('/tmp/x')",
        "from shutil import rmtree\nrmtree('/tmp/x')",
        "import os\nos.unlink(p)",
        "import os\nos.remove(p)",
        "import os\nos.rmdir(p)",
        "p.unlink()",
    ):
        assert _deletion_calls(ast.parse(bad)), bad
    safe = "from pathlib import Path\nPath(p).unlink()"
    assert _deletion_calls(ast.parse(safe)) == [("unlink", "Path(p)")]
    for banned in TREE_DELETE_LITERALS:
        assert banned not in safe


def test_release_source_imports_nothing_banned():
    """`shutil` is the only banned import this one module could have wanted."""
    import auto_ingest.custody.release_source as module

    banned = {"subprocess", "socket", "requests", "urllib", "http", "neo4j",
              "sqlite3", "paramiko", "inotify", "watchdog", "pyudev", "schedule",
              "apscheduler", "shutil"}
    for node in ast.walk(ast.parse(Path(module.__file__).read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            roots = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            roots = [(node.module or "").split(".")[0]]
        else:
            continue
        assert not (set(roots) & banned), roots
