"""Phase B: `custody hash` - the first real producer of a custody ledger.

Everything before this was a state machine over asserted evidence. `67,644`
existed only in tests and a hand-written fixture. These tests pin the producer
and, just as importantly, prove it cannot write to the source.

The fail-closed contract is load-bearing and tested here from the producer's
side: a ledger whose final line is unterminated reads as a crashed producer, and
the reconciler must void the whole diff rather than trust the records it did
manage to read.

The name tests below cover the other half of the problem: the card is **vfat**
and the destination is **SMB2/exFAT**, so both ends are case-insensitive and
both cap a path component. `A.mp4` and `a.mp4` are therefore one file, and code
that treats them as two objects would schedule a copy onto a name that already
exists - which `os.replace` resolves by clobbering or by failing, depending on the
server. Detection is the fix; renaming, deduplicating and picking a winner are
all forbidden, so every test here also asserts the keys came through unchanged.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from custody_helpers import campaign, evidence, write_bundle

from auto_ingest.custody.cli import EXIT_OK, main
from auto_ingest.custody.hashing import (
    CASE_COLLISION,
    CHUNK_BYTES,
    DEFAULT_ALGORITHM,
    DESTINATION_CASE_COLLISION,
    HASHED_STATUS,
    NAME_COMPONENT_LIMIT_BYTES,
    SIMPLE_UPPERCASE_EXPANSION,
    UNREPRESENTABLE_NAME,
    HashProgress,
    already_hashed,
    detect_name_problems,
    digest_file,
    filename_problem,
    fold_forms,
    fold_key,
    hash_source,
    simple_upcase,
    to_evidence,
)
from auto_ingest.custody.ledger import ledger_dir, reconcile_ledgers, summarize_ledger
from auto_ingest.custody.policy import MAX_SUMMARY_ENTRIES
from auto_ingest.custody.store import load_status

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI = REPO_ROOT / "bin" / "auto-ingest"



# The source end of a campaign is a fact this module controls, not a fact
# about whether the developer's card happens to be plugged in.
pytestmark = pytest.mark.usefixtures("hermetic_mounts")
def make_card(root: Path, count: int = 4, size: int = 128) -> "dict[str, Path]":
    root.mkdir(parents=True, exist_ok=True)
    keys = {}
    for i in range(count):
        path = root / f"clip_{i}.mp4"
        path.write_bytes(bytes([65 + i]) * size)
        keys[f"clip_{i}.mp4"] = path
    return keys


# ---------------------------------------------------------------------------
# digesting
# ---------------------------------------------------------------------------
def test_digest_matches_hashlib(tmp_path):
    import hashlib

    path = tmp_path / "a.bin"
    payload = os.urandom(CHUNK_BYTES + 17)
    path.write_bytes(payload)
    digest, read = digest_file(path)
    assert digest == hashlib.sha256(payload).hexdigest()
    assert read == len(payload)


def test_empty_file_hashes_cleanly(tmp_path):
    path = tmp_path / "empty"
    path.write_bytes(b"")
    digest, read = digest_file(path)
    assert read == 0
    assert len(digest) == 64


def test_digest_does_not_modify_the_source(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"payload")
    before = (path.stat().st_mtime_ns, path.stat().st_size)
    digest_file(path)
    assert (path.stat().st_mtime_ns, path.stat().st_size) == before


def test_missing_file_raises_rather_than_inventing_a_digest(tmp_path):
    with pytest.raises(OSError):
        digest_file(tmp_path / "nope")


# ---------------------------------------------------------------------------
# producing a ledger
# ---------------------------------------------------------------------------
def test_writes_one_record_per_object(tmp_path):
    keys = make_card(tmp_path / "card")
    result = hash_source(tmp_path / "bundle", keys)
    assert result.hashed == 4
    assert result.complete is True
    assert result.failed == 0
    assert result.bytes_read == 4 * 128

    ledger = ledger_dir(tmp_path / "bundle") / "hash.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(rows) == 4
    for row in rows:
        assert row["status"] == HASHED_STATUS
        assert len(row["digest"]) == 64
        assert row["size"] == 128
        assert set(row) >= {"key", "digest", "size", "status", "algorithm"}


def test_every_line_is_newline_terminated(tmp_path):
    """Load-bearing: an unterminated final line reads as a crashed producer."""
    keys = make_card(tmp_path / "card")
    hash_source(tmp_path / "bundle", keys)
    raw = (ledger_dir(tmp_path / "bundle") / "hash.jsonl").read_bytes()
    assert raw.endswith(b"\n")
    assert raw.count(b"\n") == 4
    assert b"\n\n" not in raw


def test_the_ledger_the_reconciler_accepts(tmp_path):
    """Producer and reader agree: a clean pass produces a usable ledger."""
    keys = make_card(tmp_path / "card")
    hash_source(tmp_path / "bundle", keys)
    ledger = ledger_dir(tmp_path / "bundle") / "hash.jsonl"
    result = reconcile_ledgers(ledger, ledger)
    assert result.usable is True
    assert result.verified == 4
    assert result.complete is True


def test_a_simulated_crash_voids_the_whole_diff(tmp_path):
    """A half-written line must cost the reconciliation, not shrink it."""
    keys = make_card(tmp_path / "card", count=6)
    hash_source(tmp_path / "bundle", keys, limit=4)
    ledger = ledger_dir(tmp_path / "bundle") / "hash.jsonl"
    assert len(already_hashed(ledger)) == 4

    with ledger.open("a", encoding="utf-8") as handle:
        handle.write('{"key":"clip_4.mp4","size":128,"dig')

    result = reconcile_ledgers(ledger, ledger)
    assert result.usable is False
    assert result.proposal() is None
    assert any("truncated" in reason for reason in result.incoherent)


def test_rerun_resumes_and_skips(tmp_path):
    keys = make_card(tmp_path / "card", count=6)
    first = hash_source(tmp_path / "bundle", keys, limit=3)
    assert (first.hashed, first.complete, first.interrupted) == (3, False, True)
    second = hash_source(tmp_path / "bundle", keys)
    assert second.hashed == 3
    assert second.skipped_existing == 3
    assert second.complete is True
    assert len(already_hashed(ledger_dir(tmp_path / "bundle") / "hash.jsonl")) == 6


def test_two_runs_over_an_unchanged_source_are_byte_identical(tmp_path):
    keys = make_card(tmp_path / "card")
    hash_source(tmp_path / "bundle-a", keys)
    hash_source(tmp_path / "bundle-b", keys)
    a = (ledger_dir(tmp_path / "bundle-a") / "hash.jsonl").read_bytes()
    b = (ledger_dir(tmp_path / "bundle-b") / "hash.jsonl").read_bytes()
    assert a == b


def test_a_corrupt_trailing_line_does_not_block_resuming(tmp_path):
    keys = make_card(tmp_path / "card", count=5)
    bundle = tmp_path / "bundle"
    hash_source(bundle, keys, limit=3)
    ledger = ledger_dir(bundle) / "hash.jsonl"
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write('{"key":"broken"')  # no newline: a killed producer
    resumed = hash_source(bundle, keys)
    assert resumed.hashed == 2
    assert resumed.skipped_existing == 3
    assert len(already_hashed(ledger)) == 5


def test_resuming_never_glues_a_record_onto_a_partial_line(tmp_path):
    """A defect this test was written to catch, in one assertion.

    Appending straight onto an unterminated line merges the new record into it,
    so the object is lost *and* the ledger keeps a corrupt line: one crash would
    cost two objects. The producer must terminate the damaged line first.
    """
    keys = make_card(tmp_path / "card", count=5)
    bundle = tmp_path / "bundle"
    hash_source(bundle, keys, limit=3)
    ledger = ledger_dir(bundle) / "hash.jsonl"
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write('{"key":"broken","dig')

    result = hash_source(bundle, keys)
    assert result.repaired_partial_line is True
    # all five keys must be independently recoverable
    recovered = already_hashed(ledger)
    assert len(recovered) == 5
    assert set(recovered) == set(keys)
    # ...and the damaged line is still visible as damage, not silently repaired
    assert reconcile_ledgers(ledger, ledger).usable is False


def test_a_clean_ledger_is_never_repaired(tmp_path):
    keys = make_card(tmp_path / "card")
    bundle = tmp_path / "bundle"
    first = hash_source(bundle, keys)
    assert first.repaired_partial_line is False
    second = hash_source(bundle, keys)
    assert second.repaired_partial_line is False


def test_unreadable_objects_are_counted_not_swallowed(tmp_path):
    keys = make_card(tmp_path / "card", count=2)
    keys["vanished.mp4"] = tmp_path / "card" / "not-there"
    result = hash_source(tmp_path / "bundle", keys)
    assert result.hashed == 2
    assert result.failed == 1
    assert result.complete is False
    assert any("vanished.mp4" in e for e in result.errors)


def test_error_samples_are_capped(tmp_path):
    keys = {f"missing{i}": tmp_path / f"no{i}" for i in range(30)}
    result = hash_source(tmp_path / "bundle", keys, max_errors=5)
    assert result.failed == 30
    assert len(result.errors) == 5


def test_the_ledger_summarizer_agrees_with_the_producer(tmp_path):
    keys = make_card(tmp_path / "card")
    hash_source(tmp_path / "bundle", keys)
    summary = summarize_ledger(ledger_dir(tmp_path / "bundle") / "hash.jsonl")
    assert summary.coherent is True
    assert summary.records == 4
    assert summary.by_status == {HASHED_STATUS: 4}


# ---------------------------------------------------------------------------
# evidence
# ---------------------------------------------------------------------------
def test_only_a_complete_pass_claims_completion(tmp_path):
    keys = make_card(tmp_path / "card", count=5)
    partial = to_evidence(hash_source(tmp_path / "b1", keys, limit=2))
    assert partial["hash"]["complete"] is False
    assert partial["hash"]["verified_files"] == 2

    full = to_evidence(hash_source(tmp_path / "b2", keys))
    assert full["hash"]["complete"] is True
    assert full["hash"]["verified_files"] == 5
    assert full["hash"]["algorithm"] == DEFAULT_ALGORITHM


def test_a_real_pass_moves_the_state_machine(tmp_path, capsys):
    """DISCOVERED -> HASH_COMPLETE: the number is measured, not asserted.

    The pass also records the inventory it walked, so the bundle is not left
    self-contradictory (`hash.verified_files = 4` beside
    `inventory.discovered_files = 0`), which the machine correctly refuses as
    BLOCKED.
    """
    make_card(tmp_path / "card")
    bundle = write_bundle(tmp_path / "b", campaign(), evidence())
    assert load_status(bundle).state.value == "DISCOVERED"

    code = main(["hash", "--bundle", str(bundle), "--root", str(tmp_path / "card"),
                 "--apply", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == EXIT_OK
    assert payload["hashed"] == 4
    assert payload["applied"] is True

    status = load_status(bundle)
    assert status.state.value == "HASH_COMPLETE"
    assert status.evidence.hashing.verified_files == 4
    assert status.evidence.inventory.discovered_files == 4
    assert status.source_release_allowed is False


def test_a_partial_pass_does_not_claim_completion(tmp_path, capsys):
    make_card(tmp_path / "card", count=6)
    bundle = write_bundle(tmp_path / "b", campaign(), evidence())
    main(["hash", "--bundle", str(bundle), "--root", str(tmp_path / "card"),
          "--limit", "2", "--apply", "--json"])
    capsys.readouterr()
    status = load_status(bundle)
    assert status.evidence.hashing.verified_files == 2
    assert status.evidence.hashing.complete is False
    assert status.source_release_allowed is False


def test_hash_does_not_apply_without_the_flag(tmp_path, capsys):
    make_card(tmp_path / "card")
    bundle = write_bundle(tmp_path / "b", campaign(), evidence())
    before = (bundle / "evidence.json").read_text(encoding="utf-8")
    main(["hash", "--bundle", str(bundle), "--root", str(tmp_path / "card"), "--json"])
    assert (bundle / "evidence.json").read_text(encoding="utf-8") == before


def test_an_empty_card_exits_clean(tmp_path, capsys):
    bundle = write_bundle(tmp_path / "b", campaign(), evidence())
    missing = tmp_path / "empty-card"
    missing.mkdir()
    code = main(["hash", "--bundle", str(bundle), "--root", str(missing),
                 "--apply", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == EXIT_OK   # nothing to hash is not a failure
    assert payload["complete"] is True
    assert payload["discovered"] == 0


# ---------------------------------------------------------------------------
# read-only proof
# ---------------------------------------------------------------------------
def test_hashing_leaves_the_source_untouched(tmp_path):
    keys = make_card(tmp_path / "card")
    before = {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in keys.values()}
    hash_source(tmp_path / "bundle", keys)
    after = {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in keys.values()}
    assert before == after


AUDIT_SCRIPT = r"""
import json, os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, %(repo)r)
os.chdir(%(repo)r)
from auto_ingest.custody.hashing import digest_file, hash_source

WRITES = set()
BUNDLE = %(bundle)r
MUTATING = %(mutating)r
violations = []

def hook(event, args):
    if event in MUTATING:
        violations.append({"event": event, "args": [str(a) for a in args]})
        return
    if event == "open":
        path, mode = str(args[0]), args[1]
        if mode and any(ch in mode for ch in ("w", "a", "x", "+")):
            # the only permitted writer is the bundle's own ledger
            if not path.startswith(BUNDLE):
                violations.append({"event": "open-write", "path": path, "mode": str(mode)})

sys.addaudithook(hook)

keys = dict(%(keys)r)
result = hash_source(BUNDLE, keys)
print(json.dumps({"violations": violations, "hashed": result.hashed}))
"""


def test_no_write_outside_the_bundle(tmp_path):
    """The audit hook is scoped: writes inside the bundle are the ledger's job."""
    keys = make_card(tmp_path / "card", count=3)
    bundle = str((tmp_path / "bundle").resolve())
    script = AUDIT_SCRIPT % {
        "repo": str(REPO_ROOT),
        "bundle": bundle,
        "keys": [(k, str(v)) for k, v in keys.items()],
        "source_files": [str(p) for p in keys.values()],
        "mutating": ["os.remove", "os.rename", "os.rmdir", "os.symlink",
                     "os.chmod", "os.chown", "os.truncate", "subprocess.Popen"],
    }
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        cwd=str(REPO_ROOT), timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["violations"] == [], result["violations"]
    assert result["hashed"] == 3


def test_the_source_module_cannot_write():
    import inspect

    from auto_ingest.custody import hashing

    source = inspect.getsource(hashing)
    for banned in ("os.remove", "os.rename", "os.rmdir", "os.symlink",
                   "shutil", "subprocess"):
        assert banned not in source, banned


# ---------------------------------------------------------------------------
# the real entry point
# ---------------------------------------------------------------------------
def test_hash_through_the_real_cli(tmp_path):
    make_card(tmp_path / "card")
    bundle = write_bundle(tmp_path / "b", campaign(), evidence())
    proc = subprocess.run(
        [sys.executable, str(CLI), "custody", "hash", "--bundle", str(bundle),
         "--root", str(tmp_path / "card"), "--apply", "--json"],
        capture_output=True, text=True, cwd=str(tmp_path),
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["hashed"] == 4
    assert payload["applied"] is True

    proc2 = subprocess.run(
        [sys.executable, str(CLI), "custody", "status", "--bundle", str(bundle), "--json"],
        capture_output=True, text=True, cwd=str(tmp_path),
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, timeout=120,
    )
    assert json.loads(proc2.stdout)["hash"]["verified"] == 4


def test_hash_help_is_reachable():
    proc = subprocess.run(
        [sys.executable, str(CLI), "custody", "hash", "--help"],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=60,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert proc.returncode == 0, proc.stderr
    assert "--root" in proc.stdout
    assert "--limit" in proc.stdout


def test_a_ledger_at_card_scale_is_produced_efficiently(tmp_path):
    """67,644 objects: the producer must be linear, not quadratic."""
    import time

    card = tmp_path / "card"
    card.mkdir()
    keys = {}
    for i in range(2000):
        p = card / f"c{i:05d}.mp4"
        p.write_bytes(b"\0" * 64)
        keys[f"c{i:05d}.mp4"] = p

    started = time.monotonic()
    result = hash_source(tmp_path / "bundle", keys)
    elapsed = time.monotonic() - started
    assert result.hashed == 2000
    # generous, but an fsync-per-record regression would blow straight past it
    assert elapsed < 60.0, f"hashing 2000 tiny files took {elapsed:.1f}s"


# ---------------------------------------------------------------------------
# names: a vfat card copied to a case-insensitive share
# ---------------------------------------------------------------------------
def make_colliding_card(root: Path, pairs: int = 1) -> "dict[str, Path]":
    """A card that cannot exist as two files: every clip has two spellings.

    Written on a case-sensitive filesystem on purpose - the point is that the
    *detector* notices what the real vfat card would have collapsed into one name.
    """
    root.mkdir(parents=True, exist_ok=True)
    keys = {}
    for i in range(pairs):
        for spelling in (f"clip_{i:03d}.mp4", f"CLIP_{i:03d}.mp4"):
            path = root / spelling
            path.write_bytes(spelling.encode("utf-8") * 8)
            keys[spelling] = path
    return keys


def collision_rows(bundle: Path) -> "list[dict]":
    ledger = ledger_dir(bundle) / "collisions.jsonl"
    assert ledger.is_file(), "no collision ledger was written"
    return [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]


def test_two_keys_differing_only_in_case_are_a_collision_and_both_are_kept(tmp_path):
    """The core defect: on vfat these are ONE file, and we must not pick a winner.

    Both objects are still hashed and still recorded under their exact spellings.
    A dedupe here would silently drop one of two real clips.
    """
    keys = make_colliding_card(tmp_path / "card", pairs=1)
    keys["unique.mp4"] = tmp_path / "card" / "unique.mp4"
    keys["unique.mp4"].write_bytes(b"unique")

    result = hash_source(tmp_path / "bundle", keys)

    assert result.hashed == 3
    assert result.failed == 0
    assert result.collisions == 2
    assert result.destination_collisions == 0
    assert result.unrepresentable_names == 0

    # ...and both spellings survive in the hash ledger, byte-accurate.
    hashed = already_hashed(ledger_dir(tmp_path / "bundle") / "hash.jsonl")
    assert set(hashed) == {"clip_000.mp4", "CLIP_000.mp4", "unique.mp4"}
    hash_rows = [json.loads(line) for line in
                 (ledger_dir(tmp_path / "bundle") / "hash.jsonl").read_text().splitlines()]
    assert sorted(r["key"] for r in hash_rows) == ["CLIP_000.mp4", "clip_000.mp4",
                                                    "unique.mp4"]
    assert all(r["status"] == HASHED_STATUS and len(r["digest"]) == 64
               for r in hash_rows)

    rows = collision_rows(tmp_path / "bundle")
    assert [row["key"] for row in rows] == ["CLIP_000.mp4", "clip_000.mp4"]
    assert {row["status"] for row in rows} == {CASE_COLLISION}
    # `folded` is the comparison form - the destination's UPCASE form, so it is
    # upper. It is a field on the row, never a rewrite of the key beside it, which
    # is why the key assertions above are unchanged by it.
    assert rows[0]["folded"] == rows[1]["folded"] == "CLIP_000.MP4"
    # the sample line names both sides, so an operator sees which two
    assert "clip_000.mp4" in result.name_problems[0]
    assert "CLIP_000.mp4" in result.name_problems[0]


def test_a_name_problem_is_never_custody_and_never_a_failure(tmp_path):
    """Detection is not rejection: the pass still succeeds and still proves bytes.

    The object is recorded and verified; what is refused is the *release*, and the
    refusal is carried by the evidence rather than by dropping the file.
    """
    keys = make_colliding_card(tmp_path / "card", pairs=1)
    result = hash_source(tmp_path / "bundle", keys)
    assert result.complete is True
    assert result.failed == 0
    # the collision rows are not source-verified statuses, so no reader can
    # mistake a name problem for custody
    summary = summarize_ledger(ledger_dir(tmp_path / "bundle") / "collisions.jsonl")
    assert summary.files == 0
    assert summary.by_status == {CASE_COLLISION: 2}
    # ...and the diff still reads every object as verified on the source side
    ledger = ledger_dir(tmp_path / "bundle") / "hash.jsonl"
    assert reconcile_ledgers(ledger, ledger).verified == 2


def test_a_collision_blocks_release_through_the_existing_error_counters(tmp_path, capsys):
    """End to end: CLI hash -> evidence -> the release gate refuses.

    The collision rides `errors.unresolved`, the counter the gate already refuses
    on, so it can never be a pass without the gate learning new vocabulary.
    """
    make_colliding_card(tmp_path / "card", pairs=1)
    bundle = write_bundle(tmp_path / "b", campaign(), evidence())
    code = main(["hash", "--bundle", str(bundle), "--root", str(tmp_path / "card"),
                 "--apply", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == EXIT_OK
    assert payload["case_collisions"] == 2
    assert payload["name_problem_count"] == 2

    status = load_status(bundle)
    assert status.evidence.errors.unresolved == 2
    assert status.source_release_allowed is False
    codes = [b.code for b in status.release.blockers]
    assert "unresolved_errors" in codes
    assert any("case_collision" in s for s in status.evidence.errors.summaries)


def test_a_case_fold_match_against_the_destination_is_reported(tmp_path):
    """The dangerous one: the copy would land on a name that already exists.

    `os.replace` resolves this by clobbering the existing object or by failing
    against the server, and the executor's "never overwrite" check is an
    exact-name check, so neither protects against it.
    """
    keys = make_card(tmp_path / "card", count=1)
    result = hash_source(tmp_path / "bundle", keys, destination_keys=["CLIP_0.MP4"])

    assert result.destination_collisions == 1
    assert result.collisions == 0  # exact-case source keys: not a source collision
    sample = result.name_problems[0]
    assert sample.startswith(f"{DESTINATION_CASE_COLLISION}: ")
    assert "'CLIP_0.MP4'" in sample and "'clip_0.mp4'" in sample
    rows = collision_rows(tmp_path / "bundle")
    assert [row["status"] for row in rows] == [DESTINATION_CASE_COLLISION]


def test_an_object_already_at_the_destination_under_its_own_name_is_not_a_collision(tmp_path):
    """The ordinary case: same name, already in custody. Reporting that would
    block every campaign that has ever copied anything."""
    keys = make_card(tmp_path / "card", count=1)
    result = hash_source(tmp_path / "bundle", keys, destination_keys=["clip_0.mp4"])
    assert result.destination_collisions == 0
    assert result.name_problem_count == 0
    assert not (ledger_dir(tmp_path / "bundle") / "collisions.jsonl").exists()


def test_the_destination_ledger_is_used_when_no_listing_is_passed(tmp_path):
    """The cross-boundary check needs no extra plumbing: the ledger is the account.

    CARD-01's own sample ledger records a join key and the destination path
    beside it, and a source key is only safe against every name the destination
    really holds.
    """
    from auto_ingest.custody.ledger import DESTINATION_LEDGER

    keys = make_card(tmp_path / "card", count=1)
    bundle = tmp_path / "bundle"
    root = ledger_dir(bundle)
    root.mkdir(parents=True, exist_ok=True)
    (root / DESTINATION_LEDGER).write_text(
        json.dumps({"key": "2026_0412_100205_F", "path": "CLIP_0.MP4",
                    "status": "verified_at_destination"}) + "\n",
        encoding="utf-8",
    )

    result = hash_source(bundle, keys)
    assert result.destination_collisions == 1
    assert "'clip_0.mp4' is already present at the destination as " \
           "'CLIP_0.MP4'" in result.name_problems[0]


def test_the_upcase_table_folds_dotless_i_which_casefold_does_not(tmp_path):
    """The headline: `I` and dotless `ı` are ONE filename on exFAT, and casefold missed it.

    U+0049 and U+0131 both upcase to U+0049, so the destination's one-to-one upcase
    table makes `ILKAY.mp4` and `ılkay.mp4` a single name - the second copy lands on
    the first and one object is gone. `str.casefold` keeps them apart ('i' vs 'ı'),
    which is how this pair reached the planner as two distinct objects with a copy
    queued onto a name the destination already holds.

    Detection is not deduplication: both objects are still hashed, both are still
    recorded under their exact spellings, no winner is chosen, and the campaign is
    blocked for a human to resolve.
    """
    capital_i, dotless_i = "I", "ı"
    assert capital_i.upper() == dotless_i.upper()      # the on-disk mechanism
    assert capital_i.casefold() != dotless_i.casefold()   # the fold that missed it
    assert simple_upcase(capital_i) == simple_upcase(dotless_i) == "I"

    keys = {}
    for spelling in (f"{capital_i}LKAY.mp4", f"{dotless_i}lkay.mp4"):
        path = tmp_path / "card" / spelling
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(spelling.encode("utf-8") * 8)
        keys[spelling] = path

    result = hash_source(tmp_path / "bundle", keys)

    assert result.collisions == 2, "one collision row per colliding key, not one per pair"
    assert result.name_problem_count == 2
    rows = collision_rows(tmp_path / "bundle")
    assert sorted(r["key"] for r in rows) == sorted(keys)
    assert {r["status"] for r in rows} == {CASE_COLLISION}
    # `folded` is the upcase form both names share - the one the destination compares.
    assert {r["folded"] for r in rows} == {"ILKAY.MP4"}
    # both sides named in the sample, so an operator sees which two to choose between
    assert f"{capital_i}LKAY.mp4" in result.name_problems[0]
    assert f"{dotless_i}lkay.mp4" in result.name_problems[0]

    # BOTH objects survive, byte-accurately, with their own digests: no winner,
    # no skip, no rename. This is the invariant a dedupe would have broken.
    hashed = already_hashed(ledger_dir(tmp_path / "bundle") / "hash.jsonl")
    assert set(hashed) == set(keys)
    digests = {r["key"]: r["digest"] for r in
               [json.loads(line) for line in
                (ledger_dir(tmp_path / "bundle") / "hash.jsonl").read_text().splitlines()]}
    assert len(set(digests.values())) == 2, "two distinct objects, not one name twice"


def test_dotted_capital_i_and_plain_i_are_two_names_and_are_not_reported():
    """The other direction: not every near-pair is a collision, and this one is not.

    U+0130 `İ` has an empty simple-uppercase field, so an upcase table sends it to
    itself, while `i` goes to U+0049 `I`. Two different code points are two different
    filenames on exFAT, and casefold keeps them apart too (`i`+U+0307 vs `i`), so
    neither fold reports it. Reporting it would be the false positive this design
    accepts elsewhere, but there is no reason to accept it here: it would block a
    campaign that has nothing wrong with it.

    The one place this fold is wrong is the Turkish locale, where Windows upcases
    `i` to `İ` and these two names DO collide. That is documented in the module
    docstring; the tables here are locale-invariant because a locale is not a
    property of a filesystem.
    """
    dotted, plain = "\u0130", "i"
    assert simple_upcase(dotted) == dotted != simple_upcase(plain)
    assert fold_key(f"{dotted}LKAY.mp4") != fold_key("ilkay.mp4")
    assert detect_name_problems([f"{dotted}LKAY.mp4", "ilkay.mp4"]) == []


def test_dotted_capital_i_still_collides_with_its_own_decomposed_lowercase():
    """The casefold fold earns its place: `İ` casefolds to `i` + U+0307.

    No upcase table can express that - a per-character table has no slot for a
    two-character sequence - so this pair is invisible to the primary fold and is
    found by the second one. Both rows appear, and both keys stay in the ledger.
    """
    dotted, decomposed = "\u0130", "i\u0307"
    assert simple_upcase(dotted) != simple_upcase(decomposed)   # upcase cannot see it
    assert dotted.casefold() == decomposed.casefold()           # casefold can

    problems = detect_name_problems([f"{dotted}LKAY.mp4", f"{decomposed}lkay.mp4"])
    assert [p.kind for p in problems] == [CASE_COLLISION, CASE_COLLISION]
    assert [p.key for p in problems] == [f"{decomposed}lkay.mp4", f"{dotted}LKAY.mp4"]


@pytest.mark.parametrize("left, right", [
    ("Straße.mp4", "STRASSE.mp4"),   # sharp s
    ("Faßband.MP4", "FASSBAND.mp4"),
    ("ﬁle.mp4", "FILE.mp4"),           # fi ligature
])
def test_names_the_upcase_table_keeps_apart_are_still_reported(left, right):
    """Reported anyway. This is the deliberate false positive, and here is its cost.

    `ß`.upper() is `SS` and `ﬁ`.upper() is `FI`, but a one-to-one table has a single
    slot per code point, so it maps `ß` to `ß` and the destination holds `ß.mp4`
    and `SS.mp4` as two separate files. casefold folds both to `ss`/`fi`, so this
    detector reports a collision the destination does not have.

    Kept on purpose: the cost is an operator renaming a file that did not strictly
    need renaming, and the alternative error is a file overwritten with nothing
    showing it happened. One costs five minutes; the other costs a card.
    """
    assert simple_upcase(left) != simple_upcase(right)   # the table keeps them apart
    problems = detect_name_problems([left, right])
    assert [p.kind for p in problems] == [CASE_COLLISION, CASE_COLLISION]
    assert sorted(p.key for p in problems) == sorted([left, right])


def test_long_s_is_equated_by_both_folds():
    """The converse data point: some pair agrees under both folds.

    `ſ` is U+017F LATIN SMALL LETTER LONG S. Unicode's *simple* uppercase maps it to
    U+0053 `S` - a one-character mapping, so an upcase table does the same thing -
    and casefold maps it to `s` as well. Nothing about this pair is a judgement
    call; it is what both operations independently say, and what the destination
    does.
    """
    assert simple_upcase("ſ") == simple_upcase("S") == "S"
    assert "S".casefold() == "ſ".casefold() == "s"
    problems = detect_name_problems(["S.mp4", "ſ.mp4"])
    assert [p.kind for p in problems] == [CASE_COLLISION, CASE_COLLISION]
    assert sorted(p.key for p in problems) == ["S.mp4", "ſ.mp4"]


def test_the_original_ascii_case_collision_still_collides():
    """The behaviour that was already right must stay right: `Photo.MP4`/`photo.mp4`."""
    assert fold_key("Photo.MP4") == fold_key("photo.mp4") == "PHOTO.MP4"
    problems = detect_name_problems(["Photo.MP4", "photo.mp4"])
    assert [p.kind for p in problems] == [CASE_COLLISION, CASE_COLLISION]
    assert [p.key for p in problems] == ["Photo.MP4", "photo.mp4"]
    assert detect_name_problems(["A.mp4", "a.mp4"])[0].kind == CASE_COLLISION


def test_ascii_names_are_detected_exactly_as_the_old_casefold_only_rule_detected_them():
    """No behaviour change for the common case, and that is provable rather than hoped.

    For an ASCII name the upcase form is `key.upper()` and the casefold form is
    `key.lower()`, and on ASCII "equal when uppercased" and "equal when lowercased"
    are the SAME relation - upper/lower are mutually inverse on ASCII letters. So
    adding the upcase fold cannot add an ASCII collision, and casefold was already
    there, so it cannot lose one either. Checked exhaustively rather than asserted.
    """
    import itertools

    alphabet = "aA0_.-"
    names = ["".join(chars) for chars in itertools.product(alphabet, repeat=3)]

    def old_rule(x: str, y: str) -> bool:
        return x.casefold() == y.casefold()      # what this module used to do

    def new_rule(x: str, y: str) -> bool:
        return bool(set(fold_forms(x)) & set(fold_forms(y)))

    disagreements = [(x, y) for x, y in itertools.product(names, repeat=2)
                     if old_rule(x, y) != new_rule(x, y)]
    assert disagreements == []
    # ...and the forms themselves are plain upper/lower on ASCII
    assert fold_forms("A/b_0.MP4") == ("A/B_0.MP4", "a/b_0.mp4")
    assert fold_forms("123") == ("123",)      # no case at all: one form, not two


def test_names_that_really_are_distinct_are_not_reported():
    """The guard on the other side: over-detection must not turn every card into a blocker.

    Detection decides whether a human may erase a card, so it is deliberately
    eager - but a detector that flags everything is a detector nobody reads. These
    keys share no fold form under either fold and must come back clean.
    """
    keys = ["clip_0.mp4", "clip_1.mp4", "2026/04/12/2026_0412_100205_F.MP4",
            "dcim/movie/100DSCQ.MP4", "İLKAY.mp4", "ß.mp4", "ﬁ.mp4",
            "ILKAY.mp4", "MVC1.MP4"]
    assert detect_name_problems(keys) == []
    # a near miss is still a miss: one character apart is two names
    assert detect_name_problems(["clip_0.mp4", "clip_0x.mp4"]) == []
    # ...and the ASCII pair above is still found, so this list is not just "empty in"
    assert detect_name_problems(["clip_0.mp4", "CLIP_0.mp4"])[0].kind == CASE_COLLISION


def test_simple_upcase_is_one_character_in_and_one_character_out_for_all_of_unicode():
    """The property the whole model rests on, checked against every code point.

    An upcase table has exactly one slot per code point, so the fold must never grow
    or shrink a name. Python's `str.upper` does grow it (102 characters expand, e.g.
    `ß` to `SS`), which is why the full mapping cannot be used directly.
    """
    expansions = 0
    for point in range(0x110000):
        ch = chr(point)
        folded = simple_upcase(ch)
        assert len(folded) == 1, f"U+{point:04X} did not fold 1:1"
        expansions += len(ch.upper()) != 1
    assert expansions == 102, "the expanding set moved; re-derive the table"


def test_the_expansion_table_is_the_polytonic_greek_set_and_nothing_else():
    """Pinned so a wrong entry cannot hide in here.

    A simple uppercase that differs from the full one exists for exactly 27
    characters: the polytonic Greek small letters whose precomposed
    capital-with-prosgegrammeni exists, in three regular blocks of eight at a fixed
    `+0x08` offset plus three singles. Every other expanding character has an empty
    simple-uppercase field and so maps to ITSELF - which is why `ß` is not folded to
    `SS` here, and why `ᾀ` (U+1F80) maps to `ᾈ` (U+1F88) and not to plain `Α`.
    """
    assert len(SIMPLE_UPPERCASE_EXPANSION) == 27
    for start in ("\u1f80", "\u1f90", "\u1fa0"):
        for offset in range(8):
            assert SIMPLE_UPPERCASE_EXPANSION[chr(ord(start) + offset)] == \
                chr(ord(start) + 0x08 + offset)
    assert SIMPLE_UPPERCASE_EXPANSION["\u1fb3"] == "\u1fbc"
    assert SIMPLE_UPPERCASE_EXPANSION["\u1fc3"] == "\u1fcc"
    assert SIMPLE_UPPERCASE_EXPANSION["\u1ff3"] == "\u1ffc"

    # ß, ﬁ, ﬄ, ŉ, ǰ and the Armenian ligatures have NO entry: they map to themselves
    for ch in ("ß", "ﬁ", "ﬄ", "ŉ", "ǰ", "և"):
        assert ch not in SIMPLE_UPPERCASE_EXPANSION
        assert simple_upcase(ch) == ch
    assert simple_upcase("ß") == "ß" != "SS"
    # ...and the plain-capital reading of the Greek set would have been wrong
    assert simple_upcase("\u1f80") == "\u1f88" != "\u0391"


def test_an_over_long_name_is_flagged_at_hash_time(tmp_path):
    """A name the destination cannot store must not be discovered mid-copy.

    The file cannot be created on this test root - ext4 enforces the same 255-byte
    component limit - which is the whole reason the check belongs at hash time and
    not in the middle of a copy. So the key is exercised without a real file: the
    name problem is a property of the key, and the pass still hashes everything it
    can and records the offending key byte-accurately.
    """
    long_name = f"{'n' * (NAME_COMPONENT_LIMIT_BYTES + 5)}.mp4"
    keys = make_card(tmp_path / "card", count=1)
    keys[long_name] = tmp_path / "card" / long_name

    result = hash_source(tmp_path / "bundle", keys)

    assert result.unrepresentable_names == 1
    assert result.name_problem_count == 1
    assert str(NAME_COMPONENT_LIMIT_BYTES) in result.name_problems[0]
    rows = collision_rows(tmp_path / "bundle")
    assert [(row["key"], row["status"]) for row in rows] == [
        (long_name, UNREPRESENTABLE_NAME)
    ]
    # the readable objects are unaffected by their neighbour's bad name
    assert result.hashed == 1


def test_a_long_multi_byte_name_is_flagged_even_though_it_is_legal_on_the_card():
    """The case a byte-count catches and a character count would not.

    vfat and exFAT count UTF-16 code units, so 200 CJK characters are a perfectly
    legal filename on the card - and 600 bytes, which ext4 and APFS reject. The
    rule measures bytes for exactly this reason.
    """
    cjk = "年" * 200
    assert len(cjk) == 200
    assert len(cjk.encode("utf-8")) == 600
    assert filename_problem(f"{cjk}.mp4") is not None
    assert "255" in filename_problem(f"{cjk}.mp4")

    # a long name that every filesystem in play can store is NOT flagged:
    # 250 bytes is under the limit, and detection must not cry wolf
    assert filename_problem("n" * 250) is None
    # ...and an over-long whole key is caught
    deep = "/".join(["dir"] * 1200) + "/clip.mp4"
    assert len(deep.encode("utf-8")) > 4096
    assert "4096" in filename_problem(deep)


def test_an_over_long_name_at_a_real_255_byte_boundary_is_not_flagged(tmp_path):
    """The limit is 255 bytes, not "anything long": a legal name must pass."""
    exact = f"{'n' * (NAME_COMPONENT_LIMIT_BYTES - len('.mp4'))}.mp4"
    assert len(exact.encode("utf-8")) == NAME_COMPONENT_LIMIT_BYTES
    assert filename_problem(exact) is None
    keys = make_card(tmp_path / "card", count=1)
    keys[exact] = tmp_path / "card" / exact
    exact_path = tmp_path / "card" / exact
    exact_path.write_bytes(b"payload")
    result = hash_source(tmp_path / "bundle", keys)
    assert result.unrepresentable_names == 0
    assert result.hashed == 2


@pytest.mark.parametrize("bad", ["clip:01.mp4", 'what?.mp4', "a|b.mp4", "back\\slash.mp4",
                                 "star*.mp4", "quote\".mp4", "less<.mp4", "greater>.mp4",
                                 "bell\x07.mp4", "trailing.mp4.", "trailing.mp4 "])
def test_a_name_the_destination_filesystem_cannot_store_is_flagged(tmp_path, bad):
    keys = make_card(tmp_path / "card", count=1)
    path = tmp_path / "card" / bad
    path.write_bytes(b"payload")
    keys[bad] = path

    result = hash_source(tmp_path / "bundle", keys)

    assert result.unrepresentable_names == 1
    assert result.name_problems[0].startswith(f"{UNREPRESENTABLE_NAME}: ")
    # the object is still hashed and still recorded under its exact key
    assert result.hashed == 2
    assert bad in already_hashed(ledger_dir(tmp_path / "bundle") / "hash.jsonl")


def test_an_ordinary_name_is_not_flagged():
    """The check must not cry wolf, or it would be turned off."""
    for good in ["clip_0001.mp4", "DCIM/Movie/2026_0412_100205_F.mp4",
                 "2026-04-12 10-02-05.123.mp4", "ünïcode-ß.mp4",
                 "no-extension", "..leading-dots.mp4", "two.dots.in.a.name.mp4"]:
        assert filename_problem(good) is None, good


def test_evidence_stays_bounded_at_ten_thousand_collisions():
    """The campaign summary must never grow one row per file.

    Every colliding name is still *detected* - nothing is deduplicated or dropped -
    but the evidence carries a count and at most `max_summary_entries` samples.
    Per-file detail belongs in the append-only collision ledger.
    """
    pairs = 5_000
    keys = []
    for i in range(pairs):
        keys.append(f"clip_{i:05d}.mp4")
        keys.append(f"CLIP_{i:05d}.mp4")

    problems = detect_name_problems(keys)
    assert len(problems) == 10_000          # both members of every pair reported
    assert len({p.key for p in problems}) == 10_000

    result = HashProgress(
        ledger_path=str(ledger_dir("/nonexistent") / "hash.jsonl"),
        collisions=len(problems),
        name_problems=tuple(p.sample for p in problems),
    )
    fragment = to_evidence(result, discovered=len(keys))
    assert fragment["errors"]["unresolved"] == 10_000
    assert len(fragment["errors"]["summaries"]) == MAX_SUMMARY_ENTRIES
    assert len(json.dumps(fragment)) < 4096

    # ...and the reader caps it again, so a hand-edited fragment cannot smuggle
    # ten thousand rows into a campaign bundle
    from auto_ingest.custody.evidence import CampaignEvidence

    reread = CampaignEvidence.from_dict(fragment)
    assert reread.errors.unresolved == 10_000
    assert len(reread.errors.summaries) == MAX_SUMMARY_ENTRIES


def test_the_producer_writes_one_row_per_colliding_name_and_caps_the_samples(tmp_path):
    """The same bound, through the real producer: per-file in the ledger, capped
    in the evidence."""
    pairs = 30
    keys = make_colliding_card(tmp_path / "card", pairs=pairs)
    result = hash_source(tmp_path / "bundle", keys, max_errors=MAX_SUMMARY_ENTRIES)

    assert result.collisions == 2 * pairs
    assert len(result.name_problems) == MAX_SUMMARY_ENTRIES
    assert len(collision_rows(tmp_path / "bundle")) == 2 * pairs
    assert result.hashed == 2 * pairs


def test_a_resume_neither_duplicates_nor_forgets_the_collisions(tmp_path):
    """Append-only means idempotent: a second pass leaves the ledger byte-identical.

    Detection is a property of the names, so the resume must report the same
    collisions the first pass did - not a delta, and not a doubled ledger.
    """
    keys = make_colliding_card(tmp_path / "card", pairs=2)
    bundle = tmp_path / "bundle"
    first = hash_source(bundle, keys, limit=1)
    assert first.collisions == 4          # reported before the probe ran out
    assert first.interrupted is True
    ledger = ledger_dir(bundle) / "collisions.jsonl"
    before = ledger.read_bytes()

    resumed = hash_source(bundle, keys)
    assert resumed.collisions == 4
    assert resumed.hashed == 3            # only the three it had not hashed
    assert resumed.skipped_existing == 1
    assert ledger.read_bytes() == before


def test_two_runs_over_a_colliding_card_produce_identical_ledgers(tmp_path):
    keys = make_colliding_card(tmp_path / "card", pairs=2)
    hash_source(tmp_path / "bundle-a", keys)
    hash_source(tmp_path / "bundle-b", keys)
    for name in ("hash.jsonl", "collisions.jsonl"):
        a = (ledger_dir(tmp_path / "bundle-a") / name).read_bytes()
        b = (ledger_dir(tmp_path / "bundle-b") / name).read_bytes()
        assert a == b, name
        assert a.endswith(b"\n")


def test_one_key_can_carry_two_problems_and_both_are_recorded(tmp_path):
    """A name can both fold onto its twin and be unrepresentable. Both count."""
    keys = {}
    for spelling in ("clip:000.mp4", "CLIP:000.mp4"):
        path = tmp_path / "card" / spelling
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(spelling.encode("utf-8"))
        keys[spelling] = path

    result = hash_source(tmp_path / "bundle", keys)
    assert result.collisions == 2
    assert result.unrepresentable_names == 2
    assert result.name_problem_count == 4
    kinds = {(row["key"], row["status"]) for row in collision_rows(tmp_path / "bundle")}
    assert kinds == {
        ("clip:000.mp4", CASE_COLLISION), ("CLIP:000.mp4", CASE_COLLISION),
        ("clip:000.mp4", UNREPRESENTABLE_NAME), ("CLIP:000.mp4", UNREPRESENTABLE_NAME),
    }
    # and the resume does not re-append any of them
    before = (ledger_dir(tmp_path / "bundle") / "collisions.jsonl").read_bytes()
    hash_source(tmp_path / "bundle", keys)
    assert (ledger_dir(tmp_path / "bundle") / "collisions.jsonl").read_bytes() == before


def test_a_clean_card_records_no_collision_ledger_and_touches_no_error_counter(tmp_path):
    """No regression on the normal path - the existing fixture shape.

    A clean pass must leave the summary alone: `import` merges block by block and
    this document wins, so writing `unresolved: 0` here would erase a count some
    other producer recorded.
    """
    keys = make_card(tmp_path / "card", count=4)
    result = hash_source(tmp_path / "bundle", keys)
    assert (result.collisions, result.destination_collisions,
            result.unrepresentable_names) == (0, 0, 0)
    assert result.name_problems == ()
    assert not (ledger_dir(tmp_path / "bundle") / "collisions.jsonl").exists()

    fragment = to_evidence(result, discovered=len(keys))
    assert "errors" not in fragment
    assert fragment["hash"]["verified_files"] == 4
    ledger = ledger_dir(tmp_path / "bundle") / "hash.jsonl"
    assert reconcile_ledgers(ledger, ledger).verified == 4


def test_existing_exact_case_keys_are_untouched_and_still_distinct():
    """Backwards compatibility: existing campaigns and ledgers use exact-case keys.

    Normalisation is for COMPARISON. It rewrites nothing: the ledger key stays the
    filename, so every pre-existing ledger reads exactly as it did. Note the
    comparison form of a lower-case key is now its UPPER case - that field changed
    with the fold, and nothing else did, because nothing ever read it back.
    """
    existing = ["clip_0.mp4", "clip_1.mp4", "dcim/movie/2026_0412_100205_f.mp4",
                "2026_0412_100205_f", "2026/04/12/2026_0412_100205_f.mp4"]
    assert detect_name_problems(existing) == []
    for key in existing:
        assert fold_key(key) == key.upper()

    # a key with upper case still folds for comparison, and is still stored as-is
    assert fold_key("2026/04/12/2026_0412_100205_F.MP4") == \
        "2026/04/12/2026_0412_100205_F.MP4"
    assert detect_name_problems(["2026/04/12/2026_0412_100205_F.MP4"]) == []


def test_the_card_01_fixture_keys_are_still_read_as_distinct_objects():
    """The committed fixture is a pre-existing exact-case campaign.

    It must keep reading exactly as before - including the fact that its
    destination ledger records a key the source also spells that way, which is
    ordinary custody and not a collision.
    """
    from custody_helpers import CARD_01_BUNDLE

    from auto_ingest.custody.hashing import destination_keys_in_bundle

    names = destination_keys_in_bundle(CARD_01_BUNDLE)
    assert "2026_0412_100205_F" in names
    assert "2026/04/12/2026_0412_100205_F.MP4" in names
    # the fixture's own key spelling, already in custody: not a collision
    assert detect_name_problems(["2026_0412_100205_F"],
                                destination_keys=names) == []
    # ...and a differently-spelled twin of a recorded destination name is
    assert [p.kind for p in
            detect_name_problems(["2026_0412_100205_f"], destination_keys=names)] == [
        DESTINATION_CASE_COLLISION
    ]
    assert load_status(CARD_01_BUNDLE).state.value == "RECONCILE_REQUIRED"


# ---------------------------------------------------------------------------
# Pairs that are deliberately NOT reported
# ---------------------------------------------------------------------------
# Both were previously described as blind spots. Measurement showed neither is
# an on-disk collision on the real destination, so "fixing" them would have been
# adding false positives. These tests pin the decision and its reasoning.

def test_turkish_locale_collation_is_not_an_on_disk_collision():
    """`i` and `İ` are different files on exFAT/NTFS, so must not be reported.

    Windows maps i -> İ under a Turkish locale, so Explorer sorts them together.
    That is *display collation*. The on-disk upcase table is locale-invariant:
    it maps i (U+0069) and ı (U+0131) to I, and leaves İ (U+0130) mapped to
    itself. So i.mp4 and İ.mp4 are genuinely two files, and reporting them would
    block a real campaign over nothing.
    """
    assert "i".upper() == "I"
    assert "ı".upper() == "I", "dotless i IS the upcase-equal of ASCII I"
    assert "İ".upper() == "İ", "dotted İ is NOT - it is its own upcase image"

    assert detect_name_problems(["i.mp4", "İ.mp4"]) == [], (
        "Turkish locale collation is not an on-disk collision; reporting it would "
        "be a false positive")


def test_the_locale_equivalent_dotless_pair_IS_reported():
    """The pair the upcase table really does equate must still be caught."""
    problems = detect_name_problems(["ILKAY.mp4", "ılkay.mp4"])
    assert [p.kind for p in problems] == [CASE_COLLISION, CASE_COLLISION], (
        "ILKAY and dotless ılkay are one filename on disk")
    assert sorted(p.key for p in problems) == sorted(["ILKAY.mp4", "ılkay.mp4"])


def test_nfc_and_nfd_are_distinct_on_this_destination():
    """Correct for exFAT/SMB2; wrong only for APFS.

    The destination stores bytes and compares them through the upcase table, so
    the two spellings are two files. Folding them would block any campaign with a
    decomposed accented filename against a destination that has no such problem.

    The two spellings are built from explicit code points rather than pasted
    literals: an editor or a shell can silently normalise the composed form in
    the source, which would make the test compare a string with itself and pass
    for the wrong reason.
    """
    import unicodedata

    nfc = "caf\u00e9.mp4"                      # e-acute, one code point
    nfd = "cafe\u0301.mp4"                     # e + combining acute, two
    assert nfc != nfd
    assert len(nfc) == 8 and len(nfd) == 9
    assert unicodedata.normalize("NFC", nfd) == nfc

    assert detect_name_problems([nfc, nfd]) == [], (
        "exFAT and SMB2 keep the two spellings apart; reporting them is a false "
        "positive. APFS would need a separate, deliberate normalisation pass.")


def test_a_resumed_pass_reports_cumulative_inventory_bytes(tmp_path):
    """inventory.discovered_bytes must be cumulative, like verified_files.

    Found by hashing the real 94GB card twice: the second, resumed pass wrote
    `discovered_files: 67644` beside `discovered_bytes: 0`. The ledger held
    94,278,672,670 bytes. `capacity.py:85` scales its requirement by
    discovered_bytes, so a campaign that had already hashed the whole card
    looked like a zero-byte copy - the copy would be planned with no capacity
    reservation at all.
    """
    src = tmp_path / "card"
    src.mkdir()
    payloads = {"a.mp4": b"A" * 4096, "b.mp4": b"B" * 8192}
    for name, data in payloads.items():
        (src / name).write_bytes(data)

    bundle = tmp_path / "b"
    total = sum(len(v) for v in payloads.values())

    first = hash_source(bundle, {k: src / k for k in payloads})
    assert first.complete and first.hashed == 2
    fragment = to_evidence(first, discovered=2)
    assert fragment["inventory"]["discovered_bytes"] == total

    # Second pass: everything is skipped, so bytes_read is 0. The cumulative
    # figure must still be right, or the campaign silently loses its own size.
    second = hash_source(bundle, {k: src / k for k in payloads})
    assert second.skipped_existing == 2, "resume should skip both"
    assert second.bytes_read == 0, "nothing new was read"
    resumed = to_evidence(second, discovered=2)
    assert resumed["inventory"]["discovered_files"] == 2
    assert resumed["inventory"]["discovered_bytes"] == total, (
        "a fully-skipped pass reported zero bytes for a card that holds "
        f"{total}, so capacity planning saw a zero-byte copy")
    assert resumed["hash"]["verified_bytes"] == total
