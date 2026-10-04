"""Phase B: `custody hash` - the first real producer of a custody ledger.

Everything before this was a state machine over asserted evidence. `67,644`
existed only in tests and a hand-written fixture. These tests pin the producer
and, just as importantly, prove it cannot write to the source.

The fail-closed contract is load-bearing and tested here from the producer's
side: a ledger whose final line is unterminated reads as a crashed producer, and
the reconciler must void the whole diff rather than trust the records it did
manage to read.
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
    CHUNK_BYTES,
    DEFAULT_ALGORITHM,
    HASHED_STATUS,
    already_hashed,
    digest_file,
    hash_source,
    to_evidence,
)
from auto_ingest.custody.ledger import ledger_dir, reconcile_ledgers, summarize_ledger
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
