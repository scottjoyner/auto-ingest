"""`custody verify` - the destination verification producer.

Verification is the only thing that can prove custody, so these tests are mostly
about what it refuses to count. The defect it exists to catch is `--ignore-existing`:
a destination file that exists but is truncated, corrupt or the wrong bytes. Any
implementation that treats presence as proof is worse than no verification.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from custody_helpers import campaign, destination, evidence, write_bundle

from auto_ingest.custody.cli import EXIT_GATE_CLOSED, EXIT_OK, main
from auto_ingest.custody.ledger import ledger_dir
from auto_ingest.custody.store import load_status
from auto_ingest.custody.verify import (
    FAILED,
    MISMATCH,
    MISSING,
    VERIFIED,
    already_verified,
    ends_unterminated,
    source_digests,
    to_evidence,
    verify_destination,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI = REPO_ROOT / "bin" / "auto-ingest"



# The source end of a campaign is a fact this module controls, not a fact
# about whether the developer's card happens to be plugged in.
pytestmark = pytest.mark.usefixtures("hermetic_mounts")
def build(tmp_path, count=4, size=64):
    """A hashed source, an empty destination, and a bundle with an identity."""
    src = tmp_path / "card"
    dest = tmp_path / "dest"
    src.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    bundle = write_bundle(
        tmp_path / "bundle",
        campaign(dest=destination(host_path=str(dest), mounted=True)),
        evidence(),
    )
    keys = {}
    for i in range(count):
        path = src / f"c{i}.mp4"
        path.write_bytes(bytes([65 + i]) * size)
        keys[f"c{i}.mp4"] = path
    # Through the CLI, so the evidence is recorded exactly as an operator would
    # get it - not just the ledger.
    main(["hash", "--bundle", str(bundle), "--root", str(src), "--apply", "--json"])
    return bundle, src, dest, keys


def copy_all(src: Path, dest: Path, keys):
    for key, path in keys.items():
        shutil.copyfile(path, dest / key)


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------
def test_identical_copies_are_verified(tmp_path):
    bundle, src, dest, keys = build(tmp_path)
    copy_all(src, dest, keys)
    result = verify_destination(bundle, dest)
    assert result.verified == 4
    assert (result.missing, result.mismatched, result.failed) == (0, 0, 0)
    assert result.complete is True


def test_an_absent_object_is_missing_not_failed(tmp_path):
    bundle, src, dest, keys = build(tmp_path)
    copy_all(src, dest, keys)
    (dest / "c1.mp4").unlink()
    result = verify_destination(bundle, dest)
    assert result.missing == 1
    assert result.verified == 3
    assert result.failed == 0


def test_a_present_but_corrupt_object_is_a_mismatch(tmp_path):
    """The `--ignore-existing` defect, made detectable."""
    bundle, src, dest, keys = build(tmp_path)
    copy_all(src, dest, keys)
    (dest / "c2.mp4").write_bytes(b"CORRUPT" * 16)
    result = verify_destination(bundle, dest)
    assert result.mismatched == 1
    assert result.verified == 3


def test_a_truncated_object_is_a_mismatch(tmp_path):
    bundle, src, dest, keys = build(tmp_path)
    copy_all(src, dest, keys)
    (dest / "c3.mp4").write_bytes((dest / "c3.mp4").read_bytes()[:8])
    result = verify_destination(bundle, dest)
    assert result.mismatched == 1


def test_an_unreadable_object_is_failed_never_custody(tmp_path):
    bundle, src, dest, keys = build(tmp_path)
    copy_all(src, dest, keys)
    (dest / "c0.mp4").chmod(0o000)
    try:
        result = verify_destination(bundle, dest)
        assert result.failed == 1
        assert result.verified == 3
    finally:
        (dest / "c0.mp4").chmod(0o644)


def test_an_empty_destination_reports_everything_missing(tmp_path):
    bundle, src, dest, keys = build(tmp_path)
    result = verify_destination(bundle, dest)
    assert (result.verified, result.missing) == (0, 4)
    assert result.complete is True


def test_nothing_is_ever_verified_without_a_source_digest(tmp_path):
    """The destination's own claim about itself is not evidence."""
    bundle = tmp_path / "bundle"
    dest = tmp_path / "dest"
    dest.mkdir(parents=True)
    (dest / "orphan.mp4").write_bytes(b"whatever")
    result = verify_destination(bundle, dest)
    assert result.verified == 0
    assert result.checked == 0


def test_source_digests_ignores_unhashed_objects(tmp_path):
    bundle, src, dest, keys = build(tmp_path)
    digests = source_digests(bundle)
    assert set(digests) == set(keys)


# ---------------------------------------------------------------------------
# the ledger
# ---------------------------------------------------------------------------
def test_records_carry_the_right_statuses(tmp_path):
    bundle, src, dest, keys = build(tmp_path)
    copy_all(src, dest, keys)
    (dest / "c1.mp4").unlink()
    (dest / "c2.mp4").write_bytes(b"CORRUPT" * 16)
    verify_destination(bundle, dest)
    ledger = ledger_dir(bundle) / "destination.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    by_key = {r["key"]: r["status"] for r in rows}
    assert by_key["c0.mp4"] == VERIFIED
    assert by_key["c1.mp4"] == MISSING
    assert by_key["c2.mp4"] == MISMATCH
    assert by_key["c3.mp4"] == VERIFIED


def test_every_line_is_newline_terminated(tmp_path):
    bundle, src, dest, keys = build(tmp_path)
    copy_all(src, dest, keys)
    verify_destination(bundle, dest)
    raw = (ledger_dir(bundle) / "destination.jsonl").read_bytes()
    assert raw.endswith(b"\n")
    assert raw.count(b"\n") == 4


def test_a_verified_record_carries_the_digest_it_proved(tmp_path):
    bundle, src, dest, keys = build(tmp_path)
    copy_all(src, dest, keys)
    verify_destination(bundle, dest)
    ledger = ledger_dir(bundle) / "destination.jsonl"
    rows = (json.loads(line) for line in ledger.read_text().splitlines())
    for row in rows:
        if row["status"] == VERIFIED:
            assert len(row["digest"]) == 64
            assert row["digest"] == source_digests(bundle)[row["key"]][0]


def test_a_missing_record_carries_no_digest(tmp_path):
    bundle, src, dest, keys = build(tmp_path)
    verify_destination(bundle, dest)
    ledger = ledger_dir(bundle) / "destination.jsonl"
    rows = (json.loads(line) for line in ledger.read_text().splitlines())
    for row in rows:
        if row["status"] == MISSING:
            assert "digest" not in row


def test_rerun_skips_already_verified(tmp_path):
    bundle, src, dest, keys = build(tmp_path)
    copy_all(src, dest, keys)
    first = verify_destination(bundle, dest)
    second = verify_destination(bundle, dest)
    assert first.verified == 4 and second.verified == 0
    assert second.skipped_existing == 4


def test_recheck_forces_reverification(tmp_path):
    bundle, src, dest, keys = build(tmp_path)
    copy_all(src, dest, keys)
    verify_destination(bundle, dest)
    # corrupt after proving, then re-check: the proof must not be sticky
    (dest / "c0.mp4").write_bytes(b"CHANGED" * 16)
    again = verify_destination(bundle, dest, recheck=True)
    assert again.mismatched == 1
    assert again.verified == 3


def test_limit_bounds_the_pass(tmp_path):
    bundle, src, dest, keys = build(tmp_path, count=6)
    copy_all(src, dest, keys)
    result = verify_destination(bundle, dest, limit=2)
    assert result.checked == 2
    assert result.complete is False


def test_resuming_after_a_crash_keeps_every_record_readable(tmp_path):
    bundle, src, dest, keys = build(tmp_path, count=5)
    copy_all(src, dest, keys)
    verify_destination(bundle, dest, limit=3)
    ledger = ledger_dir(bundle) / "destination.jsonl"
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write('{"key":"c3.mp4","stat')
    assert ends_unterminated(ledger) is True
    verify_destination(bundle, dest)
    assert len(already_verified(ledger)) == 5


# ---------------------------------------------------------------------------
# evidence
# ---------------------------------------------------------------------------
def test_only_a_complete_pass_claims_verification_complete(tmp_path):
    bundle, src, dest, keys = build(tmp_path, count=6)
    copy_all(src, dest, keys)
    partial = to_evidence(verify_destination(bundle, dest, limit=2))
    assert partial["destination"]["verification_complete"] is False
    full = to_evidence(verify_destination(bundle, dest))
    assert full["destination"]["verification_complete"] is True


def test_a_verification_resume_reports_the_ledger_total(tmp_path):
    """A re-run must not write 0 over a real verified count."""
    bundle, src, dest, keys = build(tmp_path)
    copy_all(src, dest, keys)
    verify_destination(bundle, dest)
    resumed = verify_destination(bundle, dest)
    assert resumed.verified == 0
    assert resumed.skipped_existing == 4
    fragment = to_evidence(resumed)
    assert fragment["destination"]["verified_files"] == 4


def test_failures_count_against_custody(tmp_path):
    bundle, src, dest, keys = build(tmp_path)
    copy_all(src, dest, keys)
    (dest / "c1.mp4").unlink()
    (dest / "c2.mp4").write_bytes(b"CORRUPT" * 16)
    fragment = to_evidence(verify_destination(bundle, dest))
    # missing + mismatched both lack proven custody
    assert fragment["reconciliation"]["source_only"] == 2
    assert fragment["reconciliation"]["mismatched"] == 1
    assert fragment["destination"]["verified_files"] == 2


def test_verification_does_not_touch_the_source(tmp_path):
    bundle, src, dest, keys = build(tmp_path)
    copy_all(src, dest, keys)
    before = {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in keys.values()}
    verify_destination(bundle, dest)
    assert before == {p: (p.stat().st_mtime_ns, p.stat().st_size)
                      for p in keys.values()}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_verify_finds_failures_and_exits_three(tmp_path, capsys):
    bundle, src, dest, keys = build(tmp_path)
    capsys.readouterr()   # discard the hash pass's output
    copy_all(src, dest, keys)
    (dest / "c2.mp4").write_bytes(b"CORRUPT" * 16)
    code = main(["verify", "--bundle", str(bundle), "--destination", str(dest), "--json"])
    assert code == EXIT_GATE_CLOSED
    payload = json.loads(capsys.readouterr().out)
    assert payload["verification"]["mismatched"] == 1
    assert payload["executed"] is False


def test_cli_verify_apply_records_the_outcome(tmp_path, capsys):
    bundle, src, dest, keys = build(tmp_path)
    capsys.readouterr()
    copy_all(src, dest, keys)
    main(["verify", "--bundle", str(bundle), "--destination", str(dest),
          "--apply", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["applied"] is True
    assert payload["state_after"] in {"HASH_COMPLETE", "COPY_COMPLETE", "VERIFIED",
                                     "VERIFYING", "RECONCILE_REQUIRED"}
    # verification alone can never release: it cannot attest that a copy happened
    assert payload["source_release_allowed_after"] is False


def test_cli_verify_does_not_apply_without_the_flag(tmp_path, capsys):
    bundle, src, dest, keys = build(tmp_path)
    copy_all(src, dest, keys)
    before = json.loads((bundle / "evidence.json").read_text(encoding="utf-8"))
    main(["verify", "--bundle", str(bundle), "--destination", str(dest), "--json"])
    capsys.readouterr()
    assert json.loads((bundle / "evidence.json").read_text(encoding="utf-8")) == before


def test_cli_verify_execute_is_still_refused(tmp_path, capsys):
    bundle, src, dest, keys = build(tmp_path)
    code = main(["verify", "--bundle", str(bundle), "--destination", str(dest),
                 "--execute"])
    assert code == EXIT_GATE_CLOSED
    assert "refusing to execute" in capsys.readouterr().err


def test_verify_without_a_destination_describes_instead_of_guessing(tmp_path, capsys):
    bundle = write_bundle(tmp_path / "b",
                          campaign(dest=destination(host_path=None)), evidence())
    code = main(["verify", "--bundle", str(bundle), "--json"])
    assert code == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert "verification" not in payload
    assert payload["executed"] is False


# ---------------------------------------------------------------------------
# the full pipeline, read-only on the source
# ---------------------------------------------------------------------------
def test_the_whole_pipeline_from_measurement_to_correct_refusal(tmp_path, capsys):
    """hash -> copy (done by the test) -> verify: custody proven, release still closed.

    The last assertion is the important one. Verification proves the bytes are
    right; it cannot attest that a *copy* happened, so the release gate must stay
    shut until copy evidence exists.
    """
    bundle, src, dest, keys = build(tmp_path)
    main(["hash", "--bundle", str(bundle), "--root", str(src), "--apply", "--json"])
    capsys.readouterr()
    copy_all(src, dest, keys)
    main(["verify", "--bundle", str(bundle), "--destination", str(dest),
          "--apply", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["verification"]["verified"] == 4

    status = load_status(bundle)
    assert status.evidence.hashing.verified_files == 4
    assert status.evidence.destination.verified_files == 4
    assert status.source_release_allowed is False
    codes = {b.code for b in status.release.blockers}
    # verification cannot attest that a copy happened, so copy evidence is the
    # blocker - not the destination, which was just proven byte-for-byte
    assert {"copy_incomplete", "copy_ledger_incomplete"} <= codes
    assert "destination_verification_incomplete" not in codes
    assert "missing_destination" not in codes


def test_the_real_cli_runs_the_pipeline(tmp_path):
    bundle, src, dest, keys = build(tmp_path)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}

    def run(*args):
        return subprocess.run(
            [sys.executable, str(CLI), "custody", *args],
            capture_output=True, text=True, cwd=str(tmp_path), env=env, timeout=120)

    assert run("hash", "--bundle", str(bundle), "--root", str(src),
               "--apply", "--json").returncode == 0
    copy_all(src, dest, keys)
    proc = run("verify", "--bundle", str(bundle), "--destination", str(dest),
               "--apply", "--json")
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["verification"]["verified"] == 4


def test_the_verify_module_cannot_copy_or_delete():
    import inspect

    from auto_ingest.custody import verify

    source = inspect.getsource(verify)
    for banned in ("shutil", "os.remove", "os.rename", "os.rmdir", "subprocess",
                   "os.link", "os.symlink"):
        assert banned not in source, banned


def test_verification_never_imports_a_copy_primitive():
    """A verifier that could move bytes would produce unfalsifiable evidence."""
    import ast
    import inspect

    from auto_ingest.custody import verify

    tree = ast.parse(inspect.getsource(verify))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in {"shutil", "subprocess"}


@pytest.mark.parametrize("status", [MISSING, MISMATCH, FAILED])
def test_no_failure_status_counts_as_custody(status):
    assert status != VERIFIED


# ---------------------------------------------------------------------------
# Cumulative verification evidence
# ---------------------------------------------------------------------------

def test_verified_bytes_are_cumulative_like_the_file_count(tmp_path):
    """A resume must not write `verified_files: 3` beside `verified_bytes: 0`.

    The count has always been cumulative, precisely so a resumed pass cannot
    regress a campaign out of custody. The byte total did not, so it did exactly
    that - and the state machine read `files: 3` as custody proven while the
    evidence said no bytes had ever been verified.
    """
    import json

    from auto_ingest.custody.verify import to_evidence, verify_destination

    src = tmp_path / "src"
    src.mkdir()
    (src / "a.MP4").write_bytes(b"a" * 100)
    (src / "b.MP4").write_bytes(b"b" * 250)
    bundle = tmp_path / "b"

    from auto_ingest.custody.hashing import hash_source

    hash_source(bundle, {"a.MP4": src / "a.MP4", "b.MP4": src / "b.MP4"})
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "a.MP4").write_bytes(b"a" * 100)
    (dest / "b.MP4").write_bytes(b"b" * 250)

    first = verify_destination(bundle, dest)
    evidence = to_evidence(first)
    assert evidence["destination"]["verified_files"] == 2
    assert evidence["destination"]["verified_bytes"] == 350

    # Second pass: everything already proven, nothing re-examined.
    second = verify_destination(bundle, dest)
    assert second.verified == 0 and second.verified_bytes == 0
    resumed = to_evidence(second)
    assert resumed["destination"]["verified_files"] == 2
    assert resumed["destination"]["verified_bytes"] == 350, (
        "the byte total regressed to 0 while the file count stayed at 2"
    )

    # The ledger, not the in-memory result, is the thing that carries this.
    rows = [json.loads(line) for line
            in (bundle / "ledgers" / "destination.jsonl").read_text().splitlines()]
    from auto_ingest.custody.verify import VERIFIED

    verified_rows = [r for r in rows if r["status"] == VERIFIED]
    assert sum(r.get("size", 0) for r in verified_rows) == 350
    assert len(verified_rows) == 2

    # A recheck re-proves the same objects and appends more rows for them. The
    # total must count each object once, not each time it was proven.
    third = verify_destination(bundle, dest, recheck=True)
    assert to_evidence(third)["destination"]["verified_bytes"] == 350
    rows = [json.loads(line) for line
            in (bundle / "ledgers" / "destination.jsonl").read_text().splitlines()
            if line.strip()]
    assert len([r for r in rows if r["status"] == VERIFIED]) == 4, "rows accumulate"
    assert to_evidence(third)["destination"]["verified_bytes"] == 350, "the total does not"
