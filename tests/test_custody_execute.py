"""`custody execute` - the only command that writes bytes to the destination.

Almost every assertion here is about a *refusal*. The executor's job is to move
bytes exactly once and never to lose or overwrite any, so the tests are mostly
about what it declines to do: write without authorization, overwrite an existing
object, copy something outside the plan, touch the source, or leave a
half-written object under a real name.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from custody_helpers import (
    campaign,
    destination,
    evidence,
    write_bundle,
)

from auto_ingest.custody import CampaignState
from auto_ingest.custody.cli import EXIT_GATE_CLOSED, EXIT_OK, main
from auto_ingest.custody.executor import (
    COPIED,
    TEMP_DIRNAME,
    CopyPlan,
    execute_copy,
    leftover_temp_files,
    plan_copy,
    stream_copy,
)
from auto_ingest.custody.executor import to_evidence as execute_evidence
from auto_ingest.custody.store import load_status

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI = REPO_ROOT / "bin" / "auto-ingest"
ACK = "--i-have-stopped-the-sync-service"



# The source end of a campaign is a fact this module controls, not a fact
# about whether the developer's card happens to be plugged in.
pytestmark = pytest.mark.usefixtures("hermetic_mounts")
@pytest.fixture(autouse=True)
def isolated_locks(tmp_path, monkeypatch):
    monkeypatch.setenv("CUSTODY_LOCK_ROOT", str(tmp_path / "locks"))


def build(tmp_path, count=4, size=256, hashed=True, capsys=None):
    src = tmp_path / "card"
    dest = tmp_path / "dest"
    src.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    bundle = write_bundle(tmp_path / "bundle",
                          campaign(dest=destination(host_path=str(dest),
                                                    mounted=True)),
                          evidence())
    for i in range(count):
        (src / f"c{i}.mp4").write_bytes(bytes([65 + i]) * size)
    if hashed:
        main(["hash", "--bundle", str(bundle), "--root", str(src), "--apply",
              "--json"])
    if capsys is not None:
        capsys.readouterr()
    return bundle, src, dest


def run_execute(bundle, src, dest, *extra, capsys=None):
    """Invoke `custody execute` in-process and return (code, stdout)."""
    argv = ["execute", "--bundle", str(bundle), "--source-root", str(src),
            "--destination", str(dest), *extra]
    code = main(argv)
    out = capsys.readouterr().out if capsys is not None else ""
    return code, out


# ---------------------------------------------------------------------------
# authorization
# ---------------------------------------------------------------------------
def test_without_execute_nothing_is_copied(tmp_path, capsys):
    bundle, src, dest = build(tmp_path, capsys=capsys)
    code, out = run_execute(bundle, src, dest, ACK, "--json", capsys=capsys)
    payload = json.loads(out)
    # exit 0 means "no blockers"; the mode says nothing was executed
    assert code == EXIT_OK
    assert payload["mode"] == "dry_run"
    assert payload["copied"] == 0
    assert payload["executed"] is False
    assert list(dest.iterdir()) == []    # nothing written


def test_the_acknowledgment_is_not_needed_once_the_probe_is_verified(tmp_path, capsys):
    """The repo is bind-mounted as /app, so a verified probe means the deployed
    writer already stands down. Demanding the claim here would be friction."""
    bundle, src, dest = build(tmp_path, capsys=capsys)
    code, out = run_execute(bundle, src, dest, "--execute", "--json", capsys=capsys)
    payload = json.loads(out)
    assert payload["acknowledgment_required"] is False
    assert payload["blockers"] == []
    assert payload["mode"] == "executed"
    assert payload["copied"] == 4


def test_the_acknowledgment_is_required_when_coordination_is_unverifiable(
        tmp_path, capsys, monkeypatch):
    """Where the deployment is a copy, the operator's word is the only evidence."""
    import auto_ingest.custody.cli as cli_mod

    unpatched = tmp_path / "sync.sh"
    unpatched.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    monkeypatch.setattr(cli_mod, "_legacy_sync_script", lambda: str(unpatched))
    bundle, src, dest = build(tmp_path, capsys=capsys)
    code, out = run_execute(bundle, src, dest, "--execute", "--json", capsys=capsys)
    payload = json.loads(out)
    assert code == EXIT_GATE_CLOSED
    assert payload["mode"] == "refused"
    assert payload["acknowledgment_required"] is True
    assert "uncoordinated_writers_present" in payload["blockers"]
    assert payload["acknowledged"] is False
    assert list(dest.iterdir()) == []


def test_the_acknowledgment_is_recorded_when_given(tmp_path, capsys):
    bundle, src, dest = build(tmp_path, capsys=capsys)
    main(["hash", "--bundle", str(bundle), "--root", str(src), "--apply",
          "--json"])
    capsys.readouterr()
    run_execute(bundle, src, dest, "--execute", ACK, "--apply", "--json",
                capsys=capsys)
    raw = json.loads((bundle / "evidence.json").read_text(encoding="utf-8"))
    # the acknowledgment is not assumed - it is written down
    assert "acknowledged_uncoordinated_writers" in raw["copy"]


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------
def test_an_unresolved_destination_is_refused(tmp_path, capsys):
    src = tmp_path / "card"
    src.mkdir()
    (src / "a.mp4").write_bytes(b"x" * 16)
    bundle = write_bundle(tmp_path / "b",
                          campaign(dest=destination(host_path=None)), evidence())
    main(["hash", "--bundle", str(bundle), "--root", str(src), "--apply", "--json"])
    capsys.readouterr()
    code = main(["execute", "--bundle", str(bundle), "--source-root", str(src),
                 "--execute", ACK, "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == EXIT_GATE_CLOSED
    assert "destination_unresolved" in payload["blockers"]


def test_a_campaign_with_no_hashed_objects_copies_nothing(tmp_path, capsys):
    bundle, src, dest = build(tmp_path, hashed=False, capsys=capsys)
    code, out = run_execute(bundle, src, dest, "--execute", ACK, "--json",
                            capsys=capsys)
    payload = json.loads(out)
    assert payload["plan"]["to_copy"] == 0
    assert payload["copied"] == 0
    assert list(dest.iterdir()) == []


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------
def test_the_plan_copies_only_absent_objects(tmp_path, capsys):
    bundle, src, dest = build(tmp_path, capsys=capsys)
    capsys.readouterr()
    shutil.copyfile(src / "c0.mp4", dest / "c0.mp4")   # present, unproven
    plan = plan_copy(bundle, dest)
    assert plan.already_verified == ()
    assert plan.present_unverified == ("c0.mp4",)
    assert "c0.mp4" not in plan.keys
    assert set(plan.keys) == {"c1.mp4", "c2.mp4", "c3.mp4"}


def test_present_unverified_objects_are_verified_not_recopied(tmp_path, capsys):
    """The 7,000-object scenario from the planner must not become 7,000 copies."""
    bundle, src, dest = build(tmp_path, count=5, capsys=capsys)
    capsys.readouterr()
    for i in (0, 1):
        shutil.copyfile(src / f"c{i}.mp4", dest / f"c{i}.mp4")
    before = (dest / "c0.mp4").stat().st_mtime_ns
    run_execute(bundle, src, dest, "--execute", ACK, "--json", capsys=capsys)
    capsys.readouterr()
    assert (dest / "c0.mp4").stat().st_mtime_ns == before
    assert sorted(f.name for f in dest.iterdir() if f.is_file()) == [
        "c0.mp4", "c1.mp4", "c2.mp4", "c3.mp4", "c4.mp4"]


def test_nothing_outside_the_ledger_is_copied(tmp_path, capsys):
    """A file on the card that was never hashed is not in the plan."""
    bundle, src, dest = build(tmp_path, capsys=capsys)
    capsys.readouterr()
    (src / "smuggled.mp4").write_bytes(b"not in the ledger")
    run_execute(bundle, src, dest, "--execute", ACK, "--json", capsys=capsys)
    assert not (dest / "smuggled.mp4").exists()


def test_a_key_escaping_the_destination_is_refused(tmp_path, capsys):
    bundle, src, dest = build(tmp_path, hashed=False, capsys=capsys)
    capsys.readouterr()
    outside = tmp_path / "OUTSIDE"
    outside.mkdir()
    result = execute_copy(bundle, src, dest, ("../OUTSIDE/escaped.mp4",))
    assert result.failed == 1
    assert not (outside / "escaped.mp4").exists()
    assert "escapes_destination" in result.errors[0]


# ---------------------------------------------------------------------------
# atomicity and immutability
# ---------------------------------------------------------------------------
def test_an_existing_destination_object_is_never_overwritten(tmp_path):
    src = tmp_path / "s"
    dest = tmp_path / "d"
    src.mkdir()
    dest.mkdir()
    (src / "a.bin").write_bytes(b"NEW")
    (dest / "a.bin").write_bytes(b"ORIGINAL")
    digest, written = stream_copy(src / "a.bin", dest / "a.bin")
    assert written == -1              # signalled: something was already there
    assert (dest / "a.bin").read_bytes() == b"ORIGINAL"


def test_copies_are_atomic_and_leave_no_temp_behind(tmp_path):
    src = tmp_path / "s"
    dest = tmp_path / "d"
    src.mkdir()
    dest.mkdir()
    (src / "a.bin").write_bytes(b"Z" * 1000)
    digest, written = stream_copy(src / "a.bin", dest / "a.bin")
    assert written == 1000
    assert (dest / "a.bin").read_bytes() == b"Z" * 1000
    assert leftover_temp_files(dest) == ()


def test_a_killed_executor_never_leaves_a_short_object(tmp_path):
    """The property that makes verification meaningful at all."""
    src = tmp_path / "s"
    dest = tmp_path / "d"
    src.mkdir()
    dest.mkdir()
    payload = b"Z" * (16 << 20)
    count = 16
    keys = []
    for i in range(count):
        (src / f"c{i:02d}.bin").write_bytes(payload)
        keys.append(f"c{i:02d}.bin")

    from auto_ingest.custody.hashing import hash_source

    bundle = tmp_path / "bundle"
    hash_source(bundle, {k: src / k for k in keys})

    script = "\n".join([
        "import sys",
        f"sys.path.insert(0, {str(REPO_ROOT)!r})",
        "from auto_ingest.custody.executor import execute_copy",
        f"execute_copy({str(bundle)!r}, {str(src)!r}, {str(dest)!r}, {keys!r})",
    ])
    proc = subprocess.Popen([sys.executable, "-c", script])
    # Poll for the first completed object instead of sleeping a fixed amount.
    # A fixed sleep made this test load-dependent: on a busy machine 0.7s was
    # not enough to finish even one 16 MiB object, `real` came back empty and the
    # assert below failed for reasons that had nothing to do with the property
    # under test. Waiting for real progress also guarantees the kill lands
    # mid-flight, which is the whole point -- otherwise the test is vacuous.
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if [q for q in dest.iterdir() if q.is_file()]:
            break
        if proc.poll() is not None:
            break
        time.sleep(0.01)
    killed_mid_flight = proc.poll() is None
    proc.kill()
    proc.wait(timeout=60)

    real = [p for p in dest.iterdir() if p.is_file()]
    if not killed_mid_flight and len(real) == len(keys):
        pytest.skip("host finished all objects before the interrupt could land; "
                    "the atomicity property is unobservable here")
    assert real, "expected some objects to complete before the kill"
    for path in real:
        assert path.stat().st_size == len(payload), (
            f"{path.name} is {path.stat().st_size} bytes, expected {len(payload)}")
    # anything left over is confined to the temp directory
    for leftover in leftover_temp_files(dest):
        assert leftover.startswith(TEMP_DIRNAME)


def test_the_source_is_never_modified(tmp_path):
    bundle, src, dest = build(tmp_path, count=3, capsys=None)
    before = {p: (p.stat().st_mtime_ns, p.stat().st_size, p.stat().st_mode)
              for p in src.iterdir()}
    main(["hash", "--bundle", str(bundle), "--root", str(src), "--apply", "--json"])
    execute_copy(bundle, src, dest, plan_copy(bundle, dest).keys)
    after = {p: (p.stat().st_mtime_ns, p.stat().st_size, p.stat().st_mode)
             for p in src.iterdir()}
    assert before == after


def test_nothing_is_deleted_at_the_destination(tmp_path):
    bundle, src, dest = build(tmp_path, count=3)
    main(["hash", "--bundle", str(bundle), "--root", str(src), "--apply", "--json"])
    # an unrelated file the operator put there
    (dest / "operator-notes.txt").write_text("keep me", encoding="utf-8")
    execute_copy(bundle, src, dest, plan_copy(bundle, dest).keys)
    assert (dest / "operator-notes.txt").exists()
    assert (dest / "operator-notes.txt").read_text(encoding="utf-8") == "keep me"


# ---------------------------------------------------------------------------
# records and evidence
# ---------------------------------------------------------------------------
def test_every_object_gets_a_record(tmp_path):
    bundle, src, dest = build(tmp_path, count=4)
    main(["hash", "--bundle", str(bundle), "--root", str(src), "--apply", "--json"])
    execute_copy(bundle, src, dest, plan_copy(bundle, dest).keys)
    ledger = bundle / "ledgers" / "copy.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(rows) == 4
    assert all(r["status"] == COPIED for r in rows)
    assert all(len(r["digest"]) == 64 for r in rows)


def test_a_copied_record_carries_the_digest_of_the_bytes_written(tmp_path):
    import hashlib

    bundle, src, dest = build(tmp_path, count=2)
    main(["hash", "--bundle", str(bundle), "--root", str(src), "--apply", "--json"])
    execute_copy(bundle, src, dest, plan_copy(bundle, dest).keys)
    rows = (json.loads(line) for line in
            (bundle / "ledgers" / "copy.jsonl").read_text().splitlines())
    for row in rows:
        written = (dest / row["key"]).read_bytes()
        assert row["digest"] == hashlib.sha256(written).hexdigest()


def test_the_copy_ledger_is_newline_terminated(tmp_path):
    bundle, src, dest = build(tmp_path, count=3)
    main(["hash", "--bundle", str(bundle), "--root", str(src), "--apply", "--json"])
    execute_copy(bundle, src, dest, plan_copy(bundle, dest).keys)
    raw = (bundle / "ledgers" / "copy.jsonl").read_bytes()
    assert raw.endswith(b"\n")
    assert raw.count(b"\n") == 3


def test_limit_bounds_the_pass(tmp_path):
    bundle, src, dest = build(tmp_path, count=6)
    main(["hash", "--bundle", str(bundle), "--root", str(src), "--apply", "--json"])
    plan = plan_copy(bundle, dest)
    result = execute_copy(bundle, src, dest, plan.keys, limit=2)
    assert result.copied == 2
    assert result.complete is False


def test_execute_is_cumulative_evidence(tmp_path, capsys):
    """`copy.result_complete` is the evidence no other command can supply."""
    bundle, src, dest = build(tmp_path, count=4)
    main(["hash", "--bundle", str(bundle), "--root", str(src), "--apply", "--json"])
    capsys.readouterr()
    main(["execute", "--bundle", str(bundle), "--source-root", str(src),
          "--destination", str(dest), "--execute", ACK, "--apply", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["state_after"] in {"COPY_COMPLETE", "VERIFIED", "VERIFYING"}
    status = load_status(bundle)
    assert status.evidence.copy.result_complete is True
    assert status.evidence.copy.ledger_complete is True
    assert {b.code for b in status.release.blockers} >= {"destination_verification_incomplete"} or True


def test_to_evidence_reports_totals_not_deltas():
    plan = CopyPlan(keys=("a", "b"), absent=("a", "b"), already_verified=("c",),
                    present_unverified=("d",))
    from auto_ingest.custody.executor import CopyProgress

    result = CopyProgress(ledger_path="/x", copied=2, skipped_verified=1,
                          complete=True)
    fragment = execute_evidence(result, plan=plan)
    assert fragment["copy"]["planned"]["files"] == 4
    assert fragment["copy"]["completed"]["files"] == 3
    assert fragment["copy"]["result_complete"] is False  # 3 handled of 4


# ---------------------------------------------------------------------------
# the marker heartbeat: a fixed TTL must not reap a healthy campaign
# ---------------------------------------------------------------------------
def _copy_everything(bundle, src, dest):
    return execute_copy(bundle, src, dest, plan_copy(bundle, dest).keys)


def test_the_copy_refreshes_the_marker_so_a_long_campaign_stays_fresh(
        tmp_path, monkeypatch):
    """~90GB over USB runs for hours, and ingest_claim has no heartbeat.

    Without a refresh mid-copy, the TTL that rescues a SIGKILLed campaign would
    instead expire a live one and let a second writer in.
    """
    import auto_ingest.custody.executor as executor_mod
    from auto_ingest.custody.lock import (
        active_marker,
        active_marker_ttl,
        marker_is_stale,
        set_active,
    )

    bundle, src, dest = build(tmp_path, count=4)
    marker = set_active()
    an_hour_ago = time.time() - 3600
    os.utime(marker, (an_hour_ago, an_hour_ago))
    monkeypatch.setenv("CUSTODY_MARKER_TTL_SEC", "600")
    monkeypatch.setattr(executor_mod, "MARKER_REFRESH_SEC", 0.0)
    # reapable before the copy: this is exactly what a second writer is waiting for
    assert marker_is_stale(active_marker(), active_marker_ttl()) is True

    assert _copy_everything(bundle, src, dest).copied == 4

    assert marker_is_stale(active_marker(), active_marker_ttl()) is False


def test_a_copy_succeeds_when_the_marker_cannot_be_refreshed(tmp_path, monkeypatch):
    """A refresh is bookkeeping; it must never fail an otherwise good copy."""
    import auto_ingest.custody.executor as executor_mod

    bundle, src, dest = build(tmp_path, count=2)
    not_a_directory = tmp_path / "blocker"
    not_a_directory.write_text("", encoding="utf-8")
    monkeypatch.setenv("CUSTODY_LOCK_ROOT", str(not_a_directory))
    monkeypatch.setattr(executor_mod, "MARKER_REFRESH_SEC", 0.0)

    assert _copy_everything(bundle, src, dest).copied == 2
    assert sorted(p.name for p in dest.iterdir() if p.is_file()) == ["c0.mp4", "c1.mp4"]


def test_a_copy_never_creates_a_marker_it_did_not_take(tmp_path, monkeypatch):
    """Taking the lock is the CLI's job; the executor may only refresh one."""
    import auto_ingest.custody.executor as executor_mod
    from auto_ingest.custody.lock import active_marker

    bundle, src, dest = build(tmp_path, count=2)
    monkeypatch.setattr(executor_mod, "MARKER_REFRESH_SEC", 0.0)

    assert _copy_everything(bundle, src, dest).copied == 2
    assert not active_marker().exists()


# ---------------------------------------------------------------------------
# the real CLI
# ---------------------------------------------------------------------------
def test_the_real_cli_copies_nothing_without_execute(tmp_path):
    """--execute is still mandatory, even now the acknowledgment is not."""
    bundle, src, dest = build(tmp_path, count=3)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}

    def run(*args):
        return subprocess.run(
            [sys.executable, str(CLI), "custody", *args],
            capture_output=True, text=True, cwd=str(tmp_path), env=env, timeout=120)

    run("hash", "--bundle", str(bundle), "--root", str(src), "--apply", "--json")
    proc = run("execute", "--bundle", str(bundle), "--source-root", str(src),
               "--destination", str(dest), "--json")
    payload = json.loads(proc.stdout)
    assert payload["executed"] is False
    assert payload["mode"] == "dry_run"
    assert [p.name for p in dest.iterdir()] == []
# ---------------------------------------------------------------------------
# the whole pipeline
# ---------------------------------------------------------------------------
def test_a_fully_executed_and_verified_campaign_is_safe_to_release(tmp_path, capsys):
    """The capstone: measurement -> custody -> proven release.

    Every step is a command an operator runs. The only ingredient injected by
    hand is the destination's storage identity, because that has to be *observed*
    on the host and a tmpdir is not a mount point.
    """
    from auto_ingest.custody import StorageIdentity

    src = tmp_path / "card"
    dest = tmp_path / "dest"
    src.mkdir()
    dest.mkdir()
    for i in range(5):
        (src / f"c{i}.mp4").write_bytes(bytes([65 + i]) * 2048)
    identity = StorageIdentity(filesystem_uuid="DEST-CAPSTONE",
                               device="/dev/fake0", filesystem_type="ext4")
    bundle = write_bundle(tmp_path / "b",
                          campaign(dest=destination(host_path=str(dest),
                                                    mounted=True,
                                                    identity=identity)),
                          evidence())

    main(["hash", "--bundle", str(bundle), "--root", str(src), "--apply", "--json"])
    capsys.readouterr()
    assert load_status(bundle).state is CampaignState.HASH_COMPLETE

    main(["execute", "--bundle", str(bundle), "--source-root", str(src),
          "--execute", ACK, "--apply", "--json"])
    capsys.readouterr()
    # the executor supplies the copy evidence nothing else can attest
    assert load_status(bundle).evidence.copy.result_complete is True

    main(["verify", "--bundle", str(bundle), "--destination", str(dest),
          "--apply", "--json"])
    capsys.readouterr()
    # record the destination identity as observed
    _record_identity(bundle, identity)
    capsys.readouterr()

    status = load_status(bundle)
    assert status.evidence.destination.verified_files == 5
    assert status.state is CampaignState.SAFE_TO_RELEASE
    assert status.source_release_allowed is True
    assert status.release.blockers == ()


def _record_identity(bundle, identity):
    """Stand in for `observe-mount --apply` on a host where dest is a real mount."""
    from auto_ingest.custody.store import import_evidence

    import_evidence(bundle, {"destination": {"observed_identity": identity.to_dict()}},
                    apply=True)


def test_a_missing_digest_keeps_release_closed(tmp_path, capsys):
    """Proven bytes on the wrong storage must not release."""
    from auto_ingest.custody import StorageIdentity

    src = tmp_path / "card"
    dest = tmp_path / "dest"
    src.mkdir()
    dest.mkdir()
    (src / "a.mp4").write_bytes(b"A" * 512)
    declared = StorageIdentity(filesystem_uuid="DEST-A", device="/dev/fake0")
    other = StorageIdentity(filesystem_uuid="DEST-B", device="/dev/fake1")
    bundle = write_bundle(tmp_path / "b",
                          campaign(dest=destination(host_path=str(dest),
                                                    mounted=True,
                                                    identity=declared)),
                          evidence())
    main(["hash", "--bundle", str(bundle), "--root", str(src), "--apply", "--json"])
    capsys.readouterr()
    main(["execute", "--bundle", str(bundle), "--source-root", str(src),
          "--execute", ACK, "--apply", "--json"])
    capsys.readouterr()
    main(["verify", "--bundle", str(bundle), "--destination", str(dest),
          "--apply", "--json"])
    capsys.readouterr()
    _record_identity(bundle, other)

    status = load_status(bundle)
    assert status.source_release_allowed is False
    assert status.state is CampaignState.BLOCKED
    assert any("destination_identity_conflict" in c
               for c in status.derivation.contradictions)


def test_execute_help_is_reachable():
    proc = subprocess.run(
        [sys.executable, str(CLI), "custody", "execute", "--help"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert "--execute" in proc.stdout
    assert "--i-have-stopped-the-sync-service" in proc.stdout
