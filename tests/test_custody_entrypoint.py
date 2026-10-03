"""The real entry point: `bin/auto-ingest custody ...`.

Everything else tests the module in-process. This exercises the surface an
operator (or Hermes) actually types - the REMAINDER forwarding in
`bin/auto-ingest`, argparse end to end, a different working directory, and the
atomic write path - because a package that only works when imported directly is
not delivered.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from custody_helpers import CARD_01_BUNDLE, campaign, evidence, write_bundle

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI = REPO_ROOT / "bin" / "auto-ingest"


def run(*args, cwd=None, env=None):
    environ = dict(os.environ)
    environ["PYTHONDONTWRITEBYTECODE"] = "1"
    if env:
        environ.update(env)
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        capture_output=True, text=True, cwd=str(cwd or REPO_ROOT),
        env=environ, timeout=120,
    )


# ---------------------------------------------------------------------------
# forwarding + help
# ---------------------------------------------------------------------------
def test_custody_is_a_subcommand_of_the_unified_cli():
    proc = run("custody", "status", "--bundle", str(CARD_01_BUNDLE), "--json")
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["state"] == "RECONCILE_REQUIRED"
    assert data["source_release_allowed"] is False


def test_custody_help_is_reachable():
    """REMAINDER forwarding swallows --help; it must not surface an argparse error."""
    for argv in (["custody"], ["custody", "--help"], ["custody", "-h"]):
        proc = run(*argv)
        assert proc.returncode == 0, proc.stderr
        assert "status" in proc.stdout and "reconcile" in proc.stdout
        assert "error" not in proc.stderr.lower()


def test_subcommand_help_is_reachable():
    for sub in ("status", "plan", "verify", "import", "new", "reconcile"):
        proc = run("custody", sub, "--help")
        assert proc.returncode == 0, proc.stderr
        assert "--bundle" in proc.stdout


def test_require_release_exit_code_propagates():
    proc = run("custody", "status", "--bundle", str(CARD_01_BUNDLE),
               "--require-release")
    assert proc.returncode == 3


def test_unknown_subcommand_is_an_error_not_a_silent_success():
    proc = run("custody", "teleport", "--bundle", str(CARD_01_BUNDLE))
    assert proc.returncode != 0
    assert "error" in proc.stderr.lower()


def test_verify_execute_is_refused_through_the_real_cli():
    proc = run("custody", "verify", "--bundle", str(CARD_01_BUNDLE), "--execute")
    assert proc.returncode == 3
    assert "refusing to execute" in proc.stderr


# ---------------------------------------------------------------------------
# working-directory independence
# ---------------------------------------------------------------------------
def test_works_from_an_unrelated_working_directory(tmp_path):
    """config.yaml is found next to the repo, not next to the caller's cwd."""
    proc = subprocess.run(
        [sys.executable, str(CLI), "custody", "status",
         "--bundle", str(CARD_01_BUNDLE), "--json"],
        capture_output=True, text=True, cwd=str(tmp_path),
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["destination"]["canonical"] == "primary:fileserver/dashcam"


def test_bundle_path_may_be_relative_to_the_caller(tmp_path):
    write_bundle(tmp_path / "b", campaign(), evidence())
    proc = run("custody", "status", "--bundle", "b", "--json", cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["state"] == "DISCOVERED"


def test_destination_env_override_reaches_the_cli(tmp_path):
    proc = run("custody", "status", "--bundle", str(CARD_01_BUNDLE), "--json",
               env={"CUSTODY_DESTINATION_ROOT": str(tmp_path / "dest")})
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    # the bundle records the destination, so env does not retro-actively rewrite it
    assert data["destination"]["host_path"] is None
    assert data["source_release_allowed"] is False


# ---------------------------------------------------------------------------
# the write path through the real CLI
# ---------------------------------------------------------------------------
def test_new_then_status_through_the_real_cli(tmp_path):
    bundle = tmp_path / "CARD-09"
    proc = run("custody", "new", "--bundle", str(bundle), "--card-id", "CARD-09",
               "--uuid", "ABCD-9999", "--device", "/dev/sdz9",
               "--label", "UNTITLED", "--mount", "/media/scott/UNTITLED",
               "--read-only", "--created-at", "2026-10-02T22:00:00Z", "--apply",
               "--json")
    assert proc.returncode == 0, proc.stderr
    created = json.loads(proc.stdout)
    assert created["mode"] == "applied"

    proc = run("custody", "status", "--bundle", str(bundle), "--json")
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["card_id"] == "CARD-09"
    assert data["state"] == "DISCOVERED"
    assert data["campaign_id"] == created["campaign_id"]


def test_read_only_commands_leave_no_temporary_files(tmp_path):
    bundle = write_bundle(tmp_path / "b", campaign(), evidence())
    for sub in ("status", "plan", "verify", "reconcile"):
        run("custody", sub, "--bundle", str(bundle))
    leftovers = [p.name for p in bundle.iterdir() if ".tmp" in p.name]
    assert leftovers == []


# ---------------------------------------------------------------------------
# concurrent atomic writes
# ---------------------------------------------------------------------------
def test_temp_names_differ_per_call_and_per_process(tmp_path):
    """A fixed `.tmp` name is the bug: two writers would share one file."""
    from auto_ingest.custody.store import _write_json_atomic

    target = tmp_path / "evidence.json"
    target.write_text('{"seed": true}', encoding="utf-8")
    seen = []
    real_replace = os.replace

    def spy(src, dst):
        seen.append(Path(src).name)
        return real_replace(src, dst)

    store_mod = sys.modules["auto_ingest.custody.store"]
    original = store_mod.os.replace
    store_mod.os.replace = spy
    try:
        for n in range(3):
            _write_json_atomic(target, {"n": n})
    finally:
        store_mod.os.replace = original

    assert len(seen) == 3
    assert len(set(seen)) == 3, f"temp names collided: {seen}"
    assert all(str(os.getpid()) in name for name in seen)
    assert all(not name.endswith(".json.tmp") for name in seen)
    assert [p.name for p in tmp_path.iterdir() if ".tmp" in p.name] == []


def test_concurrent_writers_in_separate_processes(tmp_path):
    """Two real processes racing: the target must end up valid and complete."""
    target = tmp_path / "evidence.json"
    target.write_text('{"seed": true}', encoding="utf-8")

    template = (
        "import sys\n"
        "sys.path.insert(0, {repo!r})\n"
        "from auto_ingest.custody.store import _write_json_atomic\n"
        "from pathlib import Path\n"
        "_write_json_atomic(Path({target!r}), {{'writer': {n}, 'payload': 'y' * 5000}})\n"
    )

    procs = []
    for n in range(2):
        procs.append(subprocess.Popen(
            [sys.executable, "-c",
             template.format(repo=str(REPO_ROOT), target=str(target), n=n)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        ))
    for proc in procs:
        out, err = proc.communicate(timeout=60)
        assert proc.returncode == 0, err

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["writer"] in (0, 1)
    assert payload["payload"] == "y" * 5000
    assert [p.name for p in tmp_path.iterdir() if ".tmp" in p.name] == []


def test_a_failed_write_leaves_no_temp_file(tmp_path, monkeypatch):
    sys.path.insert(0, str(REPO_ROOT))
    from auto_ingest.custody import store

    target = tmp_path / "evidence.json"
    target.write_text('{"seed": true}', encoding="utf-8")
    before = target.read_text(encoding="utf-8")

    real_replace = os.replace

    def boom(src, dst):
        raise OSError("simulated failure")

    monkeypatch.setattr(store.os, "replace", boom)
    with pytest.raises(OSError):
        store._write_json_atomic(target, {"changed": True})
    monkeypatch.setattr(store.os, "replace", real_replace)

    assert target.read_text(encoding="utf-8") == before
    assert [p.name for p in tmp_path.iterdir() if ".tmp" in p.name] == []
