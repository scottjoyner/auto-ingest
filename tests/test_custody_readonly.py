"""Zero filesystem mutation while computing custody status or plan.

The strongest available proof is CPython's audit hook, so the audit runs in a
dedicated subprocess (audit hooks cannot be uninstalled) and reports every
mutating syscall-level event it observes. A directory snapshot of the campaign
bundle is compared before and after as a second, independent check.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from custody_helpers import CARD_01_BUNDLE, fully_copied_campaign, write_bundle

from auto_ingest.custody.store import load_status

REPO_ROOT = Path(__file__).resolve().parents[1]

MUTATING_AUDIT_EVENTS = (
    "os.mkdir",
    "os.rmdir",
    "os.remove",
    "os.rename",
    "os.link",
    "os.symlink",
    "os.truncate",
    "os.chmod",
    "os.chown",
    "os.utime",
    "shutil.copyfile",
    "shutil.copymode",
    "shutil.copystat",
    "shutil.move",
    "shutil.rmtree",
    "subprocess.Popen",
    "os.system",
    "os.spawn",
    "os.fork",
    "os.posix_spawn",
)

AUDIT_SCRIPT = r"""
import json, os, sys
sys.dont_write_bytecode = True

# Import everything BEFORE the hook is installed: import machinery is allowed to
# cache, the custody code under test is not.
sys.path.insert(0, %(repo)r)
os.chdir(%(repo)r)
from auto_ingest.custody.ledger import reconcile_bundle, summarize_bundle_ledgers
from auto_ingest.custody.store import load_status, reconcile_preview

events = []
WRITE_MODES = ("w", "a", "x", "+")

def hook(event, args):
    if event in MUTATING:
        events.append({"event": event, "args": [str(a) for a in args]})
        return
    if event == "open":
        path, mode = args[0], args[1]
        if mode and any(ch in mode for ch in WRITE_MODES):
            events.append({"event": "open-write", "args": [str(path), str(mode)]})

MUTATING = %(mutating)r
sys.addaudithook(hook)

bundle = %(bundle)r
first = load_status(bundle)
second = load_status(bundle)
rec = [reconcile_bundle(bundle).to_dict(), reconcile_bundle(bundle).to_dict()]
led = [summarize_bundle_ledgers(bundle), summarize_bundle_ledgers(bundle)]
preview = reconcile_preview(bundle)
plans = [first.plan.to_dict(), second.plan.to_dict()]
statuses = [first.to_dict(), second.to_dict()]
print(json.dumps({
    "events": events,
    "identical": statuses[0] == statuses[1]
                 and plans[0] == plans[1]
                 and rec[0] == rec[1]
                 and led[0] == led[1]
                 and preview[1].to_dict()["state"] == second.to_dict()["state"],
}))
"""


def _snapshot(root: Path):
    entries = []
    for path in sorted(Path(root).rglob("*")):
        stat = path.stat()
        entries.append((str(path.relative_to(root)), path.is_dir(), stat.st_size,
                        stat.st_mtime_ns))
    return entries


def _has_backing_mount(path, mounts_path=None):
    """Whether ``path`` sits inside a non-root mount.

    ``resolve_destination(check_mount=True)`` reports ``mounted=True`` for any
    path backed by a non-root mount - tmpfs on /tmp here, or an ext4 mount at
    /media/scott/data on this host. An assertion that a destination is *not*
    mounted is only meaningful on the root filesystem, so tests use this to
    pick a suitable base (``containing_mount`` deliberately excludes ``/``).
    """
    from auto_ingest.custody.mounts import (MOUNTS_PATH, containing_mount,
                                            read_mounts)

    if mounts_path is None:
        mounts_path = MOUNTS_PATH
    return containing_mount(os.path.realpath(str(path)),
                            read_mounts(mounts_path)) is not None


def _unmounted_base(candidates):
    """First existing, writable candidate that is not inside a non-root mount."""
    for base in candidates:
        try:
            base = Path(base)
            if base.is_dir() and os.access(base, os.W_OK) and not _has_backing_mount(base):
                return base
        except OSError:
            continue
    return None


def test_backing_mount_detection(tmp_path):
    """Unit coverage for the filesystem-kind helper, using a synthetic table."""
    mounts = tmp_path / "mounts"
    mounts.write_text(
        "/dev/sda1 / ext4 rw 0 0\n"
        "tmpfs /tmp tmpfs rw 0 0\n",
        encoding="utf-8",
    )
    assert _has_backing_mount("/tmp/not-created", mounts) is True
    assert _has_backing_mount("/home/scott/not-created", mounts) is False


def test_status_and_plan_emit_no_mutating_syscall(tmp_path):
    script = AUDIT_SCRIPT % {
        "repo": str(REPO_ROOT),
        "bundle": str(CARD_01_BUNDLE),
        "mutating": list(MUTATING_AUDIT_EVENTS),
    }
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CUSTODY_DESTINATION_ROOT"] = str(tmp_path)  # must not be created or written
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT), timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["events"] == [], result["events"]
    assert result["identical"] is True


def test_reconcile_emit_no_mutating_syscall(tmp_path):
    """The set-difference computation is a pure read of the two ledgers."""
    script = AUDIT_SCRIPT % {
        "repo": str(REPO_ROOT),
        "bundle": str(CARD_01_BUNDLE),
        "mutating": list(MUTATING_AUDIT_EVENTS),
    }
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT), timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["events"] == [], result["events"]
    assert result["identical"] is True


def test_status_does_not_touch_the_bundle(tmp_path):
    bundle = CARD_01_BUNDLE
    before = _snapshot(bundle)
    for _ in range(3):
        load_status(bundle)
    assert _snapshot(bundle) == before


def test_plan_does_not_touch_a_generated_bundle(tmp_path):
    camp, ev = fully_copied_campaign(total_files=5, total_bytes=50)
    bundle = write_bundle(tmp_path / "gen", camp, ev)
    before = _snapshot(bundle)
    for _ in range(3):
        load_status(bundle)
    assert _snapshot(bundle) == before


def test_reading_a_missing_bundle_creates_nothing(tmp_path):
    missing = tmp_path / "absent"
    from auto_ingest.custody.store import BundleError

    with pytest.raises(BundleError):
        load_status(missing)
    assert not missing.exists()


def test_ledger_reading_is_read_only_and_does_not_create_ledgers(tmp_path):
    from auto_ingest.custody.ledger import summarize_bundle_ledgers

    summaries = summarize_bundle_ledgers(tmp_path)
    assert all(summary.present is False for summary in summaries.values())
    assert list(tmp_path.iterdir()) == []


def test_resolving_a_destination_never_creates_the_path(tmp_path):
    from auto_ingest.custody.destination import resolve_destination

    # /tmp is tmpfs on this host, so a target under tmp_path is reported as
    # mounted even though it never existed - the `mounted is False` assertion
    # would be meaningless there. Prefer a base on the root filesystem; skip
    # only this test if the host offers no such writable location. The custody
    # check (no path created) still runs wherever this test runs.
    base = _unmounted_base([tmp_path, Path.home(), Path("/var/tmp")])
    if base is None:
        pytest.skip("requires a non-tmpfs mount (a path outside any non-root mount)")

    target = base / f"custody-not-created-{uuid.uuid4().hex}"
    assert not target.exists()
    ref = resolve_destination({}, env={"CUSTODY_DESTINATION_ROOT": str(target)},
                              check_mount=True)
    assert ref.host_path == str(target)
    assert ref.mounted is False
    assert not target.exists()
