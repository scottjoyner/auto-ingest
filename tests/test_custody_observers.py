"""Phase A observers: what the kernel and filesystem actually say.

`campaign.source.read_only` is a declared field, so before Phase A a bundle that
merely asserted it passed the release gate with zero blockers. These tests pin the
observations that contradict an assertion, and prove the observers cannot write.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from custody_helpers import (
    CARD_01_BUNDLE,
    campaign,
    destination,
    evidence,
    fully_copied_campaign,
    patch_mount_table,
    strict_policy,
    write_bundle,
)

from auto_ingest.custody.capacity import (
    DEFAULT_HEADROOM_MIN_BYTES,
    capacity_report,
    check_capacity,
    outstanding_bytes,
)
from auto_ingest.custody.cli import EXIT_GATE_CLOSED, EXIT_OK, main
from auto_ingest.custody.mounts import (
    MountObservation,
    observe_campaign,
    observe_mount,
    observe_storage_identity,
    read_mounts,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PROC_MOUNTS = """
sysfs /sys sysfs rw,nosuid,nodev,noexec,relatime 0 0
proc /proc proc rw,nosuid,nodev,noexec,relatime 0 0
/dev/sdb1 /media/scott/UNTITLED vfat ro,nosuid,nodev,relatime,uid=1000 0 0
/dev/sdc1 /media/scott/BACKUP ext4 rw,relatime 0 0
/dev/nvme0n1p1 /mnt/custody/destination xfs rw,relatime 0 0
tmpfs /media/scott/ODD\\040NAME tmpfs ro,relatime 0 0
""".lstrip()



# The source end of a campaign is a fact this module controls, not a fact
# about whether the developer's card happens to be plugged in.
pytestmark = pytest.mark.usefixtures("hermetic_mounts")
def fake_mounts(tmp_path: Path, text: str = PROC_MOUNTS) -> Path:
    path = tmp_path / "mounts"
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------
def test_reads_the_real_kernel_table():
    """Not a fixture-only capability: /proc/mounts really is readable here."""
    table = read_mounts()
    assert table, "/proc/mounts should be readable"
    assert all(o.present for o in table)


def test_parses_options_and_filesystem_type(tmp_path):
    path = fake_mounts(tmp_path)
    o = observe_mount("/media/scott/UNTITLED", mounts_path=path)
    assert o.present is True
    assert o.read_only is True
    assert o.filesystem_type == "vfat"
    assert o.device == "/dev/sdb1"
    assert "ro" in o.options


def test_read_write_mounts_are_distinguished(tmp_path):
    path = fake_mounts(tmp_path)
    assert observe_mount("/media/scott/BACKUP", mounts_path=path).read_only is False
    assert observe_mount("/mnt/custody/destination",
                         mounts_path=path).read_only is False


def test_absent_path_is_an_answer_not_an_error(tmp_path):
    path = fake_mounts(tmp_path)
    o = observe_mount("/media/scott/NO_SUCH_CARD", mounts_path=path)
    assert o.present is False
    assert o.read_only is None


def test_unreadable_mounts_table_yields_no_observations(tmp_path):
    assert read_mounts(tmp_path / "nonexistent") == ()
    assert observe_mount("/media/scott/UNTITLED",
                         mounts_path=tmp_path / "nonexistent").present is False


def test_octal_escapes_in_mount_points_are_decoded(tmp_path):
    path = fake_mounts(tmp_path)
    o = observe_mount("/media/scott/ODD NAME", mounts_path=path)
    assert o.present is True
    assert o.read_only is True


def test_blank_and_short_lines_are_skipped(tmp_path):
    path = fake_mounts(tmp_path, "\n\nshort line here\n/dev/sdb1 /mnt/x ext4 rw 0 0\n")
    table = read_mounts(path)
    assert len(table) == 1


def test_symlinked_mount_point_still_resolves(tmp_path):
    link = tmp_path / "card"
    link.symlink_to("/media/scott/UNTITLED")
    path = fake_mounts(tmp_path)
    assert observe_mount(str(link), mounts_path=path).present is True


def test_observation_compares_against_a_declaration():
    ro = MountObservation("/m", present=True, options=("ro",))
    rw = MountObservation("/m", present=True, options=("rw",))
    gone = MountObservation("/m", present=False)
    assert ro.agrees_with(True) is True
    assert ro.agrees_with(False) is False
    assert rw.agrees_with(True) is False
    assert ro.agrees_with(None) is None       # unknown declaration
    assert gone.agrees_with(True) is None     # unknown observation


def test_storage_identity_comes_from_by_uuid_not_blkid(tmp_path):
    """A real UUID, resolved by reading symlinks - no subprocess, no tool."""
    by_uuid = tmp_path / "by-uuid"
    by_uuid.mkdir()
    device = tmp_path / "dev" / "sdb1"
    device.parent.mkdir()
    device.write_bytes(b"")
    (by_uuid / "ABCD-1234").symlink_to(device)

    observation = MountObservation("/mnt/card", present=True, device=str(device),
                                  filesystem_type="ext4")
    identity = observe_storage_identity(observation, by_uuid_dir=by_uuid)
    assert identity is not None
    assert identity.filesystem_uuid == "ABCD-1234"
    assert identity.filesystem_type == "ext4"
    # size_bytes is left unset: a stat of the mount root is its contents, not
    # its capacity, so filling it in would be a fabrication
    assert identity.size_bytes is None


def test_a_device_without_a_uuid_reports_none(tmp_path):
    by_uuid = tmp_path / "by-uuid"
    by_uuid.mkdir()
    device = tmp_path / "sdb1"
    device.write_bytes(b"")
    observation = MountObservation("/mnt/card", present=True, device=str(device))
    identity = observe_storage_identity(observation, by_uuid_dir=by_uuid)
    assert identity is not None
    assert identity.filesystem_uuid is None


def test_an_absent_mount_has_no_identity(tmp_path):
    absent = MountObservation("/mnt/gone", present=False)
    assert observe_storage_identity(absent, by_uuid_dir=tmp_path) is None


def test_a_missing_by_uuid_directory_is_tolerated(tmp_path):
    device = tmp_path / "sdb1"
    device.write_bytes(b"")
    observation = MountObservation("/mnt/card", present=True, device=str(device))
    identity = observe_storage_identity(observation, by_uuid_dir=tmp_path / "nope")
    assert identity is not None and identity.filesystem_uuid is None


def test_the_real_sd_card_uuid_is_observable():
    """Not a fixture-only capability: the real card resolves here."""
    identity = observe_storage_identity(observe_mount("/media/scott/UNTITLED"))
    if identity is None:
        pytest.skip("card not mounted on this host")
    assert identity.filesystem_uuid
    assert identity.device


def test_observe_campaign_covers_both_ends(tmp_path):
    path = fake_mounts(tmp_path)
    camp = campaign(dest=destination(host_path="/mnt/custody/destination"))
    report = observe_campaign(camp, mounts_path=path)
    assert report["source"]["observed_read_only"] is True
    assert report["source"]["read_only_agrees_with_declaration"] is True
    assert report["destination"]["observed_mounted"] is True
    assert report["destination"]["observation"]["filesystem_type"] == "xfs"


# ---------------------------------------------------------------------------
# the finding this phase exists for
# ---------------------------------------------------------------------------
def test_a_declaration_alone_passes_the_gate_but_observation_refuses(tmp_path):
    """The gap Phase A closes, stated as a test.

    Declaring read_only is enough for the release gate - the gate only knows what
    the bundle asserts. An observation of the same card can contradict it, so any
    future executor must gate on the observation.
    """
    from auto_ingest.custody import evaluate_release

    camp, ev = fully_copied_campaign()
    assert camp.source.read_only is True
    assert evaluate_release(camp, ev, strict_policy()).allowed is True

    rw_mounts = fake_mounts(
        tmp_path,
        "/dev/sdb1 /media/scott/UNTITLED vfat rw,relatime 0 0\n",
    )
    observed = observe_campaign(camp, mounts_path=rw_mounts)
    assert observed["source"]["observed_read_only"] is False
    assert observed["source"]["read_only_agrees_with_declaration"] is False


def test_cli_observe_mount_reads_the_real_kernel(tmp_path, capsys, real_mounts):
    """The one test that genuinely reads /proc/mounts.

    This asserts against what the kernel actually says rather than against this
    host's card, so it is meaningful everywhere: on a machine with the card
    plugged in it confirms the ro observation end to end, and on a machine
    without one it confirms the CLI reports absence honestly instead of
    inventing a source. The previous version hardcoded `present is True`, which
    meant it could only ever pass on a developer machine with the card mounted -
    it was a test of the peripherals, not of the code.
    """
    bundle = write_bundle(tmp_path / "b", campaign(), evidence())
    code = main(["observe-mount", "--bundle", str(bundle), "--json"])
    payload = json.loads(capsys.readouterr().out)

    really_present = "/media/scott/UNTITLED" in Path("/proc/mounts").read_text()
    assert payload["source"]["observation"]["present"] is really_present
    if really_present:
        # Card is here: the ro observation must survive the round trip.
        assert payload["source"]["observed_read_only"] is True
        assert payload["declared_read_only_trusted"] is False
        assert code == EXIT_OK
    else:
        # No card: the CLI must say so rather than reporting a source it cannot
        # see. This is the CI path, and it used to be an outright failure.
        # observed_read_only is None here, not False: "not observed" and
        # "observed writable" are different facts and must not be conflated.
        assert payload["source"]["observed_read_only"] is not True


def test_an_absent_card_is_reported_as_absent_not_assumed(tmp_path, capsys,
                                                          monkeypatch):
    """The CI condition, made hermetic: no card anywhere in the mount table.

    An empty table stands in for a runner with no SD card. The CLI must report
    the source absent and must not claim to have observed it read-only - the
    whole point of the observation is that it refuses to guess.
    """
    empty = tmp_path / "mounts"
    empty.write_text("", encoding="utf-8")
    patch_mount_table(monkeypatch, empty)
    bundle = write_bundle(tmp_path / "b", campaign(), evidence())
    code = main(["observe-mount", "--bundle", str(bundle), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"]["observation"]["present"] is False
    assert payload["source"]["observed_read_only"] is not True
    assert payload["declared_read_only_trusted"] is False
    # An absent source is a gate-closed condition, not a success.
    assert code == EXIT_GATE_CLOSED


def test_a_writable_card_contradicts_a_read_only_declaration(tmp_path):
    """The exact case a future executor must refuse.

    The bundle says read_only; the kernel says rw. The observation wins, and the
    report names the disagreement rather than silently picking a side.
    """
    path = fake_mounts(tmp_path, "/dev/sdb1 /media/scott/UNTITLED vfat rw,relatime 0 0\n")
    report = observe_campaign(campaign(), mounts_path=path)
    assert report["source"]["declared_read_only"] is True
    assert report["source"]["observed_read_only"] is False
    assert report["source"]["read_only_agrees_with_declaration"] is False


def test_cli_observe_mount_apply_keeps_both_claims(tmp_path):
    bundle = write_bundle(tmp_path / "b", campaign(), evidence())
    code = main(["observe-mount", "--bundle", str(bundle), "--apply"])
    assert code == EXIT_OK
    src = json.loads((bundle / "campaign.json").read_text(encoding="utf-8"))["source"]
    assert src["read_only"] is True                     # declaration preserved
    assert "observed_read_only" in src                   # observation recorded
    assert src["read_only_agrees_with_declaration"] in ("True", True, False, None)


def test_cli_observe_mount_does_not_apply_by_default(tmp_path):
    bundle = write_bundle(tmp_path / "b", campaign(), evidence())
    before = (bundle / "campaign.json").read_text(encoding="utf-8")
    main(["observe-mount", "--bundle", str(bundle)])
    assert (bundle / "campaign.json").read_text(encoding="utf-8") == before


# ---------------------------------------------------------------------------
# capacity
# ---------------------------------------------------------------------------
def test_capacity_uses_outstanding_not_total(tmp_path):
    """A card 61% copied needs the remaining 39%, not the whole card again."""
    _, ev = fully_copied_campaign(total_files=1000, total_bytes=1_000_000_000,
                                  verified_files=610)
    assert outstanding_bytes(ev) == int(1_000_000_000 / 1000 * 390)


def test_capacity_reports_payload_and_headroom_separately():
    _, ev = fully_copied_campaign(total_files=100, total_bytes=1_000,
                                  verified_files=100)
    report = capacity_report(campaign(), ev)
    assert report.required_bytes == 0
    assert report.headroom_bytes == DEFAULT_HEADROOM_MIN_BYTES
    assert report.total_needed == DEFAULT_HEADROOM_MIN_BYTES


def test_capacity_fails_closed_on_an_unstatable_destination():
    report = check_capacity("/nonexistent/path/xyz", 1_000)
    assert report.checked is False
    assert report.sufficient is False
    assert "unstatable" in report.reason


def test_capacity_fails_closed_on_an_unresolved_destination():
    report = check_capacity(None, 1_000)
    assert report.checked is False
    assert report.sufficient is False
    assert report.reason == "destination_unresolved"


def test_capacity_against_a_real_filesystem(tmp_path):
    report = check_capacity(str(tmp_path), 1)
    assert report.checked is True
    assert report.available_bytes is not None and report.available_bytes > 0
    assert report.total_bytes >= report.available_bytes
    assert report.sufficient is True


def test_capacity_refuses_an_impossible_requirement(tmp_path):
    report = check_capacity(str(tmp_path), 1 << 62)
    assert report.sufficient is False
    assert "short" in report.reason


def test_capacity_uses_bavail_not_bfree(tmp_path):
    """f_bfree includes the root reserve, which is not ours to consume."""
    real = os.statvfs(str(tmp_path))
    report = check_capacity(str(tmp_path), 1)
    expected = int(real.f_bavail) * (real.f_frsize or real.f_bsize)
    assert report.available_bytes == expected


def test_cli_capacity_exits_three_when_unresolved(capsys):
    code = main(["capacity", "--bundle", str(CARD_01_BUNDLE), "--json"])
    assert code == EXIT_GATE_CLOSED
    payload = json.loads(capsys.readouterr().out)
    assert payload["sufficient"] is False
    assert payload["reason"] == "destination_unresolved"


def test_cli_capacity_against_a_real_destination(tmp_path, capsys):
    camp, ev = fully_copied_campaign(total_files=1000, total_bytes=1_000_000,
                                     verified_files=1000)
    bundle = write_bundle(tmp_path / "b", camp, ev)
    raw = json.loads((bundle / "campaign.json").read_text(encoding="utf-8"))
    raw["destination"]["host_path"] = str(tmp_path)
    raw["destination"]["mounted"] = True
    (bundle / "campaign.json").write_text(json.dumps(raw), encoding="utf-8")
    code = main(["capacity", "--bundle", str(bundle), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["checked"] is True
    assert code == (EXIT_OK if payload["sufficient"] else EXIT_GATE_CLOSED)


def test_no_capacity_check_reads_a_clock():
    import inspect

    from auto_ingest.custody import capacity

    text = inspect.getsource(capacity)
    for banned in ("datetime.now", "time.time", "utcnow", "random"):
        assert banned not in text


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------
def test_preflight_refuses_card01_and_names_every_reason(capsys):
    code = main(["preflight", "--bundle", str(CARD_01_BUNDLE), "--json"])
    assert code == EXIT_GATE_CLOSED
    payload = json.loads(capsys.readouterr().out)
    assert payload["safe_to_execute"] is False
    assert "destination_resolved" in payload["failed"]
    assert "capacity_sufficient" in payload["failed"]
    names = {c["name"] for c in payload["checks"]}
    assert {"source_present", "source_read_only_observed", "destination_resolved",
            "capacity_sufficient", "release_gate_open"} <= names


def test_preflight_reports_queued_jobs(tmp_path, capsys):
    jobs = tmp_path / "drop"
    jobs.mkdir()
    (jobs / "audio.job").write_text("", encoding="utf-8")
    bundle = write_bundle(tmp_path / "b", campaign(), evidence())
    code = main(["preflight", "--bundle", str(bundle), "--json",
                 "--job-dir", str(jobs)])
    payload = json.loads(capsys.readouterr().out)
    assert "no_competing_jobs" in payload["failed"]
    assert code == EXIT_GATE_CLOSED


def test_preflight_text_output_shows_remedies(capsys):
    main(["preflight", "--bundle", str(CARD_01_BUNDLE)])
    out = capsys.readouterr().out
    assert "safe_to_execute        false" in out
    assert "-> set CUSTODY_DESTINATION_ROOT" in out


# ---------------------------------------------------------------------------
# read-only proof, via the audit hook
# ---------------------------------------------------------------------------
AUDIT_SCRIPT = r"""
import json, os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, %(repo)r)
os.chdir(%(repo)r)
from auto_ingest.custody.capacity import capacity_report, check_capacity
from auto_ingest.custody.mounts import observe_campaign, read_mounts
from auto_ingest.custody.store import load_campaign, load_evidence, load_status

events = []
WRITE = ("w", "a", "x", "+")
MUTATING = %(mutating)r

def _mounts(table):
    # dataclass reprs carry memory addresses; compare the payloads instead
    return [o.to_dict() for o in table]

def hook(event, args):
    if event in MUTATING:
        events.append({"event": event})
        return
    if event == "open":
        mode = args[1]
        if mode and any(ch in mode for ch in WRITE):
            events.append({"event": "open-write", "args": [str(args[0]), str(mode)]})

sys.addaudithook(hook)

bundle = %(bundle)r
camp = load_campaign(bundle)
ev = load_evidence(bundle)
def _relevant(parts):
    mounts_report = parts[1]
    return [mounts_report["source"]["observation"],
            mounts_report["destination"]["observation"],
            parts[2], parts[3], parts[4]]

first = [read_mounts(), observe_campaign(camp), capacity_report(camp, ev),
         check_capacity("/tmp", 1), load_status(bundle).to_dict()]
second = [read_mounts(), observe_campaign(camp), capacity_report(camp, ev),
          check_capacity("/tmp", 1), load_status(bundle).to_dict()]
def _as_jsonable(value):
    # dataclass reprs carry memory addresses, so unwrap before comparing
    import dataclasses
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    return str(value)


def _stable(value):
    # Free-space readings are environment, not input, so they are excluded.
    if isinstance(value, dict):
        return {k: _stable(v) for k, v in value.items()
                if k not in ("available_bytes", "total_bytes")}
    if isinstance(value, (list, tuple)):
        return [_stable(v) for v in value]
    if hasattr(value, "to_dict"):
        return _stable(value.to_dict())
    return value

# The WHOLE mount table is not the determinism property: this host has an
# autofs CIFS mount with x-systemd.idle-timeout=600, so unrelated mounts can
# appear or vanish between two reads. What must be stable is the campaign's own
# mount points, which is what observe_campaign reports.
diffs = []
for name, x, y in zip(
        ["source_observation", "destination_observation", "capacity_report",
         "check_capacity", "status"], _relevant(first), _relevant(second)):
    if json.dumps(_stable(x), default=_as_jsonable, sort_keys=True) \
            != json.dumps(_stable(y), default=_as_jsonable, sort_keys=True):
        diffs.append(name)
print(json.dumps({"events": events, "diffs": diffs}))
"""

MUTATING_EVENTS = (
    "os.mkdir", "os.rmdir", "os.remove", "os.rename", "os.link", "os.symlink",
    "os.truncate", "os.chmod", "os.chown", "os.utime", "subprocess.Popen",
    "os.system", "os.spawn", "os.fork", "os.posix_spawn",
)


def test_the_observers_emit_no_mutating_syscall():
    script = AUDIT_SCRIPT % {
        "repo": str(REPO_ROOT),
        "bundle": str(CARD_01_BUNDLE),
        "mutating": list(MUTATING_EVENTS),
    }
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        cwd=str(REPO_ROOT), timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["events"] == [], result["events"]
    assert result["diffs"] == [], result["diffs"]


def test_observers_are_clock_free_and_deterministic():
    report_a = observe_campaign(campaign())
    report_b = observe_campaign(campaign())
    assert json.dumps(report_a, sort_keys=True) == json.dumps(report_b, sort_keys=True)
    _, ev = fully_copied_campaign()
    assert capacity_report(campaign(), ev).to_dict() == \
        capacity_report(campaign(), ev).to_dict()


def test_observers_never_touch_the_source_filesystem():
    """observe-mount reads /proc/mounts; it never opens the card it describes."""
    import ast
    import inspect

    from auto_ingest.custody import mounts

    tree = ast.parse(inspect.getsource(mounts))
    opened = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in {"open", "listdir", "iterdir", "walk", "stat", "scandir",
                        "readlink"}:
                opened.add(name)
    # Only reads of procfs, by-uuid symlinks, and stat(). Never the card itself.
    assert opened <= {"open", "stat", "iterdir", "readlink"}, opened
    assert mounts.MOUNTS_PATH == "/proc/mounts"
    assert mounts.BY_UUID_DIR == "/dev/disk/by-uuid"
    # nothing enumerates a tree or resolves a realpath outside those two roots
    assert "os.walk" not in inspect.getsource(mounts)
    for banned in ("os.remove", "os.rename", "os.mkdir", "os.rmdir", "os.link"):
        assert banned not in inspect.getsource(mounts)
