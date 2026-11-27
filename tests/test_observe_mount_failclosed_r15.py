"""An uninspectable destination path must not be reported usable (r15).

Hermetic: a synthetic mount table and a monkeypatched ``os.path.exists``.
No /proc/mounts read, no network mount, no Neo4j, no filesystem write, no
credential access. The release gate consumes ``observed_usable``, so the
evidence-fragment test pins the value that actually reaches it.
"""
from __future__ import annotations

import os

import pytest

from auto_ingest.custody import mounts

BACKING = mounts.MountObservation(
    mount_point="/nas",
    present=True,
    device="//srv/share",
    filesystem_type="cifs",
    options=("rw",),
)
TARGET = "/nas/fileserver/headcam"
TABLE = (BACKING,)


@pytest.fixture
def real_exists():
    return os.path.exists


def _dead_mount(real_exists):
    def exists(path):
        if str(path).startswith("/nas/"):
            raise OSError(5, "Input/output error on dead network mount")
        return real_exists(str(path))
    return exists


def test_eio_on_existence_check_is_recorded_as_absent(monkeypatch, real_exists):
    monkeypatch.setattr(mounts.os.path, "exists", _dead_mount(real_exists))
    obs = mounts.observe_mount(TARGET, observations=TABLE, allow_containing=True)
    assert obs.present is True
    assert obs.backing_mount_point == "/nas"
    # Pre-fix this was None (meaning 'not checked') and usable was True.
    assert obs.path_exists is False
    assert obs.usable is False


def test_release_gate_evidence_marks_uninspectable_destination_unusable(
        monkeypatch, real_exists):
    monkeypatch.setattr(mounts.os.path, "exists", _dead_mount(real_exists))
    obs = mounts.observe_mount(TARGET, observations=TABLE, allow_containing=True)
    report = {
        "destination": {
            "observed_usable": obs.usable,
            "observation": obs.to_dict(),
        }
    }
    fragment = mounts.observations_to_evidence(report)
    assert fragment["destination"]["observed_usable"] is False
    assert fragment["destination"]["observed_backing_mount_point"] == "/nas"


def test_present_path_still_usable(monkeypatch, real_exists):
    monkeypatch.setattr(
        mounts.os.path, "exists", lambda p: True if str(p).startswith("/nas/") else real_exists(str(p)))
    obs = mounts.observe_mount(TARGET, observations=TABLE, allow_containing=True)
    assert obs.path_exists is True
    assert obs.usable is True


def test_absent_path_still_unusable(monkeypatch, real_exists):
    monkeypatch.setattr(
        mounts.os.path, "exists", lambda p: False if str(p).startswith("/nas/") else real_exists(str(p)))
    obs = mounts.observe_mount(TARGET, observations=TABLE, allow_containing=True)
    assert obs.path_exists is False
    assert obs.usable is False


def test_exact_mount_point_behaviour_unchanged(monkeypatch, real_exists):
    def never_called(path):
        raise AssertionError("exact mount match must not call os.path.exists")
    monkeypatch.setattr(mounts.os.path, "exists", never_called)
    obs = mounts.observe_mount("/nas", observations=TABLE, allow_containing=True)
    assert obs.present is True
    assert obs.path_exists is None
    assert obs.usable is True
    assert obs.backing_mount_point is None
