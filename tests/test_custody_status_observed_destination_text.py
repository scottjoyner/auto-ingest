"""The human ``custody status`` must report the destination mount observation.

``status_text()`` reads ``data["destination_observed"]`` to decide whether to
print the *observed* mount answer. ``CampaignStatus.to_dict()`` never emitted
that key, so every human-mode run fell back to the creation-time snapshot and
labelled it ``(unobserved)`` -- including network destinations whose snapshot is
always false and whose backing mount the release gate had just proven usable.
These tests are pure (in-memory dataclasses, no mounts, DB, network or writes).
"""
from __future__ import annotations

import dataclasses

from auto_ingest.custody.campaign import Campaign, CardIdentity, SourceRef
from auto_ingest.custody.destination import DestinationRef, LogicalDestination
from auto_ingest.custody.evidence import CampaignEvidence
from auto_ingest.custody.policy import CustodyPolicy
from auto_ingest.custody.report import status_text
from auto_ingest.custody.store import build_status


def _status(*, observed_usable, backing=None, snapshot_mounted=False):
    campaign = Campaign.create(
        card_id="card-1",
        source=SourceRef(
            mount_point="/mnt/card",
            read_only=True,
            card=CardIdentity(device="/dev/sdb", filesystem_uuid="ABCD-1234"),
        ),
        destination=DestinationRef(
            logical=LogicalDestination(name="primary"),
            host_path="/nas/fileserver/headcam",
            # A subdirectory of a network share is never itself a mount point.
            mounted=snapshot_mounted,
        ),
        created_at="2026-11-26T00:00:00Z",
    )
    dest = dataclasses.replace(
        CampaignEvidence().destination,
        observed_usable=observed_usable,
        observed_backing_mount_point=backing,
    )
    evidence = dataclasses.replace(CampaignEvidence(), destination=dest)
    return build_status(campaign, evidence, CustodyPolicy())


def _destination_line(status):
    lines = [ln for ln in status_text(status).splitlines()
             if ln.startswith("destination        ")]
    assert len(lines) == 2, lines
    return lines[0]


def test_observed_usable_network_destination_is_reported_as_mounted():
    status = _status(observed_usable=True, backing="/nas")
    line = _destination_line(status)
    assert "mounted=true (on /nas)" in line, line
    assert "(unobserved)" not in line, line


def test_observed_absent_subdirectory_is_reported_with_its_backing_mount():
    status = _status(observed_usable=False, backing="/nas")
    line = _destination_line(status)
    assert "mounted=false (on /nas)" in line, line
    assert "(unobserved)" not in line, line


def test_observed_mount_point_needs_no_backing_suffix():
    status = _status(observed_usable=True, backing=None, snapshot_mounted=True)
    line = _destination_line(status)
    assert "mounted=true" in line, line
    assert "(on " not in line, line
    assert "(unobserved)" not in line, line


def test_status_dict_exposes_the_observation_for_json_and_text():
    status = _status(observed_usable=True, backing="/nas")
    observed = status.to_dict()["destination_observed"]
    assert observed["observed_usable"] is True
    assert observed["observation"]["backing_mount_point"] == "/nas"
    assert observed["observation"]["is_mount_point"] is False


def test_unobserved_destination_still_falls_back_to_the_snapshot():
    # Control: must pass before and after the fix -- an absent observation is
    # reported honestly as a snapshot, never as a proven mount.
    status = _status(observed_usable=None, backing=None, snapshot_mounted=False)
    line = _destination_line(status)
    assert "mounted=false (unobserved)" in line, line
