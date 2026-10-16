"""Offline-card metadata readiness must not read or modify source media."""
from __future__ import annotations

import json

import pytest

from auto_ingest.custody import candidate_readiness as cr
from auto_ingest.custody.campaign import CardIdentity
from auto_ingest.custody.mounts import MountObservation
from auto_ingest.custody.store import new_campaign


@pytest.fixture()
def example(tmp_path, monkeypatch):
    nas = tmp_path / "nas"
    nas.mkdir()
    candidate = nas / "recording.MP4"
    candidate.write_bytes(b"data")
    index = tmp_path / "index.jsonl"
    obj = {
        "card_uuid": "ABCD-1234",
        "source_relative_path": "DCIM/recording.MP4",
        "classification": "metadata_candidate_needs_sha",
        "video_mp4_filename_not_content_verified": True,
        "bytes": 4,
        "candidate_paths": [str(candidate)],
    }
    index.write_text(json.dumps(obj) + "\n")
    campaigns = tmp_path / "campaigns"
    new_campaign(
        campaigns / "ABCD-1234",
        card_id="test",
        observed=CardIdentity(filesystem_uuid="ABCD-1234"),
        mount_point="/mnt/not-mounted-card",
        read_only=False, env={}, config={}, apply=True,
    )
    monkeypatch.setattr(cr, "observe_mount",
                        lambda _: MountObservation("/mnt/not-mounted-card", False))
    def no_device_resolution(_):
        raise AssertionError("no UUID probe for absent mount")
    monkeypatch.setattr(cr, "uuid_for_device", no_device_resolution)
    return index, campaigns, nas, obj


def test_unmounted_sd_does_not_prevent_nas_metadata_check(example):
    index, campaigns, nas, obj = example
    result = cr.inspect(index, campaigns, [], nas_root=nas)
    assert result["candidate_count"] == 1
    assert result["counts"]["nas_metadata_size_match_only"] == 1
    assert result["counts"]["blocked_unmounted_or_uuid_mismatch"] == 1
    assert result["source_card_states"]["ABCD-1234"]["source_present"] is False
    assert result["media_bytes_read"] == 0
    assert result["rows"][0]["content_sha_checked_this_run"] is False
    assert result["rows"][0]["source_release_authorized"] is False
    assert (nas / "recording.MP4").read_bytes() == b"data"


def test_missing_nas_path_not_confused_with_unmounted_source(example):
    index, campaigns, nas, obj = example
    (nas / "recording.MP4").unlink()
    result = cr.inspect(index, campaigns, [], nas_root=nas)
    assert result["counts"]["nas_candidate_path_absent"] == 1
    assert result["counts"]["blocked_unmounted_or_uuid_mismatch"] == 1
    assert result["source_release_authorized"] is False


def test_size_mismatch_or_symlink_cannot_be_ready(example, tmp_path):
    index, campaigns, nas, obj = example
    (nas / "recording.MP4").write_bytes(b"wrong-size")
    result = cr.inspect(index, campaigns, [], nas_root=nas)
    assert result["counts"]["nas_candidate_size_mismatch"] == 1
    (nas / "recording.MP4").unlink()
    target = tmp_path / "outside.MP4"
    target.write_bytes(b"data")
    (nas / "recording.MP4").symlink_to(target)
    result = cr.inspect(index, campaigns, [], nas_root=nas)
    assert result["counts"]["unsafe_destination_path"] == 1


def test_bounds_and_ambiguous_metadata_close(example):
    index, campaigns, nas, obj = example
    with pytest.raises(ValueError, match="max-candidates must"):
        cr.inspect(index, campaigns, [], max_candidates=0, nas_root=nas)
    more = [dict(obj, source_relative_path="DCIM/other.MP4")]
    with index.open("a") as file:
        for row in more:
            file.write(json.dumps(row) + "\n")
    bounded = cr.inspect(index, campaigns, [], max_candidates=1, nas_root=nas)
    assert bounded["candidate_count"] == 1
    assert bounded["total_remaining_candidates"] == 2
    assert bounded["candidates_not_inspected"] == 1
    assert bounded["complete_metadata_scan"] is False
    index.write_text(json.dumps(dict(obj, candidate_paths=["/nas/a", "/nas/b"])) + "\n")
    with pytest.raises(ValueError, match="ambiguity"):
        cr.inspect(index, campaigns, [], nas_root=nas)


def test_source_mounted_state_never_authorizes_release(example, monkeypatch):
    index, campaigns, nas, obj = example
    monkeypatch.setattr(
        cr, "observe_mount",
        lambda _: MountObservation("/mnt/not-mounted-card", True, device="/dev/test",
                                   filesystem_type="vfat", options=("ro",)),
    )
    monkeypatch.setattr(cr, "uuid_for_device", lambda _: "ABCD-1234")
    report = cr.inspect(index, campaigns, [], nas_root=nas)
    assert report["source_card_states"]["ABCD-1234"]["ready_for_source_reads"]
    assert report["source_card_states"]["ABCD-1234"]["read_only"] is True
    assert report["source_release_authorized"] is False
    assert not report["rows"][0]["source_release_authorized"]
