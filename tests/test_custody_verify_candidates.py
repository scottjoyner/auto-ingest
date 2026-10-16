"""Bounded verification is read-only, fail-closed and never release authority."""
from __future__ import annotations

import json

import pytest

from auto_ingest.custody.verify_candidates import (
    check_pair,
    choose_candidates,
    sha_stable,
)


def test_choose_candidates_smallest_first_and_budget():
    rows = [
        {"video_mp4_filename_not_content_verified": True,
         "classification": "metadata_candidate_needs_sha",
         "candidate_paths": ["/nas/old/a"], "bytes": n,
         "card_uuid": "CARD", "source_relative_path": f"DCIM/{n}.MP4"}
        for n in (40, 20, 30)
    ]
    selected = choose_candidates(rows, max_files=2, max_read_bytes=110)
    assert [x["bytes"] for x in selected] == [20, 30]
    assert all(x["bytes"] * 2 <= 110 for x in selected)
    assert rows[0]["bytes"] == 40


def test_sha_stable_and_matching_candidate_does_not_release(tmp_path):
    card = tmp_path / "card"
    nas = tmp_path / "nas"
    (card / "DCIM").mkdir(parents=True)
    nas.mkdir()
    original = card / "DCIM" / "front.MP4"
    target = nas / "front.MP4"
    original.write_bytes(b"exact-binary-video")
    target.write_bytes(b"exact-binary-video")
    row = {"card_uuid": "CARD", "source_relative_path": "DCIM/front.MP4",
           "candidate_paths": [str(target)], "bytes": original.stat().st_size}
    result = check_pair(row, card, dest_root=nas)
    assert result["sha_equal"] and result["source_stable"] and result["destination_stable"]
    assert result["source_release_authorized"] is False
    assert original.exists() and target.exists()


def test_equal_size_but_wrong_bytes_do_not_match(tmp_path):
    card = tmp_path / "card"
    nas = tmp_path / "nas"
    card.mkdir()
    nas.mkdir()
    (card / "x.MP4").write_bytes(b"abc")
    (nas / "x.MP4").write_bytes(b"xyz")
    row = {"card_uuid": "CARD", "source_relative_path": "x.MP4",
           "candidate_paths": [str(nas / "x.MP4")], "bytes": 3}
    assert check_pair(row, card, dest_root=nas)["sha_equal"] is False


def test_rejects_symlink_source_or_destination(tmp_path):
    card = tmp_path / "card"
    nas = tmp_path / "nas"
    card.mkdir()
    nas.mkdir()
    (card / "raw.MP4").write_bytes(b"abc")
    (nas / "target.MP4").write_bytes(b"abc")
    (nas / "link.MP4").symlink_to(nas / "target.MP4")
    row = {"card_uuid": "CARD", "source_relative_path": "raw.MP4",
           "candidate_paths": [str(nas / "link.MP4")], "bytes": 3}
    with pytest.raises(OSError):
        check_pair(row, card, dest_root=nas)
    (card / "slink.MP4").symlink_to(card / "raw.MP4")
    row["candidate_paths"] = [str(nas / "target.MP4")]
    row["source_relative_path"] = "slink.MP4"
    with pytest.raises(ValueError, match="source path escapes"):
        check_pair(row, card, dest_root=nas)


def test_rejects_traversal_and_outside_root(tmp_path):
    card = tmp_path / "card"
    nas = tmp_path / "nas"
    card.mkdir()
    nas.mkdir()
    (card / "x.MP4").write_bytes(b"abc")
    target = tmp_path / "outside.MP4"
    target.write_bytes(b"abc")
    row = {"card_uuid": "CARD", "source_relative_path": "x.MP4",
           "candidate_paths": [str(target)], "bytes": 3}
    with pytest.raises(ValueError, match="outside authorized"):
        check_pair(row, card, dest_root=nas)
    row["source_relative_path"] = "../outside.MP4"
    with pytest.raises(ValueError, match="safe relative"):
        check_pair(row, card, dest_root=nas)


def test_sha_stable_rejects_different_size(tmp_path):
    p = tmp_path / "x.MP4"
    p.write_bytes(b"12345")
    with pytest.raises(ValueError, match="unexpected type or size"):
        sha_stable(p, 10)


def test_prior_report_skips_stable_pairs_without_granting_release(tmp_path, monkeypatch):
    from auto_ingest.custody import verify_candidates as verifier
    from auto_ingest.custody.campaign import CardIdentity
    from auto_ingest.custody.mounts import MountObservation
    from auto_ingest.custody.store import new_campaign

    index = tmp_path / "indexed.jsonl"
    items = [
        {"video_mp4_filename_not_content_verified": True,
         "classification": "metadata_candidate_needs_sha",
         "candidate_paths": [f"/nas/candidate-{i}"], "bytes": 10,
         "card_uuid": "ABCD-1234", "source_relative_path": f"DCIM/{i}.MP4"}
        for i in range(2)
    ]
    index.write_text("".join(json.dumps(row) + "\n" for row in items))
    bundles = tmp_path / "campaigns"
    new_campaign(
        bundles / "ABCD-1234", card_id="card",
        observed=CardIdentity(filesystem_uuid="ABCD-1234"),
        mount_point="/mnt/test-card", read_only=False,
        env={}, config={}, apply=True,
    )
    monkeypatch.setattr(
        verifier, "observe_mount",
        lambda _: MountObservation("/mnt/test-card", True, device="/dev/fake",
                                   filesystem_type="vfat", options=("rw",)),
    )
    monkeypatch.setattr(verifier, "uuid_for_device", lambda _: "ABCD-1234")
    called = []
    def fake_check(row, root):
        called.append(row["source_relative_path"])
        return {
            "card_uuid": row["card_uuid"],
            "source_relative_path": row["source_relative_path"],
            "sha_equal": True, "source_stable": True,
            "destination_stable": True,
            "source_release_authorized": False, "bytes": row["bytes"],
        }
    monkeypatch.setattr(verifier, "check_pair", fake_check)
    first = verifier.verify_index(index, bundles, max_files=1)
    assert len(first["results"]) == 1
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps(first))
    second = verifier.verify_index(index, bundles, max_files=1, prior_report=prior)
    assert len(second["results"]) == 1
    assert second["results"][0]["source_relative_path"] != first["results"][0]["source_relative_path"]
    assert second["previously_compared_skipped_not_reverified"] == 1
    assert not second["source_release_authorized"]
    prior.write_text(json.dumps({**first, "source_index": "/another/index"}))
    with pytest.raises(ValueError, match="prior report does not match"):
        verifier.verify_index(index, bundles, max_files=1, prior_report=prior)
