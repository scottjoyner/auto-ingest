"""Regression tests for guarded raw SD-to-NAS transfer (never deletes source)."""
from __future__ import annotations

import hashlib
import json

import pytest

from scripts import sd_verified_transfer as transfer


def test_copy_one_is_idempotent_and_compares_payload(tmp_path):
    source, dest = tmp_path / "card", tmp_path / "nas"
    (source / "DCIM").mkdir(parents=True)
    dest.mkdir()
    file = source / "DCIM" / "camera.MP4"
    file.write_bytes(b"unaltered footage bytes")
    first = transfer.copy_one(source, dest, "DCIM/camera.MP4", file.stat().st_size)
    assert first["sha_match"] and first["source_stable"] and first["destination_stable"]
    assert first["preexisting_destination"] is False
    assert first["source_release_authorized"] is False
    second = transfer.copy_one(source, dest, "DCIM/camera.MP4", file.stat().st_size)
    assert second["preexisting_destination"] is True
    assert first["source_sha256"] == hashlib.sha256(file.read_bytes()).hexdigest()
    assert file.read_bytes() == (dest / "DCIM" / "camera.MP4").read_bytes()


def test_copy_one_refuses_existing_wrong_bytes_and_preserves_both(tmp_path):
    source, dest = tmp_path / "card", tmp_path / "nas"
    source.mkdir()
    dest.mkdir()
    (source / "camera.MP4").write_bytes(b"original")
    (dest / "camera.MP4").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="verified copy mismatch"):
        transfer.copy_one(source, dest, "camera.MP4", 8)
    assert (source / "camera.MP4").read_bytes() == b"original"
    assert (dest / "camera.MP4").read_bytes() == b"tampered"


def test_symlink_source_or_destination_is_denied(tmp_path):
    source, dest = tmp_path / "card", tmp_path / "nas"
    source.mkdir()
    dest.mkdir()
    (source / "actual.MP4").write_bytes(b"abc")
    (source / "link.MP4").symlink_to(source / "actual.MP4")
    with pytest.raises(ValueError, match="source symlink"):
        transfer.copy_one(source, dest, "link.MP4", 3)
    (dest / "actual.MP4").symlink_to(source / "actual.MP4")
    with pytest.raises(ValueError, match="NAS target symlink"):
        transfer.copy_one(source, dest, "actual.MP4", 3)
    (source / "DCIM").mkdir()
    (source / "DCIM" / "actual.MP4").write_bytes(b"abc")
    (dest / "DCIM").symlink_to(source)
    with pytest.raises(ValueError, match="parent escapes"):
        transfer.copy_one(source, dest, "DCIM/actual.MP4", 3)


def test_plan_obeys_receipt_identity_and_video_count(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    nas = tmp_path / "nas"
    nas.mkdir()
    for k, data in (("DCIM/a.MP4", b"123456"),
                    ("DCIM/b.MP4", b"abcdefg"),
                    ("DCIM/c.csv", b"x")):
        p = src / k
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(data)
    uuid = "1234-ABCD"
    rows = [{"card_uuid": uuid, "source_relative_path": p.relative_to(src).as_posix(),
             "logical_bytes": p.stat().st_size} for p in src.rglob("*") if p.is_file()]
    frozen = tmp_path / "frozen.jsonl"
    frozen.write_text("".join(json.dumps(r) + "\n" for r in rows))
    pin = hashlib.sha256(frozen.read_bytes()).hexdigest()
    monkeypatch.setattr(transfer, "_read_only_mount_check", lambda *args: None)
    monkeypatch.setattr(transfer, "card_census", lambda *args: {
        "metadata_complete": True, "matched_files": 3, "matched_bytes": 14,
        "new_files": 0, "changed_size": 0, "unsafe_entries": 0, "missing": 0,
    })
    planned, meta = transfer.make_plan(
        frozen, pin, uuid, src, nas, "//unit/share", 3, 50, False, min_videos=2)
    assert sum(k.lower().endswith(".mp4") for k, _ in planned) == 2
    assert meta["source_release_authorized"] is False
    dest = transfer.archive_root(nas, uuid)
    dest.mkdir(parents=True)
    copied = transfer.copy_one(src, dest, planned[0][0], planned[0][1])
    receipt = tmp_path / "receipt.jsonl"
    receipt.write_text(json.dumps({"card_uuid": uuid, **copied}) + "\n")
    next_plan, skipped = transfer.make_plan(
        frozen, pin, uuid, src, nas, "//unit/share", 3, 50, False,
        prior_audits=[receipt], min_videos=1)
    assert planned[0][0] not in {k for k, _ in next_plan}
    assert skipped["prior_verified_skipped_not_rechecked"] == 1
    wrong = json.loads(receipt.read_text())
    wrong["nas_sha256"] = "bad"
    receipt.write_text(json.dumps(wrong) + "\n")
    with pytest.raises(ValueError, match="prior transfer receipt"):
        transfer.make_plan(frozen, pin, uuid, src, nas, "//unit/share", 3, 50,
                           False, prior_audits=[receipt])


def test_refuses_unpinned_or_changed_manifest(tmp_path, monkeypatch):
    src, nas = tmp_path / "src", tmp_path / "nas"
    src.mkdir()
    nas.mkdir()
    frozen = tmp_path / "frozen.jsonl"
    frozen.write_text(json.dumps({"card_uuid": "1234-ABCD",
                                   "source_relative_path": "video.MP4",
                                   "logical_bytes": 5}) + "\n")
    monkeypatch.setattr(transfer, "_read_only_mount_check", lambda *args: None)
    with pytest.raises(ValueError, match="frozen ledger checksum"):
        transfer.make_plan(frozen, "f" * 64, "1234-ABCD",
                           src, nas, "test", 1, 100, True)
