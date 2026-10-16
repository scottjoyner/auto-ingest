"""Strict read-only receipt reconciliation for the three-card NAS archive."""
from __future__ import annotations

import hashlib
import json

import pytest

from scripts.sd_archive_receipt_audit import build_report


def _fixture(tmp_path, *, copies=3):
    uuids = ["69BE-B963", "68FC-E499", "5140-6932"]
    manifest = tmp_path / "frozen.jsonl"
    root = tmp_path / "receipts"
    root.mkdir()
    originals = []
    rows = []
    for i, uuid in enumerate(uuids):
        key = f"DCIM/Movie/recording{i}.MP4"
        originals.append({"card_uuid": uuid, "source_relative_path": key,
                          "logical_bytes": i + 4})
        rows.append({
            "card_uuid": uuid, "key": key, "bytes": i + 4,
            "nas_path": f"/nas/recovery/dashcam-sd-ingest/"
                        f"SDCARD-20261016-{uuid}/source/{key}",
            "source_sha256": "a" * 64, "nas_sha256": "a" * 64,
            "sha_match": True, "source_stable": True,
            "destination_stable": True, "source_release_authorized": False,
        })
    manifest.write_text("".join(json.dumps(x) + "\n" for x in originals))
    pin = hashlib.sha256(manifest.read_bytes()).hexdigest()
    for i, row in enumerate(rows[:copies]):
        (root / f"sdcopy-{i}.jsonl").write_text(json.dumps(row) + "\n")
    return manifest, pin, root, rows


def test_complete_copy_receipts_never_unlock_source_release(tmp_path):
    frozen, pin, root, rows = _fixture(tmp_path)
    report = build_report(frozen, pin, root, expected_files=3)
    assert report["copy_receipts_complete"] is True
    assert report["verified_files"] == 3
    assert report["verified_bytes"] == 15
    assert not report["source_release_authorized"]
    assert not report["beelink_snapshot_retention_attested"]
    assert not report["destination_payload_rechecked_during_this_audit"]
    assert len(report["receipt_file_sha256"]) == 3


def test_incomplete_copy_reports_missing_and_blocks_release(tmp_path):
    frozen, pin, root, rows = _fixture(tmp_path, copies=2)
    report = build_report(frozen, pin, root, expected_files=3)
    assert report["missing_receipt_files"] == 1
    assert report["copy_receipts_complete"] is False
    assert report["source_release_authorized"] is False


def test_duplicate_receipts_rejected(tmp_path):
    frozen, pin, root, rows = _fixture(tmp_path)
    (root / "sdcopy-duplicate.jsonl").write_text(json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="receipt integrity failure"):
        build_report(frozen, pin, root, expected_files=3)


@pytest.mark.parametrize("field,value", [
    ("sha_match", False),
    ("source_sha256", "f" * 64),
    ("nas_path", "/nas/different/place.MP4"),
    ("bytes", 99),
    ("source_release_authorized", True),
    ("destination_stable", False),
])
def test_tampered_receipt_denied(tmp_path, field, value):
    frozen, pin, root, rows = _fixture(tmp_path)
    bad = dict(rows[0], **{field: value})
    (root / "sdcopy-0.jsonl").write_text(json.dumps(bad) + "\n")
    with pytest.raises(ValueError, match="receipt integrity failure"):
        build_report(frozen, pin, root, expected_files=3)


def test_pinned_manifest_required(tmp_path):
    frozen, pin, root, rows = _fixture(tmp_path)
    with pytest.raises(ValueError, match="checksum differs"):
        build_report(frozen, "f" * 64, root, expected_files=3)
    with pytest.raises(ValueError, match="cardinality"):
        build_report(frozen, pin, root, expected_files=4268)


def test_unterminated_receipt_denied(tmp_path):
    frozen, pin, root, rows = _fixture(tmp_path)
    (root / "sdcopy-0.jsonl").write_text(json.dumps(rows[0]))
    with pytest.raises(ValueError, match="unterminated receipt"):
        build_report(frozen, pin, root, expected_files=3)
