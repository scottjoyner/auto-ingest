"""Guardrail tests for the supervised SD transfer controller."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts import sd_transfer_campaign as campaign


def _patch(tmp_path, monkeypatch, *, prior=False):
    root = tmp_path / "audit"
    root.mkdir()
    source = tmp_path / "card"
    source.mkdir()
    nas = tmp_path / "nas"
    nas.mkdir()
    uuid = "1234-ABCD"
    monkeypatch.setattr(campaign, "CARD_MOUNTS", {uuid: str(source)})
    monkeypatch.setattr(campaign, "_read_only_mount_check", lambda *args: None)
    prior_file = root / f"sdcopy-{uuid}-auto-000-20261016.jsonl"
    if prior:
        prior_file.write_text(json.dumps({"card_uuid": uuid, "key": "old.mp4"}) + "\n")
    called = []
    def fake_plan(*args, **kwargs):
        called.append([p.name for p in args[9]])
        chosen = [] if len(called) > 1 else [("DCIM/clip.MP4", 3)]
        return chosen, {
            "full_census_files": 1, "full_census_bytes": 3,
            "prior_verified_skipped_not_rechecked": int(prior) + len(called) - 1,
            "selected_bytes": sum(v for _, v in chosen),
        }
    monkeypatch.setattr(campaign, "make_plan", fake_plan)
    monkeypatch.setattr(campaign, "archive_root", lambda _nas, _uuid: nas / uuid)
    def fake_copy(source, dest, key, size):
        assert dest.is_dir()
        return {"key": key, "bytes": size, "sha_match": True,
                "source_stable": True, "destination_stable": True,
                "source_release_authorized": False}
    monkeypatch.setattr(campaign, "copy_one", fake_copy)
    monkeypatch.setattr(campaign.os, "statvfs", lambda _: SimpleNamespace(
        f_bavail=10**6, f_frsize=10**6,
    ))
    return root, nas, uuid, called, prior_file


def test_supervisor_writes_audit_and_reports_without_deletion(tmp_path, monkeypatch):
    root, nas, uuid, called, previous = _patch(tmp_path, monkeypatch)
    output = root / "complete.json"
    data = campaign.copy_campaign(
        tmp_path / "manifest", "0" * 64, root, nas, "//test/share",
        max_batches_per_card=2, max_files=2, max_bytes=1000,
        report_path=output,
    )
    assert data["cards"][uuid]["complete_against_frozen_manifest"]
    assert data["cards"][uuid]["newly_copied_files"] == 1
    assert data["source_release_allowed"] is False
    assert output.is_file()
    audit = root / f"sdcopy-{uuid}-auto-000-20261016.jsonl"
    assert json.loads(audit.read_text())["source_release_authorized"] is False
    assert called and len(called) == 2


def test_restart_uses_new_audit_id_instead_of_replacing(tmp_path, monkeypatch):
    root, nas, uuid, called, previous = _patch(tmp_path, monkeypatch, prior=True)
    old = previous.read_bytes()
    campaign.copy_campaign(
        tmp_path / "manifest", "0" * 64, root, nas, "//test/share",
        max_batches_per_card=2, max_files=2, max_bytes=1000,
    )
    assert previous.read_bytes() == old
    assert (root / f"sdcopy-{uuid}-auto-001-20261016.jsonl").is_file()


def test_low_capacity_refuses_before_copy(tmp_path, monkeypatch):
    root, nas, uuid, called, previous = _patch(tmp_path, monkeypatch)
    monkeypatch.setattr(campaign.os, "statvfs", lambda _: SimpleNamespace(
        f_bavail=0, f_frsize=4096,
    ))
    with pytest.raises(ValueError, match="low-space safety"):
        campaign.copy_campaign(
            tmp_path / "manifest", "0" * 64, root, nas, "//test/share",
            max_batches_per_card=1, max_files=2, max_bytes=1000,
        )
    assert not list(root.glob("*auto*.jsonl"))


def test_report_not_complete_when_batch_cap_hits(tmp_path, monkeypatch):
    root, nas, uuid, called, previous = _patch(tmp_path, monkeypatch)
    result = campaign.copy_campaign(
        tmp_path / "manifest", "0" * 64, root, nas, "//test/share",
        max_batches_per_card=1, max_files=2, max_bytes=1000,
    )
    assert result["cards"][uuid]["complete_against_frozen_manifest"] is False
    assert result["all_cards_copy_complete"] is False
    assert result["source_release_allowed"] is False
