"""No source unlink, cross-volume path escape, or unbounded digest reads."""
from __future__ import annotations

import hashlib
import json

import pytest

from auto_ingest.custody import bounded_media_probe as probe
from auto_ingest.custody.campaign import CardIdentity
from auto_ingest.custody.mounts import MountObservation
from auto_ingest.custody.store import new_campaign


@pytest.fixture
def setup(tmp_path, monkeypatch):
    card = tmp_path / "card"
    nas = tmp_path / "nas"
    campaigns = tmp_path / "campaigns"
    card.mkdir()
    nas.mkdir()
    card_uuid = "ABCD-1234"
    files = [
        ("DCIM/Movie/2026_0101_101010_F.MP4", b"equal source and destination"),
        ("DCIM/Movie/2026_0101_101011_R.MP4", b"this media changes sometimes"),
    ]
    for rel, data in files:
        for root in (card, nas):
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    new_campaign(
        campaigns / card_uuid,
        card_id="SD-" + card_uuid,
        observed=CardIdentity(filesystem_uuid=card_uuid),
        mount_point=str(card), read_only=False,
        created_at="2026-10-16T15:00:00Z",
        config={}, env={}, apply=True,
    )
    rows = []
    for rel, data in files:
        rows.append({
            "card_uuid": card_uuid, "source_relative_path": rel,
            "bytes": len(data), "candidate_paths": [str(nas / rel)],
            "classification": "metadata_candidate_needs_sha",
            "mp4_staging": "stageable_filename_only",
            "current_nas_sha_verified": False,
            "source_release_allowed": False,
        })
    index = tmp_path / "index.jsonl"
    index.write_text("".join(json.dumps(row) + "\n" for row in rows))
    mount = MountObservation(str(card), True, device="/dev/fakeCARD",
                             filesystem_type="vfat", options=("rw",))
    nasmount = MountObservation(str(nas), True, device="/dev/fakeNAS",
                                filesystem_type="ext4", options=("rw",))
    monkeypatch.setattr(probe, "observe_mount", lambda _: mount)
    monkeypatch.setattr(probe, "read_mounts", lambda: (nasmount,))
    monkeypatch.setattr(probe, "uuid_for_device", lambda _: card_uuid)
    return dict(card=card, nas=nas, campaigns=campaigns, index=index,
                sha=hashlib.sha256(index.read_bytes()).hexdigest(),
                files=files, uuid=card_uuid)


def run(x, **kwargs):
    return probe.inspect(
        x["index"], kwargs.pop("sha", x["sha"]),
        x["campaigns"], x["nas"], destination_filesystem="ext4",
        max_files=kwargs.pop("max_files", 2),
        max_read_bytes=kwargs.pop("max_read_bytes", 1000),
        max_file_bytes=kwargs.pop("max_file_bytes", 100),
        expected_destination_device=kwargs.pop("expected_destination_device", "/dev/fakeNAS"),
        **kwargs,
    )


def test_explicit_rw_observation_matches_bytes_without_release_authority(setup):
    result = run(setup, allow_writable_source=True)
    assert len(result["results"]) == 2
    assert result["matches"] == 2
    assert result["bytes_read"] == 2 * sum(len(data) for _, data in setup["files"])
    assert not result["source_release_allowed"]
    assert not result["custody_evidence_imported"]
    assert not result["source_quiescence_attested"]
    assert result["source_mounts"][setup["uuid"]]["read_only"] is False
    assert all(item["sha256_match"] and not item["release_source_allowed"]
               for item in result["results"])
    assert all((setup["card"] / rel).read_bytes() == data
               for rel, data in setup["files"])


def test_writable_source_refused_by_default(setup):
    with pytest.raises(ValueError, match="source is writable"):
        run(setup)


def test_digest_mismatch_remains_no_go(setup):
    rel, data = setup["files"][0]
    (setup["nas"] / rel).write_bytes(b"x" * len(data))
    result = run(setup, allow_writable_source=True)
    assert result["matches"] == 1
    assert any(not item["sha256_match"] for item in result["results"])
    assert result["source_release_allowed"] is False


def test_symlinked_destination_is_denied(setup):
    rel, _ = setup["files"][0]
    path = setup["nas"] / rel
    path.unlink()
    path.symlink_to(setup["card"] / rel)
    with pytest.raises(OSError):
        run(setup, allow_writable_source=True)


def test_pinned_sha_cannot_change(setup):
    with pytest.raises(ValueError, match="pinned candidate"):
        run(setup, sha="0" * 64, allow_writable_source=True)


def test_wrong_kernel_uuid_or_missing_cifs_mount_blocks_read(setup, monkeypatch):
    monkeypatch.setattr(probe, "uuid_for_device", lambda _: "FFFF-FFFF")
    with pytest.raises(ValueError, match="mount identity"):
        run(setup, allow_writable_source=True)
    monkeypatch.setattr(probe, "read_mounts", lambda: ())
    with pytest.raises(ValueError, match="backing mount"):
        run(setup, allow_writable_source=True)


def test_probe_respects_file_count_and_read_budget(setup):
    result = run(setup, allow_writable_source=True, max_files=1)
    assert len(result["results"]) == 1
    assert result["bytes_read"] <= 1000
    with pytest.raises(ValueError, match="safe policy"):
        run(setup, allow_writable_source=True,
            max_read_bytes=probe.MAX_PROBE_BYTES + 1)


def test_duplicate_candidate_destination_blocked(setup):
    rows = [json.loads(line) for line in setup["index"].read_text().splitlines()]
    rows[1]["candidate_paths"][0] = rows[0]["candidate_paths"][0]
    setup["index"].write_text("".join(json.dumps(row) + "\n" for row in rows))
    sha = hashlib.sha256(setup["index"].read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="duplicate NAS path"):
        run(setup, sha=sha, allow_writable_source=True)


def test_report_exclusive_os_disk_and_no_overwrite(setup, tmp_path):
    result = run(setup, allow_writable_source=True)
    target = tmp_path / "witness.json"
    probe._write_report(target, result)
    assert json.loads(target.read_text())["source_release_allowed"] is False
    with pytest.raises(FileExistsError):
        probe._write_report(target, result)


def test_automount_is_not_mistaken_for_actual_backing_storage(setup, monkeypatch):
    autofs = MountObservation(str(setup["nas"]), True, device="systemd-1",
                              filesystem_type="autofs", options=("rw",))
    backing = MountObservation(str(setup["nas"]), True, device="/dev/fakeNAS",
                               filesystem_type="ext4", options=("rw",))
    monkeypatch.setattr(probe, "read_mounts", lambda: (autofs, backing))
    result = run(setup, allow_writable_source=True)
    assert result["matches"] == 2
    monkeypatch.setattr(probe, "read_mounts", lambda: (autofs,))
    with pytest.raises(ValueError, match="backing mount"):
        run(setup, allow_writable_source=True)


def test_intermediate_symlink_cannot_escape_directory_fd(setup):
    original = setup["nas"] / "DCIM"
    renamed = setup["nas"] / "DCIM_SAFE"
    original.rename(renamed)
    original.symlink_to(renamed, target_is_directory=True)
    with pytest.raises(OSError):
        run(setup, allow_writable_source=True)


def test_outside_nas_path_denied_even_when_index_sha_matches(setup):
    rows = [json.loads(line) for line in setup["index"].read_text().splitlines()]
    rows[0]["candidate_paths"] = ["/tmp/outside.mp4"]
    setup["index"].write_text("".join(json.dumps(row) + "\n" for row in rows))
    digest = hashlib.sha256(setup["index"].read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="outside declared destination"):
        run(setup, sha=digest, allow_writable_source=True)


def test_backing_share_device_identity_must_match_pinned_value(setup):
    with pytest.raises(ValueError, match="backing device differs"):
        run(setup, expected_destination_device="//wrong-share",
            allow_writable_source=True)


def test_pinned_previous_receipt_skips_already_compared_file(setup, tmp_path):
    first = run(setup, allow_writable_source=True, max_files=1)
    first_path = tmp_path / "first.json"
    probe._write_report(first_path, first)
    first_sha = hashlib.sha256(first_path.read_bytes()).hexdigest()
    next_one = run(setup, allow_writable_source=True, max_files=1,
                   exclude_report=first_path, expected_exclude_sha256=first_sha)
    assert next_one["excluded_prior_matching_file_count"] == 1
    assert len(next_one["results"]) == 1
    assert next_one["results"][0]["source_relative_path"] != first["results"][0]["source_relative_path"]
    assert next_one["source_release_allowed"] is False
    with pytest.raises(ValueError, match="receipt SHA256 mismatch"):
        run(setup, allow_writable_source=True, exclude_report=first_path,
            expected_exclude_sha256="0" * 64)


def test_multiple_prior_receipts_exclude_their_union(setup, tmp_path):
    first = run(setup, allow_writable_source=True, max_files=1)
    one = tmp_path / "one.json"
    probe._write_report(one, first)
    sha_one = hashlib.sha256(one.read_bytes()).hexdigest()
    second = run(setup, allow_writable_source=True, max_files=1,
                 exclude_report=one, expected_exclude_sha256=sha_one)
    two = tmp_path / "two.json"
    probe._write_report(two, second)
    sha_two = hashlib.sha256(two.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="no candidates within safe read budget"):
        run(setup, allow_writable_source=True,
            exclude_report=[one, two], expected_exclude_sha256=[sha_one, sha_two])
    with pytest.raises(ValueError, match="both path and pinned digest"):
        run(setup, allow_writable_source=True,
            exclude_report=[one, two], expected_exclude_sha256=[sha_one])
