from __future__ import annotations

import json
from pathlib import Path

import pytest

from auto_ingest.custody.device import (
    DeviceInspection,
    DeviceInspectionError,
    inspect_device,
)
from auto_ingest.custody.mounts import MountObservation


def _fake_block_tree(tmp_path: Path, *, partition_count: int = 1):
    sys_root = tmp_path / "sys" / "class" / "block"
    devices_root = tmp_path / "sys" / "devices" / "pci-test" / "usb1" / "block" / "sdz"
    dev_root = tmp_path / "dev"
    by_uuid = dev_root / "disk" / "by-uuid"
    by_label = dev_root / "disk" / "by-label"
    mounts = tmp_path / "mounts"

    for p in (sys_root, devices_root, dev_root, by_uuid, by_label):
        p.mkdir(parents=True, exist_ok=True)

    (dev_root / "sdz").touch()
    (devices_root / "size").write_text("999999\n")
    (devices_root / "removable").write_text("1\n")
    (sys_root / "sdz").symlink_to(devices_root, target_is_directory=True)

    partitions = []
    for idx in range(1, partition_count + 1):
        name = f"sdz{idx}"
        part = devices_root / name
        (part / "device").mkdir(parents=True)
        (part / "partition").write_text(f"{idx}\n")
        (part / "size").write_text(str(240 + idx) + "\n")
        (part / "removable").write_text("1\n")
        (part / "device" / "serial").write_text(f"SER{idx}\n")
        (part / "device" / "model").write_text("FAKE-CARD\n")
        (dev_root / name).touch()
        (sys_root / name).symlink_to(part, target_is_directory=True)
        partitions.append((name, part))

    if partition_count == 1:
        name = partitions[0][0]
        (by_uuid / "ABCD-1234").symlink_to(dev_root / name)
        (by_label / "CARD").symlink_to(dev_root / name)
        mounts.write_text(
            f"/dev/{name} /media/card vfat ro,nosuid,nodev,relatime 0 0\n"
        )
    else:
        mounts.write_text("")

    return {
        "sys_root": sys_root,
        "dev_root": dev_root,
        "by_uuid": by_uuid,
        "by_label": by_label,
        "mounts": mounts,
    }


def _inspect(device: str, tree):
    return inspect_device(
        device,
        sys_block_root=tree["sys_root"],
        dev_root=tree["dev_root"],
        by_uuid_dir=tree["by_uuid"],
        by_label_dir=tree["by_label"],
        mounts_path=tree["mounts"],
    )


def test_inspect_partition_reports_identity_and_read_only_mount(tmp_path):
    tree = _fake_block_tree(tmp_path)
    result = _inspect("/dev/sdz1", tree)

    assert result.status == "PASS"
    assert result.ready_for_campaign is True
    assert result.identity["filesystem_uuid"] == "ABCD-1234"
    assert result.identity["serial"] == "SER1"
    assert result.identity["transport"] == "usb"
    assert result.identity["label"] == "CARD"
    assert result.identity["size_bytes"] == 241 * 512
    assert result.mount is not None
    assert result.mount.mount_point == "/media/card"
    assert result.mount.read_only is True


def test_inspect_disk_selects_its_only_partition(tmp_path):
    tree = _fake_block_tree(tmp_path)
    result = _inspect("/dev/sdz", tree)

    assert result.status == "PASS"
    assert result.selected_device == "/dev/sdz1"
    assert result.identity["filesystem_uuid"] == "ABCD-1234"


def test_inspect_disk_refuses_to_guess_between_multiple_partitions(tmp_path):
    tree = _fake_block_tree(tmp_path, partition_count=2)
    result = _inspect("/dev/sdz", tree)

    assert result.status == "ATTENTION_REQUIRED"
    assert result.selected_device is None
    assert result.reason == "ambiguous_partitions:2"
    assert len(result.partitions) == 2
    assert result.ready_for_campaign is False


def test_inspect_device_rejects_non_device_path():
    with pytest.raises(DeviceInspectionError, match="not a direct block-device path"):
        inspect_device("/tmp/not-a-device")


def test_inspect_device_cli_is_read_only_and_machine_readable(monkeypatch, capsys):
    from auto_ingest.custody import cli

    observed = DeviceInspection(
        requested_device="/dev/sdz1",
        status="PASS",
        selected_device="/dev/sdz1",
        partitions=({"device": "/dev/sdz1"},),
        identity={
            "device": "/dev/sdz1",
            "filesystem_uuid": "ABCD-1234",
            "serial": "SER123",
            "label": "CARD",
            "filesystem_type": "vfat",
        },
        mount=MountObservation(
            mount_point="/media/card",
            present=True,
            device="/dev/sdz1",
            filesystem_type="vfat",
            options=("ro",),
        ),
    )
    monkeypatch.setattr(cli, "inspect_device", lambda device: observed)

    assert cli.main(["inspect-device", "/dev/sdz1", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready_for_campaign"] is True
    assert payload["mount"]["read_only"] is True
    assert payload["identity"]["filesystem_uuid"] == "ABCD-1234"


def test_inspect_device_cli_returns_gate_closed_for_ambiguous(monkeypatch, capsys):
    from auto_ingest.custody import cli

    observed = DeviceInspection(
        requested_device="/dev/sdz",
        status="ATTENTION_REQUIRED",
        selected_device=None,
        partitions=(
            {"device": "/dev/sdz1"},
            {"device": "/dev/sdz2"},
        ),
        identity={},
        mount=None,
        reason="ambiguous_partitions:2",
    )
    monkeypatch.setattr(cli, "inspect_device", lambda device: observed)

    assert cli.main(["inspect-device", "/dev/sdz"]) == cli.EXIT_GATE_CLOSED
    out = capsys.readouterr().out
    assert "ATTENTION_REQUIRED" in out
    assert "ambiguous_partitions:2" in out
