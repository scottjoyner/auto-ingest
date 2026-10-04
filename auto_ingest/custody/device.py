"""Read-only physical source-device discovery for custody onboarding.

No subprocesses, udev monitors, mounts, remounts, or writes are used here.
Identity comes from sysfs, /dev/disk symlinks, and the kernel mount table.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .mounts import MOUNTS_PATH, MountObservation, read_mounts

SYS_BLOCK_ROOT = "/sys/class/block"
DEV_ROOT = "/dev"
BY_UUID_DIR = "/dev/disk/by-uuid"
BY_LABEL_DIR = "/dev/disk/by-label"


class DeviceInspectionError(RuntimeError):
    """The requested block device could not be inspected safely."""


@dataclass(frozen=True)
class DeviceInspection:
    requested_device: str
    status: str
    selected_device: Optional[str]
    partitions: Tuple[Mapping[str, Any], ...]
    identity: Mapping[str, Any]
    mount: Optional[MountObservation]
    reason: Optional[str] = None

    @property
    def ready_for_campaign(self) -> bool:
        return self.status == "PASS" and bool(
            self.identity.get("filesystem_uuid")
            or self.identity.get("serial")
            or self.identity.get("device")
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identity": dict(self.identity),
            "mount": self.mount.to_dict() if self.mount else {
                "present": False,
                "read_only": None,
            },
            "partitions": [dict(p) for p in self.partitions],
            "ready_for_campaign": self.ready_for_campaign,
            "reason": self.reason,
            "requested_device": self.requested_device,
            "selected_device": self.selected_device,
            "status": self.status,
        }


def _direct_device_name(value: str) -> str:
    requested = str(value).strip()
    p = Path(requested)
    if p.parent != Path("/dev") or not p.name:
        raise DeviceInspectionError(
            f"not a direct block-device path: {requested or '<empty>'}"
        )
    return p.name


def _read_text(path: Path) -> Optional[str]:
    try:
        value = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    return value or None


def _read_int(path: Path) -> Optional[int]:
    value = _read_text(path)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _sys_path(name: str, root: str | Path) -> Path:
    path = Path(root) / name
    if not path.exists():
        raise DeviceInspectionError(f"block device not present in sysfs: /dev/{name}")
    return path


def _parent_block_name(sys_path: Path) -> Optional[str]:
    try:
        resolved = Path(os.path.realpath(sys_path))
    except OSError:
        return None
    if not (sys_path / "partition").exists():
        return None
    return resolved.parent.name or None


def _partition_names(name: str, root: str | Path) -> Tuple[str, ...]:
    sys_root = Path(root)
    target = _sys_path(name, sys_root)
    if (target / "partition").exists():
        return (name,)

    found = []
    try:
        entries = tuple(sys_root.iterdir())
    except OSError as exc:
        raise DeviceInspectionError(f"cannot read sysfs block inventory: {exc}") from exc
    for candidate in entries:
        if not (candidate / "partition").exists():
            continue
        if _parent_block_name(candidate) == name:
            found.append(candidate.name)
    return tuple(sorted(found)) if found else (name,)


def _device_target(name: str, dev_root: str | Path) -> str:
    return os.path.realpath(str(Path(dev_root) / name))


def _alias_for_device(
    directory: str | Path,
    name: str,
    dev_root: str | Path,
) -> Optional[str]:
    base = Path(directory)
    if not base.is_dir():
        return None
    target = _device_target(name, dev_root)
    try:
        entries = tuple(base.iterdir())
    except OSError:
        return None
    for entry in entries:
        try:
            if os.path.realpath(entry) == target:
                return entry.name
        except OSError:
            continue
    return None


def _mount_for_device(
    name: str,
    observations: Sequence[MountObservation],
) -> Optional[MountObservation]:
    direct = f"/dev/{name}"
    for obs in observations:
        if not obs.device:
            continue
        if obs.device == direct:
            return obs
        try:
            if os.path.realpath(obs.device) == os.path.realpath(direct):
                return obs
        except OSError:
            continue
    return None


def _transport(name: str, sys_path: Path) -> Optional[str]:
    low = os.path.realpath(sys_path).lower()
    if "/usb" in low:
        return "usb"
    if name.startswith("nvme"):
        return "nvme"
    if name.startswith("mmcblk"):
        return "mmc"
    return None


def _describe_partition(
    name: str,
    *,
    sys_block_root: str | Path,
    dev_root: str | Path,
    by_uuid_dir: str | Path,
    by_label_dir: str | Path,
    observations: Sequence[MountObservation],
) -> Dict[str, Any]:
    sys_path = _sys_path(name, sys_block_root)
    mount = _mount_for_device(name, observations)
    sectors = _read_int(sys_path / "size")
    removable = _read_int(sys_path / "removable")
    serial = _read_text(sys_path / "device" / "serial")
    model = _read_text(sys_path / "device" / "model")
    uuid = _alias_for_device(by_uuid_dir, name, dev_root)
    label = _alias_for_device(by_label_dir, name, dev_root)
    return {
        "device": f"/dev/{name}",
        "filesystem_type": mount.filesystem_type if mount else None,
        "filesystem_uuid": uuid,
        "label": label,
        "model": model,
        "removable": bool(removable) if removable is not None else None,
        "serial": serial,
        "size_bytes": sectors * 512 if sectors is not None else None,
        "transport": _transport(name, sys_path),
    }


def inspect_device(
    device: str,
    *,
    sys_block_root: str | Path = SYS_BLOCK_ROOT,
    dev_root: str | Path = DEV_ROOT,
    by_uuid_dir: str | Path = BY_UUID_DIR,
    by_label_dir: str | Path = BY_LABEL_DIR,
    mounts_path: str | Path = MOUNTS_PATH,
) -> DeviceInspection:
    """Inspect one physical source candidate without changing system state."""

    name = _direct_device_name(device)
    _sys_path(name, sys_block_root)
    observations = read_mounts(mounts_path)
    names = _partition_names(name, sys_block_root)
    partitions = tuple(
        _describe_partition(
            candidate,
            sys_block_root=sys_block_root,
            dev_root=dev_root,
            by_uuid_dir=by_uuid_dir,
            by_label_dir=by_label_dir,
            observations=observations,
        )
        for candidate in names
    )

    if len(partitions) != 1:
        return DeviceInspection(
            requested_device=f"/dev/{name}",
            status="ATTENTION_REQUIRED",
            selected_device=None,
            partitions=partitions,
            identity={},
            mount=None,
            reason=f"ambiguous_partitions:{len(partitions)}",
        )

    identity = dict(partitions[0])
    selected_name = Path(str(identity["device"])).name
    mount = _mount_for_device(selected_name, observations)

    if not (
        identity.get("filesystem_uuid")
        or identity.get("serial")
        or identity.get("device")
    ):
        status = "ATTENTION_REQUIRED"
        reason = "identity_unprovable"
    else:
        status = "PASS"
        reason = None

    return DeviceInspection(
        requested_device=f"/dev/{name}",
        status=status,
        selected_device=str(identity["device"]),
        partitions=partitions,
        identity=identity,
        mount=mount,
        reason=reason,
    )


__all__ = [
    "DeviceInspection",
    "DeviceInspectionError",
    "inspect_device",
]
