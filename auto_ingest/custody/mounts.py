"""auto_ingest.custody.mounts - observe what is actually mounted.

The release gate blocks on ``campaign.source.read_only``, which is a **declared**
field from ``campaign.json``. A bundle that merely says ``read_only: true`` passes
the gate with zero blockers. That is the weakest load-bearing condition in the
package: the one thing protecting the source is a claim in a file.

This module reads the kernel's own answer instead. ``/proc/mounts`` states the
mount options the kernel actually applied, so ``ro`` there is evidence rather than
an assertion. Reading it is a single ``open()`` on procfs: no mounting, no
unmounting, no writes, and it works whether or not the card is present (an absent
path simply is not found, which is itself the answer).

Deliberately not used by the state machine yet. ``observe`` records what it saw
alongside the declared value so the two can be compared; flipping the gate to
require an observation is a separate, explicit decision (see
``docs/sd-card-campaign-custody.md``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .destination import StorageIdentity

MOUNTS_PATH = "/proc/mounts"

#: Symlinks named by filesystem UUID. Read for storage identity; never written.
BY_UUID_DIR = "/dev/disk/by-uuid"

#: Mount options we surface; everything else on the line is left alone.
_RELEVANT_OPTIONS = ("ro", "rw", "relatime", "noatime", "nosuid", "nodev", "noexec")


def _unescape(field: str) -> str:
    """Decode the octal escapes the kernel uses for space/tab/backslash."""
    out: List[str] = []
    i = 0
    raw = field
    while i < len(raw):
        ch = raw[i]
        if ch == "\\" and i + 3 < len(raw) + 1:
            chunk = raw[i + 1: i + 4]
            if len(chunk) == 3 and all(c in "01234567" for c in chunk):
                out.append(chr(int(chunk, 8)))
                i += 4
                continue
        out.append(ch)
        i += 1
    return "".join(out)


@dataclass(frozen=True)
class MountObservation:
    """What the kernel says about one mount point."""

    mount_point: str
    present: bool
    device: Optional[str] = None
    filesystem_type: Optional[str] = None
    options: Tuple[str, ...] = ()

    @property
    def read_only(self) -> Optional[bool]:
        """Observed read-only state, or ``None`` when the path is not mounted."""
        if not self.present:
            return None
        if "ro" in self.options:
            return True
        if "rw" in self.options:
            return False
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "filesystem_type": self.filesystem_type,
            "mount_point": self.mount_point,
            "options": list(self.options),
            "present": self.present,
            "read_only": self.read_only,
        }

    def agrees_with(self, declared: Optional[bool]) -> Optional[bool]:
        """Whether an observation and a declaration can both be true.

        ``None`` when either side is unknown, so a caller can distinguish
        "confirmed" from "no idea" instead of collapsing both into False.
        """
        observed = self.read_only
        if observed is None or declared is None:
            return None
        return observed is bool(declared)


def read_mounts(path: str | Path = MOUNTS_PATH) -> Tuple[MountObservation, ...]:
    """Parse ``/proc/mounts``. Read-only; an unreadable file yields nothing."""
    p = Path(path)
    if not p.is_file():
        return ()
    observations: List[MountObservation] = []
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        device, mount_point, fstype, options = parts[0], parts[1], parts[2], parts[3]
        mount_point = _unescape(mount_point)
        observations.append(MountObservation(
            mount_point=mount_point,
            present=True,
            device=_unescape(device),
            filesystem_type=fstype,
            options=tuple(o for o in options.split(",") if o in _RELEVANT_OPTIONS),
        ))
    return tuple(observations)


def _normalize(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    # Compare resolved paths so a symlinked alias of a real mount point still matches.
    try:
        return os.path.realpath(str(path))
    except OSError:  # pragma: no cover - realpath does not raise in practice
        return str(path)


def observe_mount(
    mount_point: Optional[str],
    *,
    mounts_path: str | Path = MOUNTS_PATH,
    observations: Optional[Tuple[MountObservation, ...]] = None,
) -> MountObservation:
    """Observe one mount point.

    Exact match first (the kernel's own spelling), then a realpath match so a
    symlinked mount is still recognised. A path that is not a mount point comes
    back ``present=False`` - which for a custody source is the answer "the card is
    not there", not an error.
    """
    if not mount_point:
        return MountObservation(mount_point="", present=False)
    table = read_mounts(mounts_path) if observations is None else observations
    if not table:
        return MountObservation(mount_point=mount_point, present=False)
    for observation in table:
        if observation.mount_point == mount_point:
            return observation
    target = _normalize(mount_point)
    for observation in table:
        if _normalize(observation.mount_point) == target:
            return observation
    return MountObservation(mount_point=mount_point, present=False)


def uuid_for_device(device: Optional[str], by_uuid_dir: str | Path = BY_UUID_DIR) -> Optional[str]:
    """Find the filesystem UUID for a block device, without running ``blkid``.

    ``/dev/disk/by-uuid`` is a directory of symlinks named by UUID, pointing at
    the block device. Reading those symlinks is a few ``readlink`` calls: no
    subprocess, no privileged tool, no mount inspection beyond what we already
    read. Returns ``None`` when the device carries no UUID (some filesystems and
    most network mounts), which is the honest answer.
    """
    if not device:
        return None
    base = Path(by_uuid_dir)
    if not base.is_dir():
        return None
    try:
        target = os.path.realpath(str(device))
    except OSError:  # pragma: no cover
        return None
    try:
        entries = sorted(base.iterdir())
    except OSError:
        return None
    for entry in entries:
        try:
            if os.path.realpath(str(entry)) == target:
                return entry.name
        except OSError:  # pragma: no cover
            continue
    return None


def observe_storage_identity(
    observation: MountObservation,
    *,
    by_uuid_dir: str | Path = BY_UUID_DIR,
) -> Optional[StorageIdentity]:
    """Build a :class:`StorageIdentity` from an observed mount, if possible.

    Combines the kernel's own view (device, filesystem type, mount options) with
    the UUID resolved from ``/dev/disk/by-uuid``. ``size_bytes`` is left unset: a
    stat of the mount root is the filesystem's *contents*, not its capacity, so
    filling it in would be a fabrication.
    """
    if not observation.present or not observation.device:
        return None
    return StorageIdentity(
        filesystem_uuid=uuid_for_device(observation.device, by_uuid_dir),
        device=observation.device,
        filesystem_type=observation.filesystem_type,
        observed_at=None,
    )


def observe_campaign(
    campaign: Any,
    *,
    mounts_path: str | Path = MOUNTS_PATH,
) -> Dict[str, Any]:
    """Observe both ends of a campaign: the source and the destination.

    Pure read. The result is a report, not evidence on its own - it carries the
    observed values next to the declared ones so a mismatch is visible rather
    than silently resolved in favour of either.
    """
    table = read_mounts(mounts_path)
    source_point = campaign.source.mount_point
    dest_path = campaign.destination.host_path

    source = observe_mount(source_point, observations=table)
    destination = observe_mount(dest_path, observations=table)

    declared_ro = campaign.source.read_only
    return {
        "campaign_id": campaign.campaign_id,
        "destination": {
            "declared_mounted": campaign.destination.mounted,
            "host_path": dest_path,
            "identity": (identity.to_dict()
                         if (identity := observe_storage_identity(destination)) else None),
            "observation": destination.to_dict(),
            "observed_mounted": destination.present,
        },
        "source": {
            "declared_read_only": declared_ro,
            "mount_point": source_point,
            "observation": source.to_dict(),
            "observed_read_only": source.read_only,
            "read_only_agrees_with_declaration": source.agrees_with(declared_ro),
        },
    }


def observations_to_evidence(report: Mapping[str, Any]) -> Dict[str, Any]:
    """The evidence fragment an observed mount should contribute.

    Only *observations* are recorded. The declared fields are left untouched, so
    merging this cannot quietly turn an unverified claim into a verified fact -
    which is why ``store.import_evidence`` never auto-applies it.

    The destination identity is the important one: without an observed identity
    the release gate stays closed on ``destination_identity_unproven``, so this is
    what turns "the bytes are right" into "the bytes are on the storage we meant".
    """
    destination = report.get("destination") or {}
    fragment: Dict[str, Any] = {}
    identity = destination.get("identity")
    if identity:
        fragment["destination"] = {"observed_identity": identity}
    return fragment


__all__ = [
    "BY_UUID_DIR",
    "MOUNTS_PATH",
    "MountObservation",
    "observe_campaign",
    "observe_mount",
    "observe_storage_identity",
    "observations_to_evidence",
    "uuid_for_device",
    "read_mounts",
]
