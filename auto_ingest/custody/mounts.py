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
    #: When the observed path is not itself a mount point but is backed by one,
    #: the mount that backs it. ``None`` when the path is the mount point or when
    #: nothing backs it.
    #:
    #: A destination is routinely a *subdirectory* of a mount - the CIFS share
    #: here is mounted at /nas and the archive lives at /nas/fileserver/headcam.
    #: Reporting that as "not mounted" is true of the path and useless as an
    #: answer about the storage, which is what custody needs to know.
    backing_mount_point: Optional[str] = None
    #: Whether the observed path itself exists. ``None`` when not checked.
    #:
    #: Kept apart from ``present`` because the two come apart for a destination:
    #: /nas is mounted, and /nas/fileserver/headcam may still not exist. Reporting
    #: "mounted" for a path that is not there answers a question nobody asked and
    #: hides the one they did.
    path_exists: Optional[bool] = None

    @property
    def is_mount_point(self) -> bool:
        """Whether the observed path is itself the mount point."""
        return self.present and self.backing_mount_point is None

    @property
    def usable(self) -> bool:
        """Mounted, and the path is actually there.

        The conjunction the release gate wants: the storage being mounted does not
        by itself make a subdirectory destination reachable.
        """
        return bool(self.present) and self.path_exists is not False

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
            "backing_mount_point": self.backing_mount_point,
            "device": self.device,
            "filesystem_type": self.filesystem_type,
            "is_mount_point": self.is_mount_point,
            "mount_point": self.mount_point,
            "options": list(self.options),
            "path_exists": self.path_exists,
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


def containing_mount(target: str, table: Tuple[MountObservation, ...],
                      ) -> Optional[MountObservation]:
    """The deepest mount that contains ``target``, or None.

    Longest match wins, because mounts nest: /nas, /nas/fileserver and
    /nas/fileserver/headcam can all be mount points at once, and the nearest
    enclosing one is the filesystem actually holding the path. Comparing on
    path *segments* rather than a string prefix, so /nasx is not "inside" /nas.

    The root filesystem is excluded, and that exclusion is the whole reason this
    function is not simply "any prefix". ``/`` is in every mount table and
    contains every path, so including it makes the answer to "is this destination
    backed by a mount?" true for every ordinary directory on the machine - which
    is not a statement about any mount, and made `mounted` true for a destination
    path that did not exist.
    """
    target_parts = [p for p in os.path.normpath(target).split("/") if p]
    best: Optional[MountObservation] = None
    for observation in table:
        if not observation.mount_point or not observation.present:
            continue
        mount_parts = [p for p in os.path.normpath(
            observation.mount_point).split("/") if p]
        if not mount_parts:
            continue        # the root filesystem backs everything; says nothing
        if len(mount_parts) >= len(target_parts):
            continue
        if target_parts[:len(mount_parts)] == mount_parts:
            if best is None or len(mount_parts) > len(
                    [p for p in best.mount_point.split("/") if p]):
                best = observation
    return best


def observe_mount(
    mount_point: Optional[str],
    *,
    mounts_path: str | Path = MOUNTS_PATH,
    observations: Optional[Tuple[MountObservation, ...]] = None,
    allow_containing: bool = False,
) -> MountObservation:
    """Observe one mount point.

    Exact match first (the kernel's own spelling), then a realpath match so a
    symlinked mount is still recognised. A path that is not a mount point comes
    back ``present=False`` - which for a custody source is the answer "the card is
    not there", not an error.

    ``allow_containing`` extends the answer to "not a mount point itself, but
    backed by one". It is opt-in and **only ever right for a destination**. A
    custody source is the thing being proven present: if the card is unmounted and
    its mount point is an empty directory that happens to sit inside some other
    filesystem, then claiming the card is there is exactly the false positive this
    subsystem exists to prevent. For a destination the question is different - not
    "is this path a mount" but "is the storage behind this path mounted", and for
    a network share those come apart.
    """
    if not mount_point:
        return MountObservation(mount_point="", present=False)
    table = read_mounts(mounts_path) if observations is None else observations
    if not table:
        return MountObservation(mount_point=mount_point, present=False)
    for observation in table:
        if observation.mount_point == mount_point:
            return observation
    # Cheap pass first. `os.path.normpath` is pure string work with no I/O;
    # `os.path.realpath` performs a network round-trip for any path under a
    # CIFS/SMB mount. Measured on this host: realpath over the mount table cost
    # 28s for a single lookup of /nas/fileserver/dashcam, and `observe_campaign`
    # does this twice, so every `status`/`preflight` on a network destination paid
    # it. String comparison resolves the overwhelming majority of lookups.
    target = os.path.normpath(str(mount_point))
    for observation in table:
        if observation.mount_point and os.path.normpath(observation.mount_point) == target:
            return observation
    # Slow path, only when the cheap pass found nothing - i.e. the caller really
    # did hand us a symlinked alias of a mount point. Correctness is unchanged;
    # only the cost of the common case is.
    resolved = _normalize(mount_point)
    for observation in table:
        if _normalize(observation.mount_point) == resolved:
            return observation
    if allow_containing:
        backing = containing_mount(str(mount_point), table)
        if backing is not None:
            from dataclasses import replace as _replace

            try:
                exists: Optional[bool] = os.path.exists(mount_point)
            except OSError:  # pragma: no cover - e.g. EIO on a dead network mount
                exists = None
            return _replace(backing, mount_point=str(mount_point),
                            backing_mount_point=backing.mount_point,
                            path_exists=exists)
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
    by_uuid_dir: str | Path = BY_UUID_DIR,
) -> Dict[str, Any]:
    """Observe both ends of a campaign: the source and the destination.

    Pure read. The result is a report, not evidence on its own - it carries the
    observed values next to the declared ones so a mismatch is visible rather than
    silently resolved in favour of either.

    ``by_uuid_dir`` is overridable so a test can point the UUID lookup at a
    fixture rather than the host's real ``/dev/disk/by-uuid``. Without it a test
    asserting on an identity would be asserting on whatever card the machine
    happens to have plugged in - the same class of bug as a test reading the real
    mount table.
    """
    table = read_mounts(mounts_path)
    source_point = campaign.source.mount_point
    dest_path = campaign.destination.host_path

    # The source gets no containing-mount fallback, ever: "is the card there" must
    # not be answered by "is some filesystem here". The destination does get it,
    # because a network share's archive is routinely a subdirectory of the mount.
    source = observe_mount(source_point, observations=table)
    destination = observe_mount(dest_path, observations=table, allow_containing=True)

    declared_ro = campaign.source.read_only
    return {
        "campaign_id": campaign.campaign_id,
        "destination": {
            "declared_mounted": campaign.destination.mounted,
            "host_path": dest_path,
            "identity": (identity.to_dict()
                         if (identity := observe_storage_identity(
                             destination, by_uuid_dir=by_uuid_dir)) else None),
            "observation": destination.to_dict(),
            "observed_mounted": destination.present,
            # Distinct from observed_mounted: the share can be mounted while the
            # subdirectory the campaign names does not exist.
            "observed_usable": destination.usable,
        },
        "source": {
            "declared_read_only": declared_ro,
            "mount_point": source_point,
            "identity": (identity.to_dict()
                         if (identity := observe_storage_identity(
                             source, by_uuid_dir=by_uuid_dir)) else None),
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

    The source identity is recorded too. Without it the campaign's declared
    ``filesystem_uuid`` - the *authoritative* card identity, per CardIdentity -
    can never be confirmed against anything, so the cross-check reports
    ``unprovable`` forever and the "is this the card the campaign is for?"
    question has no answer. Found hashing a real card: ``observe_campaign``
    resolved an identity for the destination only, so the source UUID was
    reported as unobservable even though ``uuid_for_device`` resolved it fine
    from ``/dev/disk/by-uuid``.

    Both go under an ``observed`` key rather than being merged into the declared
    fields, so importing this can still never turn a claim into a fact.
    """
    destination = report.get("destination") or {}
    source = report.get("source") or {}
    fragment: Dict[str, Any] = {}
    identity = destination.get("identity")
    if identity or isinstance(destination.get("observed_usable"), bool):
        # Only what was actually observed. Writing `observed_usable: null` into
        # evidence would record the absence of an observation as though it were an
        # observation - the same reason the `errors` block is written only when
        # there is something to say.
        dest_fragment: Dict[str, Any] = {}
        if identity:
            dest_fragment["observed_identity"] = identity
        if isinstance(destination.get("observed_usable"), bool):
            dest_fragment["observed_usable"] = destination["observed_usable"]
        backing = (destination.get("observation") or {}).get("backing_mount_point")
        if backing:
            dest_fragment["observed_backing_mount_point"] = backing
        fragment["destination"] = dest_fragment
    source_identity = source.get("identity")
    if source_identity:
        # Under `inventory`, not a new top-level block: the inventory is the
        # source-side evidence, and this is what makes it attributable to a card.
        fragment["inventory"] = {"observed_identity": source_identity}
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
