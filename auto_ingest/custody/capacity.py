"""auto_ingest.custody.capacity - can the destination hold what is coming?

Nothing in this repository checks free space before copying. The only ``statvfs``
guard that exists (``timelapse.py``) protects derived video encodes. So a 412.9 GB
campaign would copy until the destination fills and then fail as an opaque
``cp`` I/O error, half way through, with the ledger recording an interruption
rather than a precondition refusal.

This module makes the check a first-class, testable answer. ``os.statvfs`` is a
single syscall against a path - no writes, no mounting, no enumeration - and it
reports *the filesystem the path actually resolves to*, which is what matters:
a destination symlink into a full volume is caught.

Three quantities, reported separately so an operator can tell them apart:

``required_bytes``
    What the campaign still needs, taken from the evidence rather than from a
    caller-supplied number. Uses bytes outstanding at the destination, not total
    inventory: the already-copied portion needs no further space.
``available_bytes``
    Space available to an unprivileged writer (f_bavail * f_frsize), not
    f_bfree - the ~5% root reserve is not ours to consume.
``sufficient``
    ``available >= required + headroom``. Headroom exists because "exactly
    enough" is how a copy dies at 99%.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

#: Extra space demanded beyond the requirement, as a fraction of it.
DEFAULT_HEADROOM_FRACTION = 0.05

#: Also demand a floor, so a tiny requirement on a nearly-full volume still fails.
DEFAULT_HEADROOM_MIN_BYTES = 1 << 30  # 1 GiB


@dataclass(frozen=True)
class CapacityReport:
    """Whether the destination can hold the outstanding bytes, with margin."""

    path: Optional[str]
    checked: bool
    required_bytes: int = 0
    available_bytes: Optional[int] = None
    total_bytes: Optional[int] = None
    headroom_bytes: int = 0
    sufficient: Optional[bool] = None
    reason: str = ""

    @property
    def total_needed(self) -> int:
        """Payload plus margin - the figure a copy actually has to fit."""
        return self.required_bytes + self.headroom_bytes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available_bytes": self.available_bytes,
            "checked": self.checked,
            "headroom_bytes": self.headroom_bytes,
            "path": self.path,
            "reason": self.reason,
            "required_bytes": self.required_bytes,
            "sufficient": self.sufficient,
            "total_bytes": self.total_bytes,
            "total_needed_bytes": self.total_needed,
        }


def outstanding_bytes(evidence: Any) -> int:
    """Bytes still needed at the destination, before any headroom.

    Outstanding is ``inventory - verified``, scaled by the mean object size.
    Using outstanding rather than total matters on resume: a card that is already
    61% copied needs the remaining 39%, not the whole card again.
    """
    inventory_files = max(evidence.inventory_files, 0)
    verified = max(evidence.verified_at_destination, 0)
    outstanding = max(inventory_files - verified, 0)
    if inventory_files <= 0:
        return 0
    return int(round(evidence.inventory.discovered_bytes / inventory_files * outstanding))


def headroom_for(outstanding: int,
                 headroom_fraction: float = DEFAULT_HEADROOM_FRACTION,
                 headroom_min_bytes: int = DEFAULT_HEADROOM_MIN_BYTES) -> int:
    """Margin demanded beyond ``outstanding``.

    "Exactly enough" is how a copy dies at 99%, so there is always a margin - and
    a floor, so a tiny requirement on a nearly-full volume still fails.
    """
    return max(int(outstanding * headroom_fraction), headroom_min_bytes)


def required_bytes(evidence: Any, *, headroom_fraction: float = DEFAULT_HEADROOM_FRACTION,
                   headroom_min_bytes: int = DEFAULT_HEADROOM_MIN_BYTES) -> int:
    """Bytes still needed at the destination, including headroom."""
    return outstanding_bytes(evidence) + headroom_for(
        outstanding_bytes(evidence), headroom_fraction, headroom_min_bytes
    )


def check_capacity(
    path: Optional[str],
    needed_bytes: int,
    *,
    headroom_bytes: int = 0,
) -> CapacityReport:
    """Ask the filesystem whether ``needed_bytes`` fits at ``path``.

    A path that cannot be stat'ed is reported as ``sufficient=False`` with a
    reason rather than raising: for custody, "I could not check" must never read
    as "fine".
    """
    if not path:
        return CapacityReport(
            path=None, checked=False, required_bytes=needed_bytes,
            sufficient=False, reason="destination_unresolved",
        )
    try:
        st = os.statvfs(path)
    except OSError as exc:
        return CapacityReport(
            path=path, checked=False, required_bytes=needed_bytes,
            sufficient=False, reason=f"destination_unstatable:{exc.strerror or exc}",
        )

    fragment = st.f_frsize or st.f_bsize
    available = int(st.f_bavail) * fragment
    total = int(st.f_blocks) * fragment
    need = max(int(needed_bytes) + max(int(headroom_bytes), 0), 0)
    sufficient = available >= need
    return CapacityReport(
        path=path,
        checked=True,
        required_bytes=need,
        available_bytes=available,
        total_bytes=total,
        headroom_bytes=max(int(headroom_bytes), 0),
        sufficient=sufficient,
        reason="" if sufficient else (
            f"insufficient_space:need={need},available={available},short={need - available}"
        ),
    )


def capacity_report(
    campaign: Any,
    evidence: Any,
    *,
    headroom_fraction: float = DEFAULT_HEADROOM_FRACTION,
    headroom_min_bytes: int = DEFAULT_HEADROOM_MIN_BYTES,
) -> CapacityReport:
    """Full capacity answer for a campaign: required vs available at the destination.

    ``required_bytes`` is reported *without* the margin so the number answers "how
    much data is left", and ``headroom_bytes`` is reported separately so the
    margin is never mistaken for payload.
    """
    outstanding = outstanding_bytes(evidence)
    headroom = headroom_for(outstanding, headroom_fraction, headroom_min_bytes)
    report = check_capacity(campaign.destination.host_path, outstanding,
                            headroom_bytes=headroom)
    if not report.checked:
        return CapacityReport(
            path=report.path,
            checked=False,
            required_bytes=outstanding,
            headroom_bytes=headroom,
            sufficient=False,
            reason=report.reason,
        )
    return CapacityReport(
        path=report.path,
        checked=True,
        required_bytes=outstanding,
        available_bytes=report.available_bytes,
        total_bytes=report.total_bytes,
        headroom_bytes=headroom,
        sufficient=report.sufficient,
        reason=report.reason,
    )


__all__ = [
    "DEFAULT_HEADROOM_FRACTION",
    "DEFAULT_HEADROOM_MIN_BYTES",
    "CapacityReport",
    "capacity_report",
    "check_capacity",
    "headroom_for",
    "outstanding_bytes",
    "required_bytes",
]
