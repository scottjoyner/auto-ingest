"""auto_ingest.custody.campaign - stable identity for an SD-card ingest campaign.

A campaign is "one physical card, presented once, on its way to one logical
custody destination". Its identity is anchored on the *card*, not on where the
card happened to be mounted:

    card_key = sha256(device | filesystem_uuid | serial)

A different physical card appearing at the same mount point (``UNTITLED`` is the
classic offender - every unformatted card shares that label) therefore cannot
inherit the previous card's campaign. ``resolve_campaign`` is the only supported
way to pick a campaign for observed hardware, and it refuses to reuse one whose
device / filesystem UUID no longer matches.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .destination import DestinationRef, LogicalDestination, StorageIdentity

_ID_SEP = "|"


def _norm(value: Optional[str]) -> str:
    return (value or "").strip().upper()


def _fingerprint(*parts: Optional[str]) -> str:
    blob = _ID_SEP.join(_norm(p) for p in parts)
    if not blob.replace(_ID_SEP, ""):
        return ""
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


#: Identity fields, most authoritative first. The filesystem UUID (a FAT volume
#: id on an SD card) identifies the *card*; the device node only identifies the
#: slot it is plugged into, so it is the last resort, never an override.
IDENTITY_FIELDS: Tuple[str, ...] = ("filesystem_uuid", "serial", "device")


@dataclass(frozen=True)
class CardIdentity:
    """Who the physical card is, independent of where it was mounted."""

    device: Optional[str] = None
    filesystem_uuid: Optional[str] = None
    label: Optional[str] = None
    serial: Optional[str] = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "CardIdentity":
        raw = raw or {}
        card = raw.get("card") if isinstance(raw.get("card"), Mapping) else raw
        return cls(
            device=_opt(card.get("device")),
            filesystem_uuid=_opt(card.get("filesystem_uuid") or card.get("volume_id")
                                 or card.get("uuid")),
            label=_opt(card.get("label")),
            serial=_opt(card.get("serial")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "filesystem_uuid": self.filesystem_uuid,
            "label": self.label,
            "serial": self.serial,
        }

    @property
    def key(self) -> str:
        """Stable card fingerprint. Empty only when nothing at all is known."""
        return _fingerprint(self.device, self.filesystem_uuid, self.serial)

    @property
    def known(self) -> bool:
        return bool(self.device or self.filesystem_uuid or self.serial)

    def differences(self, other: "CardIdentity") -> Tuple[str, ...]:
        """Fields that mean "this is a *different* card", or that it cannot be proven.

        Compared most-authoritative-first. A matching filesystem UUID is enough
        even if the device node changed (the card moved to another slot). An
        identity field present on one side only makes the comparison
        *unprovable*, which is reported the same way as a mismatch: reuse is
        refused.
        """
        unverifiable: List[Tuple[int, str]] = []
        for index, name in enumerate(IDENTITY_FIELDS):
            mine = getattr(self, name)
            theirs = getattr(other, name)
            if bool(mine) != bool(theirs):
                unverifiable.append((index, name))
                continue
            if not mine:
                continue
            a = _norm(mine) if name == "filesystem_uuid" else mine
            b = _norm(theirs) if name == "filesystem_uuid" else theirs
            if a != b:
                return (name,)
            # This field matches and is the most authoritative one both sides
            # carry: less authoritative fields (a changed USB slot) are noise.
            blocked = [n for i, n in unverifiable if i < index]
            return tuple(blocked)
        return tuple(n for _, n in unverifiable)


def _opt(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@dataclass(frozen=True)
class SourceRef:
    """The removable source as observed."""

    mount_point: Optional[str] = None
    read_only: bool = False
    card: CardIdentity = CardIdentity()

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "SourceRef":
        raw = raw or {}
        return cls(
            mount_point=_opt(raw.get("mount_point") or raw.get("mount")),
            read_only=bool(raw.get("read_only", False)),
            card=CardIdentity.from_dict(raw),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "card": self.card.to_dict(),
            "mount_point": self.mount_point,
            "read_only": self.read_only,
        }


@dataclass(frozen=True)
class Campaign:
    """One SD-card ingest campaign with stable, hardware-anchored identity."""

    campaign_id: str
    card_id: str
    source: SourceRef
    destination: DestinationRef
    created_at: Optional[str] = None
    last_observed_at: Optional[str] = None

    # -- construction ---------------------------------------------------
    @classmethod
    def create(
        cls,
        *,
        card_id: str,
        source: SourceRef,
        destination: DestinationRef,
        created_at: Optional[str] = None,
        campaign_id: Optional[str] = None,
    ) -> "Campaign":
        """Create a campaign whose id is derived from the card, not the mount."""
        cid = campaign_id or derive_campaign_id(source.card, created_at)
        return cls(
            campaign_id=cid,
            card_id=card_id,
            source=source,
            destination=destination,
            created_at=created_at,
            last_observed_at=created_at,
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Campaign":
        raw = raw or {}
        dest_raw = raw.get("destination")
        logical_raw = (dest_raw or {}).get("logical") if isinstance(dest_raw, Mapping) else None
        destination = DestinationRef(
            logical=LogicalDestination(
                name=str((logical_raw or {}).get("name") or "primary"),
                relative_path=str((logical_raw or {}).get("relative_path") or ""),
            ),
            host_path=(dest_raw or {}).get("host_path"),
            resolved_from=str((dest_raw or {}).get("resolved_from") or "unresolved"),
            mounted=(dest_raw or {}).get("mounted"),
            identity=StorageIdentity.from_dict((dest_raw or {}).get("identity"))
            if isinstance((dest_raw or {}).get("identity"), Mapping) else None,
        )
        source_raw = raw.get("source") if isinstance(raw.get("source"), Mapping) else {}
        campaign = cls(
            campaign_id=str(raw.get("campaign_id") or ""),
            card_id=str(raw.get("card_id") or ""),
            source=SourceRef.from_dict(source_raw),
            destination=destination,
            created_at=_opt(raw.get("created_at")),
            last_observed_at=_opt(raw.get("last_observed_at")),
        )
        if not campaign.campaign_id:
            # Recover deterministically instead of inventing a random id.
            return replace(
                campaign,
                campaign_id=derive_campaign_id(campaign.source.card, campaign.created_at),
            )
        return campaign

    def to_dict(self) -> Dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "card_id": self.card_id,
            "created_at": self.created_at,
            "destination": self.destination.to_dict(),
            "last_observed_at": self.last_observed_at,
            "source": self.source.to_dict(),
        }

    # -- identity -------------------------------------------------------
    @property
    def card_key(self) -> str:
        return self.source.card.key

    def with_observation(
        self,
        observed: CardIdentity,
        *,
        observed_at: Optional[str] = None,
        read_only: Optional[bool] = None,
        mount_point: Optional[str] = None,
    ) -> Optional["Campaign"]:
        """Refresh *observation* fields only, or return ``None`` on a new card.

        Returns ``None`` when ``observed`` is not the same physical card, which
        is the mechanism that stops a second card at ``UNTITLED`` from
        inheriting this campaign's evidence.
        """
        if not self.source.card.known or not observed.known:
            return None
        if observed.differences(self.source.card):
            return None
        return replace(
            self,
            source=replace(
                self.source,
                mount_point=mount_point or self.source.mount_point,
                read_only=self.source.read_only if read_only is None else bool(read_only),
            ),
            last_observed_at=observed_at or self.last_observed_at,
        )


def derive_campaign_id(card: CardIdentity, created_at: Optional[str] = None) -> str:
    """Deterministic campaign id: ``sdcard-<card_key>-<created_at digest>``.

    Including the creation instant means a card that is pulled, re-imaged and
    re-presented gets a *new* campaign rather than silently resuming an old one,
    even when the hardware identity happens to repeat.
    """
    key = card.key or _fingerprint(card.label, created_at)
    stamp = hashlib.sha256((created_at or "").encode("utf-8")).hexdigest()[:8]
    return f"sdcard-{key}-{stamp}"


@dataclass(frozen=True)
class CampaignResolution:
    """Outcome of picking a campaign for observed hardware."""

    campaign: Optional[Campaign]
    reused: bool
    reason: str
    conflicting_fields: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conflicting_fields": list(self.conflicting_fields),
            "campaign_id": self.campaign.campaign_id if self.campaign else None,
            "reason": self.reason,
            "reused": self.reused,
        }


def resolve_campaign(
    observed: CardIdentity,
    existing: Sequence[Campaign],
    *,
    mount_point: Optional[str] = None,
    card_id: str = "",
    destination: Optional[DestinationRef] = None,
    created_at: Optional[str] = None,
) -> CampaignResolution:
    """Pick the campaign that owns ``observed`` hardware.

    Reuse requires the *card identity* to match. Mount point and label are
    reported but never sufficient: a second unformatted card mounted at
    ``UNTITLED`` matches on label only and is therefore a new campaign.
    """
    if not observed.known:
        return CampaignResolution(None, False, "observed_identity_unknown")

    for campaign in existing:
        diffs = observed.differences(campaign.source.card)
        if not diffs:
            return CampaignResolution(campaign, True, "card_identity_match")

    prior = _prior_at_mount(existing, observed, mount_point)
    if prior is not None:
        return CampaignResolution(
            campaign=None,
            reused=False,
            reason="card_identity_changed",
            conflicting_fields=tuple(prior.source.card.differences(observed)),
        )
    return CampaignResolution(None, False, "no_matching_campaign")


def _prior_at_mount(
    existing: Iterable[Campaign],
    observed: CardIdentity,
    mount_point: Optional[str] = None,
) -> Optional[Campaign]:
    """A campaign previously seen for the same label or mount (diagnostics)."""
    for campaign in existing:
        if observed.label and campaign.source.card.label == observed.label:
            return campaign
        if mount_point and campaign.source.mount_point == mount_point:
            return campaign
    return None


def campaigns_sharing_mount(existing: Iterable[Campaign], mount_point: Optional[str]) -> List[str]:
    """Campaign ids previously observed at ``mount_point`` (diagnostics only)."""
    if not mount_point:
        return []
    return sorted(
        c.campaign_id for c in existing if c.source.mount_point == mount_point
    )


__all__ = [
    "Campaign",
    "CampaignResolution",
    "CardIdentity",
    "SourceRef",
    "campaigns_sharing_mount",
    "derive_campaign_id",
    "resolve_campaign",
]
