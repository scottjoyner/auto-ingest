"""Campaign identity: a different card must never inherit CARD-01's state."""
from __future__ import annotations

from custody_helpers import CREATED_AT, campaign, card, destination

from auto_ingest.custody import (
    Campaign,
    CardIdentity,
    SourceRef,
    resolve_campaign,
)
from auto_ingest.custody.destination import (
    LogicalDestination,
    StorageIdentity,
    match_identity,
    resolve_destination,
)


def test_card_key_is_stable_for_the_same_hardware():
    assert card().key == card().key
    assert card().key != card(uuid="1111-2222").key


def test_label_alone_is_not_identity():
    """Every unformatted card is called UNTITLED; that proves nothing."""
    a = CardIdentity(device="/dev/sdb1", filesystem_uuid="AAAA-1111", label="UNTITLED")
    b = CardIdentity(device="/dev/sdc1", filesystem_uuid="BBBB-2222", label="UNTITLED")
    assert a.key != b.key
    assert a.differences(b) == ("filesystem_uuid",)


def test_same_card_reused_across_mount_points():
    existing = [campaign()]
    observed = card(device="/dev/sdc9")  # same uuid/serial, different device slot
    result = resolve_campaign(observed, existing, mount_point="/run/media/scotty/UNTITLED")
    assert result.reused is True
    assert result.campaign is existing[0]


def test_different_card_at_the_same_mount_point_is_not_reused():
    existing = [campaign()]
    other = card(uuid="FFFF-9999", device="/dev/sdc1")
    result = resolve_campaign(other, existing, mount_point="/media/scott/UNTITLED")
    assert result.reused is False
    assert result.campaign is None
    assert result.reason == "card_identity_changed"
    assert "filesystem_uuid" in result.conflicting_fields


def test_different_card_at_untitled_does_not_inherit_state():
    existing = [campaign(card_id="CARD-01")]
    other = card(uuid="FFFF-9999", device="/dev/sdc1", label="UNTITLED")
    result = resolve_campaign(other, existing, mount_point="/media/scott/UNTITLED")
    assert result.campaign is None
    replacement = Campaign.create(
        card_id="CARD-02",
        source=SourceRef(mount_point="/media/scott/UNTITLED", read_only=True, card=other),
        destination=destination(),
        created_at="2026-10-02T19:00:00Z",
    )
    assert replacement.campaign_id != existing[0].campaign_id


def test_device_change_alone_is_enough_to_refuse_reuse():
    """A uuid we cannot compare (absent on the stored record) must not be assumed."""
    stored = Campaign.create(
        card_id="CARD-09",
        source=SourceRef(mount_point="/media/scott/UNTITLED", read_only=True,
                         card=CardIdentity(device="/dev/sdb1", label="UNTITLED")),
        destination=destination(),
        created_at=CREATED_AT,
    )
    result = resolve_campaign(
        CardIdentity(device="/dev/sdb1", filesystem_uuid="NEW-0001", label="UNTITLED"),
        [stored],
        mount_point="/media/scott/UNTITLED",
    )
    assert result.reused is False


def test_unknown_observed_identity_is_refused():
    result = resolve_campaign(CardIdentity(), [campaign()])
    assert result.reused is False
    assert result.reason == "observed_identity_unknown"


def test_campaign_id_is_deterministic_from_the_card():
    a = Campaign.create(card_id="CARD-01",
                        source=SourceRef(read_only=True, card=card()),
                        destination=destination(), created_at=CREATED_AT)
    b = Campaign.create(card_id="CARD-01",
                        source=SourceRef(read_only=True, card=card()),
                        destination=destination(), created_at=CREATED_AT)
    assert a.campaign_id == b.campaign_id
    assert a.campaign_id.startswith("sdcard-")


def test_campaign_id_changes_when_the_card_changes():
    a = campaign()
    b = campaign(card_identity=card(uuid="OTHER-0001"))
    assert a.campaign_id != b.campaign_id


def test_campaign_round_trips_through_dict():
    original = campaign()
    restored = Campaign.from_dict(original.to_dict())
    assert restored == original


def test_campaign_recovers_a_missing_id_deterministically():
    raw = campaign().to_dict()
    raw.pop("campaign_id")
    restored = Campaign.from_dict(raw)
    assert restored.campaign_id == campaign().campaign_id


def test_with_observation_refuses_a_different_card():
    original = campaign()
    assert original.with_observation(card()) is not None
    assert original.with_observation(card(uuid="OTHER-0001")) is None


def test_with_observation_refreshes_only_observation_fields():
    original = campaign()
    refreshed = original.with_observation(
        card(), observed_at="2026-10-02T20:00:00Z", read_only=True
    )
    assert refreshed.campaign_id == original.campaign_id
    assert refreshed.last_observed_at == "2026-10-02T20:00:00Z"
    assert refreshed.source.card == original.source.card


# ---------------------------------------------------------------------------
# canonical destination abstraction
# ---------------------------------------------------------------------------
def test_destination_is_unresolved_without_configuration():
    ref = resolve_destination({}, env={})
    assert ref.resolved is False
    assert ref.host_path is None
    assert ref.resolved_from == "unresolved"
    assert ref.logical.canonical == "primary:fileserver/dashcam"


def test_destination_resolves_from_env():
    ref = resolve_destination({}, env={"CUSTODY_DESTINATION_ROOT": "/srv/custody"},
                             check_mount=False)
    assert ref.host_path == "/srv/custody"
    assert ref.resolved_from == "env:CUSTODY_DESTINATION_ROOT"
    assert ref.campaign_path() == "/srv/custody/fileserver/dashcam"


def test_destination_resolves_from_config_placeholder():
    ref = resolve_destination(
        {"destination_root": "${SOME_HOST_ROOT}"},
        env={"SOME_HOST_ROOT": "/mnt/custody"},
        check_mount=False,
    )
    assert ref.host_path == "/mnt/custody"
    assert ref.resolved_from == "config:custody.destination_root"


def test_destination_unresolved_placeholder_fails_closed():
    ref = resolve_destination({"destination_root": "${NOT_SET_ANYWHERE}"}, env={},
                              check_mount=False)
    assert ref.resolved is False
    assert ref.host_path is None


def test_env_beats_config():
    ref = resolve_destination(
        {"destination_root": "/from/config"},
        env={"CUSTODY_DESTINATION_ROOT": "/from/env"},
        check_mount=False,
    )
    assert ref.host_path == "/from/env"


def test_destination_mount_state_is_reported_not_assumed():
    ref = resolve_destination({"destination_root": "/definitely/not/a/mount"},
                              env={}, check_mount=True)
    assert ref.resolved is True
    assert ref.mounted is False


def test_identity_match_is_fail_closed():
    assert match_identity(None, None).matched is False
    a = StorageIdentity(filesystem_uuid="AAAA")
    b = StorageIdentity(filesystem_uuid="aaaa")  # case-insensitive
    assert match_identity(a, b).matched is True
    assert match_identity(a, StorageIdentity(filesystem_uuid="BBBB")).matched is False
    assert match_identity(a, StorageIdentity()).matched is False
    assert match_identity(a, StorageIdentity()).comparable is False


def test_logical_destination_is_host_independent():
    ref = resolve_destination({}, env={}, check_mount=False)
    assert ref.logical == LogicalDestination("primary", "fileserver/dashcam")
    assert "/" not in ref.logical.name


def test_no_historical_destination_is_embedded_in_the_package():
    """The contract must not name NAS3/NAS5/SSD_4TB as the canonical destination."""
    from pathlib import Path

    import auto_ingest.custody as pkg

    root = Path(pkg.__file__).parent
    banned = ("NAS3", "NAS5", "SSD_4TB", "/media/scott/")
    for path in sorted(root.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for literal in banned:
            assert literal not in text, f"{path.name} hardcodes {literal}"
