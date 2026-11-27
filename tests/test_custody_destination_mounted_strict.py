"""Destination `mounted` must be a boolean, not a truthy JSON string.

`Campaign.from_dict` copied `destination.mounted` raw. A campaign.json with
`"mounted": "false"` was therefore truthy, and preflight's
`destination_mounted` check (`campaign.destination.mounted is not False`) passed
for a destination explicitly declared unmounted.
"""
from __future__ import annotations

import pytest

from auto_ingest.custody.campaign import Campaign


def _campaign(mounted):
    raw = {
        "campaign_id": "campaign-1", "card_id": "CARD-01",
        "destination": {"logical": {"name": "primary"},
                        "host_path": "/archive", "mounted": mounted},
    }
    return Campaign.from_dict(raw)


def test_string_false_is_not_mounted():
    assert _campaign("false").destination.mounted is False
    assert _campaign("0").destination.mounted is False
    assert _campaign(0).destination.mounted is False


@pytest.mark.parametrize("value", [True, "true", "yes", "on", 1])
def test_true_spellings_still_parse(value):
    assert _campaign(value).destination.mounted is True


def test_missing_flag_is_unknown_not_false():
    assert Campaign.from_dict({
        "campaign_id": "campaign-1", "card_id": "CARD-01",
        "destination": {"host_path": "/archive"},
    }).destination.mounted is None


@pytest.mark.parametrize("value", ["maybe", "false-ish", [], {}])
def test_unparseable_flags_are_refused(value):
    with pytest.raises(ValueError, match="mounted"):
        _campaign(value)
