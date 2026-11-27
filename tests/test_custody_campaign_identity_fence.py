"""A campaign record with no identity must not be loadable.

`load_campaign` accepted a campaign.json whose campaign_id/card_id were empty,
so an unattributable bundle could be reported, planned and gated as if it were
a real campaign — and could collide with any other identity-less bundle.
"""
from __future__ import annotations

import json

import pytest

from auto_ingest.custody.store import BundleError, load_campaign


def _write(tmp_path, raw):
    (tmp_path / "campaign.json").write_text(json.dumps(raw), encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("field", ["campaign_id", "card_id"])
def test_missing_identity_is_refused(tmp_path, field):
    raw = {"campaign_id": "campaign-1", "card_id": "CARD-01"}
    raw[field] = "  "
    with pytest.raises(BundleError, match="identity"):
        load_campaign(_write(tmp_path, raw))


def test_valid_identity_still_loads(tmp_path):
    bundle = _write(tmp_path, {"campaign_id": "campaign-1", "card_id": "CARD-01"})
    campaign = load_campaign(bundle)
    assert campaign.campaign_id == "campaign-1"
    assert campaign.card_id == "CARD-01"
