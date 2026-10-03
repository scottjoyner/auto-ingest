"""`custody new`: the entry point of the operator loop.

Creating a campaign is the one moment a campaign id is minted, so it is where
identity mistakes are most expensive. These tests pin the three fail-closed
refusals and prove the created record derives `DISCOVERED` on its own.
"""
from __future__ import annotations

import json

import pytest
from custody_helpers import card

from auto_ingest.custody import CampaignState
from auto_ingest.custody.cli import EXIT_GATE_CLOSED, EXIT_OK, main
from auto_ingest.custody.store import (
    CampaignCreationError,
    load_campaign,
    load_status,
    new_campaign,
)

NEW_ARGS = [
    "--card-id", "CARD-02",
    "--uuid", "ABCD-1234",
    "--device", "/dev/sdc1",
    "--label", "UNTITLED",
    "--mount", "/media/scott/UNTITLED",
    "--read-only",
    "--created-at", "2026-10-02T21:00:00Z",
]


def new(bundle, *extra, json_mode=True):
    argv = ["new", "--bundle", str(bundle), *NEW_ARGS, *extra]
    if json_mode:
        argv.append("--json")
    return argv


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------
def test_label_only_is_refused(tmp_path):
    with pytest.raises(CampaignCreationError) as exc:
        new_campaign(tmp_path / "b", card_id="CARD-04",
                     observed=type(card())(label="UNTITLED"),
                     created_at="2026-10-02T21:00:00Z", apply=True)
    assert "unprovable" in str(exc.value)
    assert not (tmp_path / "b").exists()


def test_empty_identity_is_refused(tmp_path):
    with pytest.raises(CampaignCreationError):
        new_campaign(tmp_path / "b", card_id="CARD-05", observed=type(card())(),
                     created_at="2026-10-02T21:00:00Z", apply=True)


def test_existing_campaign_is_never_overwritten(tmp_path):
    bundle = tmp_path / "b"
    first = new_campaign(bundle, card_id="CARD-02", observed=card(uuid="ABCD-1234"),
                         mount_point="/media/scott/UNTITLED", read_only=True,
                         created_at="2026-10-02T21:00:00Z", apply=True)
    assert first["applied"] is True
    before = (bundle / "campaign.json").read_text(encoding="utf-8")
    with pytest.raises(CampaignCreationError) as exc:
        new_campaign(bundle, card_id="CARD-03", observed=card(uuid="ABCD-1234"),
                     created_at="2026-10-02T22:00:00Z", apply=True)
    assert "already exists" in str(exc.value)
    assert (bundle / "campaign.json").read_text(encoding="utf-8") == before


# ---------------------------------------------------------------------------
# creation
# ---------------------------------------------------------------------------
def test_new_is_validate_only_by_default(tmp_path, capsys):
    code = main(new(tmp_path / "b"))
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert json.loads(out)["mode"] == "validate_only"
    assert json.loads(out)["applied"] is False
    assert not (tmp_path / "b").exists()


def test_new_apply_writes_a_loadable_campaign(tmp_path, capsys):
    code = main(new(tmp_path / "b", "--apply"))
    assert code == EXIT_OK
    data = json.loads(capsys.readouterr().out)
    assert data["mode"] == "applied"
    assert data["state"] == "DISCOVERED"
    campaign = load_campaign(tmp_path / "b")
    assert campaign.card_id == "CARD-02"
    assert campaign.campaign_id == data["campaign_id"]
    assert campaign.source.read_only is True
    assert campaign.source.mount_point == "/media/scott/UNTITLED"
    assert campaign.source.card.filesystem_uuid == "ABCD-1234"


def test_new_campaign_is_identical_for_identical_hardware(tmp_path, capsys):
    main(new(tmp_path / "a", "--apply"))
    first = capsys.readouterr().out
    main(new(tmp_path / "b", "--apply"))
    second = capsys.readouterr().out
    assert json.loads(first)["campaign_id"] == json.loads(second)["campaign_id"]


def test_different_hardware_gets_a_different_campaign_id(tmp_path, capsys):
    main(new(tmp_path / "a", "--apply"))
    first = json.loads(capsys.readouterr().out)["campaign_id"]
    argv = list(new(tmp_path / "b", "--apply"))
    argv[argv.index("ABCD-1234")] = "ZZZZ-9999"
    main(argv)
    second = json.loads(capsys.readouterr().out)["campaign_id"]
    assert first != second


def test_unresolved_destination_is_recorded_not_guessed(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("CUSTODY_DESTINATION_ROOT", raising=False)
    monkeypatch.delenv("CUSTODY_DESTINATION_PATH", raising=False)
    main(new(tmp_path / "b", "--apply"))
    capsys.readouterr()
    campaign = load_campaign(tmp_path / "b")
    assert campaign.destination.resolved is False
    assert campaign.destination.host_path is None
    assert campaign.destination.logical.canonical == "primary:fileserver/dashcam"


def test_destination_is_resolved_from_env_when_configured(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("CUSTODY_DESTINATION_ROOT", str(tmp_path / "dest"))
    main(new(tmp_path / "b", "--apply"))
    capsys.readouterr()
    campaign = load_campaign(tmp_path / "b")
    assert campaign.destination.host_path == str(tmp_path / "dest")
    assert campaign.destination.resolved_from == "env:CUSTODY_DESTINATION_ROOT"


def test_a_fresh_campaign_derives_discovered(tmp_path, capsys):
    main(new(tmp_path / "b", "--apply"))
    capsys.readouterr()
    status = load_status(tmp_path / "b")
    assert status.state is CampaignState.DISCOVERED
    assert status.source_release_allowed is False
    assert status.plan.next_safe_action == "inventory_source"
    assert status.evidence.inventory.discovered_files == 0


def test_new_then_import_then_status_is_the_documented_loop(tmp_path, capsys):
    main(new(tmp_path / "b", "--apply"))
    capsys.readouterr()
    evidence = tmp_path / "ev.json"
    evidence.write_text(json.dumps({
        "campaign_id": load_campaign(tmp_path / "b").campaign_id,
        "inventory": {"discovered_files": 12, "discovered_bytes": 1200,
                      "complete": True, "verified": True},
        "hash": {"verified_files": 12, "verified_bytes": 1200, "complete": True},
    }), encoding="utf-8")
    code = main(["import", "--bundle", str(tmp_path / "b"), "--evidence", str(evidence),
                 "--apply", "--json"])
    assert code == EXIT_OK
    capsys.readouterr()
    status = load_status(tmp_path / "b")
    assert status.state is CampaignState.HASH_COMPLETE
    assert status.source_release_allowed is False


def test_cli_refusal_exits_three(tmp_path, capsys):
    main(new(tmp_path / "b", "--apply"))
    capsys.readouterr()
    code = main(new(tmp_path / "b", "--apply"))
    assert code == EXIT_GATE_CLOSED
    assert "refusing to create campaign" in capsys.readouterr().err
