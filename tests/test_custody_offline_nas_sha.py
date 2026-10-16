"""Offline NAS rechecks require independently qualified historical digests."""
from __future__ import annotations

import hashlib
import json

import pytest

from auto_ingest.custody import offline_nas_sha as offline


def _setup(tmp_path):
    ledger = tmp_path / "frozen.jsonl"
    report = tmp_path / "ready.json"
    rows = []
    candidates = []
    for i, qualified in enumerate((True, False, True)):
        uuid = "CARD-0001"
        key = f"DCIM/{i}.MP4"
        rows.append({
            "card_uuid": uuid, "source_relative_path": key,
            "logical_bytes": 10 + i,
            "xwing_full_source_sha256": "a" * 64 if qualified else None,
            "xwing_local_nvme_independent_full_sha_verified": qualified,
            "source_release_authorized": False,
            "canonical_nas_verified": False,
        })
        candidates.append({
            "card_uuid": uuid, "source_relative_path": key,
            "bytes": 10 + i, "nas_candidate": f"/nas/archive/{i}.MP4",
            "nas_metadata_state": "nas_metadata_size_match_only",
            "source_release_authorized": False,
            "content_sha_checked_this_run": False,
        })
    ledger.write_text("".join(json.dumps(x) + "\n" for x in rows))
    report.write_text(json.dumps({
        "scope": "metadata_only_no_source_or_nas_payload_reads",
        "candidate_count": len(candidates), "complete_metadata_scan": True,
        "source_release_authorized": False, "rows": candidates,
    }))
    pin = hashlib.sha256(ledger.read_bytes()).hexdigest()
    return ledger, report, pin


def test_select_denies_unqualified_and_budgets(tmp_path):
    ledger, ready, pin = _setup(tmp_path)
    records, audit = offline.select(
        ledger, ready, expected_frozen_sha=pin, max_files=1,
        max_nas_bytes=15)
    assert [x["source_relative_path"] for x in records] == ["DCIM/0.MP4"]
    assert audit["qualified_candidates_total"] == 2
    assert audit["selected_nas_bytes"] == 10
    assert records[0]["source_receipt_kind"].startswith("xwing_local_")


def test_frozen_pin_mismatch_denied(tmp_path):
    ledger, ready, pin = _setup(tmp_path)
    with pytest.raises(ValueError, match="frozen source ledger SHA mismatch"):
        offline.select(ledger, ready, expected_frozen_sha="f" * 64)
    ledger.write_text(ledger.read_text() + "\n")
    with pytest.raises(ValueError, match="frozen source ledger SHA mismatch"):
        offline.select(ledger, ready, expected_frozen_sha=pin)


def test_incomplete_report_and_wrong_bytes_denied(tmp_path):
    ledger, ready, pin = _setup(tmp_path)
    x = json.loads(ready.read_text())
    x["complete_metadata_scan"] = False
    ready.write_text(json.dumps(x))
    with pytest.raises(ValueError, match="readiness report incomplete"):
        offline.select(ledger, ready, expected_frozen_sha=pin)
    x["complete_metadata_scan"] = True
    x["rows"][0]["bytes"] += 1
    ready.write_text(json.dumps(x))
    with pytest.raises(ValueError, match="candidate not bound"):
        offline.select(ledger, ready, expected_frozen_sha=pin)


def test_unsafe_path_denied(tmp_path):
    ledger, ready, pin = _setup(tmp_path)
    x = json.loads(ready.read_text())
    x["rows"][0]["nas_candidate"] = "/nas/archive/../../etc/passwd"
    ready.write_text(json.dumps(x))
    with pytest.raises(ValueError, match="unsafe NAS destination"):
        offline.select(ledger, ready, expected_frozen_sha=pin)


def test_prior_exact_provenance_required(tmp_path):
    ledger, ready, pin = _setup(tmp_path)
    selected, audit = offline.select(ledger, ready, expected_frozen_sha=pin)
    old = tmp_path / "prior.json"
    old.write_text(json.dumps({
        "scope": offline.SCOPE,
        "source_ledger_sha256": pin,
        "readiness_sha256": audit["readiness_sha256"],
        "source_release_authorized": False,
        "results": [{
            "card_uuid": selected[0]["card_uuid"],
            "source_relative_path": selected[0]["source_relative_path"],
            "nas_sha256": "a" * 64,
            "sha_matches_historical_source": True,
            "destination_stable": True,
            "nas_candidate": selected[0]["nas_candidate"],
            "bytes": selected[0]["bytes"],
            "historic_source_sha256": "a" * 64,
            "source_card_read_this_run": False,
            "source_release_authorized": False,
        }],
    }))
    later, stats = offline.select(
        ledger, ready, expected_frozen_sha=pin, prior_reports=[old])
    assert stats["previously_compared_matching"] == 1
    assert len(later) == 1
    assert later[0]["source_relative_path"] != selected[0]["source_relative_path"]
    wrong = json.loads(old.read_text())
    wrong["readiness_sha256"] = "b" * 64
    old.write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="provenance mismatch"):
        offline.select(ledger, ready, expected_frozen_sha=pin, prior_reports=[old])


def test_unqualified_digest_value_is_not_evidence(tmp_path):
    ledger, ready, pin = _setup(tmp_path)
    rows = [json.loads(x) for x in ledger.read_text().splitlines()]
    rows[1]["xwing_full_source_sha256"] = "b" * 64
    ledger.write_text("".join(json.dumps(x) + "\n" for x in rows))
    pin = hashlib.sha256(ledger.read_bytes()).hexdigest()
    selected, stats = offline.select(ledger, ready, expected_frozen_sha=pin)
    assert stats["qualified_candidates_total"] == 2
    assert len(selected) == 2
