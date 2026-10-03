"""Scale: the commands Hermes will poll must stay cheap on a real card.

CARD-01 is 67,644 objects. "Bounded summary" only means something if the status
and reconcile paths stay fast with full ledgers on disk, so these tests build a
card-sized bundle and assert the read paths complete well inside an interactive
budget. The bounds are deliberately generous - they are guardrails against an
accidental O(n^2) or an accidental full-materialisation, not benchmarks.
"""
from __future__ import annotations

import json
import time

from custody_helpers import (
    campaign,
    copying,
    destination_evidence,
    evidence,
    hashing,
    inventory,
    strict_policy,
    worker,
    write_bundle,
)

from auto_ingest.custody import CampaignState
from auto_ingest.custody.ledger import reconcile_bundle, summarize_bundle_ledgers
from auto_ingest.custody.store import load_status, reconcile_preview

#: CARD-01's observed size.
OBJECTS = 67644

#: Generous ceilings. A healthy run is far below these; the point is to fail on
#: an algorithmic regression, not to benchmark.
STATUS_BUDGET_SEC = 20.0
RECONCILE_BUDGET_SEC = 30.0


def _ledger(path, keys, *, verified_at_destination):
    status = "verified_at_destination" if verified_at_destination else "verified"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for i, key in enumerate(keys):
            handle.write(json.dumps({
                "key": key,
                "path": f"2026/04/12/{key}.MP4",
                "size": 1024 * 1024,
                "digest": f"{i:064x}",
                "status": status,
            }) + "\n")


def card_sized_bundle(tmp_path, *, verified_at_destination):
    """A bundle with a full hash ledger and a partially-filled destination ledger."""
    keys = [f"2026_0412_{i:06d}_F" for i in range(OBJECTS)]
    verified = keys[: int(OBJECTS * 0.61)]
    camp = campaign()
    ev = evidence(
        inv=inventory(OBJECTS, OBJECTS * 1024 * 1024, complete=True, verified=True),
        hsh=hashing(OBJECTS, verified_bytes=OBJECTS * 1024 * 1024, complete=True),
        cpy=copying(planned_files=OBJECTS, planned_bytes=OBJECTS * 1024 * 1024,
                    started=True, result_complete=False, interrupted=True),
        dst=destination_evidence(verified_files=0),
        wkr=worker(identity="legacy", status="stopped"),
    )
    bundle = write_bundle(tmp_path / "card01", camp, ev)
    _ledger(bundle / "ledgers" / "hash.jsonl", keys, verified_at_destination=False)
    _ledger(bundle / "ledgers" / "destination.jsonl", verified,
            verified_at_destination=True)
    return bundle


def test_status_over_card_sized_ledgers_is_interactive(tmp_path):
    bundle = card_sized_bundle(tmp_path, verified_at_destination=False)
    started = time.monotonic()
    status = load_status(bundle, strict_policy())
    elapsed = time.monotonic() - started
    assert elapsed < STATUS_BUDGET_SEC, f"status took {elapsed:.1f}s over {OBJECTS} objects"
    assert status.evidence.inventory.discovered_files == OBJECTS
    assert status.state is CampaignState.RECONCILE_REQUIRED
    # the summary stays bounded no matter how big the ledgers are
    payload = status.to_dict()
    assert len(json.dumps(payload)) < 32_000
    for summary in status.ledgers.values():
        assert summary.coherent is True


def test_summarize_is_linear_enough_at_card_scale(tmp_path):
    bundle = card_sized_bundle(tmp_path, verified_at_destination=False)
    started = time.monotonic()
    summaries = summarize_bundle_ledgers(bundle)
    elapsed = time.monotonic() - started
    assert elapsed < STATUS_BUDGET_SEC, f"summarize took {elapsed:.1f}s"
    assert summaries["hash.jsonl"].records == OBJECTS
    assert summaries["destination.jsonl"].files == int(OBJECTS * 0.61)


def test_reconcile_over_card_sized_ledgers_is_interactive(tmp_path):
    bundle = card_sized_bundle(tmp_path, verified_at_destination=False)
    started = time.monotonic()
    result = reconcile_bundle(bundle, expected_source_objects=OBJECTS)
    elapsed = time.monotonic() - started
    assert elapsed < RECONCILE_BUDGET_SEC, f"reconcile took {elapsed:.1f}s"
    verified = int(OBJECTS * 0.61)
    assert result.verified == verified
    assert result.source_only == OBJECTS - verified
    assert result.usable is True


def test_reconcile_preview_keeps_the_plan_bounded_at_card_scale(tmp_path):
    bundle = card_sized_bundle(tmp_path, verified_at_destination=False)
    result, status = reconcile_preview(bundle, strict_policy())
    assert result.verified == int(OBJECTS * 0.61)
    assert status.source_release_allowed is False
    payload = status.to_dict()
    # counts are exact, samples are capped: the report does not grow with the card
    assert payload["reconciliation"]["source_only"] == OBJECTS - int(OBJECTS * 0.61)
    assert len(json.dumps(payload)) < 32_000
    copy_action = next(a for a in status.plan.actions if a.operation == "copy_objects")
    assert copy_action.estimated_files == OBJECTS - int(OBJECTS * 0.61)
    assert copy_action.excludes_verified is True


def test_repeated_status_is_stable_at_card_scale(tmp_path):
    """Idempotency has to hold at the size the card actually is."""
    bundle = card_sized_bundle(tmp_path, verified_at_destination=False)
    first = load_status(bundle, strict_policy()).to_dict()
    second = load_status(bundle, strict_policy()).to_dict()
    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_a_huge_inventory_still_does_not_explode_the_summary(tmp_path):
    """One object, enormous byte count: nothing here scales with bytes."""
    camp = campaign()
    ev = evidence(inv=inventory(1, 900_000_000_000_000, complete=True, verified=True),
                  hsh=hashing(1, verified_bytes=900_000_000_000_000, complete=True))
    bundle = write_bundle(tmp_path / "one", camp, ev)
    payload = load_status(bundle, strict_policy()).to_dict()
    assert payload["inventory"]["discovered_bytes"] == 900_000_000_000_000
    assert len(json.dumps(payload)) < 32_000
