"""Foreign destination objects must survive the evidence round trip.

``reconcile_bundle`` walks the destination root and counts objects this campaign
never recorded (``foreign_objects``). The reconciliation proposal hands that
count to ``CampaignEvidence.from_dict``, and the strict-scope release gate reads
it back to decide whether those objects block a source release.

The count was produced and reported by the CLI, but ``from_dict`` never read it:
the round trip zeroed it, so the gate could not fail closed on stray files at the
destination. These tests are hermetic - in-memory evidence plus a tmp_path tree.
"""
from __future__ import annotations

import pytest
from custody_helpers import (
    campaign,
    copying,
    destination,
    destination_evidence,
    evidence,
    fully_copied_campaign,
    hashing,
    inventory,
    strict_policy,
    worker,
    write_bundle,
)

from auto_ingest.custody import CampaignEvidence, CampaignState
from auto_ingest.custody.ledger import LedgerRead, ReconciliationResult
from auto_ingest.custody.release import evaluate_release
from auto_ingest.custody.store import merge_evidence_documents, reconcile_preview

pytestmark = pytest.mark.usefixtures("hermetic_mounts")


def _write_ledger(path, rows):
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_foreign_objects_survive_evidence_from_dict():
    """A declared foreign count is not silently zeroed by normalisation.

    ``to_dict`` emits `foreign_objects` and ``from_dict`` must rehydrate it. The
    round trip is exactly what `custody import --apply` performs.
    """
    doc = {
        "reconciliation": {
            "source_only": 0,
            "destination_only": 0,
            "mismatched": 0,
            "foreign_objects": 7,
        }
    }
    ev = CampaignEvidence.from_dict(doc)
    assert ev.reconciliation.foreign_objects == 7
    assert ev.to_dict()["reconciliation"]["foreign_objects"] == 7


def test_reconciliation_proposal_foreign_count_reaches_the_release_gate():
    """The fail-open: a proposal says 3 foreign objects, the gate sees 0.

    This is the production chain without any filesystem: proposal -> merge ->
    normalise -> evaluate. With strict destination scope the foreign objects must
    be a release blocker, not an advisory that vanished on the way.
    """
    camp, ev = fully_copied_campaign()
    coherent = LedgerRead(present=True)
    result = ReconciliationResult(
        hash_ledger_present=True,
        destination_ledger_present=True,
        verified=100,
        foreign_objects=3,
        source_read=coherent,
        destination_read=coherent,
    )
    proposal = result.proposal()
    assert proposal is not None
    assert proposal["reconciliation"]["foreign_objects"] == 3  # precondition

    merged = merge_evidence_documents(ev.to_dict(), proposal)
    loaded = CampaignEvidence.from_dict(merged)
    decision = evaluate_release(
        camp, loaded, strict_policy(strict_destination_scope=True))
    assert "foreign_destination_objects" in {b.code for b in decision.blockers}
    assert decision.allowed is False


def test_preview_strict_scope_blocks_on_stray_destination_object(tmp_path):
    """End to end through reconcile_preview, which builds status from evidence.

    The CLI's reconcile text output already showed `foreign_at_destination 1`;
    the status derived from the same diff did not, so the strict gate stayed open.
    """
    dest_root = tmp_path / "dest"
    dest_root.mkdir()
    (dest_root / "stray.MP4").write_bytes(b"x")

    camp = campaign(dest=destination(host_path=str(dest_root)))
    ev = evidence(
        inv=inventory(1, 100, complete=True, verified=True),
        hsh=hashing(1, verified_bytes=100, complete=True),
        cpy=copying(planned_files=1, planned_bytes=100, completed_files=1,
                    completed_bytes=100, started=True, result_complete=True,
                    ledger_complete=True),
        dst=destination_evidence(
            verified_files=1, verified_bytes=100,
            verification_started=True, verification_complete=True,
            observed_identity=camp.destination.identity),
        wkr=worker(status="stopped"),
    )
    bundle = write_bundle(tmp_path / "b", camp, ev)
    _write_ledger(bundle / "ledgers" / "hash.jsonl", [{"key": "a", "digest": "A1",
                                                       "status": "verified"}])
    _write_ledger(bundle / "ledgers" / "destination.jsonl",
                  [{"key": "a", "digest": "A1", "status": "verified_at_destination"}])

    result, status = reconcile_preview(
        bundle, strict_policy(strict_destination_scope=True))

    assert result.foreign_objects == 1  # the walk found it
    assert status.evidence.reconciliation.foreign_objects == 1  # status must carry it
    assert "foreign_destination_objects" in {b.code for b in status.release.blockers}
    assert status.source_release_allowed is False
    assert status.derivation.state is not CampaignState.SAFE_TO_RELEASE
