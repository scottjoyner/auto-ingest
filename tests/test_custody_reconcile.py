"""Ledger-driven reconciliation: the source-vs-destination set difference.

This is the work Hermes should not have to do by hand. The diff is computed
from the two JSONL ledgers, it is read-only, and it only *proposes* evidence -
applying it stays an explicit `custody import --apply`.
"""
from __future__ import annotations

import json

import pytest
from custody_helpers import (
    CARD_01_BUNDLE,
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
from auto_ingest.custody.cli import EXIT_GATE_CLOSED, EXIT_OK, main
from auto_ingest.custody.ledger import (
    ReconciliationUnavailable,
    reconcile_bundle,
    reconcile_ledgers,
    scan_foreign_objects,
)
from auto_ingest.custody.store import load_status, reconcile_preview

# The source end of a campaign is a fact this module controls, not a fact
# about whether the developer's card happens to be plugged in.
pytestmark = pytest.mark.usefixtures("hermetic_mounts")
def write_ledger(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def src(key, digest, **kw):
    return {"key": key, "digest": digest, "status": "verified", **kw}


def dst(key, digest, **kw):
    return {"key": key, "digest": digest, "status": "verified_at_destination", **kw}


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------
def test_identical_ledgers_reconcile_clean(tmp_path):
    h = write_ledger(tmp_path / "h.jsonl", [src("a", "A1"), src("b", "B2")])
    d = write_ledger(tmp_path / "d.jsonl", [dst("a", "A1"), dst("b", "B2")])
    r = reconcile_ledgers(h, d)
    assert (r.verified, r.source_only, r.destination_only, r.mismatched) == (2, 0, 0, 0)
    assert r.complete is True
    assert r.usable is True


def test_digest_comparison_is_case_insensitive(tmp_path):
    h = write_ledger(tmp_path / "h.jsonl", [src("a", "AABBCC")])
    d = write_ledger(tmp_path / "d.jsonl", [dst("a", "aabbcc")])
    assert reconcile_ledgers(h, d).verified == 1


def test_partial_copy_yields_source_only_for_the_rest(tmp_path):
    """The CARD-01 shape: 3 of 10 made it, 7 did not."""
    h = write_ledger(tmp_path / "h.jsonl", [src(f"k{i}", f"d{i}") for i in range(10)])
    d = write_ledger(tmp_path / "d.jsonl", [dst(f"k{i}", f"d{i}") for i in range(3)])
    r = reconcile_ledgers(h, d)
    assert (r.verified, r.source_only) == (3, 7)
    assert r.complete is False
    assert sorted(r.source_only_samples) == [f"k{i}" for i in range(3, 10)]


def test_digest_disagreement_is_a_mismatch_not_a_verification(tmp_path):
    h = write_ledger(tmp_path / "h.jsonl", [src("a", "GOOD"), src("b", "GOOD")])
    d = write_ledger(tmp_path / "d.jsonl", [dst("a", "GOOD"), dst("b", "CORRUPT")])
    r = reconcile_ledgers(h, d)
    assert (r.verified, r.mismatched, r.mismatched_samples) == (1, 1, ("b",))
    assert r.complete is False


def test_extra_destination_object_is_destination_only(tmp_path):
    h = write_ledger(tmp_path / "h.jsonl", [src("a", "A")])
    d = write_ledger(tmp_path / "d.jsonl", [dst("a", "A"), dst("stowaway", "Z")])
    r = reconcile_ledgers(h, d)
    assert r.destination_only == 1
    assert r.destination_only_samples == ("stowaway",)


def test_a_record_without_a_digest_never_counts_as_custody(tmp_path):
    h = write_ledger(tmp_path / "h.jsonl", [src("a", None), src("b", "B")])
    d = write_ledger(tmp_path / "d.jsonl", [dst("a", "A"), dst("b", "B")])
    r = reconcile_ledgers(h, d)
    assert r.verified == 1
    assert r.unverifiable == 1
    assert r.unverifiable_samples == ("a",)
    assert r.complete is False
    # ...and importing the proposal counts it as lacking custody, not as verified
    assert r.proposal()["reconciliation"]["source_only"] == 1


def test_destination_record_without_a_digest_is_unverifiable(tmp_path):
    h = write_ledger(tmp_path / "h.jsonl", [src("a", "A")])
    d = write_ledger(tmp_path / "d.jsonl", [dst("a", None)])
    r = reconcile_ledgers(h, d)
    assert r.verified == 0 and r.unverifiable == 1


def test_non_verified_statuses_are_excluded_from_both_sides(tmp_path):
    h = write_ledger(tmp_path / "h.jsonl", [
        src("good", "G"), {"key": "bad", "status": "failed", "digest": "B"},
        {"key": "pending", "status": "pending", "digest": "P"},
    ])
    d = write_ledger(tmp_path / "d.jsonl", [dst("good", "G")])
    r = reconcile_ledgers(h, d)
    assert r.source_objects == 1
    assert r.verified == 1
    assert r.source_only == 0


def test_destination_status_vocabulary_is_honoured(tmp_path):
    h = write_ledger(tmp_path / "h.jsonl", [src("a", "A"), src("b", "B")])
    d = write_ledger(tmp_path / "d.jsonl", [
        {"key": "a", "digest": "A", "status": "copied"},
        {"key": "b", "digest": "B", "status": "pending"},
    ])
    r = reconcile_ledgers(h, d)
    assert r.verified == 1
    assert r.source_only == 1


def test_malformed_lines_are_skipped_not_fatal(tmp_path):
    h = write_ledger(tmp_path / "h.jsonl", [src("a", "A")])
    h.write_text(h.read_text(encoding="utf-8") + "not json\n\n[1,2,3]\n", encoding="utf-8")
    d = write_ledger(tmp_path / "d.jsonl", [dst("a", "A")])
    assert reconcile_ledgers(h, d).verified == 1


# ---------------------------------------------------------------------------
# fail closed
# ---------------------------------------------------------------------------
def test_missing_ledgers_are_unusable_and_propose_nothing(tmp_path):
    h = write_ledger(tmp_path / "h.jsonl", [src("a", "A")])
    r = reconcile_ledgers(h, tmp_path / "nope.jsonl")
    assert r.usable is False
    assert r.proposal() is None
    assert r.complete is False


def test_to_evidence_refuses_when_unusable(tmp_path):
    with pytest.raises(ReconciliationUnavailable) as exc:
        reconcile_ledgers(tmp_path / "nope.jsonl", tmp_path / "nope2.jsonl").to_evidence()
    assert "refusing to propose a partial or untrusted diff" in str(exc.value)
    assert "hash_ledger_absent" in str(exc.value)


def test_an_empty_but_present_ledger_is_usable_and_says_zero_source_only(tmp_path):
    """Distinguishable from "absent": an empty ledger is a real, if alarming, fact."""
    h = write_ledger(tmp_path / "h.jsonl", [])
    d = write_ledger(tmp_path / "d.jsonl", [])
    r = reconcile_ledgers(h, d)
    assert r.usable is True
    assert r.complete is True
    assert r.source_objects == 0


# ---------------------------------------------------------------------------
# corruption must void the proposal, never shrink it
# ---------------------------------------------------------------------------
def test_interrupted_producer_does_not_claim_custody_for_rows_it_reached(tmp_path):
    """THE dangerous case: a ledger truncated mid-append.

    The 40 rows that made it to disk look perfectly healthy. A presence-only
    gate would propose "40 verified, 0 source_only" and say nothing about the
    60 it never reached - which then reads as proven custody for the whole card.
    """
    full = [{"key": f"k{i}", "digest": f"d{i}", "status": "verified"} for i in range(100)]
    d = write_ledger(tmp_path / "d.jsonl", full)
    h = tmp_path / "h.jsonl"
    h.write_text("".join(json.dumps(r) + "\n" for r in full[:40])
                  + '{"key": "k40", "diges', encoding="utf-8")
    r = reconcile_ledgers(h, d, expected_source_objects=100)
    assert r.usable is False
    assert r.proposal() is None
    assert "hash_ledger_truncated" in r.incoherent
    assert "hash_ledger_malformed_lines=1" in r.incoherent


def test_truncated_destination_ledger_also_voids_the_proposal(tmp_path):
    full = [{"key": f"k{i}", "digest": f"d{i}", "status": "verified"} for i in range(10)]
    h = write_ledger(tmp_path / "h.jsonl", full)
    d = tmp_path / "d.jsonl"
    d.write_text("".join(json.dumps(r) + "\n" for r in full[:5]) + '{"key":"k5","dig',
                 encoding="utf-8")
    r = reconcile_ledgers(h, d)
    assert r.usable is False
    assert "destination_ledger_truncated" in r.incoherent


def test_a_partially_written_hash_ledger_cannot_masquerade_as_complete(tmp_path):
    """Cross-checked against the recorded inventory, not just its own line count."""
    full = [{"key": f"k{i}", "digest": f"d{i}", "status": "verified"} for i in range(100)]
    d = write_ledger(tmp_path / "d.jsonl", full)
    h = write_ledger(tmp_path / "h.jsonl", full[:40])
    r = reconcile_ledgers(h, d, expected_source_objects=67644)
    assert r.usable is False
    assert any("covers_40_of_67644" in reason for reason in r.incoherent)


def test_a_garbage_ledger_is_not_an_empty_one(tmp_path):
    h = tmp_path / "h.jsonl"
    h.write_text("\x00\x01not json\n[]\n{}\n\n", encoding="utf-8")
    d = write_ledger(tmp_path / "d.jsonl", [{"key": "a", "digest": "A",
                                             "status": "verified"}])
    r = reconcile_ledgers(h, d)
    assert r.usable is False
    assert r.proposal() is None
        # 3 junk lines: unparseable, a JSON array, and an object with no status
    assert "hash_ledger_malformed_lines=2" in r.incoherent


def test_undecodable_bytes_do_not_raise(tmp_path):
    h = tmp_path / "h.jsonl"
    h.write_bytes(b'{"key": "a", "digest": "A", "status": "verified"}\n\xff\xfe\x00bad\n')
    d = write_ledger(tmp_path / "d.jsonl", [{"key": "a", "digest": "A",
                                             "status": "verified"}])
    r = reconcile_ledgers(h, d)  # must not raise
    assert r.usable is False
    assert r.verified == 1


def test_coherence_ignores_trailing_blank_lines(tmp_path):
    """A final newline after the last record is normal, not truncation."""
    rows = [{"key": f"k{i}", "digest": f"d{i}", "status": "verified"} for i in range(3)]
    h = write_ledger(tmp_path / "h.jsonl", rows)
    h.write_text(h.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    d = write_ledger(tmp_path / "d.jsonl", rows)
    r = reconcile_ledgers(h, d)
    assert r.usable is True
    assert r.verified == 3


def test_no_inventory_cross_check_available_still_works(tmp_path):
    """Nothing inventoried yet -> no cross-check, but coherence still required."""
    rows = [{"key": f"k{i}", "digest": f"d{i}", "status": "verified"} for i in range(3)]
    h = write_ledger(tmp_path / "h.jsonl", rows)
    d = write_ledger(tmp_path / "d.jsonl", rows)
    r = reconcile_ledgers(h, d, expected_source_objects=None)
    assert r.usable is True
    assert r.complete is True


def test_to_evidence_names_the_corruption(tmp_path):
    rows = [{"key": "a", "digest": "A", "status": "verified"}]
    h = write_ledger(tmp_path / "h.jsonl", rows)
    d = tmp_path / "d.jsonl"
    d.write_text('{"key": "a", "dig', encoding="utf-8")
    with pytest.raises(ReconciliationUnavailable) as exc:
        reconcile_ledgers(h, d).to_evidence()
    assert "destination_ledger_truncated" in str(exc.value)


# ---------------------------------------------------------------------------
# corruption is visible in status too
# ---------------------------------------------------------------------------
def test_summarize_reports_coherence(tmp_path):
    from auto_ingest.custody.ledger import summarize_ledger

    good = summarize_ledger(write_ledger(tmp_path / "g.jsonl", [
        {"key": "a", "digest": "A", "status": "verified"}]))
    assert (good.coherent, good.malformed, good.truncated) == (True, 0, False)

    bad = tmp_path / "b.jsonl"
    bad.write_text('{"key": "a", "diges', encoding="utf-8")
    summary = summarize_ledger(bad)
    assert summary.coherent is False
    assert summary.truncated is True


def test_a_corrupt_ledger_is_reported_as_a_status_disagreement(tmp_path):
    from auto_ingest.custody.ledger import ledger_disagreements, summarize_ledger

    good = write_ledger(tmp_path / "g.jsonl", [{"key": "a", "status": "verified"}])
    bad = tmp_path / "b.jsonl"
    bad.write_text('{"key": "a", "diges', encoding="utf-8")
    problems = ledger_disagreements(
        {"verified_files": 1, "hashed_files": 1},
        {"destination.jsonl": summarize_ledger(good),
         "hash.jsonl": summarize_ledger(bad)},
    )
    assert any("not coherent" in p for p in problems)


def test_status_surfaces_an_incoherent_ledger(tmp_path, capsys):
    camp = campaign()
    bundle = write_bundle(tmp_path / "b", camp, evidence())
    write_ledger(bundle / "ledgers" / "hash.jsonl", [{"key": "a", "status": "verified"}])
    (bundle / "ledgers" / "destination.jsonl").write_text(
        '{"key": "a", "dig', encoding="utf-8")
    main(["status", "--bundle", str(bundle), "--json"])
    data = json.loads(capsys.readouterr().out)
    assert data["ledgers"]["destination.jsonl"]["coherent"] is False
    assert any("not coherent" in d for d in data["ledger_disagreements"])


# ---------------------------------------------------------------------------
# determinism + boundedness
# ---------------------------------------------------------------------------
def test_reconciliation_is_deterministic(tmp_path):
    h = write_ledger(tmp_path / "h.jsonl", [src(f"k{i}", f"d{i}") for i in range(50)])
    d = write_ledger(tmp_path / "d.jsonl", [dst(f"k{i}", f"d{i}") for i in range(20)])
    first = reconcile_ledgers(h, d).to_dict()
    for _ in range(3):
        assert reconcile_ledgers(h, d).to_dict() == first


def test_samples_are_capped(tmp_path):
    h = write_ledger(tmp_path / "h.jsonl", [src(f"k{i:04d}", f"d{i}") for i in range(500)])
    d = write_ledger(tmp_path / "d.jsonl", [])
    r = reconcile_ledgers(h, d, max_samples=10)
    assert r.source_only == 500
    assert len(r.source_only_samples) == 10


def test_cardsized_ledger_reconciles_in_bulk(tmp_path):
    """A realistic shape: 67,644 hashed, 41,203 verified at the destination."""
    total = 67644
    h = write_ledger(tmp_path / "h.jsonl",
                     [src(f"2026_0412_{i:06d}_F", f"{i:064x}") for i in range(total)])
    d = write_ledger(tmp_path / "d.jsonl",
                     [dst(f"2026_0412_{i:06d}_F", f"{i:064x}") for i in range(41203)])
    r = reconcile_ledgers(h, d, max_samples=5)
    assert (r.verified, r.source_only) == (41203, total - 41203)
    assert len(r.source_only_samples) == 5


# ---------------------------------------------------------------------------
# read-only guarantees
# ---------------------------------------------------------------------------
def test_reconcile_does_not_touch_the_bundle(tmp_path):
    camp = campaign()
    ev = evidence(inv=inventory(10, 100, complete=True, verified=True),
                  hsh=hashing(10, verified_bytes=100, complete=True))
    bundle = write_bundle(tmp_path / "b", camp, ev)
    write_ledger(bundle / "ledgers" / "hash.jsonl", [src("a", "A"), src("b", "B")])
    write_ledger(bundle / "ledgers" / "destination.jsonl", [dst("a", "A")])
    before = {p: p.stat().st_mtime_ns for p in sorted(bundle.rglob("*"))}
    evidence_before = (bundle / "evidence.json").read_text(encoding="utf-8")
    for _ in range(3):
        reconcile_bundle(bundle)
    after = {p: p.stat().st_mtime_ns for p in sorted(bundle.rglob("*"))}
    assert after == before
    assert (bundle / "evidence.json").read_text(encoding="utf-8") == evidence_before


def test_unusable_reconciliation_leaves_existing_evidence_untouched(tmp_path):
    """No hash ledger -> no proposal -> the recorded source_only must survive."""
    result, status = reconcile_preview(CARD_01_BUNDLE)
    assert result.usable is False
    assert result.proposal() is None
    assert status.evidence.reconciliation.source_only == 67644
    assert status.evidence.hashing.verified_files == 67644


# ---------------------------------------------------------------------------
# preview semantics
# ---------------------------------------------------------------------------
def test_preview_reports_the_state_the_proposal_would_produce(tmp_path):
    camp = campaign()
    ev = evidence(
        inv=inventory(10, 1000, complete=True, verified=True),
        hsh=hashing(10, verified_bytes=1000, complete=True),
        cpy=copying(planned_files=10, planned_bytes=1000, completed_files=4,
                    completed_bytes=400, started=True, result_complete=False,
                    interrupted=True),
        dst=destination_evidence(verified_files=4, verified_bytes=400),
        rec=evidence().reconciliation,
        wkr=worker(status="stopped"),
    )
    bundle = write_bundle(tmp_path / "b", camp, ev)
    write_ledger(bundle / "ledgers" / "hash.jsonl",
                 [src(f"k{i}", f"d{i}") for i in range(10)])
    write_ledger(bundle / "ledgers" / "destination.jsonl",
                 [dst(f"k{i}", f"d{i}") for i in range(4)])

    result, status = reconcile_preview(bundle, strict_policy())
    assert result.usable is True
    assert result.verified == 4 and result.source_only == 6
    assert status.derivation.state is CampaignState.RECONCILE_REQUIRED
    assert status.source_release_allowed is False
    copy_action = next(a for a in status.plan.actions if a.operation == "copy_objects")
    assert copy_action.estimated_files == 6  # 10 source objects - 4 verified


def test_a_fully_reconciled_bundle_previews_safe_to_release(tmp_path):
    camp = campaign()
    ev = evidence(inv=inventory(3, 300, complete=True, verified=True),
                  hsh=hashing(3, verified_bytes=300, complete=True),
                  cpy=copying(planned_files=3, planned_bytes=300, completed_files=3,
                              completed_bytes=300, started=True, result_complete=True,
                              ledger_complete=True),
                  dst=destination_evidence(
                      verified_files=3, verified_bytes=300,
                      verification_started=True, verification_complete=True,
                      observed_identity=camp.destination.identity),
                  wkr=worker(status="stopped"))
    bundle = write_bundle(tmp_path / "b", camp, ev)
    write_ledger(bundle / "ledgers" / "hash.jsonl", [src(f"k{i}", f"d{i}") for i in range(3)])
    write_ledger(bundle / "ledgers" / "destination.jsonl",
                 [dst(f"k{i}", f"d{i}") for i in range(3)])
    result, status = reconcile_preview(bundle, strict_policy())
    assert result.complete is True
    assert status.derivation.state is CampaignState.SAFE_TO_RELEASE
    assert status.source_release_allowed is True


def test_reconciliation_alone_cannot_prove_destination_identity(tmp_path):
    """A perfect diff still does not prove *which storage* was written to.

    The ledger says objects matched; it cannot say the destination filesystem was
    the one the campaign named. That stays a separate, separately-recorded fact.
    """
    camp = campaign()
    ev = evidence(inv=inventory(3, 300, complete=True, verified=True),
                  hsh=hashing(3, verified_bytes=300, complete=True),
                  cpy=copying(planned_files=3, planned_bytes=300, completed_files=3,
                              completed_bytes=300, started=True, result_complete=True,
                              ledger_complete=True),
                  wkr=worker(status="stopped"))
    bundle = write_bundle(tmp_path / "b", camp, ev)
    write_ledger(bundle / "ledgers" / "hash.jsonl", [src(f"k{i}", f"d{i}") for i in range(3)])
    write_ledger(bundle / "ledgers" / "destination.jsonl",
                 [dst(f"k{i}", f"d{i}") for i in range(3)])
    result, status = reconcile_preview(bundle, strict_policy())
    assert result.complete is True
    assert status.derivation.state is CampaignState.VERIFIED
    assert "destination_identity_unproven:identity_not_proven" in status.derivation.blockers


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_reconcile_is_read_only(tmp_path, capsys):
    camp = campaign()
    bundle = write_bundle(tmp_path / "b", camp, evidence())
    write_ledger(bundle / "ledgers" / "hash.jsonl", [src("a", "A"), src("b", "B")])
    write_ledger(bundle / "ledgers" / "destination.jsonl", [dst("a", "A")])
    before = (bundle / "evidence.json").read_text(encoding="utf-8")
    code = main(["reconcile", "--bundle", str(bundle), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == EXIT_OK
    assert payload["applied"] is False
    assert payload["requires_operator_authorization"] is True
    assert payload["reconciliation"]["verified"] == 1
    assert payload["reconciliation"]["source_only"] == 1
    assert payload["proposal"]["reconciliation"]["source_only"] == 1
    assert (bundle / "evidence.json").read_text(encoding="utf-8") == before


def test_cli_reconcile_text_output_flags_a_missing_ledger(capsys):
    main(["reconcile", "--bundle", str(CARD_01_BUNDLE)])
    out = capsys.readouterr().out
    assert "ledgers_readable           false" in out
    assert "no proposal is offered - hash_ledger_absent" in out


def test_cli_reconcile_require_release_gates(tmp_path, capsys):
    camp = campaign()
    bundle = write_bundle(tmp_path / "b", camp, evidence())
    write_ledger(bundle / "ledgers" / "hash.jsonl", [src("a", "A")])
    write_ledger(bundle / "ledgers" / "destination.jsonl", [])
    code = main(["reconcile", "--bundle", str(bundle), "--require-release"])
    capsys.readouterr()
    assert code == EXIT_GATE_CLOSED


def test_cli_reconcile_then_import_then_status(tmp_path, capsys):
    """The full reconciliation loop, still without any execution."""
    camp = campaign()
    ev = evidence(inv=inventory(3, 300, complete=True, verified=True),
                  hsh=hashing(3, verified_bytes=300, complete=True),
                  cpy=copying(planned_files=3, planned_bytes=300, started=True,
                              result_complete=False, interrupted=True),
                  wkr=worker(status="stopped"))
    bundle = write_bundle(tmp_path / "b", camp, ev)
    write_ledger(bundle / "ledgers" / "hash.jsonl", [src(f"k{i}", f"d{i}") for i in range(3)])
    write_ledger(bundle / "ledgers" / "destination.jsonl",
                 [dst(f"k{i}", f"d{i}") for i in range(3)])

    main(["reconcile", "--bundle", str(bundle), "--json"])
    proposal = json.loads(capsys.readouterr().out)["proposal"]
    assert proposal["destination"]["verified_files"] == 3

    proposal_path = tmp_path / "proposal.json"
    proposal_path.write_text(json.dumps(proposal), encoding="utf-8")
    code = main(["import", "--bundle", str(bundle), "--evidence", str(proposal_path),
                 "--apply", "--json"])
    capsys.readouterr()
    assert code == EXIT_OK

    status = load_status(bundle)
    # hash/copy evidence is still incomplete, so this cannot release yet
    assert status.evidence.destination.verified_files == 3
    assert status.source_release_allowed is False
    assert status.derivation.state is CampaignState.RECONCILE_REQUIRED


def test_card_01_reconcile_does_not_offer_a_proposal(capsys):
    main(["reconcile", "--bundle", str(CARD_01_BUNDLE), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["usable"] is False
    assert payload["proposal"] is None
    assert payload["source_release_allowed_if_imported"] is False
    assert payload["current_state"] == "RECONCILE_REQUIRED"


# ---------------------------------------------------------------------------
# import merges; it never replaces
# ---------------------------------------------------------------------------
def test_import_merges_a_narrow_document_instead_of_replacing(tmp_path, capsys):
    """A reconciliation-only document must not erase the hash evidence."""
    camp = campaign()
    ev = evidence(inv=inventory(10, 1000, complete=True, verified=True),
                  hsh=hashing(10, verified_bytes=1000, complete=True),
                  cpy=copying(planned_files=10, planned_bytes=1000, started=True,
                              result_complete=False, interrupted=True),
                  wkr=worker(status="stopped"))
    bundle = write_bundle(tmp_path / "b", camp, ev)
    doc = tmp_path / "narrow.json"
    doc.write_text(json.dumps({"reconciliation": {"source_only": 10}}), encoding="utf-8")

    code = main(["import", "--bundle", str(bundle), "--evidence", str(doc),
                 "--apply", "--json"])
    result = json.loads(capsys.readouterr().out)
    assert code == EXIT_OK
    assert result["merged_with_existing"] is True

    status = load_status(bundle)
    assert status.evidence.hashing.verified_files == 10
    assert status.evidence.inventory.discovered_files == 10
    assert status.evidence.copy.planned.files == 10
    assert status.evidence.reconciliation.source_only == 10


def test_import_preserves_recorded_subfields_the_document_does_not_mention(tmp_path):
    from auto_ingest.custody.store import merge_evidence_documents

    existing = {"destination": {"verified_files": 3, "failures": 1,
                                "observed_identity": {"filesystem_uuid": "DEST-1"},
                                "error_summary": ["one bad frame"]}}
    merged = merge_evidence_documents(existing, {"destination": {"verified_files": 5}})
    assert merged["destination"]["verified_files"] == 5
    assert merged["destination"]["failures"] == 1
    assert merged["destination"]["observed_identity"] == {"filesystem_uuid": "DEST-1"}
    assert merged["destination"]["error_summary"] == ["one bad frame"]


def test_an_explicit_zero_in_the_document_wins(tmp_path):
    """A declared 0 is a statement, not an absence."""
    from auto_ingest.custody.store import merge_evidence_documents

    existing = {"destination": {"verified_files": 42}}
    merged = merge_evidence_documents(existing, {"destination": {"verified_files": 0}})
    assert merged["destination"]["verified_files"] == 0


def test_declared_state_keys_are_still_dropped_by_the_merge():
    from auto_ingest.custody.store import merge_evidence_documents

    merged = merge_evidence_documents({}, {"state": "SAFE_TO_RELEASE",
                                           "inventory": {"discovered_files": 1}})
    assert "state" not in merged
    assert merged["inventory"] == {"discovered_files": 1}


def test_import_reports_what_it_preserved(tmp_path, capsys):
    camp = campaign()
    ev = evidence(inv=inventory(10, 1000, complete=True, verified=True),
                  hsh=hashing(10, verified_bytes=1000, complete=True))
    bundle = write_bundle(tmp_path / "b", camp, ev)
    doc = tmp_path / "narrow.json"
    doc.write_text(json.dumps({"worker": {"status": "stopped"}}), encoding="utf-8")
    main(["import", "--bundle", str(bundle), "--evidence", str(doc), "--apply", "--json"])
    result = json.loads(capsys.readouterr().out)
    assert "hash.verified_files" in result["preserved_fields"]
    assert "inventory.discovered_files" in result["preserved_fields"]


def test_preview_agrees_with_import(tmp_path):
    """The preview must be exactly what `import --apply` would produce."""
    camp = campaign()
    ev = evidence(inv=inventory(4, 400, complete=True, verified=True),
                  hsh=hashing(4, verified_bytes=400, complete=True),
                  cpy=copying(planned_files=4, planned_bytes=400, started=True,
                              result_complete=False, interrupted=True),
                  wkr=worker(status="stopped"))
    bundle = write_bundle(tmp_path / "b", camp, ev)
    write_ledger(bundle / "ledgers" / "hash.jsonl", [src(f"k{i}", f"d{i}") for i in range(4)])
    write_ledger(bundle / "ledgers" / "destination.jsonl",
                 [dst(f"k{i}", f"d{i}") for i in range(2)])

    result, preview = reconcile_preview(bundle, strict_policy())
    assert result.usable is True
    doc = tmp_path / "proposal.json"
    doc.write_text(json.dumps(result.proposal()), encoding="utf-8")
    main(["import", "--bundle", str(bundle), "--evidence", str(doc), "--apply", "--json"])
    after = load_status(bundle, strict_policy())
    assert after.to_dict() == preview.to_dict()


# ---------------------------------------------------------------------------
# Files at the destination that this campaign never put there
# ---------------------------------------------------------------------------
# Reconciliation is a ledger-to-ledger set difference, so it cannot see a file the
# campaign never recorded. That is invisible in testing and obvious in production:
# the real destination root holds 332 files / 47.7 GB from a *different*, already
# archived campaign, and reconcile reported destination_only=0 - which reads as
# "the destination holds exactly this campaign and nothing else".

def test_a_foreign_file_at_the_destination_is_invisible_to_ledger_reconciliation(tmp_path):
    """The gap, stated as a test.

    Reconciliation is a set difference between the campaign's own two ledgers, so a
    file the campaign never recorded cannot appear in it. In production the live
    destination root holds 332 files / 47.7 GB from a *different*, already
    archived campaign, and reconcile reported destination_only=0 - which reads as
    "the destination holds exactly this campaign and nothing else".

    Both halves are asserted: the ledger difference really does miss it, and the
    filesystem scan really does find it.
    """
    dest = tmp_path / "dest"
    (dest / "someone-elses-campaign").mkdir(parents=True)
    (dest / "someone-elses-campaign" / "clip.MP4").write_bytes(b"not ours")

    h = write_ledger(tmp_path / "h.jsonl", [src("ours.MP4", "A1")])
    d = write_ledger(tmp_path / "d.jsonl", [dst("ours.MP4", "A1")])
    result = reconcile_ledgers(h, d)

    assert result.destination_only == 0, "the ledger difference cannot see it"
    foreign, samples = scan_foreign_objects(dest, d)
    assert foreign == 1, "the filesystem walk does"
    assert samples == ("someone-elses-campaign/clip.MP4",)


def test_objects_this_campaign_verified_are_not_foreign(tmp_path):
    dest = tmp_path / "dest"
    (dest / "2026" / "08" / "29").mkdir(parents=True)
    landed = dest / "2026" / "08" / "29" / "a.MP4"
    landed.write_bytes(b"ours")
    # The destination ledger records absolute destination paths, which is what the
    # scan matches on - staging may put an object somewhere other than its key.
    d = write_ledger(tmp_path / "d.jsonl", [dst("DCIM/a.MP4", "A1", path=str(landed))])
    assert scan_foreign_objects(dest, d) == (0, ())


def test_a_declared_sibling_campaign_is_not_foreign(tmp_path):
    """A shared destination root has to be representable, or the strict check is
    permanently red and therefore permanently ignored."""
    dest = tmp_path / "dest"
    (dest / "other-campaign").mkdir(parents=True)
    (dest / "other-campaign" / "x.MP4").write_bytes(b"theirs")
    (dest / "stray.MP4").write_bytes(b"nobodys")
    d = write_ledger(tmp_path / "d.jsonl", [dst("a.MP4", "A1", path=str(dest / "a.MP4"))])

    undeclared, samples = scan_foreign_objects(dest, d)
    assert undeclared == 2

    declared, samples = scan_foreign_objects(dest, d, tolerate=("other-campaign",))
    assert declared == 1
    assert samples == ("stray.MP4",)


def test_reconcile_bundle_carries_the_foreign_count_when_given_a_destination(tmp_path):
    from auto_ingest.custody.ledger import ledger_dir

    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "stray.MP4").write_bytes(b"x")
    bundle = tmp_path / "b"
    write_ledger(ledger_dir(bundle) / "hash.jsonl", [src("a.MP4", "A1")])
    write_ledger(ledger_dir(bundle) / "destination.jsonl", [dst("a.MP4", "A1")])

    without = reconcile_bundle(bundle)
    assert without.foreign_objects == 0, "no destination given, no walk"

    with_root = reconcile_bundle(bundle, destination_root=dest)
    assert with_root.foreign_objects == 1
    assert with_root.to_dict()["foreign_objects"] == 1


def test_the_foreign_count_is_complete_not_truncated(tmp_path):
    """A count that stops early and is reported as the total is worse than no
    count: it reads as "a couple of stray files" when it is a whole second
    campaign.

    The first version of the scan did exactly that - reported 25 for a tree
    holding 332 - by stopping the walk once the sample was full. Only the sample
    may be bounded.
    """
    dest = tmp_path / "dest"
    (dest / "other").mkdir(parents=True)
    for i in range(400):
        (dest / "other" / f"f{i:03d}.MP4").write_bytes(b"x")
    d = write_ledger(tmp_path / "d.jsonl", [dst("a.MP4", "A1")])

    count, samples = scan_foreign_objects(dest, d, max_samples=3)
    assert count == 400, "the count must be the whole, not the sample"
    assert len(samples) == 3
