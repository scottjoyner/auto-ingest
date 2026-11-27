"""Pure derived-predicate contract on custody evidence (r35).

Hermetic: every subject is a frozen dataclass built from literals plus a pure
``CustodyPolicy``. No clock, filesystem, DB, network, subprocess or service is
read, and no production state is written. These are the boolean/int projections
the state machine and release gate consume; misclassifying one of them silently
changes a custody verdict, so the exact contract is pinned here.

Fresh surface (verified before writing): none of ``has_copy_ambiguity``,
``in_flight``, ``Counts.__add__``, ``ReconciliationEvidence.clean``,
``WorkerEvidence.active``/``idle``, ``SourceReleaseEvidence.attempted``/``clean``/
``observed`` had any literal test reference on ``origin.main``, any test
reference in any branch, or any mention in the review-report corpus.
"""

from __future__ import annotations

from auto_ingest.custody.evidence import (
    CampaignEvidence,
    CopyEvidence,
    Counts,
    DestinationEvidence,
    HashEvidence,
    InventoryEvidence,
    ReconciliationEvidence,
    SourceReleaseEvidence,
    WorkerEvidence,
)
from auto_ingest.custody.policy import CustodyPolicy

# ---------------------------------------------------------------------------
# Counts: additive (files, bytes) pair
# ---------------------------------------------------------------------------


def test_counts_add_is_componentwise():
    assert Counts(2, 100) + Counts(3, 50) == Counts(5, 150)


def test_counts_add_is_identity_with_zero():
    c = Counts(4, 9)
    assert Counts() + c == c
    assert c + Counts() == c


def test_counts_add_does_not_mutate_operands():
    a, b = Counts(2, 100), Counts(3, 50)
    _ = a + b
    assert a == Counts(2, 100)
    assert b == Counts(3, 50)


def test_counts_from_dict_and_to_dict_round_trip():
    c = Counts.from_dict({"files": 4, "bytes": 9})
    assert c == Counts(4, 9)
    assert c.to_dict() == {"bytes": 9, "files": 4}


def test_counts_from_dict_non_mapping_puts_value_in_files_only():
    assert Counts.from_dict(7) == Counts(7, 0)


# ---------------------------------------------------------------------------
# CopyEvidence.in_flight: remaining planned files, floored at zero
# ---------------------------------------------------------------------------


def test_in_flight_is_remaining_planned_files():
    e = CopyEvidence(planned=Counts(10, 1000), completed=Counts(4, 999))
    assert e.in_flight == 6


def test_in_flight_is_zero_when_complete():
    e = CopyEvidence(planned=Counts(10, 5), completed=Counts(10, 5))
    assert e.in_flight == 0


def test_in_flight_clamps_negative_to_zero():
    e = CopyEvidence(planned=Counts(3, 9), completed=Counts(9, 3))
    assert e.in_flight == 0


def test_in_flight_counts_files_not_bytes():
    # bytes are wildly different but files are equal -> nothing in flight.
    e = CopyEvidence(planned=Counts(4, 1), completed=Counts(4, 1_000_000))
    assert e.in_flight == 0


def test_in_flight_independent_of_status_flags():
    base = dict(planned=Counts(5, 1), completed=Counts(2, 1))
    assert CopyEvidence(**base).in_flight == 3
    assert CopyEvidence(**base, started=True).in_flight == 3
    assert CopyEvidence(**base, interrupted=True).in_flight == 3
    assert CopyEvidence(**base, result_complete=True).in_flight == 3


# ---------------------------------------------------------------------------
# ReconciliationEvidence.clean: only this campaign's ledger differences matter
# ---------------------------------------------------------------------------


def test_reconciliation_clean_default_true():
    assert ReconciliationEvidence().clean is True


def test_reconciliation_clean_false_for_source_only():
    assert ReconciliationEvidence(source_only=1).clean is False


def test_reconciliation_clean_false_for_destination_only():
    assert ReconciliationEvidence(destination_only=1).clean is False


def test_reconciliation_clean_false_for_mismatched():
    assert ReconciliationEvidence(mismatched=1).clean is False


def test_reconciliation_clean_ignores_foreign_objects():
    # Foreign objects are unrecorded by this campaign; they must not dirty the
    # ledger-to-ledger reconciliation. This is the exact distinction that keeps
    # a shared destination root from being permanently unclean.
    assert ReconciliationEvidence(foreign_objects=500).clean is True


def test_reconciliation_clean_ignores_timestamp():
    assert ReconciliationEvidence(reconciled_at="2026-11-27T00:00:00Z").clean is True


# ---------------------------------------------------------------------------
# WorkerEvidence.active / idle: exact, disjoint status vocabularies
# ---------------------------------------------------------------------------

_ACTIVE = ("running", "copying", "hashing", "verifying", "active")
_IDLE = ("stopped", "absent", "exited", "dead", "inactive", "not_started")


def test_worker_active_statuses():
    for status in _ACTIVE:
        e = WorkerEvidence(status=status)
        assert e.active is True, status
        assert e.idle is False, status


def test_worker_idle_statuses():
    for status in _IDLE:
        e = WorkerEvidence(status=status)
        assert e.idle is True, status
        assert e.active is False, status


def test_worker_unknown_and_arbitrary_status_is_neither():
    for status in ("unknown", "", "paused", "queued"):
        e = WorkerEvidence(status=status)
        assert e.active is False, status
        assert e.idle is False, status


def test_worker_active_and_idle_are_disjoint():
    assert not (set(_ACTIVE) & set(_IDLE))


def test_worker_from_dict_normalizes_status_before_classifying():
    e = WorkerEvidence.from_dict({"status": "  RUNNING  "})
    assert e.status == "running"
    assert e.active is True
    assert WorkerEvidence.from_dict({"status": " STOPPED "}).idle is True


def test_worker_from_dict_empty_status_defaults_to_unknown():
    assert WorkerEvidence.from_dict({"status": "   "}).status == "unknown"


# ---------------------------------------------------------------------------
# SourceReleaseEvidence: source-side facts, never destination custody
# ---------------------------------------------------------------------------


def test_source_release_attempted_sums_all_decisions():
    e = SourceReleaseEvidence(released=Counts(5, 100), absent=2, failed=1, refused=3)
    assert e.attempted == 11


def test_source_release_clean_ignores_absent_and_released():
    assert SourceReleaseEvidence(released=Counts(9, 1), absent=4).clean is True


def test_source_release_clean_false_on_failed():
    assert SourceReleaseEvidence(failed=1).clean is False


def test_source_release_clean_false_on_refused():
    assert SourceReleaseEvidence(refused=1).clean is False


def test_source_release_observed_defaults_false():
    assert SourceReleaseEvidence().observed is False


def test_source_release_observed_true_when_started_alone():
    assert SourceReleaseEvidence(started=True).observed is True


def test_source_release_observed_true_when_audit_records_alone():
    assert SourceReleaseEvidence(audit_records=1).observed is True


def test_source_release_observed_true_when_attempted_alone():
    assert SourceReleaseEvidence(absent=1).observed is True
    assert SourceReleaseEvidence(released=Counts(1, 1)).observed is True


def test_source_release_started_inferred_from_release_facts():
    assert SourceReleaseEvidence.from_dict({"released": {"files": 3}}, 20).started is True
    assert SourceReleaseEvidence.from_dict({"complete": True}, 20).started is True


# ---------------------------------------------------------------------------
# CampaignEvidence projections: keep source and destination facts separate
# ---------------------------------------------------------------------------


def _evidence(**overrides) -> CampaignEvidence:
    base = dict(
        inventory=InventoryEvidence(discovered_files=10, discovered_bytes=1000),
        destination=DestinationEvidence(verified_files=3, verified_bytes=300),
        source_release=SourceReleaseEvidence(released=Counts(7, 700)),
    )
    base.update(overrides)
    return CampaignEvidence(**base)


def test_projections_read_their_own_blocks():
    e = _evidence()
    assert e.inventory_files == 10
    assert e.verified_at_destination == 3
    assert e.source_released == 7


def test_source_released_is_not_destination_verified():
    e = _evidence(source_release=SourceReleaseEvidence(released=Counts(7, 700)))
    assert e.source_released == 7
    assert e.verified_at_destination == 3


def test_hash_coverage_counts_every_exemption():
    e = _evidence(hashing=HashEvidence(verified_files=2, exemptions=("a", "b")))
    assert e.hash_coverage == 4


def test_hash_coverage_under_counts_only_declared_exemptions():
    e = _evidence(hashing=HashEvidence(verified_files=2, exemptions=("a", "b")))
    policy = CustodyPolicy(allow_hash_exemptions=True, declared_hash_exemptions=("a",))
    assert e.hash_coverage == 4
    assert e.hash_coverage_under(policy) == 3


def test_hash_coverage_under_zero_when_exemptions_not_allowed():
    e = _evidence(hashing=HashEvidence(verified_files=2, exemptions=("a", "b")))
    policy = CustodyPolicy(allow_hash_exemptions=False, declared_hash_exemptions=("a", "b"))
    assert e.hash_coverage_under(policy) == 2


# ---------------------------------------------------------------------------
# CampaignEvidence.has_copy_ambiguity: the CARD-01 reconciliation shape
# ---------------------------------------------------------------------------


def test_ambiguity_false_when_copy_never_started():
    e = _evidence(copy=CopyEvidence(started=False, interrupted=True))
    assert e.has_copy_ambiguity is False


def test_ambiguity_false_when_copy_and_ledger_complete():
    e = _evidence(
        copy=CopyEvidence(
            started=True,
            result_complete=True,
            ledger_complete=True,
            interrupted=True,
        ),
        worker=WorkerEvidence(status="running"),
    )
    # The early "both complete" return wins even over interrupted=True.
    assert e.has_copy_ambiguity is False


def test_ambiguity_true_when_interrupted_before_completion():
    e = _evidence(
        copy=CopyEvidence(started=True, interrupted=True), worker=WorkerEvidence(status="running")
    )
    assert e.has_copy_ambiguity is True


def test_ambiguity_true_when_worker_idle_before_completion():
    e = _evidence(
        copy=CopyEvidence(started=True, interrupted=False), worker=WorkerEvidence(status="stopped")
    )
    assert e.has_copy_ambiguity is True


def test_ambiguity_false_when_active_worker_and_incomplete_but_not_interrupted():
    e = _evidence(
        copy=CopyEvidence(started=True, interrupted=False), worker=WorkerEvidence(status="copying")
    )
    assert e.has_copy_ambiguity is False


def test_ambiguity_true_on_incomplete_ledger_with_idle_worker():
    e = _evidence(
        copy=CopyEvidence(
            started=True,
            result_complete=True,
            ledger_complete=False,
        ),
        worker=WorkerEvidence(status="exited"),
    )
    assert e.has_copy_ambiguity is True
