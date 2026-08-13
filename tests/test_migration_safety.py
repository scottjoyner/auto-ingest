from __future__ import annotations

import pytest

from auto_ingest.ops.migration_safety import (
    SafetyViolation,
    batches_required,
    preflight_summary,
    validate_batch_size,
)


def test_validate_batch_size_enforces_bounds():
    assert validate_batch_size(5000, max_batch_size=100000) == 5000
    with pytest.raises(SafetyViolation):
        validate_batch_size(0, max_batch_size=100000)
    with pytest.raises(SafetyViolation):
        validate_batch_size(100001, max_batch_size=100000)
    with pytest.raises(SafetyViolation):
        validate_batch_size(True, max_batch_size=100000)


def test_batches_required_is_deterministic():
    assert batches_required(0, 5000) == 0
    assert batches_required(1, 5000) == 1
    assert batches_required(5001, 5000) == 2


def test_preflight_summary_fails_closed_on_inconsistent_counts():
    with pytest.raises(SafetyViolation):
        preflight_summary(
            operation="bad",
            total_candidates=1,
            eligible_candidates=2,
            batch_size=100,
            max_batch_size=1000,
            dry_run=True,
        )


def test_preflight_summary_reports_bounded_plan():
    summary = preflight_summary(
        operation="phonelog_spatial",
        total_candidates=12001,
        eligible_candidates=10001,
        batch_size=5000,
        max_batch_size=100000,
        dry_run=True,
    )
    assert summary["batches_required"] == 3
    assert summary["ineligible_candidates"] == 2000
    assert summary["dry_run"] is True
