from __future__ import annotations

import json
from pathlib import Path

import pytest

from auto_ingest.regression import RegressionFixtureError, run_case, run_fixture


def test_core_regression_fixture_passes():
    results = run_fixture(Path("tests/regressions/core_contracts.json"))
    assert results
    assert all(result.passed for result in results)


def test_unknown_target_is_rejected_without_execution():
    result = run_case(
        {
            "name": "no arbitrary imports",
            "target": "os.system",
            "input": {"command": "anything"},
            "expect": {"contains": {}},
        }
    )
    assert result.passed is False
    assert "unknown target" in (result.error or "")


def test_fixture_version_and_case_shape_are_strict(tmp_path: Path):
    bad_version = tmp_path / "bad-version.json"
    bad_version.write_text(json.dumps({"version": 999, "cases": [{}]}), encoding="utf-8")
    with pytest.raises(RegressionFixtureError):
        run_fixture(bad_version)

    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"version": 1, "cases": []}), encoding="utf-8")
    with pytest.raises(RegressionFixtureError):
        run_fixture(empty)


def test_expected_exception_and_mismatch_paths():
    expected_error = run_case(
        {
            "name": "bad queue field",
            "target": "queue.job",
            "input": {
                "version": 1,
                "job_id": "x",
                "profile": "sync",
                "created_at": 1,
                "metadata": {},
                "command": "not allowed",
            },
            "raises": "QueueJobError",
        }
    )
    assert expected_error.passed is True

    mismatch = run_case(
        {
            "name": "mismatch",
            "target": "queue.job",
            "input": {
                "version": 1,
                "job_id": "x",
                "profile": "sync",
                "created_at": 1,
                "metadata": {},
            },
            "expect": {"contains": {"profile": "full"}},
        }
    )
    assert mismatch.passed is False
    assert mismatch.error == "expectation mismatch"
