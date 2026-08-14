from __future__ import annotations

import pytest

from auto_ingest import invariants


class Result:
    def __init__(self, rows):
        self._rows = rows

    def data(self):
        return self._rows


class Session:
    def __init__(self, rows_by_name):
        self.rows_by_name = rows_by_name

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def run(self, query, **kwargs):
        for invariant in invariants.INVARIANTS:
            if invariant.query == query:
                return Result(self.rows_by_name.get(invariant.name, []))
        raise AssertionError("unexpected invariant query")


class Driver:
    def __init__(self, rows_by_name):
        self.rows_by_name = rows_by_name

    def session(self):
        return Session(self.rows_by_name)


def test_audit_passes_when_all_invariants_are_clean():
    report = invariants.audit(Driver({}))
    assert report["ok"] is True
    assert all(item["ok"] for item in report["invariants"].values())


def test_audit_surfaces_named_violation_and_preserves_evidence():
    report = invariants.audit(
        Driver({"active_job_requires_lease": [{"key": "bad", "state": "RUNNING"}]})
    )
    assert report["ok"] is False
    item = report["invariants"]["active_job_requires_lease"]
    assert item["ok"] is False
    assert item["violations"] == [{"key": "bad", "state": "RUNNING"}]


def test_audit_rejects_unbounded_or_invalid_limit():
    with pytest.raises(ValueError, match="limit must be positive"):
        invariants.audit(Driver({}), limit=0)
