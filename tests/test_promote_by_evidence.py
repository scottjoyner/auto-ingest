"""
Tests for the lightweight, evidence-based GlobalSpeaker promotion pass.

Exercises ``auto_ingest.diarize.link_global_speakers.promote_by_evidence``
against a fake neo4j driver. The key invariant: the function pages over
GlobalSpeaker by id (bounded ``LIMIT`` per query) and NEVER issues a single
query that aggregates the whole graph. We assert the paging query is bounded
and that confirmed writes respect count.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _lgs():
    try:
        from auto_ingest.diarize import link_global_speakers as lgs  # noqa: E402
    except Exception:
        pytest.skip("link_global_speakers requires torch (not in this env)")
    return lgs


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def data(self):
        return self._rows

    def single(self):
        return None


class _FakeSession:
    def __init__(self, ids):
        self._ids = ids
        self.closed = False
        self.runs = []  # record (query, params)
        self._exhausted = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False

    def run(self, query, **params):
        self.runs.append((query, params))
        if "weight_sum" in query:
            if self._exhausted:
                return _FakeResult([])
            self._exhausted = True
            return _FakeResult([{"gid": g} for g in self._ids])
        return _FakeResult([])


class _FakeDriver:
    def __init__(self, ids):
        self._ids = ids
        self.closed = False
        self.session_obj = None

    def session(self, database=None):
        self.session_obj = _FakeSession(self._ids)
        return self.session_obj

    def close(self):
        self.closed = True


def test_promote_by_evidence_is_importable():
    assert callable(_lgs().promote_by_evidence)


def test_promote_by_evidence_promotes_and_pages():
    ids = [f"gs_{i:04d}" for i in range(3)]
    drv = _FakeDriver(ids)
    n = _lgs().promote_by_evidence(drv, min_weight=30.0, dry_run=False)
    assert n == 3
    sess = drv.session_obj
    selects = [q for q, _ in sess.runs if "weight_sum" in q]
    writes = [q for q, _ in sess.runs if "SET g.status = 'confirmed'" in q]
    # Paging: one results page (the 3 ids) + one empty terminating page.
    # Critically, NO query aggregates the whole graph in a single transaction.
    assert len(selects) == 2
    assert len(writes) == 1
    # Every selection query is bounded and paginates by id (index-backed).
    assert all("LIMIT $batch" in q for q in selects)
    assert all("g.id > coalesce($after" in q for q in selects)


def test_promote_by_evidence_dry_run_writes_nothing():
    ids = ["gs_0001", "gs_0002"]
    drv = _FakeDriver(ids)
    n = _lgs().promote_by_evidence(drv, min_weight=30.0, dry_run=True)
    assert n == 2
    sess = drv.session_obj
    writes = [q for q, _ in sess.runs if "SET g.status = 'confirmed'" in q]
    assert writes == []
