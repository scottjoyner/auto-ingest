"""Tests for the cleaned-intermediate writer + dynamic-tx importer (G5).

No live Neo4j: a fake driver captures every Cypher ``run`` so we can assert the
importer (a) uses ``UNWIND $rows``, (b) loops in bounded batches of ``batch``,
and (c) is resumable (MERGE, not CREATE). The writer is tested with a real temp
dir for the json + csv round-trip and discovery.
"""
from __future__ import annotations

import os
import tempfile

from auto_ingest import ingest_import, ingest_write

# ---------------------------------------------------------------------------
# Fake neo4j driver/session: records every run() call + Cypher
# ---------------------------------------------------------------------------

class _Call:
    __slots__ = ("query", "params")

    def __init__(self, query, params):
        self.query = query
        self.params = params


class _FakeSession:
    def __init__(self, log):
        self._log = log

    def __enter__(self):
        return self

    def __exit__(self, *e):
        return False

    def run(self, query, **params):
        self._log.append(_Call(query, dict(params)))
        return _FakeResult()


class _FakeResult:
    def data(self):
        return []

    def single(self):
        return None


class _FakeDriver:
    def __init__(self):
        self.calls: list[_Call] = []

    def session(self):
        return _FakeSession(self.calls)


# ---------------------------------------------------------------------------
# ingest_write round-trips
# ---------------------------------------------------------------------------

def test_write_read_json_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        recs = [{"id": "a1", "mph": 55, "lat": 35.1, "lon": -80.1},
                {"id": "a2", "mph": 60, "lat": 35.2, "lon": -80.2}]
        p = ingest_write.write_intermediate(d, "key1", "metadata", recs, fmt="json")
        assert p.endswith("key1.metadata.json")
        assert os.path.exists(p)
        got = ingest_write.read_intermediate(d, "key1", "metadata", fmt="json")
        assert got == recs


def test_write_read_csv_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        recs = [{"id": "a1", "mph": "55"}, {"id": "a2", "mph": "60"}]
        p = ingest_write.write_intermediate(d, "key1", "metadata", recs, fmt="csv")
        assert p.endswith(".csv")
        got = ingest_write.read_intermediate(d, "key1", "metadata", fmt="csv")
        assert got == recs


def test_write_overwrites_idempotent():
    with tempfile.TemporaryDirectory() as d:
        ingest_write.write_intermediate(d, "k", "s", [{"id": "1"}], fmt="json")
        ingest_write.write_intermediate(d, "k", "s", [{"id": "1"}, {"id": "2"}], fmt="json")
        got = ingest_write.read_intermediate(d, "k", "s", fmt="json")
        assert [r["id"] for r in got] == ["1", "2"]


def test_key_with_slashes_sanitized():
    with tempfile.TemporaryDirectory() as d:
        ingest_write.write_intermediate(d, "dashcam/2026-07-14/CLT", "metadata",
                                         [{"id": "x"}], fmt="json")
        # Discovery must reverse the sanitization back to the real key.
        found = ingest_write.list_intermediates(d, key="dashcam/2026-07-14/CLT")
        assert len(found) == 1
        assert found[0]["key"] == "dashcam/2026-07-14/CLT"
        assert found[0]["stage"] == "metadata"


def test_list_intermediates_filter_by_stage():
    with tempfile.TemporaryDirectory() as d:
        ingest_write.write_intermediate(d, "k", "metadata", [{"id": "1"}], fmt="json")
        ingest_write.write_intermediate(d, "k", "segments", [{"id": "2"}], fmt="json")
        only_meta = ingest_write.list_intermediates(d, stage="metadata")
        assert [m["stage"] for m in only_meta] == ["metadata"]
        all_ = ingest_write.list_intermediates(d)
        assert len(all_) == 2


def test_list_intermediates_missing_dir():
    assert ingest_write.list_intermediates("/no/such/dir") == []


# ---------------------------------------------------------------------------
# ingest_import
# ---------------------------------------------------------------------------

def test_import_uses_unwind_and_merges():
    with tempfile.TemporaryDirectory() as d:
        p = ingest_write.write_intermediate(d, "k", "metadata",
                                             [{"id": "a", "mph": 5}], fmt="json")
        drv = _FakeDriver()
        n = ingest_import.import_intermediate(drv, p, batch=1000,
                                               label="LocationSample", key_field="id")
        assert n == 1
        assert len(drv.calls) == 1
        q = " ".join(drv.calls[0].query.split())
        assert "UNWIND $rows AS r" in q
        assert "MERGE (n:LocationSample {id: r.key})" in q
        assert "SET n += r.props" in q
        # props payload should be the non-key fields only
        assert drv.calls[0].params["rows"][0] == {"key": "a", "props": {"mph": 5}}


def test_import_batches_by_size():
    # 2500 rows with batch=1000 => 3 UNWIND calls (1000 + 1000 + 500).
    with tempfile.TemporaryDirectory() as d:
        recs = [{"id": f"r{i}", "v": i} for i in range(2500)]
        p = ingest_write.write_intermediate(d, "k", "segments", recs, fmt="json")
        drv = _FakeDriver()
        n = ingest_import.import_intermediate(drv, p, batch=1000,
                                               label="Segment", key_field="id")
        assert n == 2500
        unwind_calls = [c for c in drv.calls if "UNWIND $rows AS r" in " ".join(c.query.split())]
        assert len(unwind_calls) >= 3
        # Each batch bounded by `batch`.
        assert all(len(c.params["rows"]) <= 1000 for c in unwind_calls)


def test_import_resumable_merge_not_create():
    with tempfile.TemporaryDirectory() as d:
        p = ingest_write.write_intermediate(d, "k", "metadata",
                                             [{"id": "a", "mph": 5}], fmt="json")
        drv = _FakeDriver()
        ingest_import.import_intermediate(drv, p, label="LocationSample", key_field="id")
        # Re-run: still MERGE (no CREATE), still only one call, no duplication.
        ingest_import.import_intermediate(drv, p, label="LocationSample", key_field="id")
        q = " ".join(drv.calls[0].query.split())
        assert "CREATE" not in q
        assert len(drv.calls) == 2


def test_import_links_to_parent():
    with tempfile.TemporaryDirectory() as d:
        recs = [{"id": "s1", "parent": "t1"}, {"id": "s2", "parent": "t1"}]
        p = ingest_write.write_intermediate(d, "k", "metadata", recs, fmt="json")
        drv = _FakeDriver()
        n = ingest_import.import_intermediate(
            drv, p, batch=1000, label="LocationSample", key_field="id",
            parent_label="Transcription", parent_field="id", edge_label="HAS_LOCATION")
        assert n == 2
        q = " ".join(drv.calls[0].query.split())
        assert "MATCH (p:Transcription {id: r.parent})" in q
        assert "MERGE (p)-[:HAS_LOCATION]->(n)" in q


def test_import_all_multiple_stages():
    with tempfile.TemporaryDirectory() as d:
        ingest_write.write_intermediate(d, "k", "transcription",
                                         [{"id": "t1", "text": "hi"}], fmt="json")
        ingest_write.write_intermediate(d, "k", "metadata",
                                         [{"id": "m1", "parent": "t1"}], fmt="json")
        drv = _FakeDriver()
        summary = ingest_import.import_all(d, drv, key="k", batch=500)
        # transcription imported before metadata (canonical order)
        keys = list(summary.keys())
        assert keys[0].endswith("/transcription")
        assert keys[1].endswith("/metadata")
        # metadata call links to parent
        meta_call = [c for c in drv.calls if "HAS_LOCATION" in " ".join(c.query.split())]
        assert len(meta_call) == 1


def test_import_csv_file():
    with tempfile.TemporaryDirectory() as d:
        p = ingest_write.write_intermediate(d, "k", "segments",
                                             [{"id": "a", "v": "1"}], fmt="csv")
        drv = _FakeDriver()
        n = ingest_import.import_intermediate(drv, p, label="Segment", key_field="id")
        assert n == 1
        assert drv.calls[0].params["rows"][0]["key"] == "a"
