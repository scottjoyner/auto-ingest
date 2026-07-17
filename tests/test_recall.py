"""Unit tests for personal recall (fake driver, no live Neo4j / model)."""
from __future__ import annotations

from typing import List

import auto_ingest.personal.embed as embed
import auto_ingest.personal.recall as recall


class FakeRecord:
    def __init__(self, row):
        self._row = row

    def data(self):
        return self._row

    def get(self, key, default=None):
        return self._row.get(key, default)


class FakeResult:
    def __init__(self, rows):
        self._recs = [FakeRecord(r) for r in rows]

    def data(self):
        return [r.data() for r in self._recs]

    def single(self):
        return self._recs[0] if self._recs else None

    def consume(self):
        return None

    def __iter__(self):
        return iter(self._recs)


class FakeSession:
    def __init__(self, queries: List[str], canned, emb_canned, driver):
        self.queries = queries
        self.canned = canned
        self.emb_canned = emb_canned
        self.driver = driver

    def run(self, cypher, **params):
        self.queries.append(cypher)
        self.driver._params.append(params)
        if "RETURN m.embedding" in cypher:
            return FakeResult(self.emb_canned)
        return FakeResult(self.canned)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeDriver:
    def __init__(self, canned, emb_canned=None):
        self.queries: List[str] = []
        self._params: List[dict] = []
        self.canned = canned
        self.emb_canned = emb_canned or [{"emb": [0.0] * recall.CLIP_DIM}]

    def session(self, database=None):
        return FakeSession(self.queries, self.canned, self.emb_canned, self)


ANN_ROWS = [
    {"sha256": "seed", "path": "/a.jpg", "kind": "picture", "date": "2024-01-01", "score": 0.99},
    {"sha256": "other", "path": "/b.jpg", "kind": "picture", "date": "2024-01-02", "score": 0.8},
]


def test_similar_media_returns_rows():
    drv = FakeDriver(ANN_ROWS)
    rows = recall.similar_media(drv, "seed", top_k=5, include_seed=True)
    assert rows == ANN_ROWS


def test_similar_media_drops_seed():
    drv = FakeDriver(ANN_ROWS)
    rows = recall.similar_media(drv, "seed", top_k=5, include_seed=False)
    assert all(r["sha256"] != "seed" for r in rows)
    assert len(rows) == 1


def test_recall_media_place_filter(monkeypatch):
    monkeypatch.setattr(recall, "ensure_media_indexes_with_retry", lambda: None)
    seed = [0.1] * recall.CLIP_DIM
    monkeypatch.setattr(embed, "embed_text", lambda text: seed)

    drv = FakeDriver(ANN_ROWS)
    recall.recall_media(drv, "beach", top_k=5, place="Boston")
    joined = " ".join(drv.queries)
    assert "AT_PLACE" in joined
    assert "SummaryPlace" in joined
    assert "media_embedding_index" in joined
    assert "media_text_embedding_index" not in joined
    assert any(p.get("qvec") == seed for p in drv._params)


def test_recall_media_uses_embed_text_seed(monkeypatch):
    seed = [0.42] * recall.CLIP_DIM
    monkeypatch.setattr(recall, "ensure_media_indexes_with_retry", lambda: None)
    monkeypatch.setattr(embed, "embed_text", lambda text: seed)

    drv = FakeDriver(ANN_ROWS)
    recall.recall_media(drv, "cats", top_k=3)
    assert any(p.get("qvec") == seed for p in drv._params)


def test_geo_media_bounding_box():
    drv = FakeDriver([
        {"sha256": "x", "path": "/x.jpg", "kind": "picture", "date": "2024-01-01",
         "lat": 42.36, "lon": -71.06, "meters": 10.0},
    ])
    rows = recall.geo_media(drv, 42.36, -71.06, radius_m=200.0, limit=50)
    assert rows and rows[0]["sha256"] == "x"
    joined = " ".join(drv.queries)
    assert "gps_lat" in joined
    assert "gps_lon" in joined
    assert "R" in joined  # radius present


def test_sha256_of_stable(tmp_path):
    f = tmp_path / "sample.bin"
    f.write_bytes(b"hello world")
    h1 = recall.sha256_of(str(f))
    h2 = recall.sha256_of(str(f))
    assert h1 == h2
    assert len(h1) == 64
    assert all(c in "0123456789abcdef" for c in h1)
