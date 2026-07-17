"""Tests for the distributed ingest claim protocol (no live Neo4j).

Uses an in-memory fake driver that understands the exact Cypher shapes issued
by ``auto_ingest.ingest_claim`` (conditional SET, indexed MATCH, paged LIMIT).
"""
from __future__ import annotations

from auto_ingest import ingest_claim


class _FakeSession:
    def __init__(self, store):
        self._store = store  # dict key -> {"owner","claimed_at","status"}

    def __enter__(self):
        return self

    def __exit__(self, *e):
        return False

    def run(self, query, **params):
        q = " ".join(query.split())
        if q.startswith("MERGE (j:IngestJob {key:$key})"):
            key = params["key"]
            job = self._store.setdefault(
                key, {"key": key, "owner": "", "claimed_at": 0, "status": None})
            # conditional SET: owner empty OR claimed_at < expires
            if job["owner"] == "" or job["claimed_at"] < params["expires"]:
                job["owner"] = params["owner"]
                job["claimed_at"] = params["now"]
                if job["status"] is None:
                    job["status"] = "claimed"
            return _Single({"owner": job["owner"], "claimed_at": job["claimed_at"]})
        if q.startswith("MATCH (j:IngestJob {key:$key})"):
            key = params["key"]
            job = self._store.get(key)
            if job and job["owner"] == params["owner"]:
                job["owner"] = ""
                job["claimed_at"] = 0
                job["status"] = "queued"
                return _Single({"key": key})
            return _Single(None)
        if q.startswith("MATCH (j:IngestJob)"):
            since = params["since"]
            limit = params["limit"]
            rows = [
                {"key": k, "owner": j["owner"], "claimed_at": j["claimed_at"],
                 "status": j["status"]}
                for k, j in self._store.items()
                if j["owner"] != "" and j["claimed_at"] >= since
            ]
            rows.sort(key=lambda r: r["claimed_at"], reverse=True)
            return _Rows(rows[:limit])
        raise AssertionError(f"unexpected query: {q}")


class _Single:
    def __init__(self, rec):
        self._rec = rec
    def single(self):
        return self._rec


class _Rows:
    def __init__(self, rows):
        self._rows = rows
    def data(self):
        return list(self._rows)


class _FakeDriver:
    def __init__(self):
        self.store: dict = {}
    def session(self):
        return _FakeSession(self.store)


def test_claim_takes_ownership():
    d = _FakeDriver()
    assert ingest_claim.claim(d, "k1", "hostA") is True
    assert d.store["k1"]["owner"] == "hostA"
    assert d.store["k1"]["status"] == "claimed"


def test_claim_is_mutually_exclusive():
    d = _FakeDriver()
    assert ingest_claim.claim(d, "k1", "hostA") is True
    # Second worker loses the race (claim not expired yet).
    assert ingest_claim.claim(d, "k1", "hostB") is False
    assert d.store["k1"]["owner"] == "hostA"


def test_claim_expired_ttl_is_reclaimable():
    d = _FakeDriver()
    t0 = 1_000_000
    assert ingest_claim.claim(d, "k1", "hostA", ttl_sec=3600, now_ms=t0) is True
    # Far in the future: previous claim expired -> new owner wins.
    t1 = t0 + 7200 * 1000
    assert ingest_claim.claim(d, "k1", "hostB", ttl_sec=3600, now_ms=t1) is True
    assert d.store["k1"]["owner"] == "hostB"


def test_release_only_by_owner():
    d = _FakeDriver()
    ingest_claim.claim(d, "k1", "hostA")
    assert ingest_claim.release(d, "k1", "hostB") is False  # not the owner
    assert d.store["k1"]["owner"] == "hostA"
    assert ingest_claim.release(d, "k1", "hostA") is True
    assert d.store["k1"]["owner"] == ""
    assert d.store["k1"]["status"] == "queued"


def test_list_claims_read_only_and_excludes_expired():
    d = _FakeDriver()
    base = 10_000_000
    ingest_claim.claim(d, "fresh", "hostA", ttl_sec=3600, now_ms=base)
    ingest_claim.claim(d, "old", "hostB", ttl_sec=3600, now_ms=base - 7200 * 1000)
    claims = ingest_claim.list_claims(d, ttl_sec=3600, limit=50, now_ms=base)
    keys = {c["key"] for c in claims}
    assert "fresh" in keys
    assert "old" not in keys  # expired, excluded


def test_list_claims_paged():
    d = _FakeDriver()
    base = 20_000_000
    for i in range(10):
        ingest_claim.claim(d, f"k{i}", f"host{i}", ttl_sec=3600,
                           now_ms=base + i * 1000)
    claims = ingest_claim.list_claims(d, ttl_sec=3600, limit=3, now_ms=base + 9999)
    assert len(claims) == 3
    # newest first
    assert claims[0]["key"] == "k9"
