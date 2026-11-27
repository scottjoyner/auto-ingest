"""Hermetic tests: claim() must work on a key with no pre-created IngestJob.

No live Neo4j. The fake driver below is deliberately faithful to Cypher's
three-valued logic: a node created by ``MERGE (j:IngestJob {key:$key})`` has
**no** owner/claimed_at/status properties, so each is NULL. Comparisons with
NULL are *unknown* (not true), and a WHERE clause drops the row. The repo's
other claim fake initialises fresh nodes with ``owner=''``, which masks this
case; this one does not.

Fail->pass contract:
  * base: ``claim()`` on an unknown key returns False and the node stays
    unclaimable forever (owner NULL on every retry).
  * fixed: the claim query guards the NULL owner, so the first claim wins.
"""
from __future__ import annotations

import re

from auto_ingest import ingest_claim


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


def _claim_where(query: str) -> str:
    """The conditional predicate between ``WITH j`` and the ownership SET."""
    match = re.search(r"WITH\s+j\s+WHERE\s+(.+?)\s+SET\s+j\.owner\s*=", query, re.DOTALL)
    assert match, "claim query has no conditional WHERE before SET j.owner"
    return " ".join(match.group(1).split())


def _cypher_where_holds(predicate: str, job: dict, expires: int) -> bool:
    """Evaluate the small predicate vocabulary used by the claim query.

    Cypher NULL semantics: ``x = y`` / ``x < y`` involving NULL is unknown and
    does not satisfy WHERE; only ``IS NULL`` can match a NULL property.
    """
    def term(text: str) -> bool:
        if text == "j.owner IS NULL":
            return job["owner"] is None
        if text == "j.owner = ''":
            return job["owner"] == ""
        if text == "j.claimed_at < $expires":
            return job["claimed_at"] is not None and job["claimed_at"] < expires
        raise AssertionError(f"unsupported predicate term: {text!r}")

    return any(term(t.strip()) for t in predicate.split(" OR "))


class _FakeSession:
    def __init__(self, store):
        self._store = store

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, **params):
        q = " ".join(query.split())
        if q.startswith("MERGE (j:IngestJob {key:$key}) SET j.updated_at = timestamp()"):
            key = params["key"]
            job = self._store.get(key)
            if job is None:
                # What Cypher actually creates: no owner/claimed_at/status at
                # all -> all three read back as NULL.
                job = {"key": key, "owner": None, "claimed_at": None, "status": None}
                self._store[key] = job
            if not _cypher_where_holds(_claim_where(q), job, params["expires"]):
                return _Single(None)  # zero rows survive WHERE -> .single() is None
            job["owner"] = params["owner"]
            job["claimed_at"] = params["now"]
            if job["status"] is None:
                job["status"] = "claimed"
            return _Single({"owner": job["owner"], "claimed_at": job["claimed_at"]})
        if q.startswith("MERGE (j:IngestJob {key:$key}) SET j.owner = '', j.status = 'pending'"):
            key = params["key"]
            job = self._store.setdefault(
                key, {"key": key, "owner": "", "claimed_at": None,
                      "status": "pending", "stages": dict(params.get("stages", {}))}
            )
            return _Single({"key": key, "owner": job["owner"],
                            "status": job["status"], "stages": job["stages"]})
        raise AssertionError(f"unexpected query: {q}")


class _FakeDriver:
    def __init__(self):
        self.store = {}

    def session(self):
        return _FakeSession(self.store)


# --- defect cases -----------------------------------------------------------

def test_claim_without_precreated_job_takes_ownership():
    """MERGE-created node has NULL owner; the very first claim must still win."""
    driver = _FakeDriver()
    assert ingest_claim.claim(driver, "bare-key", "worker-A") is True
    assert driver.store["bare-key"]["owner"] == "worker-A"


def test_claim_query_guards_null_owner():
    """The claim predicate must handle the NULL owner of a freshly MERGE'd node."""
    predicate = _claim_where(" ".join(ingest_claim._CLAIM_QUERY.split()))
    assert "j.owner IS NULL" in predicate


# --- controls (must pass before and after the fix) --------------------------

def test_create_job_then_claim_still_takes_ownership():
    driver = _FakeDriver()
    ingest_claim.create_job(driver, "coordinated")
    assert ingest_claim.claim(driver, "coordinated", "worker-A") is True
    assert driver.store["coordinated"]["owner"] == "worker-A"
    assert driver.store["coordinated"]["status"] == "pending"


def test_second_worker_cannot_steal_live_claim():
    driver = _FakeDriver()
    ingest_claim.create_job(driver, "k")
    assert ingest_claim.claim(driver, "k", "worker-A") is True
    assert ingest_claim.claim(driver, "k", "worker-B") is False
    assert driver.store["k"]["owner"] == "worker-A"


def test_expired_claim_is_reclaimable():
    driver = _FakeDriver()
    t0 = 1_000_000
    ingest_claim.create_job(driver, "k")
    assert ingest_claim.claim(driver, "k", "worker-A", ttl_sec=10, now_ms=t0) is True
    t1 = t0 + 60_000
    assert ingest_claim.claim(driver, "k", "worker-B", ttl_sec=10, now_ms=t1) is True
    assert driver.store["k"]["owner"] == "worker-B"
