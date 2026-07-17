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
        # --- index DDL (no graph effect in the fake) ---
        if q.startswith("CREATE INDEX ingestjob_key IF NOT EXISTS"):
            return _Rows([])
        if q.startswith("CREATE INDEX ingestjob_owner IF NOT EXISTS"):
            return _Rows([])
        # --- create_job: MERGE that sets pending + empty owner + stages ---
        if q.startswith("MERGE (j:IngestJob {key:$key}) SET j.owner = '', j.status = 'pending'"):
            key = params["key"]
            job = self._store.setdefault(
                key, {"key": key, "owner": "", "claimed_at": 0,
                      "status": "pending", "stages": dict(params.get("stages", {}))})
            return _Single({"key": job["key"], "owner": job["owner"],
                            "status": job["status"], "stages": job["stages"]})
        # --- claim: MERGE + conditional guarded SET ---
        if q.startswith("MERGE (j:IngestJob {key:$key}) SET j.updated_at = timestamp()"):
            key = params["key"]
            job = self._store.setdefault(
                key, {"key": key, "owner": "", "claimed_at": 0,
                      "status": "pending", "stages": {}})
            if job["owner"] == "" or job["claimed_at"] < params["expires"]:
                job["owner"] = params["owner"]
                job["claimed_at"] = params["now"]
                if job["status"] is None:
                    job["status"] = "claimed"
            return _Single({"owner": job["owner"], "claimed_at": job["claimed_at"]})
        # --- release: MATCH {key} WHERE owner = $owner ---
        if q.startswith("MATCH (j:IngestJob {key:$key}) WHERE j.owner = $owner SET j.owner = '', j.claimed_at = 0, j.status = 'queued'"):
            key = params["key"]
            job = self._store.get(key)
            if job and job["owner"] == params["owner"]:
                job["owner"] = ""
                job["claimed_at"] = 0
                job["status"] = "queued"
                return _Single({"key": key})
            return _Single(None)
        # --- update_stage: MATCH {key} WHERE owner = $owner OR $owner = '' ---
        if q.startswith("MATCH (j:IngestJob {key:$key}) WHERE j.owner = $owner OR $owner = ''"):
            key = params["key"]
            job = self._store.get(key)
            if job is None:
                return _Single(None)
            if params["owner"] != "" and job["owner"] != params["owner"]:
                return _Single(None)
            stage = params["stage"]
            job["stages"] = dict(job["stages"])
            job["stages"][stage] = True
            if stage == "graph_written":
                job["status"] = "done"
            return _Single({"key": key, "stages": job["stages"], "status": job["status"]})
        # --- stage_status: MATCH {key} RETURN ... LIMIT 1 ---
        if q.startswith("MATCH (j:IngestJob {key:$key}) RETURN j.key AS key, j.owner AS owner"):
            key = params["key"]
            job = self._store.get(key)
            if job is None:
                return _Single(None)
            return _Single({"key": key, "owner": job["owner"], "status": job["status"],
                            "stages": job["stages"]})
        # --- list_claims: MATCH (j:IngestJob) WHERE owner <> '' AND claimed_at >= since ---
        if q.startswith("MATCH (j:IngestJob) WHERE j.owner <> '' AND j.claimed_at >= $since"):
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
        # --- expired_claims: MATCH (j:IngestJob) WHERE owner <> '' AND claimed_at < expires ---
        if q.startswith("MATCH (j:IngestJob) WHERE j.owner <> '' AND j.claimed_at < $expires RETURN j.key AS key, j.owner AS owner, j.claimed_at AS claimed_at ORDER BY j.claimed_at ASC"):
            expires = params["expires"]
            limit = params["limit"]
            rows = [
                {"key": k, "owner": j["owner"], "claimed_at": j["claimed_at"]}
                for k, j in self._store.items()
                if j["owner"] != "" and j["claimed_at"] < expires
            ]
            rows.sort(key=lambda r: r["claimed_at"])
            return _Rows(rows[:limit])
        # --- reap: same predicate + bounded SET ---
        if q.startswith("MATCH (j:IngestJob) WHERE j.owner <> '' AND j.claimed_at < $expires SET j.owner = '', j.claimed_at = 0, j.status = 'pending'"):
            expires = params["expires"]
            limit = params["limit"]
            cleared = 0
            for k, j in self._store.items():
                if cleared >= limit:
                    break
                if j["owner"] != "" and j["claimed_at"] < expires:
                    j["owner"] = ""
                    j["claimed_at"] = 0
                    j["status"] = "pending"
                    cleared += 1
            return _Single({"cleared": cleared})
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
    # A key is first created (pending) by the coordinator, then claimed.
    ingest_claim.create_job(d, "k1")
    assert ingest_claim.claim(d, "k1", "hostA") is True
    assert d.store["k1"]["owner"] == "hostA"
    # claim preserves an existing (pending) status; only sets 'claimed' when null.
    assert d.store["k1"]["status"] == "pending"


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


# --- manifest / stage-tracking tests -----------------------------------------
class _QueryCaptureDriver:
    """Fake driver that records every Cypher statement issued (for DDL checks)."""
    def __init__(self):
        self.store: dict = {}
        self.queries: list = []
        self.calls: int = 0

    def session(self):
        capture = self

        class _S:
            def __enter__(self):
                return self

            def __exit__(self, *e):
                return False

            def run(self, query, **params):
                capture.queries.append(" ".join(query.split()))
                capture.calls += 1
                return _Rows([])
        return _S()


def test_ensure_indexes_issues_both_create_statements():
    d = _QueryCaptureDriver()
    ingest_claim.ensure_indexes(d)
    assert any(q == ("CREATE INDEX ingestjob_key IF NOT EXISTS "
                     "FOR (j:IngestJob) ON (j.key)") for q in d.queries)
    assert any(q == ("CREATE INDEX ingestjob_owner IF NOT EXISTS "
                     "FOR (j:IngestJob) ON (j.owner)") for q in d.queries)
    # two one-line DDL statements, nothing else
    assert d.calls == 2


def test_create_job_pending_empty_owner():
    d = _FakeDriver()
    job = ingest_claim.create_job(d, "k1")
    assert d.store["k1"]["owner"] == ""
    assert d.store["k1"]["status"] == "pending"
    assert d.store["k1"]["stages"] == {s: False for s in ingest_claim.STAGES}
    assert job["owner"] == ""
    assert job["status"] == "pending"


def test_update_stage_guards_owner():
    d = _FakeDriver()
    ingest_claim.create_job(d, "k1")
    # claim so owner is set
    ingest_claim.claim(d, "k1", "hostA", ttl_sec=3600, now_ms=1_000_000)
    # wrong owner may not update
    assert ingest_claim.update_stage(d, "k1", "copied", owner="hostB") is None
    assert d.store["k1"]["stages"]["copied"] is False
    # correct owner updates
    res = ingest_claim.update_stage(d, "k1", "copied", owner="hostA")
    assert res is not None
    assert d.store["k1"]["stages"]["copied"] is True
    # coordinator (empty owner) may update regardless
    res = ingest_claim.update_stage(d, "k1", "transcribed", owner="")
    assert d.store["k1"]["stages"]["transcribed"] is True


def test_update_stage_graph_written_flips_done():
    d = _FakeDriver()
    ingest_claim.create_job(d, "k1")
    res = ingest_claim.update_stage(d, "k1", "graph_written", owner="")
    assert res["status"] == "done"
    assert d.store["k1"]["stages"]["graph_written"] is True


def test_update_stage_rejects_unknown_stage():
    d = _FakeDriver()
    ingest_claim.create_job(d, "k1")
    import pytest
    with pytest.raises(ValueError):
        ingest_claim.update_stage(d, "k1", "not_a_stage", owner="")


def test_stage_status_read_only():
    d = _FakeDriver()
    ingest_claim.create_job(d, "k1")
    ingest_claim.update_stage(d, "k1", "copied", owner="")
    ss = ingest_claim.stage_status(d, "k1")
    assert ss["key"] == "k1"
    assert ss["stages"]["copied"] is True
    assert ss["stages"]["transcribed"] is False
    assert ingest_claim.stage_status(d, "missing") is None


def test_expired_claims_and_reap_bounded():
    d = _FakeDriver()
    base = 100_000_000
    # two claimed, within ttl; two expired
    ingest_claim.claim(d, "fresh", "hostA", ttl_sec=3600, now_ms=base)
    ingest_claim.claim(d, "fresh2", "hostB", ttl_sec=3600, now_ms=base)
    ingest_claim.claim(d, "old1", "hostC", ttl_sec=3600, now_ms=base - 7200 * 1000)
    ingest_claim.claim(d, "old2", "hostD", ttl_sec=3600, now_ms=base - 7200 * 1000)
    # expired listing excludes fresh ones
    expired = ingest_claim.expired_claims(d, ttl_sec=3600, limit=50, now_ms=base)
    keys = {e["key"] for e in expired}
    assert keys == {"old1", "old2"}
    # reap clears only expired, bounded by limit
    cleared = ingest_claim.reap(d, ttl_sec=3600, limit=1, now_ms=base)
    assert cleared == 1
    remaining = ingest_claim.expired_claims(d, ttl_sec=3600, limit=50, now_ms=base)
    assert len(remaining) == 1  # only one more expired
    # fresh claims untouched
    assert d.store["fresh"]["owner"] == "hostA"
    assert d.store["fresh2"]["owner"] == "hostB"
