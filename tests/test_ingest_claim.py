"""Unit tests for the ingest claim protocol without a live Neo4j service."""
from __future__ import annotations

import pytest

from auto_ingest import ingest_claim


class _Result:
    def __init__(self, rows=None, single=None):
        self._rows = rows or []
        self._single = single

    def single(self):
        return self._single

    def data(self):
        return list(self._rows)

    def consume(self):
        return None


class _FakeSession:
    def __init__(self, store, queries):
        self.store = store
        self.queries = queries

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, **params):
        q = " ".join(query.split())
        self.queries.append(q)

        if q.startswith("CREATE INDEX ingestjob_"):
            return _Result()

        if q.startswith("MERGE (j:IngestJob {key:$key}) ON CREATE SET j.owner = ''"):
            key = params["key"]
            job = self.store.setdefault(
                key,
                {
                    "key": key,
                    "owner": "",
                    "claimed_at": 0,
                    "status": "pending",
                    "completed_stages": [],
                    "attempt_count": 0,
                },
            )
            if "RETURN j.key AS key, j.owner AS owner" in q:
                return _Result(single=dict(job))

            # claim path
            if job["owner"] == "" or job["claimed_at"] < params["expires"]:
                job["owner"] = params["owner"]
                job["claimed_at"] = params["now"]
                job["status"] = "claimed"
                job["attempt_count"] += 1
                return _Result(
                    single={"owner": job["owner"], "claimed_at": job["claimed_at"]}
                )
            return _Result(single=None)

        if q.startswith("MATCH (j:IngestJob {key:$key}) WHERE j.owner = $owner SET j.owner = ''"):
            job = self.store.get(params["key"])
            if not job or job["owner"] != params["owner"]:
                return _Result(single=None)
            job["owner"] = ""
            job["claimed_at"] = 0
            if job["status"] != "done":
                job["status"] = "pending"
            return _Result(single={"key": job["key"]})

        if q.startswith("MATCH (j:IngestJob {key:$key}) WHERE j.owner = $owner OR $owner = ''"):
            job = self.store.get(params["key"])
            if not job:
                return _Result(single=None)
            if params["owner"] and job["owner"] != params["owner"]:
                return _Result(single=None)
            stage = params["stage"]
            if stage not in job["completed_stages"]:
                job["completed_stages"].append(stage)
            job["status"] = "done" if stage == "graph_written" else "running"
            return _Result(
                single={
                    "key": job["key"],
                    "completed_stages": list(job["completed_stages"]),
                    "status": job["status"],
                }
            )

        if q.startswith("MATCH (j:IngestJob {key:$key}) RETURN j.key AS key"):
            job = self.store.get(params["key"])
            return _Result(single=dict(job) if job else None)

        if q.startswith("MATCH (j:IngestJob) WHERE coalesce(j.owner, '') <> ''"):
            if "coalesce(j.claimed_at, 0) >= $since" in q:
                rows = [
                    {
                        "key": j["key"],
                        "owner": j["owner"],
                        "claimed_at": j["claimed_at"],
                        "status": j["status"],
                    }
                    for j in self.store.values()
                    if j["owner"] and j["claimed_at"] >= params["since"]
                ]
                rows.sort(key=lambda x: x["claimed_at"], reverse=True)
                return _Result(rows=rows[: params["limit"]])

            expired = [
                j
                for j in self.store.values()
                if j["owner"] and j["claimed_at"] < params["expires"]
            ]
            expired.sort(key=lambda x: x["claimed_at"])
            expired = expired[: params["limit"]]
            if "SET j.owner = ''" in q:
                for j in expired:
                    j["owner"] = ""
                    j["claimed_at"] = 0
                    if j["status"] != "done":
                        j["status"] = "pending"
                return _Result(single={"cleared": len(expired)})
            return _Result(
                rows=[
                    {
                        "key": j["key"],
                        "owner": j["owner"],
                        "claimed_at": j["claimed_at"],
                    }
                    for j in expired
                ]
            )

        raise AssertionError(f"unexpected query: {q}")


class _FakeDriver:
    def __init__(self):
        self.store = {}
        self.queries = []

    def session(self):
        return _FakeSession(self.store, self.queries)


def test_indexes_are_explicit_and_idempotent():
    d = _FakeDriver()
    ingest_claim.ensure_indexes(d)
    assert len(d.queries) == 2
    assert "ingestjob_key" in d.queries[0]
    assert "ingestjob_owner" in d.queries[1]


def test_create_job_is_idempotent_and_does_not_reset_progress():
    d = _FakeDriver()
    first = ingest_claim.create_job(d, "k1")
    assert first["status"] == "pending"
    ingest_claim.claim(d, "k1", "hostA", now_ms=1_000_000)
    ingest_claim.update_stage(d, "k1", "copied", owner="hostA")

    second = ingest_claim.create_job(d, "k1")
    assert second["owner"] == "hostA"
    assert second["status"] == "running"
    assert second["stages"]["copied"] is True


def test_claim_is_exclusive_and_expired_claim_can_be_reclaimed():
    d = _FakeDriver()
    t0 = 1_000_000
    assert ingest_claim.claim(d, "k1", "hostA", ttl_sec=3600, now_ms=t0)
    assert not ingest_claim.claim(d, "k1", "hostB", ttl_sec=3600, now_ms=t0 + 1)
    assert ingest_claim.claim(
        d, "k1", "hostB", ttl_sec=3600, now_ms=t0 + 7200 * 1000
    )
    assert d.store["k1"]["attempt_count"] == 2


def test_release_is_owner_guarded():
    d = _FakeDriver()
    ingest_claim.claim(d, "k1", "hostA")
    assert not ingest_claim.release(d, "k1", "hostB")
    assert ingest_claim.release(d, "k1", "hostA")
    assert d.store["k1"]["owner"] == ""
    assert d.store["k1"]["status"] == "pending"


def test_stage_updates_are_owner_guarded_and_idempotent():
    d = _FakeDriver()
    ingest_claim.create_job(d, "k1")
    ingest_claim.claim(d, "k1", "hostA")
    assert ingest_claim.update_stage(d, "k1", "copied", owner="hostB") is None

    first = ingest_claim.update_stage(d, "k1", "copied", owner="hostA")
    second = ingest_claim.update_stage(d, "k1", "copied", owner="hostA")
    assert first["stages"]["copied"] is True
    assert second["stages"]["copied"] is True
    assert d.store["k1"]["completed_stages"].count("copied") == 1


def test_terminal_stage_sets_done_and_release_preserves_done():
    d = _FakeDriver()
    ingest_claim.create_job(d, "k1")
    ingest_claim.claim(d, "k1", "hostA")
    result = ingest_claim.update_stage(d, "k1", "graph_written", owner="hostA")
    assert result["status"] == "done"
    assert ingest_claim.release(d, "k1", "hostA")
    assert d.store["k1"]["status"] == "done"


def test_stage_status_expands_primitive_completed_stage_list():
    d = _FakeDriver()
    ingest_claim.create_job(d, "k1")
    ingest_claim.update_stage(d, "k1", "copied", owner="")
    status = ingest_claim.stage_status(d, "k1")
    assert status["stages"]["copied"] is True
    assert status["stages"]["transcribed"] is False


def test_list_claims_excludes_expired_and_is_bounded():
    d = _FakeDriver()
    now = 20_000_000
    for i in range(5):
        ingest_claim.claim(d, f"fresh{i}", f"host{i}", now_ms=now + i)
    ingest_claim.claim(d, "old", "old-host", now_ms=now - 7200 * 1000)
    claims = ingest_claim.list_claims(d, ttl_sec=3600, limit=3, now_ms=now + 10)
    assert len(claims) == 3
    assert all(c["key"] != "old" for c in claims)


def test_reap_limit_is_applied_before_mutation():
    d = _FakeDriver()
    now = 100_000_000
    for i in range(3):
        ingest_claim.claim(
            d,
            f"old{i}",
            f"host{i}",
            now_ms=now - (7200 + i) * 1000,
        )
    cleared = ingest_claim.reap(d, ttl_sec=3600, limit=1, now_ms=now)
    assert cleared == 1
    assert sum(bool(j["owner"]) for j in d.store.values()) == 2


def test_invalid_inputs_fail_closed():
    d = _FakeDriver()
    with pytest.raises(ValueError):
        ingest_claim.claim(d, "k1", "")
    with pytest.raises(ValueError):
        ingest_claim.claim(d, "k1", "host", ttl_sec=0)
    with pytest.raises(ValueError):
        ingest_claim.release(d, "k1", "")
    with pytest.raises(ValueError):
        ingest_claim.update_stage(d, "k1", "not-a-stage")
    with pytest.raises(ValueError):
        ingest_claim.reap(d, limit=0)
