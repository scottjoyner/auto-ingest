"""Durable distributed ingest claim and stage-state protocol.

The protocol deliberately uses only Neo4j-supported primitive/list node
properties. Stage state is persisted as ``completed_stages: [STRING]`` and is
expanded into the historical ``{stage: bool}`` API shape at the Python boundary.

For production distributed execution, use ``claim_fenced`` and the fenced
mutation APIs. Each successful takeover increments ``fence_token``; stale
workers cannot update stages or release a lease after a newer owner acquires it.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

STAGES: List[str] = [
    "copied",
    "transcribed",
    "diarized",
    "embedded",
    "linked",
    "graph_written",
]

STATUS_PENDING = "pending"
STATUS_CLAIMED = "claimed"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"

_INDEX_KEY_QUERY = "CREATE INDEX ingestjob_key IF NOT EXISTS FOR (j:IngestJob) ON (j.key)"
_INDEX_OWNER_QUERY = "CREATE INDEX ingestjob_owner IF NOT EXISTS FOR (j:IngestJob) ON (j.owner)"


def ensure_indexes(driver) -> None:
    with driver.session() as sess:
        sess.run(_INDEX_KEY_QUERY).consume()
        sess.run(_INDEX_OWNER_QUERY).consume()


def _stage_map(completed: Optional[List[str]] = None) -> Dict[str, bool]:
    done = set(completed or [])
    return {stage: stage in done for stage in STAGES}


def _now_ms() -> int:
    return int(time.time() * 1000)


_CLAIM_QUERY = """
MERGE (j:IngestJob {key:$key})
ON CREATE SET j.owner = '', j.claimed_at = 0, j.status = 'pending',
              j.completed_stages = [], j.attempt_count = 0,
              j.fence_token = 0, j.created_at = $now
WITH j
WHERE coalesce(j.owner, '') = '' OR coalesce(j.claimed_at, 0) < $expires
SET j.owner = $owner, j.claimed_at = $now, j.status = 'claimed',
    j.attempt_count = coalesce(j.attempt_count, 0) + 1, j.updated_at = $now
RETURN j.owner AS owner, j.claimed_at AS claimed_at
"""

_FENCED_CLAIM_QUERY = """
MERGE (j:IngestJob {key:$key})
ON CREATE SET j.owner = '', j.claimed_at = 0, j.status = 'pending',
              j.completed_stages = [], j.attempt_count = 0,
              j.fence_token = 0, j.created_at = $now
WITH j
WHERE coalesce(j.owner, '') = '' OR coalesce(j.claimed_at, 0) < $expires
SET j.owner = $owner, j.claimed_at = $now, j.status = 'claimed',
    j.attempt_count = coalesce(j.attempt_count, 0) + 1,
    j.fence_token = coalesce(j.fence_token, 0) + 1,
    j.updated_at = $now
RETURN j.owner AS owner, j.claimed_at AS claimed_at,
       j.fence_token AS fence_token
"""

_RELEASE_QUERY = """
MATCH (j:IngestJob {key:$key}) WHERE j.owner = $owner
SET j.owner = '', j.claimed_at = 0,
    j.status = CASE WHEN j.status = 'done' THEN 'done' ELSE 'pending' END,
    j.updated_at = timestamp()
RETURN j.key AS key
"""

_FENCED_RELEASE_QUERY = """
MATCH (j:IngestJob {key:$key})
WHERE j.owner = $owner AND coalesce(j.fence_token, 0) = $fence_token
SET j.owner = '', j.claimed_at = 0,
    j.status = CASE WHEN j.status = 'done' THEN 'done' ELSE 'pending' END,
    j.updated_at = timestamp()
RETURN j.key AS key
"""

_LIST_QUERY = """
MATCH (j:IngestJob)
WHERE coalesce(j.owner, '') <> '' AND coalesce(j.claimed_at, 0) >= $since
RETURN j.key AS key, j.owner AS owner, j.claimed_at AS claimed_at,
       j.status AS status, coalesce(j.fence_token, 0) AS fence_token
ORDER BY j.claimed_at DESC LIMIT $limit
"""

_CREATE_JOB_QUERY = """
MERGE (j:IngestJob {key:$key})
ON CREATE SET j.owner = '', j.claimed_at = 0, j.status = 'pending',
              j.completed_stages = [], j.attempt_count = 0,
              j.fence_token = 0, j.created_at = $now
SET j.updated_at = $now
RETURN j.key AS key, j.owner AS owner, j.status AS status,
       j.completed_stages AS completed_stages,
       j.attempt_count AS attempt_count,
       coalesce(j.fence_token, 0) AS fence_token
"""

_UPDATE_STAGE_QUERY = """
MATCH (j:IngestJob {key:$key}) WHERE j.owner = $owner OR $owner = ''
SET j.completed_stages = CASE
        WHEN $stage IN coalesce(j.completed_stages, []) THEN coalesce(j.completed_stages, [])
        ELSE coalesce(j.completed_stages, []) + $stage END,
    j.status = CASE WHEN $stage = 'graph_written' THEN 'done' ELSE 'running' END,
    j.updated_at = timestamp()
RETURN j.key AS key, j.completed_stages AS completed_stages, j.status AS status
"""

_FENCED_UPDATE_STAGE_QUERY = """
MATCH (j:IngestJob {key:$key})
WHERE j.owner = $owner AND coalesce(j.fence_token, 0) = $fence_token
SET j.completed_stages = CASE
        WHEN $stage IN coalesce(j.completed_stages, []) THEN coalesce(j.completed_stages, [])
        ELSE coalesce(j.completed_stages, []) + $stage END,
    j.status = CASE WHEN $stage = 'graph_written' THEN 'done' ELSE 'running' END,
    j.updated_at = timestamp()
RETURN j.key AS key, j.completed_stages AS completed_stages,
       j.status AS status, j.fence_token AS fence_token
"""

_STAGE_STATUS_QUERY = """
MATCH (j:IngestJob {key:$key})
RETURN j.key AS key, j.owner AS owner, j.status AS status,
       j.completed_stages AS completed_stages,
       j.attempt_count AS attempt_count,
       coalesce(j.fence_token, 0) AS fence_token
LIMIT 1
"""

_EXPIRED_QUERY = """
MATCH (j:IngestJob)
WHERE coalesce(j.owner, '') <> '' AND coalesce(j.claimed_at, 0) < $expires
RETURN j.key AS key, j.owner AS owner, j.claimed_at AS claimed_at,
       coalesce(j.fence_token, 0) AS fence_token
ORDER BY j.claimed_at ASC LIMIT $limit
"""

_REAP_QUERY = """
MATCH (j:IngestJob)
WHERE coalesce(j.owner, '') <> '' AND coalesce(j.claimed_at, 0) < $expires
WITH j ORDER BY j.claimed_at ASC LIMIT $limit
SET j.owner = '', j.claimed_at = 0,
    j.status = CASE WHEN j.status = 'done' THEN 'done' ELSE 'pending' END,
    j.updated_at = timestamp()
RETURN count(j) AS cleared
"""


def claim(driver, key: str, owner: str, ttl_sec: int = 3600,
          now_ms: Optional[int] = None) -> bool:
    if not key:
        raise ValueError("claim key must be non-empty")
    if not owner:
        raise ValueError("claim owner must be a non-empty worker id")
    if ttl_sec <= 0:
        raise ValueError("claim ttl_sec must be positive")
    now = now_ms if now_ms is not None else _now_ms()
    expires = now - int(ttl_sec * 1000)
    with driver.session() as sess:
        rec = sess.run(_CLAIM_QUERY, key=key, owner=owner, now=now, expires=expires).single()
    return bool(rec) and rec.get("owner") == owner


def claim_fenced(driver, key: str, owner: str, ttl_sec: int = 3600,
                 now_ms: Optional[int] = None) -> Optional[int]:
    """Acquire a lease and return its monotonically increasing fencing token."""
    if not key or not owner:
        raise ValueError("key and owner must be non-empty")
    if ttl_sec <= 0:
        raise ValueError("ttl_sec must be positive")
    now = now_ms if now_ms is not None else _now_ms()
    expires = now - int(ttl_sec * 1000)
    with driver.session() as sess:
        rec = sess.run(_FENCED_CLAIM_QUERY, key=key, owner=owner,
                       now=now, expires=expires).single()
    if not rec or rec.get("owner") != owner:
        return None
    return int(rec.get("fence_token"))


def release(driver, key: str, owner: str) -> bool:
    if not owner:
        raise ValueError("release owner must be non-empty")
    with driver.session() as sess:
        rec = sess.run(_RELEASE_QUERY, key=key, owner=owner).single()
    return rec is not None


def release_fenced(driver, key: str, owner: str, fence_token: int) -> bool:
    if not owner or fence_token < 1:
        raise ValueError("owner and positive fence_token are required")
    with driver.session() as sess:
        rec = sess.run(_FENCED_RELEASE_QUERY, key=key, owner=owner,
                       fence_token=fence_token).single()
    return rec is not None


def list_claims(driver, *, ttl_sec: int = 3600, limit: int = 200,
                now_ms: Optional[int] = None) -> List[Dict]:
    if limit < 1:
        raise ValueError("limit must be positive")
    now = now_ms if now_ms is not None else _now_ms()
    since = now - int(ttl_sec * 1000)
    with driver.session() as sess:
        rows = sess.run(_LIST_QUERY, since=since, limit=limit).data()
    return [dict(r) for r in rows]


def create_job(driver, key: str, stages: Optional[List[str]] = None) -> Dict:
    if not key:
        raise ValueError("job key must be non-empty")
    if stages is not None:
        unknown = set(stages) - set(STAGES)
        if unknown:
            raise ValueError(f"unknown stages: {sorted(unknown)}")
    now = _now_ms()
    with driver.session() as sess:
        rec = sess.run(_CREATE_JOB_QUERY, key=key, now=now).single()
    if rec is None:
        raise RuntimeError(f"failed to create/read ingest job {key!r}")
    return {
        "key": rec.get("key"), "owner": rec.get("owner") or "",
        "status": rec.get("status") or STATUS_PENDING,
        "stages": _stage_map(rec.get("completed_stages")),
        "attempt_count": int(rec.get("attempt_count") or 0),
        "fence_token": int(rec.get("fence_token") or 0),
    }


def update_stage(driver, key: str, stage: str,
                 owner: Optional[str] = None) -> Optional[Dict]:
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage!r} (expected one of {STAGES})")
    with driver.session() as sess:
        rec = sess.run(_UPDATE_STAGE_QUERY, key=key, stage=stage,
                       owner=owner or "").single()
    if rec is None:
        return None
    return {"key": rec.get("key"), "stages": _stage_map(rec.get("completed_stages")),
            "status": rec.get("status")}


def update_stage_fenced(driver, key: str, stage: str, *, owner: str,
                        fence_token: int) -> Optional[Dict]:
    """Update stage only if caller still owns the exact lease generation."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage!r}")
    if not owner or fence_token < 1:
        raise ValueError("owner and positive fence_token are required")
    with driver.session() as sess:
        rec = sess.run(_FENCED_UPDATE_STAGE_QUERY, key=key, stage=stage,
                       owner=owner, fence_token=fence_token).single()
    if rec is None:
        return None
    return {"key": rec.get("key"), "stages": _stage_map(rec.get("completed_stages")),
            "status": rec.get("status"), "fence_token": int(rec.get("fence_token"))}


def stage_status(driver, key: str) -> Optional[Dict]:
    with driver.session() as sess:
        rec = sess.run(_STAGE_STATUS_QUERY, key=key).single()
    if rec is None:
        return None
    return {
        "key": rec.get("key"), "owner": rec.get("owner") or "",
        "status": rec.get("status"), "stages": _stage_map(rec.get("completed_stages")),
        "attempt_count": int(rec.get("attempt_count") or 0),
        "fence_token": int(rec.get("fence_token") or 0),
    }


def expired_claims(driver, ttl_sec: int = 3600, limit: int = 200,
                   now_ms: Optional[int] = None) -> List[Dict]:
    if ttl_sec <= 0 or limit < 1:
        raise ValueError("ttl_sec and limit must be positive")
    now = now_ms if now_ms is not None else _now_ms()
    expires = now - int(ttl_sec * 1000)
    with driver.session() as sess:
        return sess.run(_EXPIRED_QUERY, expires=expires, limit=limit).data()


def reap(driver, ttl_sec: int = 3600, limit: int = 200,
         now_ms: Optional[int] = None) -> int:
    if ttl_sec <= 0 or limit < 1:
        raise ValueError("ttl_sec and limit must be positive")
    now = now_ms if now_ms is not None else _now_ms()
    expires = now - int(ttl_sec * 1000)
    with driver.session() as sess:
        rec = sess.run(_REAP_QUERY, expires=expires, limit=limit).single()
    return int(rec.get("cleared")) if rec else 0
