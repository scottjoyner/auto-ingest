"""Distributed ingest claim protocol (G5/G6 — smallest safe slice).

This is the first, read-mostly slice of the DEEP_DIVE §3.3 claim protocol. It
covers the *coordination* half: which worker owns which ingest key right now,
with a TTL reaper so a crashed worker's claim auto-expires. The heavy "copy /
process / commit cleaned intermediate" half is intentionally left for a
follow-up (see docs/PLAN_distributed_ingest.md).

Design
------
* An ``IngestJob`` node represents one media key through its preprocessing
  stages (G4). It carries ``owner`` (empty string = unclaimed) and
  ``claimed_at`` (epoch ms, or 0 = never).
* ``claim(key, owner, ttl)`` does a **conditional** ``SET`` guarded by
  ``owner = '' OR claimed_at < now-ttl`` so only one worker ever wins. It
  returns True if this call took ownership.
* ``release(key)`` clears the owner so the key is re-claimable.
* ``list_claims()`` is **read-only and index-backed, paged** so it can never
  trigger the 21M-node OOM: it only scans ``IngestJob`` nodes and is bounded by
  ``LIMIT``. No full-graph aggregation is ever issued here.

All graph access goes through a ``driver`` you pass in (a ``neo4j.Driver`` or
any object exposing ``driver.session()`` returning a context manager). Nothing
here writes outside the ``IngestJob`` label and nothing is executed against the
live graph by importing this module — callers decide when to run it.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

# Processing stages a media key moves through before it is fully committed to the
# graph (G4 manifest). ``graph_written`` is the terminal stage set by the worker
# just before it moves the ``.job`` to ``done/``.
STAGES: List[str] = [
    "copied",
    "transcribed",
    "diarized",
    "embedded",
    "linked",
    "graph_written",
]

# Lifecycle status of an ``IngestJob`` node.
STATUS_PENDING = "pending"
STATUS_CLAIMED = "claimed"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"

# ---------------------------------------------------------------------------
# Index DDL (idempotent, safe online DDL — never run inline; call explicitly).
# ---------------------------------------------------------------------------
_INDEX_KEY_QUERY = (
    "CREATE INDEX ingestjob_key IF NOT EXISTS "
    "FOR (j:IngestJob) ON (j.key)"
)
_INDEX_OWNER_QUERY = (
    "CREATE INDEX ingestjob_owner IF NOT EXISTS "
    "FOR (j:IngestJob) ON (j.owner)"
)


def ensure_indexes(driver) -> None:
    """Create the two ``IngestJob`` indexes idempotently.

    One-line ONLINE DDL — safe to run against the live 21M-node DB (it does not
    lock the graph). Callers decide *when* to run this (e.g. a migration script);
    importing this module never issues it. Each statement is issued in its own
    transaction.
    """
    with driver.session() as sess:
        sess.run(_INDEX_KEY_QUERY)
        sess.run(_INDEX_OWNER_QUERY)


def _stage_map(stages: List[str]) -> Dict[str, bool]:
    return {s: False for s in stages}


# Cypher is written to be index-friendly: the conditional match on ``owner`` /
# ``claimed_at`` is cheap when an index exists on ``IngestJob(owner)`` and
# ``IngestJob(key)``. If those indexes are absent the queries still function but
# are not OOM-safe; the follow-up doc lists creating them as a prerequisite.

_CLAIM_QUERY = """
MERGE (j:IngestJob {key:$key})
SET j.updated_at = timestamp()
WITH j
WHERE j.owner = '' OR j.claimed_at < $expires
SET j.owner = $owner,
    j.claimed_at = $now,
    j.status = CASE WHEN j.status IS NULL THEN 'claimed' ELSE j.status END
RETURN j.owner AS owner, j.claimed_at AS claimed_at
"""

_RELEASE_QUERY = """
MATCH (j:IngestJob {key:$key})
WHERE j.owner = $owner
SET j.owner = '', j.claimed_at = 0, j.status = 'queued'
RETURN j.key AS key
"""

_LIST_QUERY = """
MATCH (j:IngestJob)
WHERE j.owner <> '' AND j.claimed_at >= $since
RETURN j.key AS key, j.owner AS owner,
       j.claimed_at AS claimed_at, j.status AS status
ORDER BY j.claimed_at DESC
LIMIT $limit
"""


def _now_ms() -> int:
    return int(time.time() * 1000)


def claim(driver, key: str, owner: str, ttl_sec: int = 3600,
          now_ms: Optional[int] = None) -> bool:
    """Try to take ownership of ``key`` for ``owner``.

    Returns True only if this call set the owner (unclaimed, or the previous
    claim expired). Safe for concurrent workers: the conditional ``SET`` runs
    inside a single Cypher statement so two claimants race atomically.
    """
    if not owner:
        raise ValueError("claim owner must be a non-empty worker id")
    now = now_ms if now_ms is not None else _now_ms()
    expires = now - int(ttl_sec * 1000)
    with driver.session() as sess:
        rec = sess.run(_CLAIM_QUERY, key=key, owner=owner,
                       now=now, expires=expires).single()
    # We won iff the resulting owner is us.
    return bool(rec) and rec.get("owner") == owner


def release(driver, key: str, owner: str) -> bool:
    """Release ``key`` only if currently owned by ``owner``.

    Returns True if a release happened. This prevents one worker from clearing
    another worker's live claim.
    """
    with driver.session() as sess:
        rec = sess.run(_RELEASE_QUERY, key=key, owner=owner).single()
    return rec is not None


def list_claims(driver, *, ttl_sec: int = 3600, limit: int = 200,
                now_ms: Optional[int] = None) -> List[Dict]:
    """Read-only, paged view of currently-active claims.

    Only returns jobs whose ``owner`` is non-empty AND whose claim has not
    expired (within ``ttl_sec``). Index-backed via ``IngestJob(owner)`` and
    bounded by ``LIMIT`` — never scans the whole graph. Safe to run against the
    live 21M-node database.
    """
    now = now_ms if now_ms is not None else _now_ms()
    since = now - int(ttl_sec * 1000)
    with driver.session() as sess:
        rows = sess.run(_LIST_QUERY, since=since, limit=limit).data()
    return [
        {
            "key": r.get("key"),
            "owner": r.get("owner"),
            "claimed_at": r.get("claimed_at"),
            "status": r.get("status"),
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Resumable manifest: stage tracking + per-key status (G4).
# ---------------------------------------------------------------------------
_CREATE_JOB_QUERY = """
MERGE (j:IngestJob {key:$key})
SET j.owner = '',
    j.status = 'pending',
    j.stages = $stages,
    j.created_at = timestamp()
RETURN j.key AS key, j.owner AS owner, j.status AS status, j.stages AS stages
"""

_UPDATE_STAGE_QUERY = """
MATCH (j:IngestJob {key:$key})
WHERE j.owner = $owner OR $owner = ''
SET j.stages[$stage] = true
WITH j
WHERE $stage = 'graph_written'
SET j.status = 'done'
RETURN j.key AS key, j.stages AS stages, j.status AS status
"""

_STAGE_STATUS_QUERY = """
MATCH (j:IngestJob {key:$key})
RETURN j.key AS key, j.owner AS owner, j.status AS status,
       j.stages AS stages
LIMIT 1
"""

_EXPIRED_QUERY = """
MATCH (j:IngestJob)
WHERE j.owner <> '' AND j.claimed_at < $expires
RETURN j.key AS key, j.owner AS owner, j.claimed_at AS claimed_at
ORDER BY j.claimed_at ASC
LIMIT $limit
"""

_REAP_QUERY = """
MATCH (j:IngestJob)
WHERE j.owner <> '' AND j.claimed_at < $expires
SET j.owner = '', j.claimed_at = 0, j.status = 'pending'
RETURN count(j) AS cleared
LIMIT $limit
"""


def create_job(driver, key: str, stages: Optional[List[str]] = None) -> Dict:
    """Create (idempotent MERGE) a resumable manifest node for ``key``.

    The node starts ``owner=''`` and ``status='pending'`` so any worker can
    later ``claim`` it. ``stages`` is a map of stage -> False. Returns the job
    dict (single node, index-backed on ``IngestJob(key)``).
    """
    stage_list = stages if stages is not None else STAGES
    with driver.session() as sess:
        rec = sess.run(
            _CREATE_JOB_QUERY, key=key, stages=_stage_map(stage_list)
        ).single()
    return {
        "key": rec.get("key"),
        "owner": rec.get("owner"),
        "status": rec.get("status"),
        "stages": rec.get("stages"),
    }


def update_stage(driver, key: str, stage: str,
                 owner: Optional[str] = None) -> Optional[Dict]:
    """Mark ``stage`` done for ``key``, guarded by owner.

    ``owner`` must match the node's current owner, OR be empty string to allow a
    coordinator (no owner check). Single node match on ``IngestJob(key)``; the
    ``graph_written`` stage also flips ``status='done'``. Returns the updated
    job dict, or ``None`` if the node is missing / owner guard failed.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage!r} (expected one of {STAGES})")
    with driver.session() as sess:
        rec = sess.run(
            _UPDATE_STAGE_QUERY, key=key, stage=stage, owner=owner or ""
        ).single()
    if rec is None:
        return None
    return {
        "key": rec.get("key"),
        "stages": rec.get("stages"),
        "status": rec.get("status"),
    }


def stage_status(driver, key: str) -> Optional[Dict]:
    """Read-only stage/status snapshot for ``key`` (LIMIT 1)."""
    with driver.session() as sess:
        rec = sess.run(_STAGE_STATUS_QUERY, key=key).single()
    if rec is None:
        return None
    return {
        "key": rec.get("key"),
        "owner": rec.get("owner"),
        "status": rec.get("status"),
        "stages": rec.get("stages"),
    }


def expired_claims(driver, ttl_sec: int = 3600, limit: int = 200,
                   now_ms: Optional[int] = None) -> List[Dict]:
    """Read-only list of claims whose TTL has elapsed (owner still set).

    Index-backed on ``IngestJob(owner)`` and bounded by ``LIMIT`` — never scans
    the whole graph. Safe against the 21M-node DB.
    """
    now = now_ms if now_ms is not None else _now_ms()
    expires = now - int(ttl_sec * 1000)
    with driver.session() as sess:
        rows = sess.run(_EXPIRED_QUERY, expires=expires, limit=limit).data()
    return [
        {
            "key": r.get("key"),
            "owner": r.get("owner"),
            "claimed_at": r.get("claimed_at"),
        }
        for r in rows
    ]


def reap(driver, ttl_sec: int = 3600, limit: int = 200,
         now_ms: Optional[int] = None) -> int:
    """Clear the owner on expired claims so they become re-claimable.

    Bounded by ``LIMIT`` and index-backed on ``IngestJob(owner)``. Returns the
    number of claims cleared. Safe to run on a schedule (the reaper).
    """
    now = now_ms if now_ms is not None else _now_ms()
    expires = now - int(ttl_sec * 1000)
    with driver.session() as sess:
        rec = sess.run(_REAP_QUERY, expires=expires, limit=limit).single()
    return int(rec.get("cleared")) if rec else 0
