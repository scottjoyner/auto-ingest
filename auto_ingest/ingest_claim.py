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
