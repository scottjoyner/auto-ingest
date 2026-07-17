"""Resilient Neo4j access for the shorts pipeline.

Under sustained host load the shared Neo4j transaction pool OOMs
(``MemoryPoolOutOfMemoryError``) or drops the connection
(``ServiceUnavailable`` / ``SessionExpired``). These are *transient*: a fresh
driver + a backoff retry recovers. This module centralizes that pattern so
``planner`` / ``curator`` / ``content_miner`` don't each re-implement it (and
don't crash the whole plan on DB pressure).

Original pattern lived in ``scripts/run_narrated_mix.py``; this is the shared,
importable version.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional, TypeVar

from neo4j import GraphDatabase

log = logging.getLogger("shorts.db_retry")

T = TypeVar("T")

# Errors that are safe to retry (DB under memory/connection pressure).
_TRANSIENT_MARKERS = (
    "MemoryPoolOutOfMemory",
    "TransientError",
    "ServiceUnavailable",
    "SessionExpired",
)


def is_transient(e: Exception) -> bool:
    etype = type(e).__name__
    text = str(e)
    return any(m in (etype + text) for m in _TRANSIENT_MARKERS)


def retry(fn: Callable[[Any], T], *,
          attempts: int = 12, base_wait: float = 20.0,
          uri: Optional[str] = None, user: Optional[str] = None,
          password: Optional[str] = None, db: Optional[str] = None) -> Optional[T]:
    """Run ``fn(driver)`` with a fresh driver per attempt and backoff retry.

    ``fn`` receives a connected :class:`neo4j.GraphDatabase.driver` and returns
    its result. Transient errors trigger a linear backoff retry; non-transient
    errors propagate immediately (returned as ``None`` after logging). Returns
    ``None`` if all attempts fail.
    """
    last: Optional[Exception] = None
    for i in range(attempts):
        drv = None
        try:
            drv = GraphDatabase.driver(
                uri or "bolt://localhost:7687",
                auth=(user or "neo4j", password or ""),
            )
            if db:
                # Validate the database exists by opening a session (cheap).
                with drv.session(database=db) as _s:
                    pass
            return fn(drv)
        except Exception as e:  # noqa: BLE001 - we inspect + decide
            last = e
            if is_transient(e):
                wait = base_wait * (i + 1)
                log.warning("DB pressure (%s); retry %d/%d in %.0fs",
                            type(e).__name__, i + 1, attempts, wait)
                time.sleep(wait)
                continue
            log.error("Non-transient DB error: %s", e)
            return None
        finally:
            if drv is not None:
                try:
                    drv.close()
                except Exception:
                    pass
    log.error("DB operation failed after %d retries: %s", attempts, last)
    return None


def with_driver(fn: Callable[[Any], T], *,
                attempts: int = 12, base_wait: float = 20.0) -> Optional[T]:
    """``retry`` that resolves creds via ``auto_ingest_config`` (no args needed).

    Convenience wrapper for the shorts package: pulls the Neo4j URI/user/password
    from the standard config loader so callers just pass a function of the driver.
    """
    import os

    from auto_ingest_config import (  # local import avoids heavy dep at module load
        get_neo4j_password,
    )
    return retry(
        fn, attempts=attempts, base_wait=base_wait,
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=get_neo4j_password(),
        db=os.getenv("NEO4J_DB", "neo4j"),
    )
