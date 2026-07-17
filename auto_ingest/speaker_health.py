"""
auto_ingest.speaker_health — read-only health checks for the Scott ("me") anchor.

The canonical "Scott" identity in the knowledge graph is a single GlobalSpeaker
with ``is_me=true`` (person_id='scott'). Past linker runs sometimes fragmented
that identity into multiple ``is_me=true`` GlobalSpeakers; this module provides
an importable, read-only query helper so both a CLI script and tests can assert
exactly ONE such node exists.
"""
from __future__ import annotations

from neo4j import GraphDatabase

try:
    from auto_ingest_config import get_neo4j_env
    _NEO4J_URI, _NEO4J_USER, _NEO4J_PASSWORD, _NEO4J_DB = get_neo4j_env()
except Exception:  # pragma: no cover - import-fallback for tests / packaging
    import os

    _NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    _NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    _NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD") or os.getenv(
        "NEO4J_PASSWORD_DEFAULT", "knowledge_graph_2026"
    )
    _NEO4J_DB = os.getenv("NEO4J_DB", "neo4j")


_COUNT_IS_ME_QUERY = "MATCH (g:GlobalSpeaker{is_me:true}) RETURN count(g) AS n"


def count_is_me_global_speakers(driver=None) -> int:
    """Return the number of ``GlobalSpeaker`` nodes with ``is_me=true``.

    Read-only. If ``driver`` is provided it is used as-is (and NOT closed);
    otherwise a short-lived driver is created and closed. Safe to call from
    tests with a monkeypatched driver.
    """
    own_driver = driver is None
    drv = driver or GraphDatabase.driver(
        _NEO4J_URI, auth=(_NEO4J_USER, _NEO4J_PASSWORD)
    )
    try:
        with drv.session(database=_NEO4J_DB) as sess:
            rec = sess.run(_COUNT_IS_ME_QUERY).single()
            return int(rec["n"]) if rec else 0
    finally:
        if own_driver:
            drv.close()


def check_anchor_health(expected: int = 1) -> int:
    """Assert exactly ``expected`` ``is_me`` GlobalSpeaker(s) exist.

    Returns the actual count. Raises ``AssertionError`` when the count differs
    from ``expected`` so the CLI can exit non-zero.
    """
    count = count_is_me_global_speakers()
    assert count == expected, (
        f"Expected exactly {expected} 'is_me' GlobalSpeaker node, "
        f"found {count}."
    )
    return count
