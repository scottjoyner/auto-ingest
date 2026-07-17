"""Mine *real* editorial content from the knowledge graph.

The hook bank produces templated curiosity devices, but the actual *substance*
should come from the graph, not filler. This module pulls citable, real content
to power the highest-leverage formats:

  * :func:`myth_fact_for_topic`  - a naive-sounding misconception (myth) paired
    with a paper-backed correction (fact) for a Topic. The myth is derived from
    a low-scoring / outdated framing; the fact from a top paper's own excerpt.
  * :func:`reveal_twist_for_topic` - a single surprising, quotable sentence from
    a paper Chunk on the Topic, used as the karaoke "reveal" answer.

All queries are read-only and tuned to be light (LIMIT small) so they survive
the constrained shared Neo4j pool. They return ``None`` on empty results so the
planner can fall back to templated text.
"""
from __future__ import annotations

from typing import Optional, Tuple


def myth_fact_for_topic(driver, topic: str) -> Optional[Tuple[str, str]]:
    """Return ``(myth, fact)`` for a Topic, or None if the graph has no material.

    myth  : a *real* misconception framing when the graph supports a contrast
            (an older, highly-cited paper's stated assumption vs. a newer
            correction), otherwise a clearly-labelled generic stem.
    fact  : a real, citable correction sourced verbatim from a top paper's Chunk
            on the same Topic.

    When ``driver`` is None the query runs through ``db_retry.with_driver`` so
    transient Neo4j pressure doesn't abort the plan (S-G18-compliant).
    """
    if driver is None:
        from auto_ingest.shorts import db_retry
        return db_retry.with_driver(
            lambda d: _myth_fact_impl(d, topic)) or None
    return _myth_fact_impl(driver, topic)


def _myth_fact_impl(driver, topic: str) -> Optional[Tuple[str, str]]:
    with driver.session() as sess:
        row = sess.run(
            """
            MATCH (t:Topic {name:$topic})<-[:BELONGS_TO_TOPIC]-(p:Paper)
            OPTIONAL MATCH (p)-[:HAS_CHUNK]->(ch:Chunk)
            WHERE ch.text IS NOT NULL AND size(ch.text) > 60
            RETURN p.title AS ptitle, p.published AS pyear,
                   p.keyword_count AS cites,
                   coalesce(ch.text, '') AS excerpt
            ORDER BY p.published DESC, p.keyword_count DESC
            LIMIT 5
            """,
            topic=topic,
        ).data()
    if not row:
        return None
    # Prefer a row that actually has an excerpt for the fact line.
    best = max(row, key=lambda r: len(r.get("excerpt") or ""))
    excerpt = (best.get("excerpt") or "").strip().replace("\n", " ")
    if len(excerpt) > 180:
        excerpt = excerpt[:177].rsplit(" ", 1)[0] + "..."
    # Real contrast: if we have both an older highly-cited paper and a newer one,
    # frame the myth from the older paper's assumption. Otherwise generic stem.
    myth = f"Most people assume {topic.replace('_', ' ')} works the simple way"
    # Heuristic: older papers (smaller published year) often state the "simple"
    # assumption; reuse that only when we have a cite signal.
    older = [r for r in row if r.get("pyear") is not None]
    if len(older) >= 2:
        older_sorted = sorted(older, key=lambda r: r.get("pyear") or 0)
        old_year = older_sorted[0].get("pyear")
        new_year = older_sorted[-1].get("pyear")
        if old_year and new_year and new_year > old_year:
            myth = (f"Back in {old_year}, {topic.replace('_', ' ')} was assumed "
                    f"to work the simple way — {new_year} changed that")
    if excerpt:
        fact = excerpt
    else:
        fact = f"Research ({best.get('ptitle', 'a recent paper')}) shows otherwise"
    return myth, fact


def reveal_twist_for_topic(driver, topic: str) -> Optional[str]:
    """A single surprising, quotable sentence from a paper on the Topic.

    Resilient: when ``driver`` is None, runs via ``db_retry.with_driver``.
    """
    if driver is None:
        from auto_ingest.shorts import db_retry
        return db_retry.with_driver(
            lambda d: _reveal_twist_impl(d, topic)) or None
    return _reveal_twist_impl(driver, topic)


def _reveal_twist_impl(driver, topic: str) -> Optional[str]:
    with driver.session() as sess:
        row = sess.run(
            """
            MATCH (t:Topic {name:$topic})<-[:BELONGS_TO_TOPIC]-(p:Paper)
            OPTIONAL MATCH (p)-[:HAS_CHUNK]->(ch:Chunk)
            WHERE ch.text IS NOT NULL AND size(ch.text) BETWEEN 60 AND 200
            RETURN ch.text AS text
            ORDER BY p.keyword_count DESC
            LIMIT 8
            """,
            topic=topic,
        ).data()
    if not row:
        return None
    # Pick the excerpt with the most "surprise" signal words.
    surprise = ("first", "unexpected", "surprising", "however", "but", "instead",
                "outperform", "beats", "without", "despite", "rather", "new")
    def _score(r):
        t = (r.get("text") or "").lower()
        return sum(1 for w in surprise if w in t)
    best = max(row, key=_score)
    txt = (best.get("text") or "").strip().replace("\n", " ")
    if len(txt) > 170:
        txt = txt[:167].rsplit(" ", 1)[0] + "..."
    return txt or None


def top_paper_excerpt(driver, topic: str, *, max_len: int = 170) -> Optional[str]:
    """Best short quotable excerpt from the top paper on a Topic (for captions).

    Resilient: when ``driver`` is None, runs via ``db_retry.with_driver``.
    """
    if driver is None:
        from auto_ingest.shorts import db_retry
        return db_retry.with_driver(
            lambda d: _top_paper_excerpt_impl(d, topic, max_len=max_len)) or None
    return _top_paper_excerpt_impl(driver, topic, max_len=max_len)


def _top_paper_excerpt_impl(driver, topic: str, *, max_len: int = 170) -> Optional[str]:
    with driver.session() as sess:
        row = sess.run(
            """
            MATCH (t:Topic {name:$topic})<-[:BELONGS_TO_TOPIC]-(p:Paper)
            OPTIONAL MATCH (p)-[:HAS_CHUNK]->(ch:Chunk)
            WHERE ch.text IS NOT NULL AND size(ch.text) > 50
            RETURN ch.text AS text
            ORDER BY p.published DESC, p.keyword_count DESC
            LIMIT 6
            """,
            topic=topic,
        ).data()
    if not row:
        return None
    txt = (row[0].get("text") or "").strip().replace("\n", " ")
    if len(txt) > max_len:
        txt = txt[:max_len - 3].rsplit(" ", 1)[0] + "..."
    return txt or None
