"""Curate research content from Neo4j into a :class:`Brief`.

The knowledge graph holds Papers, Concepts, Topics and VaultDocuments but is NOT
linked to spoken content. This module turns a Topic (e.g. ``large_language_models``)
into a tight, citable brief that downstream planning turns into shorts.

Two stages:
  1. ``curate_topic`` — pure Neo4j read: top papers, concepts, chunks for a topic.
  2. ``synthesize_brief`` — LLM (Ollama) turns the curated facts into a hook +
     bullet points + source list. Kept separate so curation is testable without
     an LLM and so iteration can swap the prompt/model.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from auto_ingest.shorts.models import Brief, SourceRef

log = logging.getLogger("shorts.curator")

DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("SHORTS_MODEL", "gemma3:4b")


@dataclass
class CuratedFacts:
    """Raw, query-shaped research facts for a topic (pre-LLM)."""

    topic: str
    topic_title: str
    papers: List[Dict[str, Any]] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)
    chunks: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------
# Neo4j curation
# ---------------------------
def curate_topic(driver, topic: str, *, top_papers: int = 6, top_concepts: int = 12,
                 chunk_per_paper: int = 1) -> CuratedFacts:
    """Read top papers/concepts/chunks for a Topic from Neo4j."""
    with driver.session() as sess:
        title_rec = sess.run(
            "MATCH (t:Topic {name:$topic}) RETURN coalesce(t.title, t.name) AS title",
            topic=topic,
        ).single()
        topic_title = title_rec["title"] if title_rec else topic

        papers = [
            dict(r)
            for r in sess.run(
                """
                MATCH (p:Paper)-[:BELONGS_TO_TOPIC]->(t:Topic {name:$topic})
                OPTIONAL MATCH (p)-[:HAS_KEYWORD]->(k:Keyword)
                WITH p, collect(DISTINCT k.name)[..5] AS kws
                RETURN p.arxiv_id AS id, p.title AS title, p.published AS year,
                       p.abs_url AS url, p.keyword_count AS cites, kws
                ORDER BY p.published DESC, p.keyword_count DESC
                LIMIT $n
                """,
                topic=topic, n=top_papers,
            )
        ]

        concepts = [
            r["name"]
            for r in sess.run(
                """
                MATCH (p:Paper)-[:BELONGS_TO_TOPIC]->(t:Topic {name:$topic})
                MATCH (p)-[:HAS_CONCEPT]->(c:Concept)
                WITH c, count(p) AS support
                RETURN c.name AS name ORDER BY support DESC LIMIT $n
                """,
                topic=topic, n=top_concepts,
            )
        ]

        chunks: List[Dict[str, Any]] = []
        for p in papers:
            for r in sess.run(
                """
                MATCH (p:Paper {arxiv_id:$pid})-[:HAS_CHUNK]->(ch:Chunk)
                WHERE ch.text IS NOT NULL AND size(ch.text) > 40
                RETURN ch.id AS id, ch.text AS text
                LIMIT $n
                """,
                pid=p.get("id") or p.get("arxiv_id"), n=chunk_per_paper,
            ):
                chunks.append(dict(r))

    return CuratedFacts(
        topic=topic, topic_title=topic_title,
        papers=papers, concepts=concepts, chunks=chunks,
    )


# ---------------------------
# LLM synthesis
# ---------------------------
_SYSTEM = (
    "You are a concise science-communicator for short-form video. Given curated "
    "research facts about ONE topic, write a tight brief: a one-sentence hook and "
    "3-5 punchy points a curious viewer would find surprising. Cite sources by "
    "their [n] index. Output STRICT JSON: "
    '{"title": str, "hook": str, "points": [str], "tags": [str]}.'
)


def _build_prompt(facts: CuratedFacts) -> str:
    lines = [f"Topic: {facts.topic_title} ({facts.topic})", ""]
    lines.append("Sources (indexed):")
    for i, p in enumerate(facts.papers, 1):
        yr = f" ({p['year']})" if p.get("year") else ""
        lines.append(f"  [{i}] {p.get('title', 'Untitled')}{yr}")
    if facts.concepts:
        lines.append("")
        lines.append("Key concepts: " + ", ".join(facts.concepts[:12]))
    if facts.chunks:
        lines.append("")
        lines.append("Notable excerpts:")
        for ch in facts.chunks[:4]:
            txt = (ch.get("text") or "").strip().replace("\n", " ")
            lines.append(f"  - {txt[:280]}")
    lines.append("")
    lines.append("Write the JSON brief now.")
    return "\n".join(lines)


def _extract_json(text: str) -> Dict[str, Any]:
    import json as _json
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object in LLM output:\n{text[:400]}")
    return _json.loads(text[start : end + 1])


def synthesize_brief(facts: CuratedFacts, *, client=None, model: str = DEFAULT_OLLAMA_MODEL,
                     ollama_url: str = DEFAULT_OLLAMA_URL) -> Brief:
    """Turn curated facts into a :class:`Brief` via Ollama.

    ``client`` may be an ``OllamaClient`` (from auto_ingest.content.build_summaries)
    or None to construct one from ``ollama_url``.
    """
    if client is None:
        from auto_ingest.content.build_summaries import OllamaClient
        client = OllamaClient(ollama_url)

    prompt = _build_prompt(facts)
    raw = client.generate(model, prompt)
    data = _extract_json(raw)

    sources = [
        SourceRef(
            kind="paper",
            ref_id=str(p.get("id", "")),
            title=str(p.get("title", "Untitled")),
            url=p.get("url"),
            year=p.get("year"),
        )
        for p in facts.papers
    ]

    brief = Brief(
        topic=facts.topic,
        title=str(data.get("title", facts.topic_title)),
        hook=str(data.get("hook", "")),
        points=[str(x) for x in data.get("points", [])],
        sources=sources,
        tags=[str(x) for x in data.get("tags", [])],
    )
    log.info("Synthesized brief for %s: %d points, %d sources",
             facts.topic, len(brief.points), len(brief.sources))
    return brief


def curate_brief(driver, topic: str, *, client=None, model: str = DEFAULT_OLLAMA_MODEL,
                 ollama_url: str = DEFAULT_OLLAMA_URL, **curate_kwargs) -> Brief:
    """Convenience: curate facts then synthesize a brief in one call."""
    facts = curate_topic(driver, topic, **curate_kwargs)
    return synthesize_brief(facts, client=client, model=model, ollama_url=ollama_url)


@dataclass
class DiscussionClip:
    """A spoken utterance (with its transcription + time anchor) that MENTIONS a
    concept belonging to ``topic`` — the basis for 'your own discussion' shorts."""

    utterance_id: str
    text: str
    transcription_key: str
    start_ms: Optional[int]
    concept: str
    score: float


def discusses_topic(driver, topic: str, *, min_score: float = 0.65,
                    limit: int = 40) -> List[DiscussionClip]:
    """Find spoken utterances that discuss a research Topic.

    Uses the Utterance-[:MENTIONS]->Concept edges (populated by
    scripts/link_utterances_to_concepts.py) joined to the Topic via
    Paper-[:BELONGS_TO_TOPIC]->Topic so a viewer's *own words* about LLMs/agents
    can be curated, not just the papers.
    """
    with driver.session() as sess:
        rows = sess.run(
            """
            MATCH (t:Topic {name:$topic})<-[:BELONGS_TO_TOPIC]-(p:Paper)
                  -[:HAS_CONCEPT]->(c:Concept)<-[m:MENTIONS]-(u:Utterance)
            MATCH (u)<-[:HAS_UTTERANCE]-(tr:Transcription)
            WHERE m.score >= $min
            RETURN u.id AS uid, u.text AS text, tr.key AS tkey,
                   tr.started_at AS started, c.name AS concept, m.score AS score
            ORDER BY m.score DESC
            LIMIT $limit
            """,
            topic=topic, min=min_score, limit=limit,
        ).data()
    return [
        DiscussionClip(
            utterance_id=r["uid"], text=r["text"] or "",
            transcription_key=r["tkey"] or "", start_ms=None,
            concept=r["concept"], score=float(r["score"]),
        )
        for r in rows
    ]
