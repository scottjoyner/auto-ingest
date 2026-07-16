#!/usr/bin/env python3
"""curious_agent.py — idle query system for the Scott knowledge graph.

Runs on a schedule. Performs a few interesting read-only Cypher queries against
Neo4j, builds a short human-readable digest, and posts it to Signal (via the
existing storage-kg-watchdog delivery path: a JSON line on stdout that the
Hermes cron layer delivers). Also monitors graph FRESHNESS and warns if the
graph hasn't received new ingests in N days.

Outputs a JSON object to stdout (the Hermes cron 'local' deliver consumes it),
and a plain-text digest. Designed to be a no_agent cron job.

Queries (all read-only, safe):
  - New places detected since last run (place_name added recently)
  - Top research themes at HOME this period
  - Research bridge growth (new RELATED_CONCEPT edges)
  - Freshness: max ingested_at / created_at across key labels
  - Stale-ingest alert if freshest record > STALE_DAYS old

Usage:
  python curious_agent.py
  python curious_agent.py --check-only   # just freshness, no insight digest
"""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

STALE_DAYS = 7  # alert if no new ingest in this many days


def log(*a):
    print("[curious_agent]", *a, file=sys.stderr, flush=True)


def get_neo4j():
    from neo4j import GraphDatabase
    from auto_ingest_config import get_neo4j_config
    cfg = get_neo4j_config()
    drv = GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))
    return drv, cfg.get("db") or "neo4j"


def q(drv, db, cypher, params=None):
    with drv.session(database=db) as s:
        return s.run(cypher, params or {}).data()


def freshness(drv, db):
    rows = q(drv, db,
        "MATCH (n) WHERE n.ingested_at IS NOT NULL "
        "RETURN max(n.ingested_at) AS newest")
    newest = rows[0]["newest"] if rows else None
    if not newest:
        return {"newest": None, "stale": True, "days": None}
    # newest may be datetime or string
    try:
        if hasattr(newest, "to_native"):
            dt = newest.to_native()
        elif isinstance(newest, str):
            dt = datetime.fromisoformat(newest.replace("Z", "+00:00"))
        else:
            dt = newest
        age = (datetime.now(timezone.utc) - dt).days
    except Exception:
        age = None
    return {"newest": str(newest), "stale": (age is not None and age > STALE_DAYS),
            "days": age}


def insights(drv, db):
    out = {}
    # new places since ~last 7 days
    try:
        out["new_places"] = q(drv, db,
            "MATCH (c:SummaryLocationCluster) WHERE c.place_name IS NOT NULL "
            "AND c.geocoded_at IS NOT NULL "
            "RETURN c.place_name AS place, c.summary_count AS n "
            "ORDER BY c.summary_count DESC LIMIT 5")
    except Exception as e:
        out["new_places"] = [{"error": str(e)[:80]}]
    # top research themes at HOME
    try:
        out["home_themes"] = q(drv, db,
            "MATCH (s:Summary)-[:AT_PLACE]->(p:SummaryPlace {place_role:'home'}) "
            "MATCH (s)-[:RELATED_CONCEPT {method:'semantic'}]->(cn:Concept) "
            "RETURN cn.name AS theme, count(*) AS w ORDER BY w DESC LIMIT 6")
    except Exception as e:
        out["home_themes"] = [{"error": str(e)[:80]}]
    # research bridge growth
    try:
        out["related_concept_edges"] = q(drv, db,
            "MATCH ()-[r:RELATED_CONCEPT]->() RETURN count(r) AS c LIMIT 1")[0]["c"]
    except Exception:
        out["related_concept_edges"] = None
    # signal message count (after bridge runs)
    try:
        out["signal_messages"] = q(drv, db,
            "MATCH (m:SignalMessage) RETURN count(m) AS c LIMIT 1")[0]["c"]
    except Exception:
        out["signal_messages"] = None
    # arxiv papers
    try:
        out["papers"] = q(drv, db,
            "MATCH (p:Paper) RETURN count(p) AS c LIMIT 1")[0]["c"]
    except Exception:
        out["papers"] = None
    # similarity coverage (768-space) + SIMILAR edges
    try:
        out["papers_768"] = q(drv, db,
            "MATCH (p:Paper) WHERE p.embedding_768 IS NOT NULL RETURN count(p) AS c LIMIT 1")[0]["c"]
    except Exception:
        out["papers_768"] = None
    try:
        out["similar_edges"] = q(drv, db,
            "MATCH ()-[r:SIMILAR]->() WHERE r.method='vector' RETURN count(r) AS c LIMIT 1")[0]["c"]
    except Exception:
        out["similar_edges"] = None
    # most-connected papers (the "papers like this" insight)
    try:
        out["top_connected_papers"] = q(drv, db,
            "MATCH (p:Paper)-[:SIMILAR]->() WITH p, count(*) AS d "
            "WHERE p.title IS NOT NULL RETURN p.arxiv_id AS id, p.title AS title, d AS degree "
            "ORDER BY d DESC LIMIT 5")
    except Exception as e:
        out["top_connected_papers"] = [{"error": str(e)[:80]}]
    return out


def build_digest(fresh, ins):
    lines = ["🧠 Knowledge Graph Digest"]
    if fresh.get("newest"):
        lines.append(f"  freshest ingest: {fresh['newest']} ({fresh.get('days')}d ago)")
    else:
        lines.append("  freshest ingest: unknown")
    if fresh.get("stale"):
        lines.append(f"  ⚠️ STALE: no new ingest in >{STALE_DAYS}d — pipeline may be paused")
    if ins.get("papers"):
        lines.append(f"  arXiv papers in graph: {ins['papers']}")
    if ins.get("papers_768") is not None:
        lines.append(f"  Papers in 768-vec space: {ins['papers_768']}")
    if ins.get("similar_edges") is not None:
        lines.append(f"  Paper SIMILAR edges: {ins['similar_edges']}")
    if ins.get("signal_messages") is not None:
        lines.append(f"  Signal messages ingested: {ins['signal_messages']}")
    if ins.get("related_concept_edges") is not None:
        lines.append(f"  Research bridge edges: {ins['related_concept_edges']}")
    if ins.get("home_themes"):
        lines.append("  Top HOME research themes:")
        for t in ins["home_themes"][:5]:
            lines.append(f"    - {t.get('theme')}: {t.get('w')}")
    if ins.get("top_connected_papers"):
        pk = [p for p in ins["top_connected_papers"] if "error" not in p]
        if pk:
            lines.append("  Most-connected papers (best paper hubs):")
            for p in pk[:5]:
                title = (p.get("title") or "")[:55]
                lines.append(f"    - [{p.get('degree')}] {title}")
    if ins.get("new_places"):
        lines.append("  Top places by activity:")
        for p in ins["new_places"][:5]:
            lines.append(f"    - {p.get('place')}: {p.get('n')}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    drv, db = get_neo4j()
    try:
        fresh = freshness(drv, db)
        if args.check_only:
            print(json.dumps({"freshness": fresh}, indent=2))
            return
        ins = insights(drv, db)
    finally:
        drv.close()

    digest = build_digest(fresh, ins)
    # stdout JSON for the Hermes cron deliver layer
    print(json.dumps({
        "freshness": fresh,
        "insights": ins,
        "digest": digest,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
