#!/usr/bin/env python3
"""Scott Graph MCP server — unified 'what Scott said, where, and when'.

Exposes the unified (Scott:Person) activity graph (Neo4j) as MCP tools so any
agent/client can query it. Built on top of the W-57/W-56 enrichment:

  (Scott)-[:SPOKE]->(Utterance)         Utterance.text + MENTIONS->Concept
  (Scott)-[:WROTE]->(Summary)           Summary.text/bullets + DISCUSSES/RELATED_CONCEPT->Concept
  (Scott)-[:COMMUNICATED_VIA]->(PhoneLog)  PhoneLog->AT_PLACE (places incl. travel)
  (Scott)-[:DOCUMENTED]->(KgNode)       docs ABOUT->Concept

Tools:
  scott_said_about(concept, limit)      utterances Scott spoke about a concept
  scott_said_at(place, limit)           utterances at a place (resolved via contemporaneous PhoneLog)
  scott_activity_on(date, limit)        what Scott said/wrote on a date
  scott_search_said(query, limit)       lexical search of spoken text
  scott_timeline(limit)                 place history (when, where)

Run: python scott_graph_mcp.py   (stdio MCP server)
"""
from __future__ import annotations
import sys
from pathlib import Path
from datetime import date

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("scott-graph")


def _cfg():
    from auto_ingest_config import get_neo4j_config
    return get_neo4j_config()


def _driver():
    import neo4j
    cfg = _cfg()
    return neo4j.GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"])), cfg.get("db")


@mcp.tool()
def scott_said_about(concept: str, limit: int = 20) -> str:
    """What did Scott SPEAK (utterances) about a research/skill concept?"""
    limit = min(max(limit, 1), 100)
    drv, db = _driver()
    try:
        rows = drv.session(database=db).run(
            """
            MATCH (s:Person {name:'Scott'})-[:SPOKE]->(u:Utterance)-[m:MENTIONS]->(c:Concept)
            WHERE c.name CONTAINS $c
            RETURN u.text AS text, u.created_at AS when, round(m.score,3) AS score
            ORDER BY m.score DESC LIMIT $lim
            """, c=concept, lim=limit,
        ).data()
    finally:
        drv.close()
    if not rows:
        return f"Scott did not speak about '{concept}' (in the linked concept graph)."
    out = [f"Scott spoke about '{concept}' ({len(rows)} hits):"]
    for r in rows:
        when = r["when"].isoformat() if r.get("when") else "?"
        out.append(f"  [{when}] (sim {r['score']}) {r['text'][:240]}")
    return "\n".join(out)


@mcp.tool()
def scott_said_at(place: str, limit: int = 20) -> str:
    """What did Scott SAY at a place? Resolves utterance time -> contemporaneous
    PhoneLog -> place (within 1h). Works even before place edges are materialized."""
    limit = min(max(limit, 1), 100)
    drv, db = _driver()
    try:
        rows = drv.session(database=db).run(
            """
            MATCH (s:Person {name:'Scott'})-[:SPOKE]->(u:Utterance)
            MATCH (p:PhoneLog)-[:AT_PLACE]->(pl:SummaryPlace)
            WHERE pl.name CONTAINS $place
              AND p.epoch_millis >= datetime(u.created_at).epochMillis - 3600000
              AND p.epoch_millis <= datetime(u.created_at).epochMillis + 3600000
            WITH u, pl, abs(p.epoch_millis - datetime(u.created_at).epochMillis) AS d
            ORDER BY d
            RETURN DISTINCT u.text AS text, u.created_at AS when, pl.name AS place
            LIMIT $lim
            """, place=place, lim=limit,
        ).data()
    finally:
        drv.close()
    if not rows:
        return f"No utterances resolved to a place matching '{place}'."
    out = [f"Scott said at '{place}' ({len(rows)} hits):"]
    for r in rows:
        when = r["when"].isoformat() if r.get("when") else "?"
        out.append(f"  [{when}] {r['text'][:240]}")
    return "\n".join(out)


@mcp.tool()
def scott_activity_on(date_str: str, limit: int = 20) -> str:
    """What did Scott say/write on a given date (YYYY-MM-DD)?"""
    limit = min(max(limit, 1), 100)
    drv, db = _driver()
    try:
        rows = drv.session(database=db).run(
            """
            MATCH (s:Person {name:'Scott'})-[:SPOKE]->(u:Utterance)
            WHERE date(u.created_at) = date($d)
            RETURN u.text AS text, u.created_at AS when
            ORDER BY u.created_at LIMIT $lim
            """, d=date_str, lim=limit,
        ).data()
        sums = drv.session(database=db).run(
            """
            MATCH (s:Person {name:'Scott'})-[:WROTE]->(sm:Summary)
            WHERE date(sm.created_at) = date($d)
            RETURN sm.text AS text, sm.created_at AS when
            ORDER BY sm.created_at LIMIT $lim
            """, d=date_str, lim=limit,
        ).data()
    finally:
        drv.close()
    if not rows and not sums:
        return f"No activity found for {date_str}."
    out = [f"Scott on {date_str}:"]
    out.append(f"  SPOKE ({len(rows)}):")
    for r in rows[:limit]:
        when = r["when"].isoformat() if r.get("when") else "?"
        out.append(f"    [{when}] {r['text'][:200]}")
    out.append(f"  WROTE ({len(sums)}):")
    for r in sums[:limit]:
        when = r["when"].isoformat() if r.get("when") else "?"
        out.append(f"    [{when}] {r['text'][:200]}")
    return "\n".join(out)


@mcp.tool()
def scott_search_said(query: str, limit: int = 20) -> str:
    """Lexical search of everything Scott SPOKE (utterance text)."""
    limit = min(max(limit, 1), 100)
    drv, db = _driver()
    try:
        rows = drv.session(database=db).run(
            """
            MATCH (s:Person {name:'Scott'})-[:SPOKE]->(u:Utterance)
            WHERE u.text CONTAINS $q
            RETURN u.text AS text, u.created_at AS when
            ORDER BY u.created_at DESC LIMIT $lim
            """, q=query, lim=limit,
        ).data()
    finally:
        drv.close()
    if not rows:
        return f"No spoken utterances match '{query}'."
    out = [f"Scott said (matching '{query}'): {len(rows)} hits"]
    for r in rows:
        when = r["when"].isoformat() if r.get("when") else "?"
        out.append(f"  [{when}] {r['text'][:240]}")
    return "\n".join(out)


@mcp.tool()
def scott_timeline(limit: int = 30) -> str:
    """Scott's place history: where, and how many activities tied to each."""
    limit = min(max(limit, 1), 200)
    drv, db = _driver()
    try:
        rows = drv.session(database=db).run(
            """
            MATCH (s:Person {name:'Scott'})-[:COMMUNICATED_VIA]->(p:PhoneLog)-[:AT_PLACE]->(pl:SummaryPlace)
            RETURN pl.name AS place, pl.place_role AS role, count(p) AS pings
            ORDER BY pings DESC LIMIT $lim
            """, lim=limit,
        ).data()
    finally:
        drv.close()
    if not rows:
        return "No place history yet (PhoneLog->AT_PLACE may still be running)."
    out = ["Scott's place history (by PhoneLog pings):"]
    for r in rows:
        out.append(f"  {r['place']} [{r['role']}]: {r['pings']:,} pings")
    return "\n".join(out)


if __name__ == "__main__":
    mcp.run(transport="stdio")
