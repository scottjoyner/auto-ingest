#!/usr/bin/env python3
"""Scott Graph MCP server — unified 'what Scott said, where, and when'.

DUMMY-PROOF DESIGN (for small/weak models):
  * Every tool is wrapped so it can NEVER throw — errors become plain-text replies.
  * Tool docstrings are plain English with clear "USE THIS WHEN..." guidance + examples.
  * All inputs are simple strings with safe defaults; dates accept YYYY-MM-DD.
  * Outputs are short, scannable plain text (hard-capped) a small model can read.
  * A `scott_help` tool lets a model discover the right tool without guessing.

Graph: (Scott:Person)-[:SPOKE]->(Utterance) etc., built by enrich_scott_unify.py.
Run: python scott_graph_mcp.py   (stdio MCP server)
"""
from __future__ import annotations
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("scott-graph")


# ---------------------------------------------------------------------------
# Safe DB helpers — a weak model must never see a stack trace.
# ---------------------------------------------------------------------------
def _driver():
    import neo4j
    from auto_ingest_config import get_neo4j_config
    cfg = get_neo4j_config()
    return neo4j.GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"])), cfg.get("db")


def _safe(fn):
    """Wrap a tool body: catch ALL errors, return a helpful plain-text string."""
    import functools
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - intentionally swallow for dumb models
            return (f"[scott-graph error] Could not answer. Reason: {type(e).__name__}: "
                    f"{str(e)[:200]}\nTry a simpler query, or call scott_help.")
    return wrapped


def _cap(n, lo=1, hi=100):
    try:
        return int(max(lo, min(hi, int(n))))
    except Exception:
        return 20


def _norm_date(s: str) -> str:
    """Accept '2026-06-14' or 'June 14 2026' etc. Best-effort -> YYYY-MM-DD."""
    s = (s or "").strip()
    if not s:
        return ""
    # Already ISO
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    # Try month-name parsing
    try:
        from datetime import datetime
        for fmt in ("%B %d %Y", "%b %d %Y", "%B %d, %Y", "%b %d, %Y", "%m/%d/%Y", "%Y/%m/%d"):
            try:
                return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
    except Exception:
        pass
    return ""  # unparseable -> caller shows friendly "give a date" message


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@mcp.tool()
@_safe
def scott_help() -> str:
    """USE THIS FIRST if you are unsure which tool to call. Lists the available
    tools in plain English with examples. Call scott_help when you don't know
    what to do, or before answering a question about Scott."""
    return (
        "SCOTT GRAPH — what Scott said, where, and when.\n"
        "Available tools (call ONE that matches the question):\n"
        "1) scott_said_about(concept) — 'What did Scott say about X?'\n"
        "     example: scott_said_about('audio')\n"
        "2) scott_said_at(place) — 'What did Scott say in/at X?'\n"
        "     example: scott_said_at('Park City')\n"
        "3) scott_activity_on(date) — 'What did Scott do on DATE?'\n"
        "     example: scott_activity_on('2026-06-14')\n"
        "4) scott_search_said(words) — 'Find when Scott said these words'\n"
        "     example: scott_search_said('the call')\n"
        "5) scott_timeline() — 'Where has Scott been? List his places.'\n"
        "6) scott_help() — show this list again.\n"
        "All inputs are simple words or a date like 2026-06-14. Keep limit small (5-20)."
    )


@mcp.tool()
@_safe
def scott_said_about(concept: str, limit: int = 10) -> str:
    """USE WHEN: someone asks what Scott SPOKE or SAID about a topic, subject,
    skill, or concept (e.g. 'audio', 'neo4j', 'skiing').
    Inputs: concept = the topic word(s); limit = max results (default 10, max 100).
    Example: scott_said_about('audio')"""
    concept = (concept or "").strip()
    if not concept:
        return "Please give a topic, e.g. scott_said_about('audio')."
    limit = _cap(limit)
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
        return f"Scott did not speak about '{concept}' in the linked graph."
    out = [f"Scott spoke about '{concept}' — {len(rows)} result(s):"]
    for r in rows:
        when = r["when"].isoformat()[:19] if r.get("when") else "unknown date"
        out.append(f"- [{when}] {r['text'][:300]}")
    return "\n".join(out)


@mcp.tool()
@_safe
def scott_said_at(place: str, limit: int = 10) -> str:
    """USE WHEN: someone asks what Scott SAID at a location/place (e.g. 'Park City',
    'Boone', 'South End', 'Denver'). Finds utterances whose time matches a PhoneLog
    recorded at that place.
    Inputs: place = place name word(s); limit = max results (default 10, max 100).
    Example: scott_said_at('Park City')"""
    place = (place or "").strip()
    if not place:
        return "Please give a place, e.g. scott_said_at('Boone')."
    limit = _cap(limit)
    drv, db = _driver()
    try:
        # Primary: use materialized Utterance->AT_PLACE edges (Phase 2, DST-correct).
        rows = drv.session(database=db).run(
            """
            MATCH (s:Person {name:'Scott'})-[:SPOKE]->(u:Utterance)-[:AT_PLACE]->(pl:SummaryPlace)
            WHERE pl.name CONTAINS $place
            RETURN u.text AS text, u.created_at AS when, pl.name AS place
            LIMIT $lim
            """, place=place, lim=limit,
        ).data()
        # Fallback (utterances with no place edge yet): time-resolved join via PhoneLog.
        if not rows:
            rows = drv.session(database=db).run(
                """
                MATCH (s:Person {name:'Scott'})-[:SPOKE]->(u:Utterance)
                  WHERE NOT (u)-[:AT_PLACE]->()
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
    out = [f"What Scott said at '{place}' — {len(rows)} result(s):"]
    for r in rows:
        when = r["when"].isoformat()[:19] if r.get("when") else "unknown date"
        out.append(f"- [{when}] {r['text'][:300]}")
    return "\n".join(out)


@mcp.tool()
@_safe
def scott_activity_on(date_str: str, limit: int = 10) -> str:
    """USE WHEN: someone asks what Scott did / said / wrote ON a specific date.
    Inputs: date_str = a date like '2026-06-14' (also accepts 'June 14 2026');
    limit = max results (default 10, max 100).
    Example: scott_activity_on('2026-06-14')"""
    d = _norm_date(date_str)
    if not d:
        return "Please give a date, e.g. scott_activity_on('2026-06-14')."
    limit = _cap(limit)
    drv, db = _driver()
    try:
        spoke = drv.session(database=db).run(
            """
            MATCH (s:Person {name:'Scott'})-[:SPOKE]->(u:Utterance)
            WITH u, substring(toString(u.created_at), 0, 10) AS ud
            WHERE ud = $d
            RETURN u.text AS text, u.created_at AS when
            ORDER BY u.created_at LIMIT $lim
            """, d=d, lim=limit,
        ).data()
        wrote = drv.session(database=db).run(
            """
            MATCH (s:Person {name:'Scott'})-[:WROTE]->(sm:Summary)
            WITH sm, substring(toString(sm.created_at), 0, 10) AS sd
            WHERE sd = $d
            RETURN sm.text AS text, sm.created_at AS when
            ORDER BY sm.created_at LIMIT $lim
            """, d=d, lim=limit,
        ).data()
    finally:
        drv.close()
    if not spoke and not wrote:
        return f"No activity found for {d}."
    out = [f"Scott on {d}:"]
    out.append(f"SAID ({len(spoke)}):")
    for r in spoke:
        when = r["when"].isoformat()[:19] if r.get("when") else "?"
        out.append(f"- [{when}] {r['text'][:300]}")
    out.append(f"WROTE ({len(wrote)}):")
    for r in wrote:
        when = r["when"].isoformat()[:19] if r.get("when") else "?"
        out.append(f"- [{when}] {r['text'][:300]}")
    return "\n".join(out)


@mcp.tool()
@_safe
def scott_search_said(query: str, limit: int = 10) -> str:
    """USE WHEN: you want to FIND a specific thing Scott said by searching his
    spoken words (e.g. 'the call', 'Junior', 'ski').
    Inputs: query = the words to find; limit = max results (default 10, max 100).
    Example: scott_search_said('the call')"""
    query = (query or "").strip()
    if not query:
        return "Please give words to search, e.g. scott_search_said('ski')."
    limit = _cap(limit)
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
    out = [f"Scott said (matching '{query}') — {len(rows)} result(s):"]
    for r in rows:
        when = r["when"].isoformat()[:19] if r.get("when") else "unknown date"
        out.append(f"- [{when}] {r['text'][:300]}")
    return "\n".join(out)


@mcp.tool()
@_safe
def scott_timeline(limit: int = 15) -> str:
    """USE WHEN: someone asks WHERE Scott has been, or for his list of places /
    travel history. No inputs needed (limit optional, default 15, max 200).
    Example: scott_timeline()"""
    limit = _cap(limit, lo=1, hi=200)
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
    out = ["Scott's places (by how much time spent):"]
    for r in rows:
        role = r.get("role") or "place"
        out.append(f"- {r['place']} [{role}]: {r['pings']:,} pings")
    return "\n".join(out)


if __name__ == "__main__":
    mcp.run(transport="stdio")
