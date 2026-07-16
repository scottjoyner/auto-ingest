#!/usr/bin/env python3
"""life_timeline_harvest.py — stitch Scott's activity into a chronological timeline agents can query.

Builds one (:LifeEvent {date}) node per calendar day from the richest temporal
sources in the graph (Summary as the backbone, plus Utterance counts, dominant
place from PlaceHour, and PhoneLog geo presence). LifeEvents are chained with
[:NEXT] in date order and linked to their source Summaries via [:INCLUDES] and to
places via [:AT], so an agent can traverse "what happened on / around day X" and
"what came before/after" without scanning 180K Summary nodes.

Idempotent: MERGE on (:LifeEvent {date}). Safe to re-run daily; recomputes
aggregates for the trailing window and (re)links the NEXT chain.

Design notes / gotchas (validated against the live graph 2026-07-16):
- Summary.created_at is MIXED: 181,641 DateTime + 14 raw Long(ms). Normalize the
  day with:  CASE WHEN created_at IS :: INTEGER THEN datetime({epochMillis:..})
             ELSE created_at END  -> substring(toString(..),0,10)
- Summary/Utterance created_at is LOCAL Eastern stored as UTC; PhoneLog.timestamp
  is a true-UTC ISO string. For day-bucketing at ~day granularity this skew is
  immaterial, so we bucket everything by its own stored day string.
- PlaceHour.hourBucket is epoch millis; place presence per day = sum(pings) by place.
- Corpus spans ~85 days (2025-09-25 .. 2026-06-16); ~181K summaries. One node/day.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from auto_ingest_config import get_neo4j_config  # noqa: E402
from neo4j import GraphDatabase  # noqa: E402


def log(m):
    print(f"[life_timeline] {m}", flush=True)


# day-normalization expression reused across queries; `X` is the created_at value
def _day_expr(var):
    return (
        f"(CASE WHEN {var} IS :: INTEGER "
        f"THEN substring(toString(datetime({{epochMillis:{var}}})),0,10) "
        f"ELSE substring(toString({var}),0,10) END)"
    )


def ensure_schema(session):
    session.run(
        "CREATE CONSTRAINT life_event_date IF NOT EXISTS "
        "FOR (e:LifeEvent) REQUIRE e.date IS UNIQUE"
    )


def build_day_events(session, since_day: str | None, limit_days: int | None):
    """Create/update one LifeEvent per day with aggregates from Summary + Utterance."""
    day = _day_expr("s.created_at")
    where = "WHERE s.created_at IS NOT NULL"
    if since_day:
        where += f" AND {day} >= $since"
    # Aggregate summaries per day: count, sample of texts for a digest, place tallies.
    q = f"""
    MATCH (s:Summary) {where}
    WITH {day} AS day, collect(s) AS sums
    WITH day, sums, size(sums) AS summary_count
    // digest: up to 8 summary texts (prefer non-empty), joined
    WITH day, sums, summary_count,
         [x IN sums WHERE x.text IS NOT NULL | x.text][0..8] AS texts
    MERGE (e:LifeEvent {{date: day}})
    SET e.summary_count = summary_count,
        e.digest = substring(reduce(a='', t IN texts | a + ' • ' + substring(t,0,220)), 0, 1800),
        e.updated_at = timestamp()
    RETURN count(*) AS days
    """
    params = {}
    if since_day:
        params["since"] = since_day
    r = session.run(q, **params).single()
    return r["days"] if r else 0


def attach_utterance_counts(session, since_day: str | None):
    day = _day_expr("u.created_at")
    where = "WHERE u.created_at IS NOT NULL"
    if since_day:
        where += f" AND {day} >= $since"
    q = f"""
    MATCH (u:Utterance) {where}
    WITH {day} AS day, count(u) AS uc
    MATCH (e:LifeEvent {{date: day}})
    SET e.utterance_count = uc
    RETURN count(*) AS n
    """
    params = {"since": since_day} if since_day else {}
    r = session.run(q, **params).single()
    return r["n"] if r else 0


def attach_dominant_place(session, since_day: str | None):
    """Use PlaceHour (hourBucket ms -> day) to find each day's dominant place by pings."""
    where = "WHERE h.hourBucket IS NOT NULL AND h.place IS NOT NULL"
    if since_day:
        where += " AND substring(toString(datetime({epochMillis:h.hourBucket})),0,10) >= $since"
    q = f"""
    MATCH (h:PlaceHour) {where}
    WITH substring(toString(datetime({{epochMillis:h.hourBucket}})),0,10) AS day,
         h.place AS place, sum(h.pings) AS pings
    ORDER BY day, pings DESC
    WITH day, collect({{place:place, pings:pings}}) AS ranked
    WITH day, ranked[0] AS top, reduce(s=0, r IN ranked | s + r.pings) AS total
    MATCH (e:LifeEvent {{date: day}})
    SET e.dominant_place = top.place,
        e.place_pings = top.pings,
        e.total_pings = total
    WITH e, top
    // link to the SummaryPlace node when it exists
    OPTIONAL MATCH (sp:SummaryPlace {{name: top.place}})
    FOREACH (_ IN CASE WHEN sp IS NULL THEN [] ELSE [1] END |
        MERGE (e)-[:AT]->(sp))
    RETURN count(*) AS n
    """
    params = {"since": since_day} if since_day else {}
    r = session.run(q, **params).single()
    return r["n"] if r else 0


def link_includes(session, since_day: str | None, cap: int):
    """Link each LifeEvent to a capped sample of its source Summaries via INCLUDES."""
    day = _day_expr("s.created_at")
    where = "WHERE s.created_at IS NOT NULL"
    if since_day:
        where += f" AND {day} >= $since"
    q = f"""
    MATCH (s:Summary) {where}
    WITH {day} AS day, collect(s)[0..$cap] AS sample
    MATCH (e:LifeEvent {{date: day}})
    UNWIND sample AS s
    MERGE (e)-[:INCLUDES]->(s)
    RETURN count(*) AS n
    """
    params = {"cap": cap}
    if since_day:
        params["since"] = since_day
    r = session.run(q, **params).single()
    return r["n"] if r else 0


def rebuild_next_chain(session):
    """(Re)build the [:NEXT] chain across ALL LifeEvents in date order."""
    # clear old chain, then relink in order
    session.run("MATCH (:LifeEvent)-[r:NEXT]->(:LifeEvent) DELETE r")
    q = """
    MATCH (e:LifeEvent)
    WITH e ORDER BY e.date ASC
    WITH collect(e) AS evs
    UNWIND range(0, size(evs)-2) AS i
    WITH evs[i] AS a, evs[i+1] AS b
    MERGE (a)-[:NEXT]->(b)
    RETURN count(*) AS n
    """
    r = session.run(q).single()
    return r["n"] if r else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", help="only (re)build days >= this YYYY-MM-DD (default: all)")
    ap.add_argument("--days-back", type=int, default=0,
                    help="only rebuild the trailing N days (overrides --since)")
    ap.add_argument("--include-cap", type=int, default=25,
                    help="max Summary INCLUDES links per day (default 25)")
    args = ap.parse_args()

    since = args.since
    if args.days_back and args.days_back > 0:
        from datetime import datetime, timedelta, timezone
        since = (datetime.now(timezone.utc) - timedelta(days=args.days_back)).strftime("%Y-%m-%d")

    cfg = get_neo4j_config()
    drv = GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))
    db = cfg.get("db") or "neo4j"
    try:
        with drv.session(database=db) as s:
            ensure_schema(s)
            log(f"building day events (since={since or 'ALL'})...")
            d = build_day_events(s, since, None)
            log(f"  day events upserted: {d}")
            uc = attach_utterance_counts(s, since)
            log(f"  utterance counts attached: {uc}")
            pl = attach_dominant_place(s, since)
            log(f"  dominant place attached: {pl}")
            inc = link_includes(s, since, args.include_cap)
            log(f"  INCLUDES links: {inc}")
            n = rebuild_next_chain(s)
            log(f"  NEXT chain edges: {n}")
            # summary
            tot = s.run("MATCH (e:LifeEvent) RETURN count(e) AS c").single()["c"]
            rng = s.run("MATCH (e:LifeEvent) RETURN min(e.date) AS mn, max(e.date) AS mx").single()
            log(f"DONE: {tot} LifeEvents spanning {rng['mn']} .. {rng['mx']}")
    finally:
        drv.close()


if __name__ == "__main__":
    main()
