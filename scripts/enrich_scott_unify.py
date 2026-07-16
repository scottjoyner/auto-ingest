#!/usr/bin/env python3
"""Unify all 'Scott' activity streams under a single (Scott:Person) node.

Phase 1 (always): MERGE (Scott:Person {name:'Scott'}) and link every activity stream:
  (Scott)-[:SPOKE]->(Utterance)
  (Scott)-[:WROTE]->(Summary)
  (Scott)-[:COMMUNICATED_VIA]->(PhoneLog)
  (Scott)-[:DOCUMENTED]->(KgNode)

Phase 2 (--with-time-place): give Utterances + Summaries a TIME-RESOLVED place by
finding the contemporaneous PhoneLog (within +/- 1h) that already has AT_PLACE, and
MERGE (Utterance)-[:AT_PLACE]->(SummaryPlace) / (Summary)-[:AT_PLACE]->(place).
This makes "what did Scott SAY at Park City / on <date>" directly queryable.
Requires PhoneLog->AT_PLACE to be complete first.

Usage:
  python enrich_scott_unify.py                 # phase 1 only
  python enrich_scott_unify.py --with-time-place
  python enrich_scott_unify.py --quiet
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _cfg():
    from auto_ingest_config import get_neo4j_config
    return get_neo4j_config()


WINDOW_MS = 3_600_000  # +/- 1 hour


def run(args) -> int:
    import neo4j
    cfg = _cfg()
    driver = neo4j.GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))
    total = 0
    try:
        with driver.session(database=cfg.get("db")) as sess:
            # ---- Phase 1: Scott node + stream links ----
            sess.run("MERGE (s:Person {name:'Scott'}) SET s:Scott")
            links = [
                ("SPOKE", "Utterance"),
                ("WROTE", "Summary"),
                ("DOCUMENTED", "KgNode"),
                # PhoneLog linked separately (--link-phonelog) to avoid deadlock
                # with the concurrent PhoneLog->AT_PLACE background passes.
            ]
            for rel, lbl in links:
                done = 0
                while True:
                    try:
                        rec = sess.run(
                            f"MATCH (s:Person {{name:'Scott'}}) "
                            f"MATCH (n:{lbl}) WHERE NOT (s)-[:{rel}]->(n) "
                            f"WITH s, n LIMIT 50000 "
                            f"MERGE (s)-[:{rel}]->(n) "
                            f"RETURN count(n) AS c"
                        ).single()
                    except neo4j.exceptions.TransientError:
                        continue  # retry on deadlock
                    c = rec["c"] if rec else 0
                    done += c
                    if c == 0:
                        break
                if not args.quiet:
                    print(f"[scott_unify] linked {done} {lbl} via {rel}", flush=True)
                total += done

            # PhoneLog link (separate, run with --link-phonelog after background passes end)
            if args.link_phonelog:
                done = 0
                while True:
                    try:
                        rec = sess.run(
                            "MATCH (s:Person {name:'Scott'}) "
                            "MATCH (n:PhoneLog) WHERE NOT (s)-[:COMMUNICATED_VIA]->(n) "
                            "WITH s, n LIMIT 50000 "
                            "MERGE (s)-[:COMMUNICATED_VIA]->(n) "
                            "RETURN count(n) AS c"
                        ).single()
                    except neo4j.exceptions.TransientError:
                        continue
                    c = rec["c"] if rec else 0
                    done += c
                    if c == 0:
                        break
                if not args.quiet:
                    print(f"[scott_unify] linked {done} PhoneLog via COMMUNICATED_VIA", flush=True)
                total += done

            # ---- Phase 2: time-resolved place for Utterance + Summary ----
            # Strategy: materialize an hour->place map ONCE (aggregate PhoneLogs by
            # 1-hour bucket into :PlaceHour nodes), then attach each Utterance/Summary
            # to the dominant place of its own hour bucket. Avoids a 21M-row scan
            # per utterance (the naive datetime join is unindexed -> deadly slow).
            if args.with_time_place:
                if not args.quiet:
                    print("[scott_unify] Phase 2: building hour->place map from PhoneLog...", flush=True)
                # Build/rebuild PlaceHour map (idempotent: deletes old first)
                sess.run("MATCH (ph:PlaceHour) DELETE ph")
                sess.run(
                    """
                    MATCH (p:PhoneLog)-[:AT_PLACE]->(pl:SummaryPlace)
                    WHERE p.epoch_millis IS NOT NULL
                    WITH p.epoch_millis - (p.epoch_millis % 3600000) AS hb,
                         pl.name AS pn, count(*) AS c
                    ORDER BY hb, c DESC
                    WITH hb, collect({pn: pn, c: c})[0] AS top
                    MERGE (ph:PlaceHour {hourBucket: hb})
                    SET ph.place = top.pn, ph.pings = top.c
                    """
                )
                nph = sess.run("MATCH (ph:PlaceHour) RETURN count(ph) AS c").single()["c"]
                if not args.quiet:
                    print(f"[scott_unify] Phase 2: {nph} PlaceHour buckets built", flush=True)

                for lbl, time_prop in [("Utterance", "created_at"), ("Summary", "created_at")]:
                    done = 0
                    while True:
                        try:
                            rec = sess.run(
                                f"""
                                MATCH (s:{lbl}) WHERE s.{time_prop} IS NOT NULL
                                  AND NOT (s)-[:AT_PLACE]->()
                                WITH s,
                                  CASE
                                    WHEN s.{time_prop} IS TYPED INTEGER OR s.{time_prop} IS TYPED FLOAT
                                      THEN toInteger(s.{time_prop})
                                    ELSE datetime(s.{time_prop}).epochMillis
                                  END AS em
                                WITH s, em - (em % 3600000) AS hb
                                // Timezone mismatch: Utterance/Summary created_at is local (Eastern),
                                // PhoneLog epoch_millis is UTC -> ~4-5h offset (DST-varying).
                                // Match the NEAREST PlaceHour bucket within +-6h (DST-agnostic).
                                UNWIND [0,3600000,7200000,10800000,14400000,18000000,21600000,
                                        -3600000,-7200000,-10800000,-14400000,-18000000,-21600000] AS off
                                MATCH (ph:PlaceHour {{hourBucket: hb + off}})
                                WITH s, ph, abs(off) AS ad
                                ORDER BY ad
                                WITH s, collect(ph)[0] AS ph0
                                MATCH (pl:SummaryPlace {{name: ph0.place}})
                                MERGE (s)-[:AT_PLACE]->(pl)
                                RETURN count(s) AS c
                                """,
                            ).single()
                        except neo4j.exceptions.TransientError:
                            continue
                        c = rec["c"] if rec else 0
                        done += c
                        if c == 0:
                            break
                    if not args.quiet:
                        print(f"[scott_unify] time-placed {done} {lbl}", flush=True)
                    total += done
    finally:
        driver.close()
    print(f"[scott_unify] done. total links/places = {total}", flush=True)
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-time-place", action="store_true")
    ap.add_argument("--link-phonelog", action="store_true",
                    help="Link PhoneLog to Scott (run after background PhoneLog passes end)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
