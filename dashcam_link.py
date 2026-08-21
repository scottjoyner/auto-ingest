#!/usr/bin/env python3
"""Backfill DashcamDay linking + timestamps for clips created before linking existed.

Derives the clip datetime from the key (2026_0709_203536_R -> 2026-07-09 20:35:36),
creates (:DashcamDay {date}) (idempotent), links (DashcamClip)-[:ON_DAY]->(:DashcamDay),
and sets Clip.timestamp/date plus each DashcamFrame.timestamp = clip_time + minute.

Usage:
  python3 dashcam_link.py
  python3 dashcam_link.py --only-unlinked
"""
import os
import sys
import argparse
from datetime import timedelta

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dashcam_frame_vision import parse_key_datetime
from neo4j import GraphDatabase


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    ap.add_argument("--neo4j-user", default="neo4j")
    ap.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", "knowledge_graph_2026"))
    ap.add_argument("--only-unlinked", action="store_true", help="skip clips already ON_DAY")
    args = ap.parse_args()

    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))
    q = ("MATCH (c:DashcamClip) "
         + ("WHERE NOT (c)-[:ON_DAY]->() " if args.only_unlinked else "")
         + "RETURN c.key AS key")
    with driver.session() as sess:
        keys = [r["key"] for r in sess.run(q).data()]
    print(f"[info] linking {len(keys)} clips", flush=True)

    linked = skipped = 0
    with driver.session() as sess:
        for key in keys:
            dt = parse_key_datetime(key)
            if dt is None:
                skipped += 1
                continue
            day_date = dt.strftime("%Y-%m-%d")
            sess.run(
                """
                MATCH (c:DashcamClip {key:$key})
                SET c.timestamp=$dt, c.date=$date
                WITH c
                MERGE (day:DashcamDay {date:$date})
                SET day.year=$y, day.month=$mo, day.day=$d
                MERGE (c)-[:ON_DAY]->(day)
                WITH c, $dt AS dt
                MATCH (c)-[:HAS_FRAME]->(f:DashcamFrame)
                SET f.timestamp = dt + duration({minutes: f.minute})
                """,
                key=key, dt=dt, date=day_date,
                y=dt.year, mo=dt.month, d=dt.day,
            )
            linked += 1
        # also ensure every DashcamDay has year/month/day even if derived elsewhere
    print(f"[done] linked={linked} skipped(no-key-date)={skipped}", flush=True)
    driver.close()


if __name__ == "__main__":
    main()
