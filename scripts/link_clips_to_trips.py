#!/usr/bin/env python3
"""Link DashcamClip -> Trip, reconstructing car trips from clip continuity when no
Overland trip covers the clip era.

Two strategies, both idempotent:
  A) Overland overlap: clip time window overlaps a Trip(startTime..endTime).
  B) Dashcam reconstruction: clips with no Trip are grouped by continuity
     (gap > --gap-seconds between consecutive recordings => new trip) and a
     Trip(source:'dashcam', reconstructed:true) is created, with IN_TRIP edges.

Why reconstruction: Overland/LocationEvent trips exist only for 2024, but dashcam
clips are 2025-2026 with no GPS trips. Grouping consecutive 1-min clips recovers
the "Trip" needed for per-trip speaker isolation.

Also: --report-trip-speakers <tripId|uniqueKey> shows the speaker breakdown of a trip,
including which local speakers resolve to Scott (is_me).
"""
import os
import re
import argparse
import logging
from neo4j import GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD") or os.getenv("NEO4J_PASSWORD_DEFAULT") or "knowledge_graph_2026"
NEO4J_DB = os.getenv("NEO4J_DB", "neo4j")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def link_overland(sess, dry_run):
    q = """
    MATCH (c:DashcamClip)
    WHERE NOT (c)-[:IN_TRIP]->(:Trip)
    WITH c, c.created_at AS cend, c.created_at - coalesce(c.duration_s,60)*1000 AS cstart
    MATCH (t:Trip)
    WHERE t.startTime <= datetime({epochMillis:cend}) AND t.endTime >= datetime({epochMillis:cstart})
    MERGE (c)-[:IN_TRIP]->(t)
    RETURN count(c) AS n
    """
    if dry_run:
        n = sess.run(
            "MATCH (c:DashcamClip) WHERE NOT (c)-[:IN_TRIP]->(:Trip) "
            "WITH c, c.created_at AS cend, c.created_at - coalesce(c.duration_s,60)*1000 AS cstart "
            "MATCH (t:Trip) WHERE t.startTime <= datetime({epochMillis:cend}) AND t.endTime >= datetime({epochMillis:cstart}) "
            "RETURN count(c) AS n"
        ).single()["n"]
        logging.info(f"[overland] would link {n} clips (dry-run).")
        return n
    n = sess.run(q).single()["n"]
    logging.info(f"[overland] linked {n} clips to existing Trips.")
    return n


def reconstruct_dashcam(sess, gap_seconds, dry_run, batch=500):
    rows = sess.run(
        "MATCH (c:DashcamClip) WHERE NOT (c)-[:IN_TRIP]->(:Trip) "
        "RETURN c.key AS k, c.created_at AS ca, coalesce(c.duration_s,60) AS dur "
        "ORDER BY c.created_at"
    ).data()
    if not rows:
        logging.info("[reconstruct] no unlinked clips remain.")
        return 0
    gap_ms = gap_seconds * 1000
    trips_made = 0
    edges = 0
    group = []
    group_start = None

    def flush(grp):
        nonlocal trips_made, edges
        if not grp:
            return
        g0 = grp[0]
        g1 = grp[-1]
        start_ms = g0["ca"] - g0["dur"] * 1000
        end_ms = g1["ca"]
        uk = f"Dashcam_{start_ms}"
        if dry_run:
            trips_made += 1
            edges += len(grp)
            return
        sess.run(
            "MERGE (t:Trip{uniqueKey:$uk}) "
            "ON CREATE SET t.source='dashcam', t.reconstructed=true, "
            "t.trackerName='Dashcam', t.startTime=datetime({epochMillis:$s}), "
            "t.endTime=datetime({epochMillis:$e}), t.clipCount=size($keys), "
            "t.created_at=datetime() "
            "ON MATCH SET t.clipCount=size($keys) "
            "WITH t UNWIND $keys AS key "
            "MATCH (c:DashcamClip{key:key}) MERGE (c)-[:IN_TRIP]->(t)",
            uk=uk, s=int(start_ms), e=int(end_ms), keys=[r["k"] for r in grp],
        )
        trips_made += 1
        edges += len(grp)

    for r in rows:
        if group and (r["ca"] - (group_start) > gap_ms):
            flush(group)
            group = []
            group_start = None
        group.append(r)
        if group_start is None:
            group_start = r["ca"] - r["dur"] * 1000
        group_start = min(group_start, r["ca"] - r["dur"] * 1000)
    flush(group)
    logging.info(f"[reconstruct] {'would create' if dry_run else 'created'} {trips_made} dashcam trips covering {edges} clips.")
    return trips_made


def report_trip_speakers(sess, ident):
    rec = sess.run(
        "MATCH (t:Trip) WHERE t.tripId=$i OR t.uniqueKey=$i RETURN t.uniqueKey AS uk, t.startTime AS s, t.endTime AS e, t.source AS src, t.clipCount AS cc",
        i=ident,
    ).single()
    if not rec:
        logging.error(f"No Trip with tripId/uniqueKey={ident}")
        return
    logging.info(f"Trip {rec['uk']} src={rec['src']} clips={rec['cc']} {rec['s']} -> {rec['e']}")
    # Precise per-trip speaker breakdown via the FOR_CLIP edges backfilled by
    # scripts/backfill_segment_clip_key.py (G13 fix). This avoids the aggregated
    # transcriptions that previously inflated counts to corpus scale.
    rows = sess.run(
        "MATCH (tr:Trip{uniqueKey:$uk})<-[:IN_TRIP]-(c:DashcamClip)<-[:FOR_CLIP]-(t:Transcription) "
        "MATCH (t)-[:HAS_SEGMENT]->(seg:Segment)-[:SPOKEN_BY]->(sp:Speaker) "
        "RETURN coalesce(sp.label,'?') AS label, coalesce(sp.is_me,false) AS me, count(seg) AS segs "
        "ORDER BY segs DESC",
        uk=rec["uk"],
    ).data()
    if not rows:
        logging.info("  No Transcriptions linked to this trip's clips yet. Run "
                     "scripts/backfill_segment_clip_key.py to attach FOR_CLIP edges.")
        return
    for row in rows:
        tag = "  <== SCOTT" if row["me"] else ""
        logging.info(f"  {row['label']:12} is_me={row['me']} segs={row['segs']}{tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gap-seconds", type=int, default=600, help="New reconstructed trip after this silence between clips")
    ap.add_argument("--no-overland", action="store_true", help="Skip Overland time-overlap linking")
    ap.add_argument("--report-trip-speakers", default=None, help="tripId or uniqueKey to break down speakers")
    args = ap.parse_args()

    drv = driver()
    with drv.session(database=NEO4J_DB) as sess:
        if args.report_trip_speakers:
            report_trip_speakers(sess, args.report_trip_speakers)
            return
        if not args.no_overland:
            link_overland(sess, args.dry_run)
        reconstruct_dashcam(sess, args.gap_seconds, args.dry_run)
        total = sess.run("MATCH (c:DashcamClip)-[:IN_TRIP]->(:Trip) RETURN count(c) AS n").single()["n"]
        trips = sess.run("MATCH (t:Trip) RETURN count(t) AS n").single()["n"]
        logging.info(f"Summary: {total} DashcamClip linked to Trips; {trips} total Trip nodes.")
    drv.close()


if __name__ == "__main__":
    main()
