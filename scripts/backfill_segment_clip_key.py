#!/usr/bin/env python3
"""Backfill Segment.clip_key (fixes G13) and drop the corrupted IN_CLIP edges.

G13: a Segment needs to know which DashcamClip it belongs to so a Trip can be
resolved precisely at the segment level:
    Trip<-[:IN_TRIP]-DashcamClip<-[:FOR_CLIP]-Transcription-HAS_SEGMENT->Segment
    then Segment.clip_key == DashcamClip.key.

The canonical field is `Segment.clip_key`. Per-clip speaker attribution is done
by matching `Segment.clip_key`, NOT by the IN_CLIP relationship.

Why IN_CLIP is dropped: an earlier backfill matched Transcriptions by *stem*
(`MATCH (t:Transcription{key:$stem})`), so one stem resolved to many
Transcription nodes and every segment got `MERGE (s)-[:IN_CLIP]->(c)` edges to
every clip sharing that stem. Result: ~334k segments gained up to 52k wrong
IN_CLIP edges. Nothing in the codebase reads IN_CLIP, so it is safe to delete.

This script:
  1. Deletes all (:Segment)-[:IN_CLIP]->(:DashcamClip) edges (idempotent).
  2. For every Transcription that already has FOR_CLIP, sets `t.clip_key` and
     `Segment.clip_key` on its HAS_SEGMENT segments using the Transcription's
     EXACT key (no stem matching -> no collision). Idempotent.

--dry-run prints what it would do without writing.
"""
import argparse
import logging
import os
import time

from neo4j import GraphDatabase
from neo4j.exceptions import TransientError

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD") or os.getenv("NEO4J_PASSWORD_DEFAULT") or "knowledge_graph_2026"
NEO4J_DB = os.getenv("NEO4J_DB", "neo4j")
BATCH = int(os.getenv("BATCH", "2000"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def _run_with_retry(sess, q, params=None, attempts=8):
    last = None
    for i in range(attempts):
        try:
            return sess.run(q, params or {})
        except TransientError as e:
            last = e
            time.sleep(0.3 * (2 ** i))
    raise last


def drop_in_clip(sess, dry_run):
    """Delete every (:Segment)-[:IN_CLIP]->(:DashcamClip) edge."""
    if dry_run:
        n = _run_with_retry(sess, "MATCH ()-[r:IN_CLIP]->() RETURN count(r) AS n").single()["n"]
        logging.info(f"[dry-run] would delete {n} IN_CLIP edges")
        return 0
    deleted = 0
    while True:
        r = _run_with_retry(
            sess,
            "MATCH ()-[r:IN_CLIP]->() WITH r LIMIT $b DELETE r RETURN count(r) AS n",
            {"b": BATCH},
        ).single()
        n = r["n"]
        deleted += n
        if n == 0:
            break
        logging.info(f"  deleted {deleted} IN_CLIP edges so far")
    logging.info(f"DONE deleted {deleted} IN_CLIP edges")
    return deleted


def backfill_clip_key(sess, dry_run, limit=0):
    """Set t.clip_key and Segment.clip_key for every Transcription with FOR_CLIP.

    Keyed by the Transcription's exact key (no stem matching) so segments are
    attributed to exactly the clip their owning Transcription points at.
    """
    rows = _run_with_retry(
        sess,
        "MATCH (t:Transcription)-[:FOR_CLIP]->(c:DashcamClip) RETURN t.key AS tk, c.key AS ck",
    ).data()
    if limit:
        rows = rows[:limit]
    logging.info(f"Processing {len(rows)} transcriptions with FOR_CLIP (dry-run={dry_run}).")
    done = 0
    for i, row in enumerate(rows):
        tk, ck = row["tk"], row["ck"]
        if dry_run:
            n = _run_with_retry(
                sess,
                "MATCH (t:Transcription{key:$tk})-[:HAS_SEGMENT]->(s:Segment) RETURN count(s) AS n",
                {"tk": tk},
            ).single()["n"]
            done += 1
            if (i + 1) % 5000 == 0:
                logging.info(f"  [dry-run] progress {i+1}/{len(rows)} segs_in_batch={n}")
            continue
        _run_with_retry(
            sess,
            ("MATCH (t:Transcription{key:$tk})-[:FOR_CLIP]->(c:DashcamClip{key:$ck}) "
             "SET t.clip_key=$ck "
             "WITH t, c MATCH (t)-[:HAS_SEGMENT]->(s:Segment) "
             "SET s.clip_key=$ck"),
            {"tk": tk, "ck": ck},
        )
        done += 1
        if (i + 1) % 5000 == 0:
            logging.info(f"  progress {i+1}/{len(rows)}")
    logging.info(f"DONE backfilled {done} transcriptions")
    return done


def backfill_audio_clip_key(sess, dry_run, limit=0):
    """Set Segment.clip_key for AUDIO transcriptions that have no DashcamClip.

    Audio (phone/recorder) transcriptions legitimately have no DashcamClip, so
    their segments' clip_key stays null under the dashcam backfill. We attribute
    them to their audio source key (the Transcription key / WAV stem) so the field
    is fully populated and queryable per-source. Idempotent.
    """
    rows = _run_with_retry(
        sess,
        "MATCH (t:Transcription)-[:HAS_SEGMENT]->(s:Segment) "
        "WHERE s.clip_key IS NULL AND t.source_media CONTAINS 'audio' "
        "WITH DISTINCT t RETURN t.key AS tk",
    ).data()
    if limit:
        rows = rows[:limit]
    logging.info(f"Audio backfill: {len(rows)} transcriptions with null clip_key (dry-run={dry_run}).")
    done = 0
    for i, row in enumerate(rows):
        tk = row["tk"]
        if dry_run:
            done += 1
            if (i + 1) % 5000 == 0:
                logging.info(f"  [dry-run] progress {i+1}/{len(rows)}")
            continue
        _run_with_retry(
            sess,
            ("MATCH (t:Transcription{key:$tk})-[:HAS_SEGMENT]->(s:Segment) "
             "WHERE s.clip_key IS NULL SET s.clip_key=$tk"),
            {"tk": tk},
        )
        done += 1
        if (i + 1) % 5000 == 0:
            logging.info(f"  progress {i+1}/{len(rows)}")
    logging.info(f"DONE audio backfill {done} transcriptions")
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-in-clip", action="store_true",
                    help="Do NOT delete the (corrupted) IN_CLIP edges.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap transcriptions processed in the backfill (0 = all).")
    ap.add_argument("--skip-drop", action="store_true",
                    help="Skip the IN_CLIP delete step (only backfill clip_key).")
    ap.add_argument("--backfill-audio", action="store_true",
                    help="Also attribute audio-only segments to their audio source key.")
    args = ap.parse_args()

    drv = driver()
    with drv.session(database=NEO4J_DB) as sess:
        if not args.skip_drop and not args.keep_in_clip:
            drop_in_clip(sess, args.dry_run)
        backfill_clip_key(sess, args.dry_run, limit=args.limit)
        if args.backfill_audio:
            backfill_audio_clip_key(sess, args.dry_run, limit=args.limit)
    drv.close()


if __name__ == "__main__":
    main()
