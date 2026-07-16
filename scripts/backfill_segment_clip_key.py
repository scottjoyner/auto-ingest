#!/usr/bin/env python3
"""Backfill Segment.clip_key + Transcription-[:FOR_CLIP]->DashcamClip (fixes G13).

Most Transcriptions are per-clip: their key equals the clip stem and they carry
~30-200 segments. A minority are aggregated (shared key, millions of segments) and
are skipped so they don't pollute per-trip segment counts. Once linked, a Trip can be
resolved precisely: Trip<-[:IN_TRIP]-Clip<-[:FOR_CLIP]-Transcription-HAS_SEGMENT->Segment.

Idempotent (only links clips without a FOR_CLIP transcription yet) and --dry-run safe.
"""
import os
import re
import time
import argparse
import logging
from neo4j import GraphDatabase
from neo4j.exceptions import TransientError

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD") or os.getenv("NEO4J_PASSWORD_DEFAULT") or "knowledge_graph_2026"
NEO4J_DB = os.getenv("NEO4J_DB", "neo4j")
MAX_SEGS = int(os.getenv("MAX_SEGS", "3000"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def stem(k):
    return re.sub(r"_[FRIL]$", "", k)


def _write_with_retry(sess, tk, ck, attempts=8):
    q = ("MATCH (t:Transcription{key:$tk}) MATCH (c:DashcamClip{key:$ck}) "
         "MERGE (t)-[:FOR_CLIP]->(c) SET t.clip_key=$ck "
         "WITH t, c MATCH (t)-[:HAS_SEGMENT]->(s:Segment) "
         "SET s.clip_key=$ck MERGE (s)-[:IN_CLIP]->(c)")
    last = None
    for i in range(attempts):
        try:
            sess.run(q, tk=tk, ck=ck)
            return
        except TransientError as e:
            last = e
            time.sleep(0.3 * (2 ** i))
    raise last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-segs", type=int, default=MAX_SEGS,
                    help="Skip transcriptions with more segments (aggregated); default 3000")
    ap.add_argument("--limit", type=int, default=0, help="Cap clips processed (0 = all unlinked)")
    args = ap.parse_args()

    drv = driver()
    with drv.session(database=NEO4J_DB) as sess:
        clips = [r["k"] for r in sess.run(
            "MATCH (c:DashcamClip) WHERE NOT (c)<-[:FOR_CLIP]-(:Transcription) RETURN c.key AS k"
        )]
        if args.limit:
            clips = clips[: args.limit]
        logging.info(f"Processing {len(clips)} unlinked clips (max-segs={args.max_segs}).")
        linked = skipped_agg = no_trans = 0
        for i, k in enumerate(clips):
            st = stem(k)
            tks = [r["k"] for r in sess.run("MATCH (t:Transcription{key:$k}) RETURN t.key AS k", k=st)]
            if not tks:
                no_trans += 1
                continue
            chosen = []
            for tk in tks:
                n = sess.run(
                    "MATCH (t:Transcription{key:$k})-[:HAS_SEGMENT]->(s:Segment) RETURN count(s) AS n",
                    k=tk,
                ).single()["n"]
                if n <= args.max_segs:
                    chosen.append(tk)
            if not chosen:
                skipped_agg += 1
                continue
            if not args.dry_run:
                for tk in chosen:
                    _write_with_retry(sess, tk, k)
            linked += len(chosen)
            if (i + 1) % 5000 == 0:
                logging.info(f"  progress {i+1}/{len(clips)} linked={linked}")
        logging.info(f"DONE linked_transcriptions={linked} skipped_aggregated={skipped_agg} no_transcription={no_trans}")
    drv.close()


if __name__ == "__main__":
    main()
