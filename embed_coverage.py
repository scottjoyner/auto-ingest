#!/usr/bin/env python3
"""
embed_coverage.py — read-only report of embedding coverage per label x KG.

For each label, counts total nodes (with a source text) vs. those carrying --prop,
and prints coverage %. Handy to confirm the embedding wave actually finished and to
spot gaps. Reads only; safe to run while other jobs write.

Run per KG:
  embed_coverage.py --uri bolt://127.0.0.1:17687 --prop emb_e5_large --labels Segment Utterance ...
  embed_coverage.py --uri bolt://100.64.43.123:7687 --prop emb_e5_large --labels Chunk
"""
from __future__ import annotations
import os
import sys
import argparse
import logging
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DEFAULT_LABELS = [
    "Segment", "Utterance", "Transcription", "Summary",
    "Entity", "Concept", "Topic", "Keyword", "KgNode", "Note",
    "Speaker", "GlobalSpeaker", "Chunk",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default=os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"))
    ap.add_argument("--user", default=os.getenv("NEO4J_USER", "neo4j"))
    ap.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", "knowledge_graph_2026"))
    ap.add_argument("--prop", default="emb_e5_large")
    ap.add_argument("--labels", nargs="*", default=DEFAULT_LABELS)
    args = ap.parse_args()

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    total_nodes = 0
    total_done = 0
    with driver.session() as s:
        logging.info("=== coverage for %s prop=%s ===", args.uri, args.prop)
        for lbl in args.labels:
            try:
                tot = s.run(
                    f"MATCH (n:{lbl}) WHERE n.text IS NOT NULL OR n.name IS NOT NULL "
                    f"OR n.title IS NOT NULL OR n.label IS NOT NULL OR n.display_label IS NOT NULL "
                    f"RETURN count(n)"
                ).single().values()[0]
            except Exception as e:
                logging.warning("%s: label query failed (%s)", lbl, e)
                continue
            done = s.run(
                f"MATCH (n:{lbl}) WHERE n.{args.prop} IS NOT NULL RETURN count(n)"
            ).single().values()[0]
            pct = (100.0 * done / tot) if tot else 0.0
            flag = "OK " if tot and done == tot else "!!!"
            logging.info("%-14s total=%-9d %-9s=%-9d %5.1f%%  %s", lbl, tot, args.prop, done, pct, flag)
            total_nodes += tot
            total_done += done
    overall = (100.0 * total_done / total_nodes) if total_nodes else 0.0
    logging.info("=== overall: %d/%d (%.1f%%) ===", total_done, total_nodes, overall)
    driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
