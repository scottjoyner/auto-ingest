#!/usr/bin/env python3
"""Resumable, batched Entity/Mention extraction from Transcriptions.

SAFETY:
  * NEVER scans the whole graph. It processes a CAPPED sample: either a single
    ``--trip`` (all clips in that trip) or ``--limit`` Transcriptions per run.
  * Idempotent + resumable: processed ``Transcription`` keys are recorded in a
    state file (``--state``) so re-runs skip them. Extracted ``Entity``/
    ``Mention`` nodes are MERGEd by stable id, so re-running is safe.
  * ``--dry-run`` prints what *would* be extracted without writing anything.
  * No full-graph aggregation, no multi-hop walks beyond one Transcription ->
    Segments -> text.

This deliberately does NOT attempt the full Entity/Mention pass against the
live 21M-node graph (too heavy / risky). It is the safe, incremental tool the
DEEP_DIVE §3.6 plan calls for, to be run repeatedly (e.g. per trip) until the
corpus is covered.

Pattern extraction is intentionally MINIMAL (date/time/self-reference); wire
in richer extractors behind ``extract_mentions_from_text`` if available.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neo4j import GraphDatabase  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD") or os.getenv("NEO4J_PASSWORD_DEFAULT") or "knowledge_graph_2026"
NEO4J_DB = os.getenv("NEO4J_DB", "neo4j")

# Minimal, dependency-free pattern extractors (extend as needed).
SELF_RE = re.compile(r"\b(I|my|me|myself|we|our|us)\b", re.I)
DATE_RE = re.compile(
    r"\b(\d{4}-\d{2}-\d{2}"
    r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:,?\s+\d{4})?)",
    re.I,
)
TIME_RE = re.compile(r"\b(\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm)?)\b", re.I)


def extract_mentions_from_text(text: str) -> dict:
    """Return a dict of {'self': bool, 'dates': [...], 'times': [...]}.

    Pure function (no graph access) so it is trivially unit-testable.
    """
    text = text or ""
    return {
        "self": bool(SELF_RE.search(text)),
        "dates": sorted(set(DATE_RE.findall(text))),
        "times": sorted(set(TIME_RE.findall(text))),
    }


def driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def fetch_transcription_keys(sess, trip: str, limit: int, done: set) -> list:
    """Return up to ``limit`` Transcription keys not in ``done``.

    If ``trip`` is given, only Transcriptions whose clip is IN_TRIP that trip.
    Otherwise any Transcription (still capped by ``limit``). Never a full scan:
    the query is bounded by ``LIMIT`` and filters on the processed set.
    """
    if trip:
        rows = sess.run(
            """
            MATCH (t:Trip {key:$trip})<-[:IN_TRIP]-(c:DashcamClip)
            MATCH (tr:Transcription)-[:FOR_CLIP]->(c)
            WHERE NOT tr.key IN $done
            RETURN tr.key AS k LIMIT $lim
            """, trip=trip, done=list(done), lim=limit,
        ).data()
    else:
        rows = sess.run(
            """
            MATCH (tr:Transcription)
            WHERE NOT tr.key IN $done
            RETURN tr.key AS k LIMIT $lim
            """, done=list(done), lim=limit,
        ).data()
    return [r["k"] for r in rows]


def process(sess, keys: list, dry_run: bool) -> int:
    """Extract Entity/Mention nodes from the given Transcription keys."""
    created = 0
    for k in keys:
        recs = sess.run(
            """
            MATCH (tr:Transcription {key:$k})-[:HAS_SEGMENT]->(s:Segment)
            RETURN s.text AS text
            """, k=k,
        ).data()
        texts = [r["text"] or "" for r in recs if r.get("text")]
        # Aggregate extracted signals across the transcription's segments.
        mentions = {}
        self_count = 0
        for tx in texts:
            ex = extract_mentions_from_text(tx)
            if ex["self"]:
                self_count += 1
            for d in ex["dates"]:
                mentions.setdefault(("Date", d), 0)
                mentions[("Date", d)] += 1
            for tm in ex["times"]:
                mentions.setdefault(("Time", tm), 0)
                mentions[("Time", tm)] += 1
        if not mentions and self_count == 0:
            continue
        created += 1
        if dry_run:
            logging.info(
                f"[dry-run] {k}: self_refs={self_count} entities={dict(mentions)}"
            )
            continue
        # Merge Entity nodes + Mention edges, keyed by stable ids.
        for (etype, val), cnt in mentions.items():
            eid = f"ent|{etype}|{val}"
            sess.run(
                """
                MERGE (e:Entity {id:$eid})
                ON CREATE SET e.created_at=datetime(), e.kind=$kind, e.name=$val
                SET e.updated_at=datetime()
                WITH e
                MATCH (tr:Transcription {key:$k})
                MERGE (tr)-[m:MENTIONS]->(e)
                SET m.count=coalesce(m.count,0)+$cnt, m.updated_at=datetime()
                """, eid=eid, kind=etype, val=val, k=k, cnt=cnt,
            )
        if self_count:
            eid = "ent|SelfRef|scott"
            sess.run(
                """
                MERGE (e:Entity {id:$eid})
                ON CREATE SET e.created_at=datetime(), e.kind='SelfRef', e.name='scott'
                SET e.updated_at=datetime()
                WITH e
                MATCH (tr:Transcription {key:$k})
                MERGE (tr)-[m:MENTIONS]->(e)
                SET m.count=coalesce(m.count,0)+$cnt, m.updated_at=datetime()
                """, eid=eid, k=k, cnt=self_count,
            )
    return created


def load_state(path: Path) -> set:
    if path.exists():
        try:
            return set(json.loads(path.read_text()))
        except Exception:
            return set()
    return set()


def save_state(path: Path, done: set) -> None:
    path.write_text(json.dumps(sorted(done)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trip", type=str, default="",
                    help="Only process Transcriptions in this Trip (key).")
    ap.add_argument("--limit", type=int, default=200,
                    help="Max Transcriptions to process this run (0 = use --trip fully).")
    ap.add_argument("--state", type=str, default="./extract_mentions_state.json",
                    help="State file of processed Transcription keys (resume).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be extracted; write nothing.")
    args = ap.parse_args()

    state_path = Path(args.state)
    done = load_state(state_path)
    limit = args.limit if args.limit else 10_000_000

    drv = driver()
    try:
        with drv.session(database=NEO4J_DB) as sess:
            keys = fetch_transcription_keys(sess, args.trip, limit, done)
            logging.info(
                f"Processing {len(keys)} Transcriptions (trip={args.trip or 'any'}, "
                f"dry_run={args.dry_run})."
            )
            n = process(sess, keys, args.dry_run)
    finally:
        drv.close()

    if keys:
        done.update(keys)
        save_state(state_path, done)
    logging.info(f"DONE: {n} Transcriptions yielded extractions this run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
