#!/usr/bin/env python3
"""Idle-harvest enrichment pipeline orchestrator (W-57).

Runs the Summary-enrichment steps in order. Every step is IDEMPOTENT
(guards on `WHERE s.lat IS NULL`, `WHERE s.embedding_768 IS NULL`,
`WHERE NOT (s)-[:AT_PLACE]->()`, etc.), so re-running only processes NEW
Summaries — safe for unattended cron.

Steps:
  1  geo-tag            (s.lat/lon + LOCATED_AT_TIME_BASED from PhoneLog)
  3  cluster place names (top clusters -> OSM)
  3b merge super-clusters (cells -> SummaryPlace + AT_PLACE)
  4  infer context role  (home/work/travel per place)
  5  lexical bridge      (DISCUSSES from concept names in text)
  5b semantic bridge     (RELATED_CONCEPT via 768-dim re-embed)

Note: the full geocode (cluster_place_enrich_FULL) is NOT in the daily path —
it only renames previously-unnamed clusters and is slow (~18 min). Run it
manually once after a big backfill. The daily path uses cluster_place_enrich
(top-N clusters), which is incremental and fast.

Watchdog behaviour: the fast daily steps (3b,4,5) are cheap (~15s) so the daily
cron runs them unconditionally in --quiet mode. The expensive steps — geo (1, slow
because it scans PhoneLog for every Summary; many pending are permanent no-match)
and semantic re-embed (5b, ~30-50 min) — are NOT in the daily path. Run 5b weekly
via --with-semantic, and re-run cluster_place_enrich_FULL.py manually after a big
backfill. Use --check to see pending-work counts.

Recommended cron:
  DAILY :  enrich_pipeline.py --steps 3b,4,5 --quiet
  WEEKLY:  enrich_pipeline.py --with-semantic --quiet   (plus manual full geocode)

Usage:
  python enrich_pipeline.py                 # run fast daily steps (1,3,3b,4,5)
  python enrich_pipeline.py --quiet         # only print lines when something changed
  python enrich_pipeline.py --with-semantic # also run 5b (weekly, expensive re-embed)
  python enrich_pipeline.py --check         # print pending-work counts and exit
  python enrich_pipeline.py --steps 1,3b,5b # run a subset

Exit 0 always (even when idle) so cron doesn't alarm on "nothing new".
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent  # auto-ingest root
SCRIPTS = REPO / "scripts"

# (step id, script filename, default-on)
STEPS = [
    ("1",  "enrich_summary_geo.py", True),
    ("3",  "enrich_cluster_place.py", True),
    ("3b", "enrich_merge_super_clusters.py", True),
    ("4",  "enrich_infer_context_role.py", True),
    ("5",  "enrich_bridge_to_concepts.py", True),
    ("5b", "enrich_bridge_semantic.py", False),  # EXPENSIVE (re-embed); --with-semantic
]

VENV_PY = REPO / ".venv" / "bin" / "python"
FALLBACK_PY = "/home/scott/git/hermes-agent/venv/bin/python"


def _py() -> str:
    return str(VENV_PY) if VENV_PY.exists() else FALLBACK_PY


def _cfg():
    sys.path.insert(0, str(REPO))
    from auto_ingest_config import get_neo4j_config
    return get_neo4j_config()


def run_step(script: str, quiet: bool) -> tuple[int, float]:
    """Run one enrichment script. Return (changed_lines, elapsed_sec)."""
    path = SCRIPTS / script
    if not path.exists():
        print(f"  [skip] {script} not found", file=sys.stderr)
        return 0, 0.0
    t0 = time.time()
    proc = subprocess.run([_py(), str(path)], capture_output=True, text=True)
    elapsed = time.time() - t0
    out = (proc.stdout or "") + (proc.stderr or "")
    changed = sum(
        1 for ln in out.splitlines()
        if any(k in ln.lower() for k in (
            "created", "updated", "linked", "tagged", "geocoded",
            "merged", "re-embed", "bridged", "done"))
    )
    if not quiet:
        for ln in out.splitlines():
            if ln.strip():
                print(f"    {script}: {ln}")
    elif changed:
        print(f"  {script}: {changed} change-lines ({elapsed:.1f}s)")
    return changed, elapsed


def pending_work() -> dict:
    """Cheap COUNT of un-enriched summaries (watchdog pre-check)."""
    import neo4j
    cfg = _cfg()
    driver = neo4j.GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))
    try:
        with driver.session(database=cfg.get("db")) as sess:
            def cnt(q: str) -> int:
                return sess.run(q).single()[0]
            geo = cnt("MATCH (s:Summary) WHERE s.lat IS NULL AND size(s.text) > 0 RETURN count(s)")
            atplace = cnt("MATCH (s:Summary) WHERE s.lat IS NOT NULL AND NOT (s)-[:AT_PLACE]->() RETURN count(s)")
            lexical = cnt("MATCH (s:Summary) WHERE s.lat IS NOT NULL AND NOT (s)-[:DISCUSSES]->() RETURN count(s)")
            semantic = cnt("MATCH (s:Summary) WHERE s.lat IS NOT NULL AND s.embedding_768 IS NULL RETURN count(s)")
        return {"geo": geo, "atplace": atplace, "lexical": lexical, "semantic": semantic}
    finally:
        driver.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true", help="only report when something changed")
    ap.add_argument("--steps", default="", help="comma subset of 1,3,3b,4,5,5b")
    ap.add_argument("--with-semantic", action="store_true",
                    help="also run 5b (semantic re-embed) — EXPENSIVE, use weekly")
    ap.add_argument("--check", action="store_true", help="print pending-work counts and exit")
    args = ap.parse_args()

    work = pending_work()
    if args.check:
        print("pending:", work)
        return 0
    if sum(work.values()) == 0:
        if not args.quiet:
            print("[enrich_pipeline] nothing new to enrich — idle.")
        return 0
    print(f"[enrich_pipeline] pending work: {work}")

    wanted = set(args.steps.split(",")) if args.steps else None
    steps = [(sid, s) for sid, s, on in STEPS
             if (wanted is not None and sid in wanted)
             or (wanted is None and (on or args.with_semantic))]

    print(f"[enrich_pipeline] {len(steps)} step(s) | py={_py()}")
    total_changed = 0
    for sid, script in steps:
        print(f"[step {sid}] {script}")
        changed, elapsed = run_step(script, args.quiet)
        total_changed += changed
        print(f"  -> {changed} change-lines in {elapsed:.1f}s")

    print(f"[enrich_pipeline] done. total change-lines={total_changed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
