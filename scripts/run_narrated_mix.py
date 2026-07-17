"""Narrated mix driver (iteration-aware).

Plans montage + highlights + discussion(s) across several seeds/topics, renders
all with owner-voice TTS into NAS5, deduped (by clip_key) + manifest-upserted.

Run via:
    .venv/bin/python scripts/run_narrated_mix.py \
        --seeds 7 11 23 --topics large_language_models knowledge_graph

Detach with: setsid bash -c '.venv/bin/python -u scripts/run_narrated_mix.py ... \
        > log 2>&1' < /dev/null & disown
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("narrated_mix")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auto_ingest.shorts import backdrop, curator, planner, render  # noqa: E402
from auto_ingest.shorts.models import Plan  # noqa: E402

NAMESPACE = "/media/scott/NAS5/fileserver/dashcam_timelapse/_mix_run/narrated"
PLANS_DIR = Path("/media/scott/SSD_4TB/hermes-home/auto-ingest/plans_narrated")
TTS_ENV = {"COQUI_TOS_AGREED": "1"}


def _driver():
    from neo4j import GraphDatabase

    from auto_ingest_config import get_neo4j_password
    d = GraphDatabase.driver("bolt://localhost:7687",
                             auth=("neo4j", get_neo4j_password()))
    return d


def _plan_with_retry(fn, *, attempts: int = 20, base_wait: float = 45.0):
    """Run a planning query, retrying on transient Neo4j memory-pool OOM.

    The DB transaction pool is tiny and shared with other sessions on this
    host; heavy planning queries intermittently OOM. Back off and retry
    instead of aborting the whole batch.
    """
    import time
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # neo4j TransientError (OOM) is transient
            last = e
            if "MemoryPoolOutOfMemory" in str(e) or "TransientError" in type(e).__name__:
                wait = base_wait * (i + 1)
                log.warning("Planning hit DB memory pressure; retry %d/%d in %.0fs",
                            i + 1, attempts, wait)
                time.sleep(wait)
                continue
            raise
    log.error("Planning failed after %d retries: %s", attempts, last)
    return None


def _cached_or_plan(cache_key: str, fn, force: bool = False):
    """Reuse a saved plan JSON if present; otherwise plan (with retry) + cache."""
    path = PLANS_DIR / f"{cache_key}.json"
    if not force and path.exists():
        try:
            return Plan.load(path)
        except Exception as e:
            log.warning("Plan cache unreadable (%s); re-planning", e)
    plan = _plan_with_retry(fn)
    if plan is not None:
        plan.save(path)
    return plan


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 23, 42])
    ap.add_argument("--topics", nargs="+",
                    default=["large_language_models", "multi_agent_systems",
                             "graph_neural_networks", "reinforcement_learning",
                             "diffusion_models", "computer_vision",
                             "retrieval_augmented_generation", "robotics"])
    ap.add_argument("--trips", type=int, default=1,
                    help="Number of real 'Trip Story' journeys to plan per seed")
    ap.add_argument("--profile", default=os.getenv("SHORTS_PROFILE", "clean"),
                    help="Caption profile: clean | karaoke | wordgrid | cinematic")
    ap.add_argument("--force-replan", action="store_true",
                    help="Ignore cached plan JSONs and re-plan from the graph")
    ap.add_argument("--namespace", default=NAMESPACE)
    args = ap.parse_args(argv)

    PLANS_DIR.mkdir(parents=True, exist_ok=True)
    out_root = Path(args.namespace)
    out_root.mkdir(parents=True, exist_ok=True)
    os.environ.update(TTS_ENV)

    drv = _driver()
    try:
        anchors = _plan_with_retry(
            lambda: backdrop.select_highway_pool(drv, limit=400))
        if not anchors:
            log.error("No highway anchors resolved; aborting")
            return 1
        log.info("Resolved %d highway anchors", len(anchors))

        plans: list[Plan] = []
        for seed in args.seeds:
            def _montage(s=seed):
                return planner.plan_montage(drv, count=3, dur=30.0, seed=s, limit=400)
            p = _cached_or_plan(f"montage__{seed}", _montage, force=args.force_replan)
            if p:
                plans.append(p)
            def _high(s=seed):
                return planner.plan_highlights(
                    drv, kinds=("music", "review", "speed"),
                    per_kind=2, dur=20.0, seed=s, limit=400)
            p = _cached_or_plan(f"highlights__{seed}", _high, force=args.force_replan)
            if p:
                plans.append(p)
            for topic in args.topics:
                def _disc(t=topic, s=seed):
                    clips = curator.discusses_topic(
                        drv, t, min_score=0.6, min_text_len=40, limit=40)
                    if not clips:
                        return None
                    return planner.plan_discussion(
                        drv, clips, topic=t, short_count=3,
                        short_dur=30.0, seed=s)
                p = _cached_or_plan(f"discussion_{topic}__{seed}", _disc, force=args.force_replan)
                if p:
                    plans.append(p)
            if args.trips:
                def _trip(s=seed):
                    return planner.plan_trip_story(
                        drv, count=args.trips, short_dur=30.0,
                        shots_per_trip=5, clip_dur=6.0, seed=s)
                p = _cached_or_plan(f"trip__{seed}", _trip, force=args.force_replan)
                if p:
                    plans.append(p)
    finally:
        drv.close()

    plans = [p for p in plans if p is not None]
    total = 0
    for p in plans:
        log.info("Planned %s -> %d shorts", p.topic, len(p.shorts))
        rendered = render.render_plan(
            p, out_root / p.topic, tts=True, skip_used_clips=True,
            skip_history=False, upsert=True, profile_name=args.profile)
        total += len(rendered)
        log.info("Rendered %d/%d for %s", len(rendered), len(p.shorts), p.topic)

    log.info("NARRATED MIX COMPLETE: %d shorts -> %s", total, out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
