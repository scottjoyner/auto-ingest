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


def build_plans(drv, anchors, seeds, topics) -> list[Plan]:
    plans: list[Plan] = []
    for seed in seeds:
        plans.append(planner.plan_montage(drv, count=3, dur=30.0, seed=seed, limit=400))
        plans.append(planner.plan_highlights(
            drv, kinds=("music", "review", "speed"),
            per_kind=2, dur=20.0, seed=seed, limit=400))
        for topic in topics:
            clips = curator.discusses_topic(
                drv, topic, min_score=0.6, min_text_len=40, limit=40)
            if clips:
                plans.append(planner.plan_discussion(
                    drv, clips, topic=topic, short_count=3,
                    short_dur=30.0, seed=seed))
            else:
                log.warning("No discussion clips for %s (seed %s); skipping", topic, seed)
    return plans


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 23, 42])
    ap.add_argument("--topics", nargs="+",
                    default=["large_language_models", "multi_agent_systems",
                             "graph_neural_networks", "reinforcement_learning",
                             "diffusion_models", "computer_vision",
                             "retrieval_augmented_generation", "robotics"])
    ap.add_argument("--namespace", default=NAMESPACE)
    args = ap.parse_args(argv)

    PLANS_DIR.mkdir(parents=True, exist_ok=True)
    out_root = Path(args.namespace)
    out_root.mkdir(parents=True, exist_ok=True)
    os.environ.update(TTS_ENV)

    drv = _driver()
    try:
        anchors = backdrop.select_highway_pool(drv, limit=400)
        log.info("Resolved %d highway anchors", len(anchors))
        plans = build_plans(drv, anchors, args.seeds, args.topics)
    finally:
        drv.close()

    for p in plans:
        path = PLANS_DIR / f"{p.topic}__{p.plan_id}.json"
        p.save(path)
        log.info("Planned %s (seed-derived) -> %s (%d shorts)", p.topic, path, len(p.shorts))

    total = 0
    for p in plans:
        rendered = render.render_plan(
            p, out_root / p.topic, tts=True, skip_used_clips=True,
            skip_history=False, upsert=True)
        total += len(rendered)
        log.info("Rendered %d/%d for %s", len(rendered), len(p.shorts), p.topic)

    log.info("NARRATED MIX COMPLETE: %d shorts -> %s", total, out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
