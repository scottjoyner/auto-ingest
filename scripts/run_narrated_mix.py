"""Narrated mix driver: plan montage + highlights + discussion, render all
with owner-voice TTS into NAS5, deduped + manifest-upserted.

Run via: .venv/bin/python scripts/run_narrated_mix.py
(Detached/low-priority from the shell wrapper.)
"""
from __future__ import annotations

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
    from auto_ingest_config import get_neo4j_password
    from neo4j import GraphDatabase
    d = GraphDatabase.driver("bolt://localhost:7687",
                             auth=("neo4j", get_neo4j_password()))
    return d


def main() -> int:
    PLANS_DIR.mkdir(parents=True, exist_ok=True)
    out_root = Path(NAMESPACE)
    out_root.mkdir(parents=True, exist_ok=True)
    os.environ.update(TTS_ENV)

    drv = _driver()
    try:
        anchors = backdrop.select_highway_pool(drv, limit=400)
        log.info("Resolved %d highway anchors", len(anchors))

        plans: list[Plan] = []

        # Montage (ambient, owner-narrated hook)
        m = planner.plan_montage(drv, count=3, dur=30.0, seed=7, limit=400)
        plans.append(m)

        # Highlights (music / review / speed events)
        h = planner.plan_highlights(drv, kinds=("music", "review", "speed"),
                                    per_kind=2, dur=20.0, seed=7, limit=400)
        plans.append(h)

        # Discussion (your own words, time-aligned) -- no LLM needed
        clips = curator.discusses_topic(
            drv, "large_language_models", min_score=0.65,
            min_text_len=40, limit=40,
        )
        if clips:
            d = planner.plan_discussion(
                drv, clips, topic="large_language_models",
                short_count=3, short_dur=30.0, seed=7,
            )
            plans.append(d)
        else:
            log.warning("No discussion clips for large_language_models; skipping")
    finally:
        drv.close()

    for p in plans:
        path = PLANS_DIR / f"{p.topic}__{p.plan_id}.json"
        p.save(path)
        log.info("Planned %s -> %s (%d shorts)", p.topic, path, len(p.shorts))

    total = 0
    for p in plans:
        rendered = render.render_plan(
            p, out_root / p.topic, tts=True, skip_used_clips=True, upsert=True,
        )
        total += len(rendered)
        log.info("Rendered %d/%d for %s", len(rendered), len(p.shorts), p.topic)

    log.info("NARRATED MIX COMPLETE: %d shorts -> %s", total, out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
