"""auto_ingest.shorts — research-scripted vertical shorts from the knowledge graph.

Pipeline:
  plan     curate a Topic from Neo4j + select highway B-roll -> write a Plan JSON
  render   render planned shorts (needs the moviepy/whisper image)
  iterate  regenerate a Plan (new seed/counts, drop rejected) for variation
  list     show plans/shorts on disk or in Neo4j

The Plan JSON on disk is the iteration artifact: re-run ``iterate`` to explore
variations, ``render`` to produce video, and the manifest is upserted to Neo4j.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List

from auto_ingest.shorts import backdrop, curator, planner, render
from auto_ingest.shorts.models import Plan

log = logging.getLogger("shorts.cli")

DEFAULT_PLANS_DIR = Path(os.getenv("SHORTS_PLANS_DIR", "./shorts_plans"))
DEFAULT_OUT_DIR = Path(os.getenv("SHORTS_OUT_DIR", "./shorts_out"))


def _driver():
    from neo4j import GraphDatabase

    from auto_ingest_config import get_neo4j_password
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    pw = get_neo4j_password()
    db = os.getenv("NEO4J_DB", "neo4j")
    return GraphDatabase.driver(uri, auth=(user, pw)), db


def _cmd_plan(args) -> int:
    driver, _ = _driver()
    try:
        if args.discusses:
            clips = curator.discusses_topic(
                driver, args.topic, min_score=args.min_score,
                min_text_len=args.min_text_len, limit=args.discuss_limit)
            if not clips:
                print(f"No spoken discussion found for topic {args.topic!r} "
                      f"(min_score={args.min_score}). Run link-concepts first.")
                return 1
            # Time-aligned: B-roll is the dashcam clip AT the moment each
            # utterance was spoken, not a random highway anchor.
            plan = planner.plan_discussion(
                driver, clips, topic=args.topic,
                short_count=args.shorts, short_dur=args.dur, seed=args.seed)
            out = args.plans_dir or DEFAULT_PLANS_DIR
            path = Path(out) / f"{args.topic}__{plan.plan_id}.json"
            plan.save(path)
            print(f"Planned {len(plan.shorts)} time-aligned discussion short(s) "
                  f"-> {path}")
            return 0
        else:
            brief = curator.curate_brief(driver, args.topic, top_papers=args.top_papers)
        anchors = backdrop.select_highway_pool(driver, limit=args.pool)
    finally:
        driver.close()
    plan = planner.plan_shorts(
        brief, anchors, short_count=args.shorts, short_dur=args.dur,
        shots_per_short=args.shots_per_short, seed=args.seed,
    )
    out = args.plans_dir or DEFAULT_PLANS_DIR
    path = Path(out) / f"{args.topic}__{plan.plan_id}.json"
    plan.save(path)
    print(f"Planned {len(plan.shorts)} shorts -> {path}")
    return 0


def _cmd_montage(args) -> int:
    driver, _ = _driver()
    try:
        plan = planner.plan_montage(
            driver, count=args.shorts, dur=args.dur, mood=args.mood,
            limit=args.pool, seed=args.seed,
        )
    finally:
        driver.close()
    out = args.plans_dir or DEFAULT_PLANS_DIR
    path = Path(out) / f"montage__{plan.plan_id}.json"
    plan.save(path)
    print(f"Planned {len(plan.shorts)} montage short(s) -> {path}")
    return 0


def _cmd_highlights(args) -> int:
    driver, _ = _driver()
    try:
        plan = planner.plan_highlights(
            driver, kinds=tuple(args.kinds), per_kind=args.per_kind,
            dur=args.dur, limit=args.pool, seed=args.seed,
        )
    finally:
        driver.close()
    out = args.plans_dir or DEFAULT_PLANS_DIR
    path = Path(out) / f"highlights__{plan.plan_id}.json"
    plan.save(path)
    print(f"Planned {len(plan.shorts)} highlight short(s) -> {path}")
    return 0


def _cmd_trip(args) -> int:
    driver, _ = _driver()
    try:
        plan = planner.plan_trip_story(
            driver, trip_key=args.trip_key or None, count=args.shorts,
            short_dur=args.dur, shots_per_trip=args.shots_per_short,
            clip_dur=args.clip_dur, seed=args.seed,
        )
    finally:
        driver.close()
    out = args.plans_dir or DEFAULT_PLANS_DIR
    path = Path(out) / f"trip__{plan.plan_id}.json"
    plan.save(path)
    print(f"Planned {len(plan.shorts)} trip-story short(s) -> {path}")
    return 0


def _cmd_iterate(args) -> int:
    prev = Plan.load(Path(args.plan))
    driver, _ = _driver()
    try:
        anchors = backdrop.select_highway_pool(driver, limit=args.pool) if args.resolve else []
    finally:
        driver.close()
    new = planner.iterate_plan(
        prev, anchors, short_count=args.shorts, seed=args.seed,
        reject_ids=args.reject,
    )
    path = Path(args.plan).parent / f"{new.topic}__{new.plan_id}.json"
    new.save(path)
    print(f"Iterated -> {path} (iteration {new.iteration}, {len(new.shorts)} shorts)")
    return 0


def _cmd_render(args) -> int:
    plan = Plan.load(Path(args.plan))
    driver, _ = _driver()
    try:
        anchors = backdrop.select_highway_pool(driver, limit=args.pool)
        # Fill any unresolved shots with locally-available highway anchors.
        if any(not s.shots for s in plan.shorts):
            plan = planner.iterate_plan(plan, anchors, short_count=len(plan.shorts))
        out_dir = args.out_dir or DEFAULT_OUT_DIR
        paths = render.render_plan(plan, out_dir, only=args.only)
        render.upsert_manifest(driver, plan, out_dir)
    finally:
        driver.close()
    print(f"Rendered {len(paths)} short(s) -> {out_dir}")
    plan.save(Path(args.plan))  # persist updated statuses
    return 0


def _cmd_list(args) -> int:
    if args.neo4j:
        driver, _ = _driver()
        try:
            for r in driver.session().run(
                "MATCH (sp:ShortPlan) RETURN sp.plan_id AS id, sp.topic AS topic, "
                "sp.iteration AS iter ORDER BY sp.updated_at DESC"
            ):
                print(f"{r['id']}  {r['topic']}  iter={r['iter']}")
        finally:
            driver.close()
        return 0
    base = Path(args.plans_dir or DEFAULT_PLANS_DIR)
    for p in sorted(base.glob("*.json")):
        try:
            plan = Plan.load(p)
        except Exception:
            continue
        print(f"{p.name}  topic={plan.topic} shorts={len(plan.shorts)} iter={plan.iteration}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="auto-ingest shorts",
                                description="Research-scripted shorts from Neo4j + highway B-roll.")
    sub = p.add_subparsers(dest="action", required=True)

    pp = sub.add_parser("plan", help="Curate a Topic and write a Plan JSON.")
    pp.add_argument("topic", help="Topic name, e.g. large_language_models")
    pp.add_argument("--shorts", type=int, default=3)
    pp.add_argument("--dur", type=float, default=30.0, help="Seconds per short")
    pp.add_argument("--shots-per-short", type=int, default=3)
    pp.add_argument("--top-papers", type=int, default=6)
    pp.add_argument("--seed", type=int, default=1)
    pp.add_argument("--pool", type=int, default=400, help="Highway anchor pool size")
    pp.add_argument("--plans-dir", type=Path, default=DEFAULT_PLANS_DIR)
    pp.add_argument("--discusses", action="store_true",
                    help="Curate from spoken Utterance-[:MENTIONS]->Concept edges "
                         "instead of papers (needs link-concepts run).")
    pp.add_argument("--min-score", type=float, default=0.65,
                    help="Min MENTIONS score when --discusses")
    pp.add_argument("--min-text-len", type=int, default=30,
                    help="Drop utterances shorter than this when --discusses")
    pp.add_argument("--discuss-limit", type=int, default=40,
                    help="Max discussion clips to curate when --discusses")
    pp.set_defaults(func=_cmd_plan)

    pi = sub.add_parser("iterate", help="Regenerate a plan (variation / drop rejects).")
    pi.add_argument("plan", type=Path)
    pi.add_argument("--shorts", type=int, default=None)
    pi.add_argument("--seed", type=int, default=None)
    pi.add_argument("--reject", nargs="*", default=[], help="Short ids to drop")
    pi.add_argument("--resolve", action="store_true", help="Re-resolve highway pool")
    pi.add_argument("--pool", type=int, default=400)
    pi.set_defaults(func=_cmd_iterate)

    pr = sub.add_parser("render", help="Render planned shorts to MP4.")
    pr.add_argument("plan", type=Path)
    pr.add_argument("--only", nargs="*", default=None, help="Render only these short ids")
    pr.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    pr.add_argument("--pool", type=int, default=400)
    pr.set_defaults(func=_cmd_render)

    pl = sub.add_parser("list", help="List plans (disk or Neo4j).")
    pl.add_argument("--plans-dir", type=Path, default=DEFAULT_PLANS_DIR)
    pl.add_argument("--neo4j", action="store_true")
    pl.set_defaults(func=_cmd_list)

    pm = sub.add_parser("montage", help="Plan a Day-in-the-Drive ambient montage (no narration).")
    pm.add_argument("--shorts", type=int, default=3)
    pm.add_argument("--dur", type=float, default=30.0, help="Seconds per short")
    pm.add_argument("--mood", type=str, default=None,
                    help="calm | focused | night (drives caption mood/road)")
    pm.add_argument("--pool", type=int, default=400, help="Highway anchor pool size")
    pm.add_argument("--seed", type=int, default=None)
    pm.add_argument("--plans-dir", type=Path, default=DEFAULT_PLANS_DIR)
    pm.set_defaults(func=_cmd_montage)

    ph = sub.add_parser("highlights", help="Plan event-highlight shorts (music/review/speed).")
    ph.add_argument("--kinds", nargs="*", default=["music", "review", "speed"],
                    choices=["music", "review", "speed"],
                    help="Event kinds to mine")
    ph.add_argument("--per-kind", type=int, default=2, help="Shorts per event kind")
    ph.add_argument("--dur", type=float, default=20.0, help="Seconds per short")
    ph.add_argument("--pool", type=int, default=400, help="Highway anchor pool size")
    ph.add_argument("--seed", type=int, default=None)
    ph.add_argument("--plans-dir", type=Path, default=DEFAULT_PLANS_DIR)
    ph.set_defaults(func=_cmd_highlights)

    pt = sub.add_parser("trip", help="Plan 'Trip Story' shorts (geo + footage sequence).")
    pt.add_argument("--trip-key", type=str, default=None,
                   help="Specific Trip uniqueKey; else the N longest trips.")
    pt.add_argument("--shorts", type=int, default=3)
    pt.add_argument("--dur", type=float, default=30.0, help="Seconds per short")
    pt.add_argument("--shots-per-short", dest="shots_per_short", type=int, default=4)
    pt.add_argument("--clip-dur", dest="clip_dur", type=float, default=6.0)
    pt.add_argument("--seed", type=int, default=1)
    pt.add_argument("--plans-dir", type=Path, default=DEFAULT_PLANS_DIR)
    pt.set_defaults(func=_cmd_trip)

    return p


def main(argv: List[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
