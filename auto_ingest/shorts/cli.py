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
import json
import logging
import os
import sys
from pathlib import Path
from typing import List

from auto_ingest.shorts import backdrop, curator, planner, publish, render
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
    # Pass the live driver so real graph content is mined (reveal + myth/fact)
    # instead of falling back to templated text (S-G10).
    plan = planner.plan_shorts(
        brief, anchors, short_count=args.shorts, short_dur=args.dur,
        shots_per_short=args.shots_per_short, seed=args.seed,
        driver=driver,
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
        paths = render.render_plan(
            plan, out_dir, only=args.only, tts=args.tts,
            skip_used_clips=True, upsert=args.upsert,
            skip_history=args.skip_history,
        )
        if not args.upsert:
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
    pr.add_argument("--tts", action="store_true",
                    help="Narrate shorts in the owner's voice (XTTS-v2 clone)")
    pr.add_argument("--skip-used", action="store_true",
                    help="Skip shorts whose B-roll was used in a prior rendered short")
    pr.add_argument("--upsert", action="store_true",
                    help="Persist :ShortPlan/:Short manifest (status/published) to Neo4j")
    pr.add_argument("--skip-history", action="store_true",
                    help="Also skip clips already rendered in PRIOR batches (off by "
                         "default so iterations keep producing fresh shorts)")
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


    ppv = sub.add_parser("publish", help="Publish workflow: queue/inspect/upload rendered shorts.")
    ppv.add_argument("action", choices=["queue", "list", "upload", "validate", "brand-check", "auth", "tiktok", "instagram", "metrics", "ab", "schedule"],
                      help="queue = stage; list = show staged; upload = push; "
                           "validate = check queue integrity; "
                           "brand-check = verify brand assets/handles/bios are present; "
                           "auth/tiktok/instagram = bootstrap an OAuth token (sign-in link); "
                           "schedule = drive the posting calendar into the queue")
    ppv.add_argument("target", nargs="?", default=None,
                     help="Optional platform for 'auth' (e.g. youtube)")
    ppv.add_argument("--platform", default="youtube_shorts",
                     help="Single platform for queue/list (default youtube_shorts)")
    ppv.add_argument("--platforms", nargs="*", default=["youtube_shorts", "tiktok", "instagram"],
                     help="Platforms for disk-based queue + upload (all by default)")
    ppv.add_argument("--disk", action="store_true",
                     help="Stage from rendered MP4s on disk instead of Neo4j (DB-independent)")
    ppv.add_argument("--out-root",
                     default="/media/scott/NAS5/fileserver/dashcam_timelapse/_mix_run/narrated",
                     help="Root dir scanned by --disk")
    ppv.add_argument("--limit", type=int, default=0)
    ppv.add_argument("--plans-dir", type=Path, default=DEFAULT_PLANS_DIR,
                     help="Plan JSON dir used to enrich queue items (hook/thumbnail)")
    ppv.add_argument("--dry-run", action="store_true", default=True,
                     help="Report what would upload; no network calls (default)")
    ppv.add_argument("--no-dry-run", dest="dry_run", action="store_false",
                     help="Actually upload (requires platform credentials in env)")
    ppv.add_argument("--headless", action="store_true",
                      help="auth: use manual code-paste instead of local browser server")
    ppv.add_argument("--brand-dir", type=Path,
                      default=Path(__file__).resolve().parent.parent.parent / "docs" / "brand",
                      help="Brand assets/manifest dir for brand-check")
    ppv.add_argument("--metrics-file", default=None,
                      help="metrics: CSV/JSON file to ingest (with metrics ingest)")
    ppv.add_argument("--metrics-store", type=Path, default=None,
                      help="metrics: override the JSONL store path")
    ppv.add_argument("--metrics-format", default="csv", choices=["csv", "json"],
                      help="metrics: ingest file format (csv or json array)")
    ppv.add_argument("--plan-file", type=Path, default=None,
                      help="plan JSON for `metrics predict` (score + store virality)")
    ppv.add_argument("--as-json", action="store_true",
                      help="metrics: print report as JSON instead of readable text")
    ppv.add_argument("--days", type=int, default=7,
                     help="schedule: build a calendar this many days long (default 7)")
    ppv.add_argument("--date", dest="date", default=None,
                     help="schedule: YYYY-MM-DD (or day index) of the day to emit; "
                          "defaults to today = day 1")
    ppv.add_argument("--apply", action="store_true",
                      help="schedule: actually enqueue the day's ready slots "
                           "(via publish.queue_for_publish); without it, report-only")
    # A/B testing args (used when action == 'ab'; sub-action via `target`).
    ppv.add_argument("--key", default=None, help="ab/metrics: item key (short id)")
    ppv.add_argument("--variants", type=int, default=2,
                      help="ab: number of variants to generate (default 2)")
    ppv.add_argument("--variant", type=int, default=None,
                      help="ab: winning variant index (for 'choose')")
    ppv.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                      help="ab: dir to write thumbnail variants into")
    ppv.set_defaults(func=_cmd_publish)

    return p


def _brand_check(brand_dir: Path) -> "tuple[bool, list]":
    """Verify brand assets + manifest are present and well-formed.

    Returns (ok, issues). Pure: no network, no credentials. Checks the manifest
    JSON (handles/bios present) and that each referenced asset file exists and
    is a valid image of the expected dimensions.
    """
    from PIL import Image

    ok = True
    issues: list = []
    brand_dir = Path(brand_dir)
    manifest = brand_dir / "brand_manifest.json"
    if not manifest.exists():
        return False, [f"missing brand_manifest.json at {manifest}"]
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception as e:
        return False, [f"brand_manifest.json unreadable: {e}"]

    for plat, handle in (data.get("handles") or {}).items():
        if not handle:
            ok = False
            issues.append(f"handle missing for {plat}")
    for plat, bio in (data.get("bios") or {}).items():
        if not bio:
            ok = False
            issues.append(f"bio missing for {plat}")

    expected = {"avatar": (1000, 1000), "banner_youtube": (2560, 1440)}
    for key, (ew, eh) in expected.items():
        rel = (data.get("assets") or {}).get(key)
        if not rel:
            ok = False
            issues.append(f"asset path missing for {key}")
            continue
        p = brand_dir / rel
        if not p.exists():
            ok = False
            issues.append(f"asset missing: {p}")
            continue
        try:
            with Image.open(p) as im:
                if im.size != (ew, eh):
                    ok = False
                    issues.append(f"{p.name} is {im.size}, expected {(ew, eh)}")
        except Exception as e:
            ok = False
            issues.append(f"{p.name} not a valid image: {e}")
    return ok, issues


def _cmd_ab_plan(args) -> int:
    from auto_ingest.shorts import abtest, uploader

    items = uploader.load_queue()
    item = next((i for i in items
                 if i.key == args.key and i.platform == args.platform), None)
    if item is None:
        print(f"No queued short for key={args.key} platform={args.platform}")
        return 1
    out_dir = Path(args.out_dir)
    mp4 = Path(item.out_path)
    if not mp4.exists():
        print(f"Rendered MP4 missing: {mp4}")
        return 1
    topic = item.topic.replace("_", " ")
    title = (item.title or item.key).replace("_", " ")
    n = args.variants
    thumbs = abtest.make_thumbnail_variants(
        mp4, out_dir / item.key, title=title, topic=topic, n=n)
    # Title variants: build a Brief-like object from the queue item.
    from auto_ingest.shorts.models import Brief
    brief = Brief(topic=item.topic, title=item.title or item.key,
                  hook=item.brief_hook or "", tags=[])
    titles = abtest.title_variants(brief, n=n)
    plan = abtest.plan_variants(item.key, args.platform, thumbs, titles)
    print(f"Planned {len(thumbs)} thumbnail + {len(titles)} title variants "
          f"for {item.key}/{args.platform}")
    for i, (t, h) in enumerate(zip(plan.thumb_variants, plan.title_variants)):
        print(f"  v{i}: thumb={t}\n       title={h}")
    return 0


def _cmd_ab_list(args) -> int:
    from auto_ingest.shorts import abtest

    plans = abtest.load_variants(short_id=args.key, platform=args.platform)
    if not plans:
        print("No A/B variant plans stored.")
        return 0
    for pl in plans:
        winner = pl.winner if pl.winner is not None else "-"
        print(f"{pl.short_id:18} {pl.platform:14} "
              f"variants={len(pl.thumb_variants)} "
              f"active={pl.active_variant} winner={winner}")
        for i, (t, h) in enumerate(zip(pl.thumb_variants, pl.title_variants)):
            mark = "*" if i == pl.active_variant else " "
            print(f"  {mark}v{i}: thumb={t}  title={h}")
    return 0


def _cmd_ab_choose(args) -> int:
    from auto_ingest.shorts import abtest

    plan = abtest.choose_winner(args.key, args.platform, args.variant)
    if plan is None:
        print(f"No variant plan for key={args.key} platform={args.platform}")
        return 1
    print(f"Recorded winner=v{plan.winner} for {args.key}/{args.platform} "
          f"at {plan.chosen_at}")
    return 0


def _cmd_publish(args) -> int:
    if args.action == "queue":
        if args.disk:
            n = publish.queue_all_on_disk(args.out_root, args.platforms, limit=args.limit)
            print(f"Staged {n} (short,platform) entries from disk under {args.out_root}")
        else:
            n = publish.queue_all_unpublished(platform=args.platform, limit=args.limit)
            print(f"Queued {n} short(s) for {args.platform}")
        return 0
    if args.action == "list":
        from auto_ingest.shorts import uploader
        items = uploader.load_queue()
        for it in items:
            print(f"{it.platform:14} {it.key}  {it.topic}  -> {it.out_path}")
        print(f"{len(items)} queued short(s)")
        return 0
    if args.action == "validate":
        from auto_ingest.shorts import uploader
        issues = uploader.validate_queue(platforms=args.platforms or None)
        if not issues:
            print(f"OK: queue intact ({len(uploader.load_queue())} items)")
            return 0
        for iss in issues:
            print(f"[{iss['issue']}] {iss['platform']} {iss['key']} -> {iss['path']}")
        print(f"{len(issues)} issue(s) found")
        return 1
    if args.action == "brand-check":
        ok, issues = _brand_check(args.brand_dir)
        if ok:
            print(f"OK: brand assets + manifest intact at {args.brand_dir}")
            return 0
        for iss in issues:
            print(f"[brand] {iss}")
        print(f"{len(issues)} brand issue(s) found")
        return 1
    if args.action in ("auth", "youtube"):
        plat = args.target or "youtube"
        if plat == "youtube":
            from auto_ingest.shorts import yt_auth
            tok = yt_auth.bootstrap_token(headless=args.headless)
        elif plat == "tiktok":
            from auto_ingest.shorts import tiktok_auth
            tok = tiktok_auth.bootstrap_token(headless=args.headless)
        elif plat == "instagram":
            from auto_ingest.shorts import instagram_auth
            tok = instagram_auth.bootstrap_token(headless=args.headless)
        else:
            print(f"Unknown platform for auth: {plat}")
            return 2
        print(f"{plat} authenticated. Token at {tok}")
        return 0
    if args.action == "tiktok":
        from auto_ingest.shorts import tiktok_auth
        tok = tiktok_auth.bootstrap_token(headless=args.headless)
        print(f"TikTok authenticated. Token at {tok}")
        return 0
    if args.action == "instagram":
        from auto_ingest.shorts import instagram_auth
        tok = instagram_auth.bootstrap_token(headless=args.headless)
        print(f"Instagram authenticated. Token at {tok}")
        return 0
    if args.action == "ab":
        sub = args.target or "list"
        if sub == "plan":
            return _cmd_ab_plan(args)
        if sub == "list":
            return _cmd_ab_list(args)
        if sub == "choose":
            return _cmd_ab_choose(args)
        print(f"Unknown ab sub-action: {sub} (use plan|list|choose)")
        return 2
    if args.action == "metrics":
        return _cmd_publish_metrics(args)
    if args.action == "schedule":
        return _cmd_schedule(args)
    # upload
    from auto_ingest.shorts import uploader
    from auto_ingest.shorts.publish_guard import LivePublishForbidden, require_live_mode
    if args.dry_run:
        items = uploader.load_queue()
        if args.platforms:
            items = [i for i in items if i.platform in args.platforms]
        print(f"[dry-run] {len(items)} queued item(s) — what WOULD upload:\n")
        for it in items:
            rep = uploader.plan_report(it, plans_dir=args.plans_dir)
            print(f"[{rep['platform']}] {rep['key']}")
            print(f"  file:      {rep['file']}  (exists={rep['file_exists']})")
            print(f"  title:     {rep['title']}")
            print(f"  desc:      {rep['description']}")
            print(f"  hashtags:  {rep['hashtags']}")
            print(f"  caption:   {rep.get('caption', '')}")
            print(f"  thumbnail: {rep['thumbnail']}  (uses_thumbnail={rep.get('uses_thumbnail')})")
        print(f"\n[dry-run] {len(items)} item(s) reported. No network/creds used.")
        return 0
    try:
        require_live_mode(where="upload")
    except LivePublishForbidden as e:
        print(f"[blocked] {e}")
        return 2
    attempted = uploader.process_queue(
        platforms=args.platforms, dry_run=False)
    print(f"[LIVE] upload: attempted {attempted} item(s)")
    return 0


def _cmd_schedule(args) -> int:
    """Activate the posting scheduler: turn a calendar day into queue entries.

    Safe by default — report-only unless ``--apply``. Never uploads, never
    touches OAuth/creds. With ``--apply``, ready slots (a short_id that exists
    on disk) are enqueued via the existing idempotent ``publish.queue_for_publish``.
    """
    import datetime as _dt

    from auto_ingest.shorts import publish, scheduling, uploader

    # Resolve the target day index. --date may be a YYYY-MM-DD (reinterpreted as
    # day 1 = today) or a plain 1-based index; default = today = day 1.
    day_index = 1
    if args.date:
        try:
            _dt.date.fromisoformat(args.date)  # valid date -> treat as "today"
            day_index = 1
        except ValueError:
            try:
                day_index = int(args.date)
            except ValueError:
                print(f"--date must be YYYY-MM-DD or a day index (got {args.date!r})")
                return 2
    if day_index < 1:
        print("--date day index must be >= 1")
        return 2

    plans = scheduling.build_calendar(days=args.days)
    slots = [s for dp in plans if dp.day == day_index for s in dp.slots]
    if not slots:
        print(f"[schedule] no slots for day {day_index} (calendar has {len(plans)} days)")
        return 0

    queued = uploader.load_queue()
    print(f"[schedule] day {day_index} — {len(slots)} slot(s) "
          f"(apply={args.apply}):\n")
    n_queued = 0
    for s in slots:
        found = None
        if s.short_id:
            found = next((q for q in queued if q.key == s.short_id
                          and q.platform == s.platform), None)
            title = found.title if found else s.short_id.replace("_", " ")
            exists = found is not None and Path(found.out_path).exists()
            status = "READY"
        else:
            title = f"({s.note or 'NEEDS_RENDER'})"
            exists = False
            status = "NEEDS_RENDER"

        # Best-effort platform-native title without creds.
        if found is not None:
            title = uploader._title_for(s.platform,
                                        uploader.QueueItem(
                                            key=found.key, topic=found.topic,
                                            title=found.title, out_path=found.out_path,
                                            platform=found.platform))
        print(f"  {s.time_local}  {s.platform:14}  {s.topic}")
        print(f"      id:    {s.short_id or '—'}  ({status})")
        print(f"      title: {title}")

        if args.apply and s.short_id and exists:
            pub = publish.Publishable(
                key=found.key, topic=found.topic, title=found.title,
                out_path=found.out_path, platform=found.platform)
            publish.queue_for_publish(pub, platform=s.platform)
            n_queued += 1
            log.info("schedule: enqueued %s -> %s", s.short_id, s.platform)

    mode = "APPLIED" if args.apply else "DRY-RUN"
    print(f"\n[{mode}] day {day_index}: {len(slots)} slot(s), "
          f"{n_queued} enqueued.")
    return 0


def _cmd_publish_metrics(args) -> int:
    from auto_ingest.shorts import feedback, metrics

    store = args.metrics_store
    sub = args.target or "report"
    if sub == "ingest":
        if not args.metrics_file:
            print("metrics ingest requires --metrics-file")
            return 2
        platform = args.platform
        if args.metrics_format == "json":
            rows = json.loads(Path(args.metrics_file).read_text(encoding="utf-8"))
            n = metrics.ingest_dicts(platform, rows, path=store)
        else:
            n = metrics.ingest_platform_csv(platform, args.metrics_file, path=store)
        print(f"Ingested {n} metric record(s) for {platform}")
        return 0
    if sub == "report":
        recs = metrics.load_metrics(path=store)
        if not recs:
            print("No metrics recorded yet. Ingest with: "
                  "publish metrics ingest --platform youtube --file x.csv")
            return 0
        rep = feedback.feedback_report(recs)
        sugg = feedback.suggest_next(recs)
        if args.as_json:
            print(json.dumps({"report": rep, "suggestions": sugg}, indent=2))
        else:
            print("=== Performance by platform ===")
            for p, v in rep["by_platform"].items():
                print(f"  {p}: avg_views={v['avg_views']}  avg_retention={v['avg_retention']}%")
            print("\n=== Top hooks/topics (by retention) ===")
            for h in rep["top_hooks"]:
                print(f"  {h['text']!r}: views={h['avg_views']} retention={h['avg_view_pct']}%")
            print("\n=== Weak hooks/topics ===")
            for h in rep["weak_hooks"]:
                print(f"  {h['text']!r}: views={h['avg_views']} retention={h['avg_view_pct']}%")
            print("\n=== Topic performance (avg views) ===")
            for t, v in rep["topic_perf"].items():
                print(f"  {t}: {v}")
            pva = rep["pred_vs_actual"]
            print(f"\nPredicted vs actual (MAE virality~views): {pva['mean_abs_error']} "
                  f"over {pva['n']} matched posts")
            print("\n=== Suggestions for next batch ===")
            for s in sugg:
                print(f"  - {s}")
        return 0
    if sub == "predict":
        if not args.plan_file:
            print("metrics predict requires --plan-file <plan.json>")
            return 2
        from auto_ingest.shorts import metrics as _m
        plan = __import__("json").loads(Path(args.plan_file).read_text(encoding="utf-8"))
        shorts = plan.get("shorts", [])
        platform = args.platform or "youtube_shorts"
        n = 0
        for sd in shorts:
            from auto_ingest.shorts.models import Plan
            sh = Plan.from_dict({"shorts": [sd]}).shorts[0]
            _m.store_prediction_for_short(sh, platform, topic=sh.brief_topic or "",
                                          path=store)
            n += 1
        print(f"Stored virality predictions for {n} short(s) on {platform} "
              f"(loop: virality.score_short -> metrics.publish_prediction)")
        return 0
    print(f"Unknown metrics sub-action: {sub} (use 'ingest', 'report', or 'predict')")
    return 2


def main(argv: List[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
