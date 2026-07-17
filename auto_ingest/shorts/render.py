"""Render planned research shorts to video + upsert a Neo4j manifest.

Rendering requires the moviepy/whisper image (Compose or .venv). Planning and
curation do not. ``render_short`` is the only module that touches the video
stack; it is imported lazily so the rest of the package is unit-testable here.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

from auto_ingest.shorts.models import Plan, PlannedShort

log = logging.getLogger("shorts.render")

PROFILE_NAME = os.getenv("SHORTS_PROFILE", "clean")
WIDTH = int(os.getenv("SHORTS_W", "1080"))
HEIGHT = int(os.getenv("SHORTS_H", "1920"))


def _neo4j_creds():
    from auto_ingest_config import get_neo4j_password
    return (
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        os.getenv("NEO4J_USER", "neo4j"),
        get_neo4j_password(),
        os.getenv("NEO4J_DB", "neo4j"),
    )


def render_short(item: PlannedShort, out_dir: Path, *,
                 profile_name: str = PROFILE_NAME, width: int = WIDTH,
                 height: int = HEIGHT, bitrate: str = "6M",
                 overwrite: bool = False, tts: bool = False) -> Path:
    """Render one :class:`PlannedShort` to a 9:16 MP4. Returns the output path.

    When ``tts`` is set, the short's scripted cues are synthesized in the
    owner's voice (XTTS-v2 voice-clone) and muxed as the narration track. If
    TTS is unavailable the short is rendered silently rather than failing.
    """
    from shorts_builder import compose_scripted_short

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{item.id}.mp4"
    if out_path.exists() and not overwrite:
        log.info("Already rendered (skip): %s", out_path)
        item.status = "rendered"
        item.out_path = str(out_path)
        return out_path

    shots = [s.to_dict() for s in item.shots]
    cues = [c.to_dict() for c in item.cues]
    # Skip shorts with no resolvable footage (audio-sourced discussion lines
    # whose dashcam clip isn't mounted here) rather than crashing the renderer.
    if not any((sh.get("fr_path") and Path(sh["fr_path"]).is_file()) for sh in shots):
        log.info("Skip %s (no resolvable footage)", item.id)
        item.status = "rejected"
        return out_path
    narration_audio = None
    if tts:
        try:
            from auto_ingest.shorts.tts import narrate
            narration_audio = narrate(item.title, cues)
        except Exception as e:  # pragma: no cover - TTS is environment-dependent
            log.info("TTS narration skipped for %s: %s", item.id, e)
            narration_audio = None
    compose_scripted_short(
        shots, cues, out_path,
        profile_name=profile_name, width=width, height=height, bitrate=bitrate,
        narration_audio=narration_audio,
        hashtag=item.brief_topic or "",
    )
    item.status = "rendered"
    item.out_path = str(out_path)
    log.info("Rendered short %s -> %s", item.id, out_path)
    return out_path


def upsert_manifest(driver, plan: Plan, out_dir: Path) -> None:
    """Persist the plan + rendered shorts into Neo4j as :ShortPlan / :Short."""
    from neo4j import GraphDatabase

    uri, user, pw, db = _neo4j_creds()
    with GraphDatabase.driver(uri, auth=(user, pw)).session(database=db) as sess:
        sess.run(
            """
            MERGE (sp:ShortPlan {plan_id:$pid})
            SET sp.topic=$topic, sp.iteration=$iter,
                sp.updated_at=timestamp(), sp.brief_title=$title
            """,
            pid=plan.plan_id, topic=plan.topic, iter=plan.iteration,
            title=plan.brief.title,
        )
        for s in plan.shorts:
            clip_keys = [sh.clip_key for sh in s.shots if sh.clip_key]
            sess.run(
                """
                MERGE (sh:Short {key:$key})
                SET sh.title=$title, sh.topic=$topic, sh.status=$status,
                    sh.plan_id=$pid, sh.out_path=$out, sh.updated_at=timestamp(),
                    sh.clip_keys=$cks, sh.published=false
                WITH sh
                MATCH (sp:ShortPlan {plan_id:$pid})
                MERGE (sp)-[:PLANS]->(sh)
                """,
                key=s.id, title=s.title, topic=s.brief_topic,
                status=s.status, pid=plan.plan_id, out=s.out_path,
                cks=clip_keys,
            )
    log.info("Upserted manifest for plan %s (%d shorts)", plan.plan_id, len(plan.shorts))


def render_plan(plan: Plan, out_dir: Path, *, only: Optional[List[str]] = None,
                skip_used_clips: bool = False, upsert: bool = False,
                tts: bool = False, skip_history: bool = False,
                **render_kwargs) -> List[Path]:
    """Render every (or selected) short in a plan; update statuses in place.

    Dedup behavior (controls clip reuse across a run):
    * ``skip_used_clips`` (default True via CLI) dedups *within the plan* so a
      B-roll clip is not reused across two shorts in the same batch.
    * ``skip_history`` additionally skips clips already rendered in a *prior*
      ``:Short`` (status 'rendered') in Neo4j. This is OFF by default so that
      repeated iterations keep producing fresh shorts from the same highway
      pool instead of being starved by earlier batches.

    When ``upsert`` is set, the plan + rendered shorts are persisted to Neo4j
    as :ShortPlan / :Short (status 'rendered'), giving a publish-aware audit
    trail (and, when ``skip_history`` is set, feeding the dedup logic).
    """
    used: set = set()
    if skip_history:
        used = _already_rendered_clips()
    out: List[Path] = []
    for item in plan.shorts:
        if only and item.id not in only:
            continue
        if item.status == "rejected":
            continue
        # Dedup: skip if any shot's clip was already used earlier in THIS
        # plan's render loop (skip_used_clips) OR in a prior batch (skip_history).
        if (skip_used_clips or skip_history) and any(s.clip_key in used for s in item.shots):
            log.info("Skip %s (clip already used)", item.id)
            continue
        try:
            p = render_short(item, out_dir, tts=tts, **render_kwargs)
            out.append(p)
            if item.out_path:
                item.status = "rendered"
                for s in item.shots:
                    if s.clip_key:
                        used.add(s.clip_key)
        except Exception as e:  # pragma: no cover - render failures are environment-specific
            log.error("Render failed for %s: %s", item.id, e)
    if upsert and out:
        try:
            from neo4j import GraphDatabase
            uri, user, pw, db = _neo4j_creds()
            with GraphDatabase.driver(uri, auth=(user, pw)) as drv:
                upsert_manifest(drv, plan, out_dir)
        except Exception as e:  # pragma: no cover
            log.warning("Manifest upsert failed: %s", e)
    return out


def _already_rendered_clips() -> set:
    """Clip keys already used by a rendered ``:Short`` (dedup source)."""
    from neo4j import GraphDatabase
    uri, user, pw, db = _neo4j_creds()
    out: set = set()
    try:
        with GraphDatabase.driver(uri, auth=(user, pw), database=db) as drv:
            with drv.session(database=db) as sess:
                for r in sess.run(
                    "MATCH (sh:Short {status:'rendered'}) "
                    "RETURN sh.clip_keys AS ks"
                ).data():
                    ks = r.get("ks") or []
                    out.update(ks)
    except Exception as e:  # graph optional for rendering
        log.warning("Could not read rendered-clip history: %s", e)
    return out
