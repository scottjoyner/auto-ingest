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
                 overwrite: bool = False) -> Path:
    """Render one :class:`PlannedShort` to a 9:16 MP4. Returns the output path."""
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
    compose_scripted_short(
        shots, cues, out_path,
        profile_name=profile_name, width=width, height=height, bitrate=bitrate,
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
            sess.run(
                """
                MERGE (sh:Short {key:$key})
                SET sh.title=$title, sh.topic=$topic, sh.status=$status,
                    sh.plan_id=$pid, sh.out_path=$out, sh.updated_at=timestamp()
                WITH sh
                MATCH (sp:ShortPlan {plan_id:$pid})
                MERGE (sp)-[:PLANS]->(sh)
                """,
                key=s.id, title=s.title, topic=s.brief_topic,
                status=s.status, pid=plan.plan_id, out=s.out_path,
            )
    log.info("Upserted manifest for plan %s (%d shorts)", plan.plan_id, len(plan.shorts))


def render_plan(plan: Plan, out_dir: Path, *, only: Optional[List[str]] = None,
                **render_kwargs) -> List[Path]:
    """Render every (or selected) short in a plan; update statuses in place."""
    out: List[Path] = []
    for item in plan.shorts:
        if only and item.id not in only:
            continue
        if item.status == "rejected":
            continue
        try:
            p = render_short(item, out_dir, **render_kwargs)
            out.append(p)
        except Exception as e:  # pragma: no cover - render failures are environment-specific
            log.error("Render failed for %s: %s", item.id, e)
    return out
