"""Publish workflow for rendered narrated shorts.

Rendered ``:Short`` nodes carry ``status='rendered'`` and ``published=false``.
This module turns that into an actual publish pipeline:

* ``list_unpublished``   - shorts ready to post (rendered, not yet published).
* ``queue_for_publish``  - append a short to a local publish queue (JSON) so a
  downstream uploader (YouTube/Shorts/TikTok) can consume it. Keeps this repo
  free of platform credentials; the uploader reads the queue + the MP4 path.
* ``mark_published``     - set ``published=true``, ``published_at``, ``platform``
  on the ``:Short`` so posted-dedup works (a short is never queued twice).

Nothing here calls a network API; it only stages + records intent.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

log = logging.getLogger("shorts.publish")

QUEUE_PATH = Path(os.environ.get(
    "SHORTS_PUBLISH_QUEUE",
    "/media/scott/NAS5/fileserver/dashcam_timelapse/_mix_run/publish_queue.jsonl",
))


def _neo4j_creds():
    from auto_ingest_config import get_neo4j_password
    return (
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        os.getenv("NEO4J_USER", "neo4j"),
        get_neo4j_password(),
        os.getenv("NEO4J_DB", "neo4j"),
    )


@dataclass
class Publishable:
    key: str
    topic: str
    title: str
    out_path: str
    platform: Optional[str] = None
    published_at: Optional[str] = None


def list_unpublished(limit: int = 50) -> List[Publishable]:
    """Shorts that are rendered but not yet published."""
    from neo4j import GraphDatabase
    uri, user, pw, db = _neo4j_creds()
    out: List[Publishable] = []
    with GraphDatabase.driver(uri, auth=(user, pw)).session(database=db) as sess:
        for r in sess.run(
            "MATCH (sh:Short {status:'rendered'}) "
            "WHERE sh.published = false OR sh.published IS NULL "
            "RETURN sh.key AS k, sh.topic AS topic, sh.title AS title, "
            "sh.out_path AS out ORDER BY sh.updated_at DESC LIMIT $lim",
            lim=limit,
        ).data():
            out.append(Publishable(
                key=r["k"], topic=r.get("topic") or "",
                title=r.get("title") or "", out_path=r.get("out") or ""))
    return out


def queue_for_publish(short: Publishable, platform: str = "youtube_shorts") -> Path:
    """Append a short to the publish queue (JSONL). Idempotent per (key,platform)."""
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "key": short.key, "topic": short.topic, "title": short.title,
        "out_path": short.out_path, "platform": platform,
        "queued_at": datetime.now(timezone.utc).isoformat(),
    }
    # Skip if already queued (posted-dedup at queue level).
    if QUEUE_PATH.exists():
        for line in QUEUE_PATH.read_text().splitlines():
            try:
                if json.loads(line).get("key") == short.key and                    json.loads(line).get("platform") == platform:
                    log.info("Already queued: %s/%s", short.key, platform)
                    return QUEUE_PATH
            except Exception:
                continue
    with QUEUE_PATH.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    log.info("Queued %s -> %s (%s)", short.key, platform, short.out_path)
    return QUEUE_PATH


def mark_published(key: str, platform: str,
                   published_at: Optional[str] = None) -> None:
    """Record that a short was published (sets :Short flags)."""
    from neo4j import GraphDatabase
    uri, user, pw, db = _neo4j_creds()
    ts = published_at or datetime.now(timezone.utc).isoformat()
    with GraphDatabase.driver(uri, auth=(user, pw)).session(database=db) as sess:
        sess.run(
            "MATCH (sh:Short {key:$k}) "
            "SET sh.published=true, sh.published_at=$ts, sh.platform=$plat",
            k=key, ts=ts, plat=platform,
        )
    log.info("Marked published: %s (%s)", key, platform)


def queue_all_unpublished(platform: str = "youtube_shorts", limit: int = 50) -> int:
    """Stage every unpublished short into the publish queue. Returns count."""
    n = 0
    for s in list_unpublished(limit=limit):
        if not s.out_path:
            continue
        queue_for_publish(s, platform=platform)
        n += 1
    return n


def scan_rendered(out_root: Path) -> List[Publishable]:
    """List rendered MP4s directly from disk (DB-independent staging).

    The parent directory is treated as the topic and the filename stem as the
    short key. This lets us stage a publish queue even when Neo4j is saturated.
    """
    root = Path(out_root)
    out: List[Publishable] = []
    if not root.exists():
        return out
    for mp4 in sorted(root.rglob("*.mp4")):
        topic = mp4.parent.name
        key = mp4.stem
        out.append(Publishable(
            key=key, topic=topic, title=key.replace("_", " "),
            out_path=str(mp4)))
    return out


def queue_all_on_disk(out_root: Path, platforms: List[str], limit: int = 0) -> int:
    """Stage every rendered MP4 on disk into the queue for each platform.

    Idempotent per (key, platform) via :func:`queue_for_publish`. Returns the
    number of (short, platform) entries appended.
    """
    items = scan_rendered(out_root)
    if limit:
        items = items[:limit]
    n = 0
    for it in items:
        for plat in platforms:
            before = _queue_file_lines()
            queue_for_publish(it, platform=plat)
            if _queue_file_lines() > before:
                n += 1
    log.info("Staged %d (short,platform) entries from disk under %s", n, out_root)
    return n


def _queue_file_lines() -> int:
    if not QUEUE_PATH.exists():
        return 0
    return sum(1 for _ in QUEUE_PATH.read_text().splitlines() if _.strip())
