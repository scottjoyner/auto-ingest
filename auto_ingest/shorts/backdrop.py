"""Select generic highway dashcam footage to use as B-roll backdrop.

The knowledge graph links frames to clips (``Frame-[:BELONGS_TO]->DashcamClip``)
and frames carry ``mph``. "Highway" footage = front-view clips (``view:'F'``)
with frames at highway speed. We return a pool of (clip, timecode) anchors the
planner draws from so every short gets neutral driving B-roll, not a specific
event.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger("shorts.backdrop")

DEFAULT_MPH_MIN = float(os.getenv("SHORTS_MPH_MIN", "45.0"))


def _dashcam_root() -> Path:
    from auto_ingest_config import get_dashcam_root
    return Path(get_dashcam_root())


def _fr_path_for_key(clip_key: str, root: Path) -> Optional[Path]:
    """Resolve the front-view MP4 for a clip key under the dashcam root.

    Clips are stored YYYY/MM/DD/<key>_F.MP4. The key in Neo4j may already carry
    the ``_F`` suffix or not.
    """
    key = clip_key
    if not key.endswith(("_F", "_R")):
        key = f"{key}_F"
    # clip keys look like 2025_1002_175045_F -> 2025/10/02/2025_1002_175045_F.MP4
    parts = key.split("_")
    if len(parts) >= 3:
        year, mmdd = parts[0], parts[1]
        month = mmdd[:2]
        day = mmdd[2:]
        cand = root / year / month / day / f"{key}.MP4"
        if cand.exists():
            return cand
    # Fallback: glob under the root for the key.
    matches = list(root.rglob(f"{key}.MP4"))
    if matches:
        return matches[0]
    return None


def select_highway_pool(driver, *, mph_min: float = DEFAULT_MPH_MIN,
                        limit: int = 400, seed_view: str = "F") -> List[Dict[str, object]]:
    """Return a pool of highway footage anchors.

    Each anchor: ``{"clip_key", "fr_path", "t_sec", "mph"}`` where ``fr_path`` is
    resolved on this host (or None if the file is not mounted here).
    """
    root = _dashcam_root()
    with driver.session() as sess:
        rows = sess.run(
            """
            MATCH (f:Frame)-[:BELONGS_TO]->(c:DashcamClip)
            WHERE c.view = $view AND f.mph >= $mph
            WITH c, f
            ORDER BY f.mph DESC
            RETURN c.key AS clip_key, f.frame AS frame_no, f.mph AS mph,
                   c.fps AS fps, c.path AS path
            LIMIT $limit
            """,
            view=seed_view, mph=mph_min, limit=limit,
        ).data()

    anchors: List[Dict[str, object]] = []
    for r in rows:
        clip_key = r.get("clip_key")
        fps = r.get("fps") or 30.0
        frame_no = r.get("frame_no") or 0
        t_sec = float(frame_no) / float(fps) if fps else 0.0
        fr = _fr_path_for_key(clip_key, root)
        anchors.append({
            "clip_key": clip_key,
            "fr_path": str(fr) if fr else None,
            "t_sec": t_sec,
            "mph": r.get("mph"),
            "path": r.get("path"),
        })
    log.info("Highway pool: %d anchors (mph>=%.0f, view=%s); %d resolvable here",
             len(anchors), mph_min, seed_view,
             sum(1 for a in anchors if a["fr_path"]))
    return anchors


def pick_shots(anchors: List[Dict[str, object]], *, count: int = 3,
               min_gap_sec: float = 8.0, clip_dur: float = 6.0,
               rng_seed: Optional[int] = None) -> List[Dict[str, object]]:
    """Pick ``count`` non-overlapping, locally-available shots from the pool."""
    import random
    rnd = random.Random(rng_seed)
    available = [a for a in anchors if a["fr_path"]]
    rnd.shuffle(available)
    picked: List[Dict[str, object]] = []
    last_end: Dict[str, float] = {}
    for a in available:
        if len(picked) >= count:
            break
        ck = a["clip_key"]
        start = float(a["t_sec"])
        # keep shots of the same clip separated
        if ck in last_end and start < last_end[ck] + min_gap_sec:
            continue
        picked.append({
            "clip_key": ck,
            "fr_path": a["fr_path"],
            "t_sec": start,
            "dur": clip_dur,
            "mph": a["mph"],
        })
        last_end[ck] = start + clip_dur
    return picked
