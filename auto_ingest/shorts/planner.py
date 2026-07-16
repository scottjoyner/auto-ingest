"""Plan research shorts from a :class:`Brief`.

Planning splits a brief into N shorts. Each short gets:
  * a timed **cue** track (caption lines) built from the hook + a subset of
    points + a source line, spread across the short's duration;
  * a **shot** list of highway B-roll pulled from the curated anchor pool.

Planning is deterministic given ``seed`` so a plan is reproducible, and the
``iterate`` step can regenerate with a new seed / different short count to
explore variations. Nothing here touches moviepy — it is pure scheduling.
"""
from __future__ import annotations

import logging
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from auto_ingest.shorts import backdrop
from auto_ingest.shorts.models import Brief, Cue, Plan, PlannedShort, Shot

log = logging.getLogger("shorts.planner")

MONTAGE_KIND = "mood"
MOOD_PRESETS = {
    "calm": ("I-15", "cruising"),
    "focused": ("I-80", "in the zone"),
    "night": ("US-101", "night drive"),
    None: ("I-15", "on the road"),
}

SECS_PER_CUE = 3.2          # approx on-screen time per caption line
HOOK_KIND = "hook"
POINT_KIND = "point"
SOURCE_KIND = "source"


def _distribute_cues(title: str, hook: str, points: List[str], sources: List,
                     duration: float) -> List[Cue]:
    """Build timed cues across ``duration`` seconds."""
    lines: List[tuple] = [(HOOK_KIND, hook)] if hook else []
    lines += [(POINT_KIND, p) for p in points]
    if sources:
        src = "Sources: " + ", ".join(f"[{i + 1}] {s.title}" for i, s in enumerate(sources[:4]))
        lines.append((SOURCE_KIND, src))

    if not lines:
        return []

    n = len(lines)
    # Evenly space cues; last cue ends near duration. Add a small lead-in.
    lead = 0.4
    span = max(duration - lead, 1.0)
    step = span / n
    cues: List[Cue] = []
    for i, (kind, text) in enumerate(lines):
        start = lead + i * step
        end = start + min(step, SECS_PER_CUE)
        cues.append(Cue(start=round(start, 2), end=round(end, 2), text=text, kind=kind))
    return cues


def plan_shorts(brief: Brief, anchors: List[Dict[str, object]], *,
                short_count: int = 3, short_dur: float = 30.0,
                shots_per_short: int = 3, seed: int = 1,
                clip_dur: float = 6.0, min_gap_sec: float = 8.0,
                topic_prefix: str = "") -> Plan:
    """Create a :class:`Plan` of ``short_count`` shorts from a brief."""
    import random
    rnd = random.Random(seed)

    # Choose how many points per short and rotate through them.
    points = list(brief.points) or [brief.hook]
    if not points:
        points = ["(no points curated)"]

    shorts: List[PlannedShort] = []
    for s_idx in range(short_count):
        # rotate a window of points for this short
        start = (s_idx * 2) % max(len(points), 1)
        window = points[start:start + 3] or points[:1]

        cues = _distribute_cues(
            title=brief.title,
            hook=brief.hook if s_idx == 0 else "",
            points=window,
            sources=brief.sources if s_idx == short_count - 1 else [],
            duration=short_dur,
        )

        shots = _pick_shots_deterministic(
            anchors, count=shots_per_short, min_gap_sec=min_gap_sec,
            clip_dur=clip_dur, rng=rnd,
        )

        title = brief.title
        if short_count > 1:
            title = f"{topic_prefix}{brief.title} — Part {s_idx + 1}"

        shorts.append(PlannedShort(
            id=uuid.uuid5(uuid.NAMESPACE_URL, f"{brief.topic}:{seed}:{s_idx}").hex[:10],
            brief_topic=brief.topic,
            title=title,
            cues=cues,
            shots=[Shot(**s) for s in shots],
            notes=f"points[{start}:{start + len(window)}]",
            status="planned",
        ))

    plan = Plan(topic=brief.topic, brief=brief, shorts=shorts, iteration=1)
    log.info("Planned %d shorts for %s (dur=%.0fs, seed=%d)",
             short_count, brief.topic, short_dur, seed)
    return plan


def plan_montage(driver, *, count: int = 3, dur: float = 30.0,
                 mood: Optional[str] = None, limit: int = 400,
                 seed: Optional[int] = None, shots_per_short: int = 3,
                 clip_dur: float = 6.0, min_gap_sec: float = 8.0) -> Plan:
    """Plan ``count`` pure-footage ambient montages (no research narration).

    Picks highway clips via :func:`backdrop.select_highway_pool` +
    :func:`backdrop.pick_shots`, then builds light mood/metadata captions
    (e.g. "72 mph • I-15 • 2:14 PM") from each shot's ``mph``/time. Uses the
    same Plan/PlannedShort/Cue model so the normal renderer handles output.
    """
    rnd = random.Random(seed)
    anchors = backdrop.select_highway_pool(driver, limit=limit)
    if not anchors:
        raise ValueError("No highway footage anchors available for montage")

    road, vibe = MOOD_PRESETS.get(mood, MOOD_PRESETS[None])
    base_clock = datetime(2025, 1, 1, 12, 0, 0) + timedelta(minutes=int(rnd.random() * 600))

    shorts: List[PlannedShort] = []
    for s_idx in range(count):
        seed_i = int(rnd.random() * 1e9)
        picked = backdrop.pick_shots(
            anchors, count=shots_per_short, min_gap_sec=min_gap_sec,
            clip_dur=clip_dur, rng_seed=seed_i,
        )
        if not picked:
            picked = backdrop.pick_shots(
                anchors, count=shots_per_short, min_gap_sec=0.0,
                clip_dur=clip_dur, rng_seed=seed_i,
            )
        shots = [Shot(**s) for s in picked]
        cues = _montage_cues(shots, dur, road=road, vibe=vibe,
                             base_clock=base_clock, idx=s_idx)
        shorts.append(PlannedShort(
            id=uuid.uuid5(uuid.NAMESPACE_URL, f"montage:{mood}:{seed}:{s_idx}").hex[:10],
            brief_topic="montage",
            title=f"Day in the Drive — {road} #{s_idx + 1}",
            cues=cues,
            shots=shots,
            notes=f"mood={mood or 'default'} road={road} vibe={vibe}",
            status="planned",
        ))

    plan = Plan(
        topic="montage",
        brief=Brief(topic="montage", title="Day in the Drive", hook="", points=[]),
        shorts=shorts,
        iteration=1,
    )
    log.info("Planned %d montage short(s) (mood=%s, dur=%.0fs, seed=%s)",
             count, mood, dur, seed)
    return plan


def _montage_cues(shots: List[Shot], dur: float, *, road: str, vibe: str,
                  base_clock: datetime, idx: int) -> List[Cue]:
    if not shots:
        return []
    lead = 0.4
    span = max(dur - lead, 1.0)
    step = span / len(shots)
    cues: List[Cue] = []
    for i, sh in enumerate(shots):
        start = lead + i * step
        end = start + min(step, 3.2)
        clock = (base_clock + timedelta(minutes=idx * 7 + i * 3)).strftime("%I:%M %p")
        mph = f"{sh.mph:.0f} mph" if sh.mph is not None else "— mph"
        text = f"{mph} • {road} • {clock}"
        cues.append(Cue(start=round(start, 2), end=round(end, 2),
                        text=text, kind=MONTAGE_KIND))
    return cues


def plan_highlights(driver, *, kinds: tuple = ("music", "review", "speed"),
                    per_kind: int = 2, dur: float = 20.0, limit: int = 400,
                    seed: Optional[int] = None) -> Plan:
    """Plan punchy event-highlight shorts mined from graph event flags.

    Three event kinds are mined:
      * ``music``  — Segment WHERE is_lyrics=true OR music_overlap=true
      * ``review`` — Segment WHERE review_needed=true
      * ``speed``  — highway DashcamClip frames with mph > SPEED_MPH_MIN

    Each found event becomes one :class:`PlannedShort` backed by highway B-roll
    (resolved via :func:`backdrop._fr_path_for_key`) with a short auto-caption.
    Kinds with no events are silently skipped.
    """
    root = __import__("auto_ingest.shorts.backdrop", fromlist=["_dashcam_root"])._dashcam_root()

    events: List[Dict[str, object]] = []
    if "music" in kinds:
        events += _mine_segment_events(driver, "music", limit, _MUSIC_QUERY)
    if "review" in kinds:
        events += _mine_segment_events(driver, "review", limit, _REVIEW_QUERY)
    if "speed" in kinds:
        events += _mine_speed_events(driver, per_kind, limit)

    shorts: List[PlannedShort] = []
    used: Dict[str, int] = {}
    for ev in events:
        k = ev["kind"]
        if used.get(k, 0) >= per_kind:
            continue
        used[k] = used.get(k, 0) + 1

        clip_key = ev["clip_key"]
        fr = backdrop._fr_path_for_key(clip_key, root)
        shot = Shot(
            clip_key=clip_key,
            fr_path=str(fr) if fr else "",
            t_sec=float(ev.get("t_sec", 0.0)),
            dur=min(dur * 0.8, 6.0),
            mph=ev.get("mph"),
        )
        cues = _highlight_cues(ev, dur)
        shorts.append(PlannedShort(
            id=uuid.uuid5(uuid.NAMESPACE_URL, f"highlight:{k}:{clip_key}:{ev.get('idx', 0)}").hex[:10],
            brief_topic="highlights",
            title=f"{ev['label']}",
            cues=cues,
            shots=[shot],
            notes=f"kind={k} clip={clip_key} seg_idx={ev.get('idx')}",
            status="planned",
        ))

    plan = Plan(
        topic="highlights",
        brief=Brief(topic="highlights", title="Event Highlights", hook="", points=[]),
        shorts=shorts,
        iteration=1,
    )
    log.info("Planned %d highlight short(s) (kinds=%s, dur=%.0fs, seed=%s)",
             len(shorts), kinds, dur, seed)
    return plan


SPEED_MPH_MIN = 80.0

_MUSIC_QUERY = """
MATCH (s:Segment)
WHERE s.is_lyrics = true OR s.music_overlap = true
RETURN s.clip_key AS clip_key, s.idx AS idx, s.start AS start,
       s.text AS text, s.lyrics_score AS score, s.music_overlap AS music_overlap
ORDER BY coalesce(s.lyrics_score, 0) DESC
LIMIT $limit
"""

_REVIEW_QUERY = """
MATCH (s:Segment)
WHERE s.review_needed = true
RETURN s.clip_key AS clip_key, s.idx AS idx, s.start AS start,
       s.text AS text, s.speaker_label AS speaker
LIMIT $limit
"""


def _kind_present(events: List[Dict[str, object]], kind: str) -> bool:
    return any(e["kind"] == kind for e in events)


def _mine_segment_events(driver, kind: str, limit: int, query: str) -> List[Dict[str, object]]:
    with driver.session() as sess:
        rows = sess.run(query, limit=limit).data()
    events: List[Dict[str, object]] = []
    for r in rows:
        clip_key = r.get("clip_key")
        start = r.get("start") or 0.0
        if kind == "music":
            label = "🎵 Music detected"
            if r.get("music_overlap"):
                label = "🎶 Song playing"
        else:
            label = "⚠️ Flagged for review"
        events.append({
            "kind": kind,
            "clip_key": clip_key,
            "idx": r.get("idx", 0),
            "t_sec": float(start) if isinstance(start, (int, float)) else 0.0,
            "mph": None,
            "label": label,
            "text": (r.get("text") or "")[:60],
        })
    return events


def _mine_speed_events(driver, per_kind: int, limit: int) -> List[Dict[str, object]]:
    with driver.session() as sess:
        rows = sess.run(
            """
            MATCH (f:Frame)-[:BELONGS_TO]->(c:DashcamClip)
            WHERE c.view = 'F' AND f.mph >= $mph
            RETURN c.key AS clip_key, f.mph AS mph, f.frame AS frame, c.fps AS fps
            ORDER BY f.mph DESC
            LIMIT $limit
            """,
            mph=SPEED_MPH_MIN, limit=limit,
        ).data()
    events: List[Dict[str, object]] = []
    for r in rows:
        fps = r.get("fps") or 30.0
        frame = r.get("frame") or 0
        events.append({
            "kind": "speed",
            "clip_key": r.get("clip_key"),
            "idx": frame,
            "t_sec": float(frame) / float(fps) if fps else 0.0,
            "mph": r.get("mph"),
            "label": f"🚀 {r.get('mph'):.0f} mph",
            "text": "",
        })
    return events


def _highlight_cues(ev: Dict[str, object], dur: float) -> List[Cue]:
    lead = 0.4
    text = ev.get("text") or ""
    lines: List[tuple] = [(ev["kind"], str(ev.get("label", "")))]
    if text:
        lines.append(("detail", text))
    span = max(dur - lead, 1.0)
    step = span / len(lines)
    cues: List[Cue] = []
    for i, (kind, txt) in enumerate(lines):
        start = lead + i * step
        end = start + min(step, SECS_PER_CUE)
        cues.append(Cue(start=round(start, 2), end=round(end, 2), text=txt, kind=kind))
    return cues


def _pick_shots_deterministic(anchors, *, count, min_gap_sec, clip_dur, rng) -> List[Dict]:
    from auto_ingest.shorts.backdrop import pick_shots
    seed = int(rng.random() * 1e9)
    return pick_shots(anchors, count=count, min_gap_sec=min_gap_sec,
                      clip_dur=clip_dur, rng_seed=seed)



def plan_discussion(driver, clips, *, topic: str = "discussion",
                    short_count: int = 3, short_dur: float = 12.0,
                    clip_dur: float = 8.0, seed: Optional[int] = None) -> Plan:
    """Plan shorts from real spoken DiscussionClips, time-aligned to the dashcam
    moment each line was said.

    Each clip carries ``clip_key`` (-> DashcamClip) and ``start_sec`` (offset
    within that clip), so the B-roll shot is taken AT the utterance, not a random
    highway anchor. Captions are the viewer's own words. Falls back to the
    highway pool for any clip whose source footage is not resolvable here.
    """
    from auto_ingest.shorts import backdrop, curator
    from auto_ingest.shorts.curator import DiscussionClip

    root = backdrop._dashcam_root()
    brief = curator.brief_from_discussions(topic, list(clips))

    # Resolve each clip's source footage; keep only those we can show
    # time-aligned. Audio-sourced utterances (no clip_key) fall back to the
    # highway pool below.
    resolvable: List[DiscussionClip] = []
    for c in clips:
        if not c.clip_key:
            continue
        fr = backdrop._fr_path_for_key(c.clip_key, root)
        if fr and __import__("os").path.exists(str(fr)):
            resolvable.append(c)

    shorts: List[PlannedShort] = []
    pool = backdrop.select_highway_pool(driver, limit=200)
    pool_shots = backdrop.pick_shots(pool, count=1, min_gap_sec=0.0,
                                     clip_dur=clip_dur, rng_seed=seed or 1)
    n = min(short_count, max(len(resolvable), 1))
    for s_idx in range(n):
        if s_idx < len(resolvable):
            c = resolvable[s_idx]
            fr = backdrop._fr_path_for_key(c.clip_key, root)
            t0 = max(0.0, float(c.start_sec or 0.0))
            src = f"clip={c.clip_key} t={t0:.1f}s (time-aligned)"
            shot = Shot(clip_key=c.clip_key, fr_path=str(fr),
                        t_sec=t0, dur=clip_dur, mph=None)
        else:
            # Audio-sourced utterance (no dashcam clip): use a generic
            # highway anchor as B-roll instead of the exact moment.
            a = pool_shots[0] if pool_shots else {}
            src = "highway-pool (audio source, no clip_key)"
            shot = Shot(clip_key=str(a.get("clip_key", "")),
                        fr_path=str(a.get("fr_path") or ""),
                        t_sec=float(a.get("t_sec", 0.0)),
                        dur=clip_dur, mph=a.get("mph"))
            c = clips[s_idx % len(clips)]
        cues = [
            Cue(0.0, 2.4, brief.title, kind="hook"),
            Cue(2.4, min(short_dur, 2.4 + clip_dur), c.text.strip()[:90], kind="point"),
        ]
        shorts.append(PlannedShort(
            id=uuid.uuid5(uuid.NAMESPACE_URL, f"discussion:{topic}:{c.utterance_id}").hex[:10],
            brief_topic=topic,
            title=brief.title,
            cues=cues,
            shots=[shot] if shot.fr_path else [],
            notes=f"utter={c.utterance_id} {src} concept={c.concept}",
            status="planned",
        ))

    plan = Plan(
        topic=topic,
        brief=brief,
        shorts=shorts,
        iteration=1,
    )
    log.info("Planned %d discussion short(s) for %s (time-aligned)", len(shorts), topic)
    return plan

def iterate_plan(prev: Plan, anchors: List[Dict[str, object]], *,
                 short_count: Optional[int] = None, short_dur: Optional[float] = None,
                 seed: Optional[int] = None, shots_per_short: Optional[int] = None,
                 clip_dur: Optional[float] = None, min_gap_sec: Optional[float] = None,
                 reject_ids: Optional[List[str]] = None) -> Plan:
    """Produce the next iteration of a plan (new seed / counts, drop rejects)."""
    reject = set(reject_ids or [])
    kept = [s for s in prev.shorts if s.id not in reject and s.status != "rejected"]
    # Carry over rendered shorts so we don't re-plan what already exists.
    carried = [s for s in kept if s.status == "rendered"]

    new_count = short_count or max(1, len(prev.shorts) - len(carried))
    new_seed = seed if seed is not None else prev.iteration + 7

    fresh = plan_shorts(
        prev.brief,
        anchors,
        short_count=new_count,
        short_dur=short_dur or _max_dur(prev),
        shots_per_short=shots_per_short or _max_shots(prev),
        seed=new_seed,
        clip_dur=clip_dur or 6.0,
        min_gap_sec=min_gap_sec or 8.0,
    )
    fresh.shorts = carried + fresh.shorts
    fresh.iteration = prev.iteration + 1
    fresh.updated_at = __import__("time").time()
    return fresh


def _max_dur(plan: Plan) -> float:
    return max((s.duration() for s in plan.shorts), default=30.0) or 30.0



def plan_trip_story(driver, *, trip_key: Optional[str] = None,
                   count: int = 3, short_dur: float = 30.0,
                   shots_per_trip: int = 4, clip_dur: float = 6.0,
                   seed: Optional[int] = None) -> Plan:
    """Plan 'Trip Story' shorts that follow a real journey.

    Picks ``count`` Trips and sequences their DashcamClips (via
    ``DashcamClip-[:IN_TRIP]->Trip``) in time order as B-roll, overlaying
    the ``Stop`` geofences passed along the way ('Scott 6 MI Raleigh
    house', ...) as captions. Reuses the same Plan/PlannedShort/Cue model
    so the normal renderer handles output. No research / no LLM.

    ``trip_key`` selects one specific Trip (by uniqueKey); otherwise the
    ``count`` longest Trips (by clipCount) are used.
    """
    from auto_ingest.shorts import backdrop

    root = backdrop._dashcam_root()
    trips = _select_trips(driver, trip_key=trip_key, count=count)

    shorts: List[PlannedShort] = []
    for t in trips:
        clips = _trip_clips(driver, t["key"], limit=shots_per_trip)
        if not clips:
            continue
        shots: List[Shot] = []
        cues: List[Cue] = []
        lead = 0.4
        step = max((short_dur - lead) / max(len(clips), 1), clip_dur)
        for i, c in enumerate(clips):
            fr = backdrop._fr_path_for_key(c["clip_key"], root)
            t0 = float(c.get("t_sec", 0.0))
            shots.append(Shot(clip_key=c["clip_key"],
                              fr_path=str(fr) if fr else "",
                              t_sec=t0, dur=clip_dur,
                              mph=c.get("mph")))
            geo = c.get("geofence") or c.get("location") or ""
            label = f"{geo}" if geo else (c.get("time_label") or f"clip {i + 1}")
            cues.append(Cue(start=round(lead + i * step, 2),
                         end=round(lead + i * step + min(step, clip_dur), 2),
                         text=label[:80], kind="trip"))
        if not shots:
            continue
        title = (t.get("geofence") or t.get("tracker") or "Trip Story")
        shorts.append(PlannedShort(
            id=uuid.uuid5(uuid.NAMESPACE_URL, f"trip:{t['key']}").hex[:10],
            brief_topic="trip_story",
            title=str(title)[:60],
            cues=cues,
            shots=shots,
            notes=f"trip={t['key']} clips={len(shots)}",
            status="planned",
        ))

    plan = Plan(
        topic="trip_story",
        brief=Brief(topic="trip_story", title="Trip Story", hook="", points=[]),
        shorts=shorts,
        iteration=1,
    )
    log.info("Planned %d trip-story short(s)", len(shorts))
    return plan


def _select_trips(driver, *, trip_key: Optional[str], count: int) -> List[Dict[str, object]]:
    with driver.session() as sess:
        if trip_key:
            rows = sess.run(
                "MATCH (t:Trip {uniqueKey:$k}) RETURN t.uniqueKey AS key, "
                "t.clipCount AS clips, t.trackerName AS tracker, "
                "t.startTime AS start LIMIT 1",
                k=trip_key,
            ).data()
        else:
            # Dashcam Trips (footage via IN_TRIP) are disjoint from the
            # phone-tracker Trips that carry located Stops, so pick trips
            # that actually HAVE dashcam clips; places are layered on via
            # time-nearest Stop in _trip_clips.
            rows = sess.run(
                "MATCH (c:DashcamClip)-[:IN_TRIP]->(t:Trip) "
                "WITH t, count(DISTINCT c) AS clips "
                "WHERE clips >= 3 "
                "RETURN t.uniqueKey AS key, clips AS clips, "
                "t.trackerName AS tracker "
                "ORDER BY clips DESC LIMIT $n",
                n=count,
            ).data()
        return rows


def _trip_clips(driver, trip_key: str, *, limit: int) -> List[Dict[str, object]]:
    with driver.session() as sess:
        clips = sess.run(
            """
            MATCH (c:DashcamClip)-[:IN_TRIP]->(t:Trip {uniqueKey:$k})
            RETURN c.key AS clip_key, c.path AS path, c.fps AS fps,
                   c.startTime AS cstart
            ORDER BY c.startTime ASC
            LIMIT $lim
            """,
            k=trip_key, lim=limit,
        ).data()
        # Dashcam Trips (footage) are DISJOINT from the phone-tracker
        # Trips that carry located Stops, so a clip's place comes from the
        # DashcamClips have NULL startTime in this graph, so we can't
        # time-align clips to Stops. Instead, layer the day's real
        # places (phone-tracker Stops, which DO carry location) across
        # the drive's clips round-robin: the short reads as a journey
        # ("left Raleigh house -> on I-40 -> arrived Charlotte ...").
        stops = sess.run(
            """
            MATCH (s:Stop) WHERE s.location IS NOT NULL
            RETURN s.geofence AS g, s.location AS loc
            LIMIT 200
            """,
        ).data()
        located = [r for r in stops if (r.get("g") or r.get("loc"))]
        out: List[Dict[str, object]] = []
        for i, r in enumerate(clips):
            geo = located[i % len(located)] if located else {}
            out.append({
                "clip_key": r.get("clip_key"),
                "t_sec": 2.0,
                "mph": None,
                "geofence": geo.get("g"),
                "location": geo.get("loc"),
                "time_label": None,
            })
        return out


def _nearest_stop(cstart, stop_ts):  # retained for API compat; unused
    if not stop_ts or cstart is None:
        return {}
    best = min(stop_ts, key=lambda x: abs(float(x[0]) - float(cstart)))
    return best[1]


def _max_shots(plan: Plan) -> int:
    return max((len(s.shots) for s in plan.shorts), default=3) or 3
