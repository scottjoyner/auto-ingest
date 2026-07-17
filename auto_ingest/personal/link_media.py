"""Link :MediaFile nodes to :SummaryPlace / :Trip / :PhoneLog via GPS haversine.

MediaFile has no millisecond timestamp (only a date), so temporal joins to Trip
are weak; we use a GPS-spatial (haversine) join instead. All writes go through
``with_driver`` because the graph (~21M nodes) OOMs under load: queries are
bounded (LIMIT), index-backed (bounding-box prefilter on gps/lat/lon), and
resumable (``m.linked_at IS NULL`` is the primary skip mechanism, with an
optional ``--state`` json checkpoint for crash recovery).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
from datetime import datetime, timezone
from typing import List, Optional

from auto_ingest.shorts.db_retry import with_driver

log = logging.getLogger("personal.link_media")

_EARTH_R = 6371000.0
# Degrees latitude per meter (approx). Longitude scaled by cos(lat) per-point.
_M_PER_DEG_LAT = 111_320.0


def haversine(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in meters between two lat/lon points."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * _EARTH_R * math.asin(math.sqrt(h))


def _bbox(lat: float, lon: float, radius_m: float):
    """Return (lat0, lat1, lon0, lon1) bounding box padded by radius_m."""
    dlat = radius_m / _M_PER_DEG_LAT
    coslat = max(math.cos(math.radians(lat)), 1e-6)
    dlon = radius_m / (_M_PER_DEG_LAT * coslat)
    return lat - dlat, lat + dlat, lon - dlon, lon + dlon


# ---------------------------------------------------------------------------
# State checkpoint (optional; linked_at IS NULL is the primary skip mechanism)
# ---------------------------------------------------------------------------
def _load_state(state_path: Optional[str]) -> set:
    if not state_path or not os.path.exists(state_path):
        return set()
    try:
        with open(state_path, "r", encoding="utf-8") as fh:
            return set(json.load(fh).get("done", []))
    except Exception as e:
        log.info("could not load state %s: %s", state_path, e)
        return set()


def _save_state(state_path: Optional[str], done: set) -> None:
    if not state_path:
        return
    try:
        with open(state_path, "w", encoding="utf-8") as fh:
            json.dump({"done": sorted(done)}, fh)
    except Exception as e:
        log.info("could not save state %s: %s", state_path, e)


def _db() -> str:
    return os.getenv("NEO4J_DB", "neo4j")


def _fetch_unlinked(driver, limit: int, done: set) -> List[dict]:
    """MediaFiles with GPS and linked_at IS NULL (bounded)."""
    q = (
        "MATCH (m:MediaFile) "
        "WHERE m.gps_lat IS NOT NULL AND m.gps_lon IS NOT NULL "
        "AND m.linked_at IS NULL "
        "RETURN m.sha256 AS sha, m.gps_lat AS lat, m.gps_lon AS lon "
        "LIMIT $limit"
    )
    with driver.session(database=_db()) as sess:
        rows = sess.run(q, limit=int(limit)).data()
    return [r for r in rows if r["sha"] not in done]


def _mark_linked(driver, sha: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with driver.session(database=_db()) as sess:
        sess.run(
            "MATCH (m:MediaFile {sha256:$sha}) SET m.linked_at = $now",
            sha=sha, now=now,
        ).consume()


# ---------------------------------------------------------------------------
# Linkers
# ---------------------------------------------------------------------------
def link_to_places(driver, max_radius_m: float = 200.0, limit: int = 500,
                   state_path: Optional[str] = None) -> int:
    """Link each unlinked MediaFile to nearest :SummaryPlace within radius."""
    done = _load_state(state_path)
    media = _fetch_unlinked(driver, limit, done)
    linked = 0
    with driver.session(database=_db()) as sess:
        for m in media:
            lat, lon = m["lat"], m["lon"]
            lat0, lat1, lon0, lon1 = _bbox(lat, lon, max_radius_m)
            cands = sess.run(
                "MATCH (p:SummaryPlace) "
                "WHERE p.lat IS NOT NULL AND p.lon IS NOT NULL "
                "AND p.lat >= $lat0 AND p.lat <= $lat1 "
                "AND p.lon >= $lon0 AND p.lon <= $lon1 "
                "RETURN p.name AS name, p.lat AS lat, p.lon AS lon LIMIT 50",
                lat0=lat0, lat1=lat1, lon0=lon0, lon1=lon1,
            ).data()
            best, best_d = None, float("inf")
            for c in cands:
                d = haversine(lat, lon, c["lat"], c["lon"])
                if d < best_d:
                    best_d, best = d, c
            if best and best_d <= max_radius_m:
                sess.run(
                    "MATCH (m:MediaFile {sha256:$sha}) "
                    "MATCH (pl:SummaryPlace {name:$name, lat:$lat, lon:$lon}) "
                    "MERGE (m)-[:AT_PLACE]->(pl)",
                    sha=m["sha"], name=best["name"], lat=best["lat"], lon=best["lon"],
                ).consume()
                linked += 1
    log.info("link_to_places: %d of %d media linked", linked, len(media))
    return linked


def link_to_trips(driver, max_radius_m: float = 300.0, limit: int = 500,
                  state_path: Optional[str] = None) -> int:
    """Link each unlinked MediaFile to :Trip(s) whose LocationEvents are nearby."""
    done = _load_state(state_path)
    media = _fetch_unlinked(driver, limit, done)
    linked = 0
    with driver.session(database=_db()) as sess:
        for m in media:
            lat, lon = m["lat"], m["lon"]
            lat0, lat1, lon0, lon1 = _bbox(lat, lon, max_radius_m)
            trips = sess.run(
                "MATCH (le:LocationEvent) "
                "WHERE le.latitude IS NOT NULL AND le.longitude IS NOT NULL "
                "AND le.latitude >= $lat0 AND le.latitude <= $lat1 "
                "AND le.longitude >= $lon0 AND le.longitude <= $lon1 "
                "WITH le, point.distance("
                "  point({latitude:$lat, longitude:$lon}), "
                "  point({latitude:le.latitude, longitude:le.longitude})) AS d "
                "WHERE d <= $r "
                "MATCH (le)-[:BELONGS_TO]->(t:Trip) "
                "RETURN DISTINCT t.uniqueKey AS uniqueKey, t.tripId AS tripId LIMIT 5",
                lat0=lat0, lat1=lat1, lon0=lon0, lon1=lon1,
                lat=lat, lon=lon, r=max_radius_m,
            ).data()
            for t in trips:
                key = t.get("uniqueKey")
                if key is not None:
                    sess.run(
                        "MATCH (m:MediaFile {sha256:$sha}) "
                        "MATCH (t:Trip {uniqueKey:$key}) "
                        "MERGE (m)-[:DURING]->(t)",
                        sha=m["sha"], key=key,
                    ).consume()
                    linked += 1
                elif t.get("tripId") is not None:
                    sess.run(
                        "MATCH (m:MediaFile {sha256:$sha}) "
                        "MATCH (t:Trip {tripId:$tid}) "
                        "MERGE (m)-[:DURING]->(t)",
                        sha=m["sha"], tid=t["tripId"],
                    ).consume()
                    linked += 1
    log.info("link_to_trips: %d trip edges from %d media", linked, len(media))
    return linked


def link_to_phonelogs(driver, max_radius_m: float = 100.0, limit: int = 500,
                      state_path: Optional[str] = None) -> int:
    """Link each unlinked MediaFile to nearest :PhoneLog within radius."""
    done = _load_state(state_path)
    media = _fetch_unlinked(driver, limit, done)
    linked = 0
    with driver.session(database=_db()) as sess:
        for m in media:
            lat, lon = m["lat"], m["lon"]
            lat0, lat1, lon0, lon1 = _bbox(lat, lon, max_radius_m)
            cands = sess.run(
                "MATCH (pl:PhoneLog) "
                "WHERE pl.latitude IS NOT NULL AND pl.longitude IS NOT NULL "
                "AND pl.latitude >= $lat0 AND pl.latitude <= $lat1 "
                "AND pl.longitude >= $lon0 AND pl.longitude <= $lon1 "
                "RETURN elementId(pl) AS pid, pl.latitude AS lat, "
                "pl.longitude AS lon LIMIT 50",
                lat0=lat0, lat1=lat1, lon0=lon0, lon1=lon1,
            ).data()
            best, best_d = None, float("inf")
            for c in cands:
                d = haversine(lat, lon, c["lat"], c["lon"])
                if d < best_d:
                    best_d, best = d, c
            if best and best_d <= max_radius_m:
                sess.run(
                    "MATCH (m:MediaFile {sha256:$sha}) "
                    "MATCH (pl:PhoneLog) WHERE elementId(pl) = $pid "
                    "MERGE (m)-[:NEAR {meters:$d}]->(pl)",
                    sha=m["sha"], pid=best["pid"], d=round(best_d, 1),
                ).consume()
                linked += 1
    log.info("link_to_phonelogs: %d of %d media linked", linked, len(media))
    return linked


def link_all(driver, max_radius_m: Optional[float] = None, limit: int = 500,
             state_path: Optional[str] = None, places: bool = True,
             trips: bool = True, phonelogs: bool = True) -> dict:
    """Run place/trip/phonelog linkers in order; mark processed media; print counts."""
    done = _load_state(state_path)
    counts = {"places": 0, "trips": 0, "phonelogs": 0}
    if places:
        counts["places"] = link_to_places(
            driver, max_radius_m or 200.0, limit, state_path)
    if trips:
        counts["trips"] = link_to_trips(
            driver, max_radius_m or 300.0, limit, state_path)
    if phonelogs:
        counts["phonelogs"] = link_to_phonelogs(
            driver, max_radius_m or 100.0, limit, state_path)

    # Mark the media processed in this pass (skip on future runs).
    processed = _fetch_unlinked(driver, limit, done)
    for m in processed:
        _mark_linked(driver, m["sha"])
        done.add(m["sha"])
    _save_state(state_path, done)

    print(
        f"[link_media] places={counts['places']} trips={counts['trips']} "
        f"phonelogs={counts['phonelogs']} processed={len(processed)}",
        flush=True,
    )
    return counts


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Link personal MediaFile nodes by GPS.")
    ap.add_argument("--places", action="store_true")
    ap.add_argument("--trips", action="store_true")
    ap.add_argument("--phonelogs", action="store_true")
    ap.add_argument("--radius-m", type=float, default=None)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--state", type=str, default=None)
    args = ap.parse_args()

    # Default: run all three if none explicitly selected.
    if not (args.places or args.trips or args.phonelogs):
        do_places = do_trips = do_phonelogs = True
    else:
        do_places, do_trips, do_phonelogs = args.places, args.trips, args.phonelogs

    with_driver(lambda drv: link_all(
        drv, max_radius_m=args.radius_m, limit=args.limit, state_path=args.state,
        places=do_places, trips=do_trips, phonelogs=do_phonelogs,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
