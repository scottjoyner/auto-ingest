#!/usr/bin/env python3
"""Second-pass: attach out-of-range PhoneLog points to named TRAVEL places.

The main enrich_phonelog_place.py attached ~17M Charlotte-area points (within 50km of a
known SummaryPlace). The remainder are REAL travel regions, identified from coordinate
clustering (2026-07-16): Park City UT (skiing), Boone NC, Central Colorado (ski),
Denver CO, NW of SLC. This script creates those travel SummaryPlace nodes (role=travel)
and distance-joins the still-unplaced PhoneLogs to them (<= TRAVEL_RADIUS_M).

Centroids are data-derived (see cluster analysis). Names are descriptive; refine later.
Idempotent: skips already-placed points; creates place only if absent.

Usage:
  python enrich_phonelog_travel.py                # full travel pass
  python enrich_phonelog_travel.py --limit 500000
  python enrich_phonelog_travel.py --quiet
"""
from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _cfg():
    from auto_ingest_config import get_neo4j_config
    return get_neo4j_config()


# Data-derived travel-region centroids (lat, lon, name) from 2026-07-16 cluster scan.
TRAVEL_PLACES = [
    (40.62, -111.68, "Park City, UT (ski)"),
    (39.25, -107.05, "Central Colorado (ski)"),
    (36.18, -81.75, "Boone / Blowing Rock, NC"),
    (39.90, -104.70, "Denver, CO"),
    (40.80, -112.00, "NW of Salt Lake City, UT"),
]
TRAVEL_RADIUS_M = 60_000
SCAN_BATCH = 200_000
WRITE_BATCH = 20_000


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def run(radius_m: int, limit: int, quiet: bool) -> int:
    import neo4j
    cfg = _cfg()
    driver = neo4j.GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))
    total = 0
    try:
        with driver.session(database=cfg.get("db")) as sess:
            # Ensure travel places exist
            for lat, lon, name in TRAVEL_PLACES:
                sess.run(
                    "MERGE (pl:SummaryPlace {name: $name}) "
                    "SET pl.lat=$lat, pl.lon=$lon, pl.place_role='travel', "
                    "pl.summary_count=0, pl.note='travel region (PhoneLog cluster, 2026-07-16)'",
                    name=name, lat=lat, lon=lon,
                )
            if not quiet:
                print(f"[phonelog_travel] {len(TRAVEL_PLACES)} travel places ready, radius={radius_m}m", flush=True)

            scanned = 0
            skip = 0
            while True:
                if limit and scanned >= limit:
                    break
                take = min(SCAN_BATCH, limit - scanned) if limit else SCAN_BATCH
                rows = list(sess.run(
                    "MATCH (p:PhoneLog) WHERE p.loc IS NOT NULL AND NOT (p)-[:AT_PLACE]->() "
                    "WITH p SKIP $skip LIMIT $take "
                    "RETURN elementId(p) AS pid, p.loc.latitude AS lat, p.loc.longitude AS lon",
                    skip=skip, take=take,
                ))
                if not rows:
                    break
                to_write = []
                for r in rows:
                    lat, lon = r["lat"], r["lon"]
                    if lat is None:
                        continue
                    best, best_d = None, float("inf")
                    for plat, plon, pname in TRAVEL_PLACES:
                        d = _haversine(lat, lon, plat, plon)
                        if d < best_d:
                            best_d, best = d, pname
                    if best and best_d <= radius_m:
                        to_write.append((r["pid"], best))
                for i in range(0, len(to_write), WRITE_BATCH):
                    batch = to_write[i:i + WRITE_BATCH]
                    with sess.begin_transaction() as tx:
                        for pid, pname in batch:
                            tx.run(
                                "MATCH (p:PhoneLog) WHERE elementId(p) = $pid "
                                "MATCH (pl:SummaryPlace {name: $pn}) "
                                "MERGE (p)-[:AT_PLACE]->(pl)",
                                pid=pid, pn=pname,
                            )
                        tx.commit()
                    total += len(batch)
                scanned += len(rows)
                skip += len(rows)
                if not quiet:
                    print(f"[phonelog_travel] scanned {scanned}, attached {total}", flush=True)
                if len(rows) < take:
                    break
    finally:
        driver.close()
    print(f"[phonelog_travel] done. attached {total} travel PhoneLog-AT_PLACE edges", flush=True)
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius-km", type=float, default=60.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    run(int(args.radius_km * 1000), args.limit, args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
