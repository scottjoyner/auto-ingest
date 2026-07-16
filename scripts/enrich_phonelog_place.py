#!/usr/bin/env python3
"""Attach PhoneLog points to the NEAREST known SummaryPlace via a distance join.

PhoneLog store is ordered: first ~1M rows are travel elsewhere (Arizona/Utah, 0% near a
known place); rows after that are ~99.8% within 50km of a Charlotte-area SummaryPlace. So
the VAST majority of PhoneLog IS attachable. This script streams all un-placed points in
batches, computes the nearest SummaryPlace (Python haversine over 114 places), and MERGEs
(PhoneLog)-[:AT_PLACE]->(SummaryPlace) when within RADIUS_M. Idempotent. Safe to run
alongside other enrichment (different nodes from Utterance/Summary).

Usage:
  python enrich_phonelog_place.py                  # full pass
  python enrich_phonelog_place.py --limit 500000  # dev
  python enrich_phonelog_place.py --quiet
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


RADIUS_M = 50_000
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
            places = sess.run(
                "MATCH (pl:SummaryPlace) RETURN pl.name AS n, pl.lat AS lat, pl.lon AS lon"
            ).data()
            if not places:
                print("[phonelog_place] no SummaryPlace nodes", flush=True)
                return 0
            if not quiet:
                print(f"[phonelog_place] {len(places)} places, radius={radius_m}m", flush=True)

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
                    for pl in places:
                        d = _haversine(lat, lon, pl["lat"], pl["lon"])
                        if d < best_d:
                            best_d, best = d, pl["n"]
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
                    print(f"[phonelog_place] scanned {scanned}, attached {total}", flush=True)
                if len(rows) < take:
                    break
    finally:
        driver.close()
    print(f"[phonelog_place] done. attached {total} PhoneLog-AT_PLACE edges", flush=True)
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius-km", type=float, default=50.0)
    ap.add_argument("--limit", type=int, default=0, help="cap points scanned (dev)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    run(int(args.radius_km * 1000), args.limit, args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
