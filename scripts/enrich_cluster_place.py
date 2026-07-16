#!/usr/bin/env python3
"""
Enrich top SummaryLocationCluster nodes with human-readable place names via
OpenStreetMap Nominatim reverse geocoding (no API key). Rate-limited to 1 req/sec.

Only enriches clusters with summary_count >= a threshold (top N by size) to keep
the API footprint small and respect Nominatim's usage policy.

Writes c.place_name, c.place_category, c.geocoded_at on each cluster.
"""
import neo4j
import requests
import time
import sys

# Use the repo's standard Neo4j config resolver (env -> config -> baked-in).
import sys as _sys
from pathlib import Path as _P
_SYS_REPO = str(_P(__file__).resolve().parent.parent)
if _SYS_REPO not in _sys.path:
    _sys.path.insert(0, _SYS_REPO)
from auto_ingest_config import get_neo4j_config as _get_cfg

MIN_COUNT = 30   # only clusters with >= this many summaries get a place name
TOP_N = 60       # cap to the N largest clusters

def main():
    _cfg = _get_cfg()
    uri, user, pw = _cfg["uri"], _cfg["user"], _cfg["password"]
    driver = neo4j.GraphDatabase.driver(uri, auth=(user, pw))
    driver.verify_connectivity()
    print("connected", flush=True)

    # Pull top clusters by size
    q = """
    MATCH (c:SummaryLocationCluster)
    WHERE c.summary_count >= $minc
    RETURN c.grid_key AS gk, c.lat AS lat, c.lon AS lon, c.summary_count AS cnt
    ORDER BY c.summary_count DESC
    LIMIT $topn
    """
    clusters = []
    with driver.session() as s:
        for rec in s.run(q, minc=MIN_COUNT, topn=TOP_N):
            clusters.append((rec["gk"], rec["lat"], rec["lon"], rec["cnt"]))
    print(f"enriching {len(clusters)} clusters", flush=True)

    done = 0
    for (gk, lat, lon, cnt) in clusters:
        try:
            url = f"https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat={lat}&lon={lon}&zoom=18&addressdetails=1"
            r = requests.get(url, headers={"User-Agent": "Hermes-KG-Enrich/1.0"}, timeout=15)
            if r.status_code == 200:
                data = r.json()
                addr = data.get("address", {})
                # Priority: neighbourhood > suburb > city > town > village > county
                place = None
                for k in ("neighbourhood", "suburb", "city", "town", "village", "municipality", "county"):
                    if k in addr:
                        place = addr[k]
                        break
                if not place:
                    place = data.get("name") or data.get("display_name", "").split(",")[0]
                category = data.get("type") or data.get("class") or "unknown"
                # write back
                with driver.session() as ws:
                    ws.run(
                        "MATCH (c:SummaryLocationCluster {grid_key: $gk}) "
                        "SET c.place_name = $pn, c.place_category = $cat, c.geocoded_at = datetime()",
                        gk=gk, pn=place, cat=category)
                done += 1
                print(f"  {gk} ({cnt}): {place} [{category}]", flush=True)
            else:
                print(f"  {gk}: HTTP {r.status_code}", flush=True)
        except Exception as e:
            print(f"  {gk}: ERROR {e}", flush=True)
        time.sleep(1.1)  # rate limit

    print(f"DONE. geocoded {done}/{len(clusters)} clusters", flush=True)
    driver.close()

if __name__ == "__main__":
    main()
