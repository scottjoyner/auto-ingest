#!/usr/bin/env python3
"""
Step 3b of neo4j-summary-geo-clustering: merge the ~1,047 fine-grained
SummaryLocationCluster grid cells into a small number of clean SUMMARY PLACES.

WHY: the 3-decimal grid fragments every real location into many cells that all
reverse-geocode to the same place_name (South End = 59 cells / 50k summaries,
Uptown = 54 cells / 7k, Charlotte = 112 scattered cells). Querying "what happened
at home" requires aggregating 50+ cells. This script collapses them.

RESULT:
- Creates `SummaryPlace` nodes keyed by `place_name` (one per real place).
- `(cluster)-[:PART_OF]->(place)` links each grid cell to its place.
- `(s)-[:AT_PLACE]->(place)` added to every geo-tagged summary for clean direct queries.
- `place_role` (home/work/travel/other) copied onto the Place from the dominant cluster role.
- Place gets: summary_count (sum), cells (count), lat/lon (volume-weighted centroid),
  first_seen/last_seen if available.

Idempotent: re-running MERGEs (does not duplicate edges or nodes). Safe to re-run after
more clusters get geocoded (it will fold new cells into existing places).

Note on "Charlotte": it's the generic Nominatim catch-all (112 scattered cells). They are
folded into a single SummaryPlace named "Charlotte" so they don't pollute South End/Uptown.
You can later split Charlotte by proximity if desired; for now one node is cleaner than 112.
"""
import neo4j
from collections import defaultdict

# Use the repo's standard Neo4j config resolver (env -> config -> baked-in).
import sys as _sys
from pathlib import Path as _P
_SYS_REPO = str(_P(__file__).resolve().parent.parent)
if _SYS_REPO not in _sys.path:
    _sys.path.insert(0, _SYS_REPO)
from auto_ingest_config import get_neo4j_config as _get_cfg

def main():
    _cfg = _get_cfg()
    uri, user, pw = _cfg["uri"], _cfg["user"], _cfg["password"]
    driver = neo4j.GraphDatabase.driver(uri, auth=(user, pw))
    driver.verify_connectivity()
    print("connected", flush=True)

    # Pull every named cluster with its role + coords + count
    q = """
    MATCH (c:SummaryLocationCluster) WHERE c.place_name IS NOT NULL
    RETURN c.place_name AS pn, c.grid_key AS gk, c.lat AS lat, c.lon AS lon,
           c.summary_count AS cnt, c.place_role AS role
    """
    places = defaultdict(lambda: {"cells": 0, "cnt": 0, "latw": 0.0, "lonw": 0.0,
                                  "roles": defaultdict(int), "gks": []})
    with driver.session() as s:
        for rec in s.run(q):
            pn = rec["pn"]
            p = places[pn]
            p["cells"] += 1
            cnt = rec["cnt"] or 0
            p["cnt"] += cnt
            p["latw"] += (rec["lat"] or 0) * cnt
            p["lonw"] += (rec["lon"] or 0) * cnt
            if rec["role"]:
                p["roles"][rec["role"]] += cnt
            p["gks"].append(rec["gk"])

    print(f"aggregating {sum(p['cells'] for p in places.values())} cells into {len(places)} places", flush=True)

    # Create/update Place nodes + PART_OF edges (per place, idempotent)
    with driver.session() as ws:
        for pn, p in places.items():
            cnt = p["cnt"]
            lat = p["latw"] / cnt if cnt else 0.0
            lon = p["lonw"] / cnt if cnt else 0.0
            role = max(p["roles"].items(), key=lambda kv: kv[1])[0] if p["roles"] else "other"
            ws.execute_write(_upsert_place, pn, cnt, p["cells"], lat, lon, role, p["gks"])

    # Bulk AT_PLACE link from IN_CLUSTER + PART_OF (reliable single pass; the per-gk
    # loop under-linked due to transaction batching). Idempotent via MERGE.
    with driver.session() as ws:
        res = ws.execute_write(lambda tx: tx.run(
            "MATCH (s:Summary)-[:IN_CLUSTER]->(c:SummaryLocationCluster)-[:PART_OF]->(pl:SummaryPlace) "
            "MERGE (s)-[:AT_PLACE]->(pl) RETURN count(s) AS n").single())
        print(f"  AT_PLACE edges: {res['n']}", flush=True)

    print(f"DONE. created/updated {len(places)} SummaryPlace nodes", flush=True)
    driver.close()

def _upsert_place(tx, pn, cnt, cells, lat, lon, role, gks):
    tx.run(
        "MERGE (pl:SummaryPlace {name: $pn}) "
        "SET pl.summary_count = $cnt, pl.cells = $cells, pl.lat = $lat, pl.lon = $lon, "
        "pl.place_role = $role, pl.updated_at = datetime()",
        pn=pn, cnt=cnt, cells=cells, lat=lat, lon=lon, role=role)
    for gk in gks:
        tx.run(
            "MATCH (c:SummaryLocationCluster {grid_key: $gk}) "
            "MERGE (pl:SummaryPlace {name: $pn}) "
            "MERGE (c)-[:PART_OF]->(pl)",
            gk=gk, pn=pn)

if __name__ == "__main__":
    main()
