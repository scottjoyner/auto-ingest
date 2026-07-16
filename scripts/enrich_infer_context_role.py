#!/usr/bin/env python3
"""
Step 4 of neo4j-summary-geo-clustering: infer a context role (home / work / travel / other)
for each SummaryLocationCluster and each geo-tagged Summary.

Strategy (data-driven, not hardcoded place names):
- For each named cluster, compute a time signature from its summaries:
    n, weekday_share, biz_hours_share (09:00-18:00), evening_share (18:00-23:00), avg_hour.
- HOME  = cluster with the largest summary_count that is NOT a clear "work" signature.
- WORK  = cluster (excluding HOME) with the highest work_score =
           weekday_share * 0.5 + biz_hours_share * 0.3 + (evening_share if weekday) * 0.2
- TRAVEL = named clusters that are clearly not home/work (low n, weekend-heavy, or
           distant from home coords) — tagged 'travel'.
- OTHER = everything else.

Then every Summary gets s.context_role from its IN_CLUSTER cluster's role. If a cluster
has no name yet, role stays null (will be filled when place names are added).

Idempotent: uses SET (not CREATE), safe to re-run.
"""
import neo4j

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

    # 1. Per-cluster time signature
    q_sig = """
    MATCH (s:Summary)-[:IN_CLUSTER]->(c:SummaryLocationCluster)
    WHERE c.place_name IS NOT NULL AND s.created_at IS NOT NULL
    WITH c,
         count(s) AS n,
         avg(CASE WHEN s.created_at.dayOfWeek IN [1,2,3,4,5] THEN 1.0 ELSE 0.0 END) AS weekday_share,
         avg(CASE WHEN s.created_at.hour >= 9 AND s.created_at.hour <= 18 THEN 1.0 ELSE 0.0 END) AS biz_hours_share,
         avg(CASE WHEN s.created_at.hour >= 18 AND s.created_at.hour <= 23 THEN 1.0 ELSE 0.0 END) AS evening_share
    RETURN c.grid_key AS gk, c.place_name AS name, c.summary_count AS cnt,
           n, weekday_share, biz_hours_share, evening_share
    """
    clusters = []
    with driver.session() as s:
        for r in s.run(q_sig):
            clusters.append({
                "gk": r["gk"], "name": r["name"], "cnt": r["cnt"] or 0,
                "n": r["n"], "wd": r["weekday_share"] or 0,
                "biz": r["biz_hours_share"] or 0, "eve": r["evening_share"] or 0,
            })
    print(f"scored {len(clusters)} named clusters", flush=True)

    if not clusters:
        print("no named clusters yet -- run cluster_place_enrich.py first", flush=True)
        return

    # 1b. Aggregate signatures by place_name (cells of the same place share a role)
    by_place = {}
    for c in clusters:
        p = by_place.setdefault(c["name"], {"name": c["name"], "cells": [], "cnt": 0,
                                            "wd": 0.0, "biz": 0.0, "eve": 0.0, "n": 0})
        p["cells"].append(c)
        p["cnt"] += c["cnt"]; p["n"] += c["n"]
    for p in by_place.values():
        k = p["n"] if p["n"] else 1
        # weight by cluster volume
        p["wd"] = sum(c["wd"] * c["n"] for c in p["cells"]) / k
        p["biz"] = sum(c["biz"] * c["n"] for c in p["cells"]) / k
        p["eve"] = sum(c["eve"] * c["n"] for c in p["cells"]) / k

    places = list(by_place.values())
    print(f"aggregated to {len(places)} named places", flush=True)

    # 2. Decide HOME and WORK at the place level
    home_p = max(places, key=lambda p: p["cnt"])
    work_score = lambda p: (p["wd"] * 0.5 + p["biz"] * 0.3 + (p["eve"] if p["wd"] > 0.5 else 0) * 0.2)
    candidates = [p for p in places if p["name"] != home_p["name"]]
    work_p = max(candidates, key=work_score) if candidates else None

    print(f"HOME  = {home_p['name']} (n={home_p['cnt']})", flush=True)
    if work_p:
        print(f"WORK  = {work_p['name']} (n={work_p['cnt']}, score={work_score(work_p):.2f})", flush=True)

    # 3. Assign roles per place, then per cell
    place_role = {}
    for p in places:
        if p["name"] == home_p["name"]:
            place_role[p["name"]] = "home"
        elif work_p and p["name"] == work_p["name"]:
            place_role[p["name"]] = "work"
        else:
            if p["cnt"] < 200 or p["wd"] < 0.15:
                place_role[p["name"]] = "travel"
            else:
                place_role[p["name"]] = "other"

    roles = {}  # gk -> role
    for p in places:
        for c in p["cells"]:
            roles[c["gk"]] = place_role[p["name"]]
    for p in places:
        print(f"  {p['name']:18s} n={p['cnt']:6d} wd={p['wd']:.2f} biz={p['biz']:.2f} -> {place_role[p['name']]}", flush=True)

    # 4. Write cluster roles + per-summary context_role
    with driver.session() as s:
        with s.begin_transaction() as tx:
            for c in clusters:
                tx.run(
                    "MATCH (cl:SummaryLocationCluster {grid_key: $gk}) "
                    "SET cl.place_role = $role",
                    gk=c["gk"], role=roles[c["gk"]])
            tx.commit()

    # Per-summary context_role from its cluster
    q_sum = """
    MATCH (s:Summary)-[:IN_CLUSTER]->(c:SummaryLocationCluster)
    WHERE c.place_role IS NOT NULL
    WITH s, c.place_role AS role
    SET s.context_role = role
    RETURN count(s) AS tagged
    """
    with driver.session() as s:
        rec = s.run(q_sum).single()
        print(f"tagged {rec['tagged']} summaries with context_role", flush=True)

    driver.close()
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
