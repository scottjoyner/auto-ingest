#!/usr/bin/env python3
"""
Summary -> PhoneLog(geo) time-based enrichment for the Neo4j KG on x1-370.

Why this script exists:
- The skill's Summary->LocationEvent clustering is dead for this dataset:
  LocationEvent only covers 2024; Summaries cover 2025-09..2026-06 (no overlap).
- PhoneLog has 18.19M geo-tagged rows (loc = point{srid:4326,x:lon,y:lat}) spanning
  2024-10..2026-05, which OVERLAPS the summaries. So PhoneLog is the real geo source.
- pl.timestamp is stored as a STRING (ISO), not DateTime. datetime(pl.timestamp) parses it.
- Per-summary nearest-neighbor joins time out against 18M rows through the MCP tool,
  so we do it in Python with the Bolt driver: bucket PhoneLogs by hour, then O(1) lookup.

Algorithm:
  For each month M in [2025-09 .. 2026-06]:
    1. Load PhoneLogs whose timestamp falls in M (indexed string range) into
       hourly buckets: hour_key(str) -> list[(epoch_sec, lat, lon)]
    2. Load Summaries in M without lat yet.
    3. For each summary, scan its own hour bucket + adjacent (+-1h) for nearest point.
    4. SET s.lat/s.lon/geo_source and MERGE (s)-[:LOCATED_AT_TIME_BASED]->(pl) in tx of 500.
"""
import neo4j
from datetime import datetime, timezone
import sys

# Use the repo's standard Neo4j config resolver (env -> config -> baked-in).
import sys as _sys
from pathlib import Path as _P
_SYS_REPO = str(_P(__file__).resolve().parent.parent)
if _SYS_REPO not in _sys.path:
    _sys.path.insert(0, _SYS_REPO)
from auto_ingest_config import get_neo4j_config as _get_cfg

# Month windows covering the summary range
MONTHS = [
    ("2025-09-01", "2025-10-01"),
    ("2025-10-01", "2025-11-01"),
    ("2025-11-01", "2025-12-01"),
    ("2025-12-01", "2026-01-01"),
    ("2026-01-01", "2026-02-01"),
    ("2026-02-01", "2026-03-01"),
    ("2026-03-01", "2026-04-01"),
    ("2026-04-01", "2026-05-01"),
    ("2026-05-01", "2026-06-01"),
    ("2026-06-01", "2026-07-01"),
]

def iso_to_epoch(s):
    # s like '2026-01-15T16:18:07Z' or with fractional
    s2 = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s2)
    except ValueError:
        dt = datetime.fromisoformat(s2[:19] + "+00:00")
    return dt.timestamp()

def build_buckets(driver, m_start, m_end):
    """Load PhoneLogs for the month into hourly buckets."""
    buckets = {}
    q = """
    MATCH (pl:PhoneLog)
    WHERE pl.timestamp >= $ms AND pl.timestamp < $me AND pl.loc IS NOT NULL
    RETURN pl.timestamp AS ts, pl.loc.latitude AS lat, pl.loc.longitude AS lon
    """
    with driver.session() as s:
        res = s.run(q, ms=m_start + "T00:00:00Z", me=m_end + "T00:00:00Z")
        for rec in res:
            ts = rec["ts"]
            if not ts:
                continue
            hk = ts[:13]  # 'YYYY-MM-DDTHH'
            buckets.setdefault(hk, []).append((iso_to_epoch(ts), rec["lat"], rec["lon"]))
    return buckets

def nearest(buckets, sum_epoch, sum_hk):
    """Search own hour + adjacent hours for nearest point."""
    best = None
    best_diff = None
    for hk in (sum_hk, _shift(sum_hk, -1), _shift(sum_hk, +1)):
        pts = buckets.get(hk)
        if not pts:
            continue
        for (ep, lat, lon) in pts:
            d = abs(ep - sum_epoch)
            if best_diff is None or d < best_diff:
                best_diff = d
                best = (lat, lon)
    return best, best_diff

def _shift(hk, delta_h):
    # hk = 'YYYY-MM-DDTHH'; shift by delta_h hours
    dt = datetime.fromisoformat(hk.replace("T", " ") + ":00")
    from datetime import timedelta
    dt2 = dt + timedelta(hours=delta_h)
    return dt2.strftime("%Y-%m-%dT%H")

def enrich_month(driver, m_start, m_end):
    buckets = build_buckets(driver, m_start, m_end)
    n_buckets_pts = sum(len(v) for v in buckets.values())
    print(f"[{m_start}] loaded {n_buckets_pts} phonelog pts into {len(buckets)} hourly buckets", flush=True)

    q_sum = """
    MATCH (s:Summary)
    WHERE s.created_at >= datetime($ms) AND s.created_at < datetime($me)
      AND size(s.text) > 0 AND s.lat IS NULL
    RETURN s.id AS id, s.created_at AS ca
    """
    q_write = """
    MATCH (s:Summary {id: $id})
    MATCH (pl:PhoneLog {timestamp: $ts})
    SET s.lat = $lat, s.lon = $lon, s.geo_source = 'phonelog'
    MERGE (s)-[r:LOCATED_AT_TIME_BASED]->(pl)
    SET r.time_difference_seconds = $diff,
        r.clustered_at = datetime(),
        r.confidence = CASE
            WHEN $diff <= 3600  THEN 1.0
            WHEN $diff <= 7200  THEN 0.8
            WHEN $diff <= 14400 THEN 0.6
            ELSE 0.4 END
    """

    total = 0
    # Collect all summaries for the month first (do NOT write while iterating the read result)
    rows = []
    with driver.session() as s:
        res = s.run(q_sum, ms=m_start + "T00:00:00Z", me=m_end + "T00:00:00Z")
        for rec in res:
            ca = rec["ca"]
            if ca is None:
                continue
            sum_epoch = ca.to_native().timestamp()
            sum_hk = str(ca)[:13].replace(" ", "T")
            best, diff = nearest(buckets, sum_epoch, sum_hk)
            if best is None:
                continue
            lat, lon = best
            ts = _nearest_ts(buckets, sum_epoch, sum_hk)
            rows.append({"id": rec["id"], "ts": ts, "lat": lat, "lon": lon, "diff": diff})

    # Write in batches of 500 using a fresh session/transaction
    for i in range(0, len(rows), 500):
        batch = rows[i:i+500]
        with driver.session() as ws:
            with ws.begin_transaction() as tx:
                for b in batch:
                    tx.run(q_write, id=b["id"], ts=b["ts"], lat=b["lat"], lon=b["lon"], diff=b["diff"])
                tx.commit()
        total += len(batch)
        print(f"[{m_start}] wrote {total}/{len(rows)}", flush=True)
    print(f"[{m_start}] enriched {total} summaries", flush=True)
    return total

def _nearest_ts(buckets, sum_epoch, sum_hk):
    best = None; best_diff = None
    for hk in (sum_hk, _shift(sum_hk, -1), _shift(sum_hk, +1)):
        pts = buckets.get(hk)
        if not pts: continue
        for (ep, lat, lon) in pts:
            d = abs(ep - sum_epoch)
            if best_diff is None or d < best_diff:
                best_diff = d; best = ep
    # convert epoch back to ISO 'Z'
    from datetime import datetime as dt2
    return dt2.fromtimestamp(best, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _flush(session, q, batch):
    with session.begin_transaction() as tx:
        for b in batch:
            tx.run(q, id=b["id"], ts=b["ts"], lat=b["lat"], lon=b["lon"], diff=b["diff"])
        tx.commit()

def main():
    _cfg = _get_cfg()
    uri, user, pw = _cfg["uri"], _cfg["user"], _cfg["password"]
    driver = neo4j.GraphDatabase.driver(uri, auth=(user, pw))
    try:
        driver.verify_connectivity()
        print("connected to neo4j", flush=True)
        grand = 0
        for (ms, me) in MONTHS:
            grand += enrich_month(driver, ms, me)
        print(f"DONE. total enriched: {grand}", flush=True)
    finally:
        driver.close()

if __name__ == "__main__":
    main()
