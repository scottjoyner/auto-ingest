#!/usr/bin/env python3
"""
Backfill Segment.speaker_* and SPOKEN_BY edges from diarization RTTM.

Default behavior:
  - Iterate Transcriptions, use t.source_rttm only.
  - Skip if RTTM missing/invalid.
Layered discovery (opt-in):
  - If --allow-discovery is passed, search provided --rttm-dir paths and
    persist discovered path to t.source_rttm.

Requires: pip install neo4j
"""

import argparse, os, re, logging, hashlib
from typing import List, Dict, Any, Tuple, Optional
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------- utils ----------
def stable_id(*parts: str) -> str:
    h = hashlib.md5()
    for p in parts:
        h.update((p or "").encode("utf-8", errors="ignore"))
        h.update(b"|")
    return h.hexdigest()

def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))

def load_rttm(path: str) -> List[Tuple[float, float, str]]:
    """
    Robust RTTM: expects lines like:
      SPEAKER <file> <chan> <tbeg> <tdur> <ortho> <stype> <name> <conf> ...
    Uses 3=tbeg, 4=tdur, 7=name when present, else last token. Filters dur<=0.
    """
    ivals: List[Tuple[float,float,str]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if not parts or parts[0].upper() != "SPEAKER" or len(parts) < 5:
                    continue
                try:
                    start = float(parts[3]); dur = float(parts[4])
                    if dur <= 0:
                        continue
                    end = start + dur
                except Exception:
                    continue
                spk = parts[7] if len(parts) >= 8 else parts[-1]
                spk = "UNKNOWN" if str(spk).upper() in ("UNK", "UNKNOWN", "<UNK>") else spk
                ivals.append((start, end, spk))
        ivals.sort(key=lambda x: x[0])
    except Exception as ex:
        logging.warning(f"RTTM read failed: {path} :: {ex}")
    return ivals

def sum_overlap(seg_start: float, seg_end: float, rttm: List[Tuple[float,float,str]]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for a,b,spk in rttm:
        ov = overlap(seg_start, seg_end, a, b)
        if ov > 0:
            totals[spk] = totals.get(spk, 0.0) + ov
    return totals

# ---------- RTTM discovery (opt-in) ----------
PAT_RTTM = re.compile(r"_speakers\.rttm$", re.IGNORECASE)

def _key_from_rttm_filename(name: str) -> str:
    base = re.sub(r"\.rttm$", "", name, flags=re.IGNORECASE)
    base = re.sub(r"_speakers$", "", base, flags=re.IGNORECASE)
    return base

def index_rttm_dirs(search_dirs: List[str]) -> Dict[str, List[str]]:
    """
    Build once: {key -> [absolute_paths]} by scanning all dirs.
    """
    index: Dict[str, List[str]] = {}
    for root_dir in search_dirs:
        if not root_dir or not os.path.isdir(root_dir):
            continue
        for r, _, files in os.walk(root_dir):
            for name in files:
                if not PAT_RTTM.search(name):
                    continue
                p = os.path.join(r, name)
                key = _key_from_rttm_filename(name)
                index.setdefault(key, []).append(os.path.abspath(p))
    logging.info(f"Indexed RTTMs: {sum(len(v) for v in index.values())} files, {len(index)} keys")
    return index

def discover_rttm_for_key(key: str, idx: Dict[str, List[str]]) -> Optional[str]:
    # Pass 1: exact filename under any indexed path
    paths = idx.get(key, [])
    if paths:
        # Prefer shortest path depth to avoid odd duplicates
        return sorted(paths, key=lambda p: (p.count(os.sep), len(p)))[0]
    # Pass 2: fallback — substring match across all keys
    key_lower = key.lower()
    candidates = [(k, v) for k, v in idx.items() if key_lower in k.lower()]
    if candidates:
        # Pick the nearest (smallest extra chars) then shortest path
        candidates.sort(key=lambda kv: (abs(len(kv[0]) - len(key)), len(kv[0])))
        return sorted(candidates[0][1], key=lambda p: (p.count(os.sep), len(p)))[0]
    return None

# ---------- Neo4j ----------
def driver_from_env(uri, user, pwd):
    return GraphDatabase.driver(uri, auth=(user, pwd))

FETCH_BATCH = """
// Transcriptions with their segments.
// If only_missing, require at least one segment with s.speaker_label IS NULL.
MATCH (t:Transcription)
WHERE ($only_missing = true AND EXISTS {
  MATCH (t)-[:HAS_SEGMENT]->(s:Segment)
  WHERE s.speaker_label IS NULL
}) OR ($only_missing = false)
WITH t
MATCH (t)-[:HAS_SEGMENT]->(s:Segment)
WHERE ($only_missing = true AND s.speaker_label IS NULL) OR ($only_missing = false)
WITH t, collect({id:s.id, start:coalesce(s.start,0.0), end:coalesce(s.end,0.0)}) AS segs
WHERE size(segs) > 0
RETURN t.id AS tid, t.key AS tkey, t.source_rttm AS rttm, segs
LIMIT $limit
"""

WRITE_BACK = """
// 1) Ensure Speakers
UNWIND $speakers AS sp
MERGE (spk:Speaker {id: sp.id})
  ON CREATE SET spk.label = sp.label, spk.key = sp.key, spk.created_at = datetime()
  ON MATCH  SET spk.label = sp.label, spk.key = sp.key, spk.updated_at = datetime()
WITH 1 AS _

// 2) Update dominant props on Segment
UNWIND $segments AS seg
MATCH (s:Segment {id: seg.id})
SET s.speaker_label = seg.speaker_label,
    s.speaker_id    = seg.speaker_id
WITH 1 AS _

// 3) Delete old SPOKEN_BY edges for these segments (idempotent writes)
UNWIND $segments AS seg
MATCH (s:Segment {id: seg.id})-[r:SPOKEN_BY]->(:Speaker)
DELETE r
WITH 1 AS _

// 4) Create mixture edges with proportions/overlap
UNWIND $edges AS e
MATCH (s:Segment {id: e.segment_id})
MATCH (spk:Speaker {id: e.speaker_id})
MERGE (s)-[r:SPOKEN_BY]->(spk)
SET r.proportion = e.proportion,
    r.overlap    = e.overlap
"""

UPDATE_T_SOURCE_RTTM = """
MATCH (t:Transcription {id:$tid})
SET t.source_rttm = $path
"""

# ---------- core ----------
def reconcile_once(sess,
                   limit: int,
                   only_missing: bool,
                   min_prop: float,
                   require_overlap: bool,
                   dry_run: bool,
                   allow_discovery: bool,
                   rttm_index: Optional[Dict[str, List[str]]]) -> int:
    res = sess.run(FETCH_BATCH, limit=limit, only_missing=only_missing)
    rows = res.data()
    if not rows:
        return 0

    updated = 0
    for row in rows:
        tid  = row["tid"]
        key  = row["tkey"]
        path = row.get("rttm")
        segs: List[Dict[str, Any]] = row["segs"]

        # RTTM-first: use t.source_rttm if valid
        need_discovery = (not path) or (not os.path.isfile(path))
        if need_discovery:
            if not allow_discovery:
                logging.warning(f"[{key}] No/invalid t.source_rttm and discovery disabled -> skip (path={path})")
                continue
            if not rttm_index:
                logging.warning(f"[{key}] Discovery enabled but no RTTM index present -> skip")
                continue
            cand = discover_rttm_for_key(key, rttm_index)
            if not cand:
                logging.warning(f"[{key}] RTTM not found in indexed dirs -> skip")
                continue
            logging.info(f"[{key}] discovered RTTM: {cand}")
            path = cand
            if not dry_run:
                sess.run(UPDATE_T_SOURCE_RTTM, tid=tid, path=cand)

        rttm = load_rttm(path)
        if not rttm:
            logging.warning(f"[{key}] RTTM parsed empty -> skip (path={path})")
            continue

        # Prepare speakers
        labels = sorted(set(spk for _,_,spk in rttm))
        unknown_id = stable_id(key, "spk", "UNKNOWN")
        speakers = [{"id": unknown_id, "label": "UNKNOWN", "key": key}]
        for lbl in labels:
            speakers.append({"id": stable_id(key, "spk", lbl), "label": lbl, "key": key})

        seg_updates: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        any_overlap = False

        for s in segs:
            sid = s["id"]; s_start = float(s["start"]); s_end = float(s["end"])
            totals = sum_overlap(s_start, s_end, rttm)
            total_ov = sum(totals.values()) or 0.0

            if totals:
                any_overlap = True
                best_lbl = max(totals.items(), key=lambda kv: kv[1])[0]
                best_id  = stable_id(key, "spk", best_lbl)
            else:
                best_lbl = "UNKNOWN"
                best_id  = unknown_id

            seg_updates.append({"id": sid, "speaker_label": best_lbl, "speaker_id": best_id})

            if total_ov > 0.0:
                for lbl, ov in totals.items():
                    prop = ov / total_ov
                    if prop >= min_prop:
                        edges.append({
                            "segment_id": sid,
                            "speaker_id": stable_id(key, "spk", lbl),
                            "proportion": float(prop),
                            "overlap": float(ov),
                        })

        if require_overlap and not any_overlap:
            logging.warning(f"[{key}] No RTTM/segment overlap -> skip write (path={path})")
            continue

        if not seg_updates:
            logging.info(f"[{key}] No segments to update -> skip")
            continue

        if dry_run:
            logging.info(f"[dry-run] [{key}] segments={len(seg_updates)} edges={len(edges)} rttm={path}")
            continue

        sess.run(WRITE_BACK, speakers=speakers, segments=seg_updates, edges=edges)
        updated += len(seg_updates)
        logging.info(f"[{key}] updated {len(seg_updates)} segments, edges={len(edges)} (rttm={path})")

    return updated

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Backfill Segment speakers from t.source_rttm, with optional discovery.")
    ap.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    ap.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"))
    ap.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", "password"))
    ap.add_argument("--db", default=os.getenv("NEO4J_DB", "neo4j"))

    ap.add_argument("--batch", type=int, default=25, help="Max Transcriptions per fetch")
    ap.add_argument("--only-missing", action="store_true", default=True,
                    help="Only Segments missing speaker_label (default true)")
    ap.add_argument("--include-labeled", action="store_true",
                    help="Process all Segments (overrides only-missing)")

    ap.add_argument("--min-proportion", type=float, default=0.05,
                    help="Minimum mixture share to keep SPOKEN_BY edges")
    ap.add_argument("--no-require-overlap", action="store_true",
                    help="Write even if RTTM had no overlap with any Segment")
    ap.add_argument("--dry-run", action="store_true", help="Log actions only; no writes")

    # Discovery (opt-in)
    ap.add_argument("--allow-discovery", action="store_true",
                    help="If t.source_rttm is missing/invalid, search RTTM dirs and persist discovered path")
    ap.add_argument("--rttm-dir", action="append", default=[],
                    help="Directory to recursively index for *_speakers.rttm (repeatable)")

    args = ap.parse_args()
    only_missing = not args.include_labeled
    require_overlap = not args.no_require_overlap

    # Build RTTM index if discovery enabled
    idx = None
    if args.allow_discovery:
        search_dirs = args.rttm_dir or [
            "/mnt/8TB_2025/fileserver/dashcam/audio",
            "/mnt/8TB_2025/fileserver/audio",
            "/mnt/8TB_2025/fileserver/dashcam",
        ]
        idx = index_rttm_dirs(search_dirs)

    drv = driver_from_env(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
    total = 0
    try:
        with drv.session(database=args.db) as sess:
            while True:
                n = reconcile_once(
                    sess=sess,
                    limit=args.batch,
                    only_missing=only_missing,
                    min_prop=args.min_proportion,
                    require_overlap=require_overlap,
                    dry_run=args.dry_run,
                    allow_discovery=args.allow_discovery,
                    rttm_index=idx,
                )
                if n == 0:
                    break
                total += n
        logging.info(f"Done. Updated {total} segment(s).")
    finally:
        drv.close()

if __name__ == "__main__":
    main()
