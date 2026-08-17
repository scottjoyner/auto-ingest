#!/usr/bin/env python3
import argparse
import json
import logging
import os
import re

from auto_ingest_config import get_fileserver_path, get_neo4j_config

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from neo4j import GraphDatabase
from tqdm import tqdm
from transformers import pipeline


# --------------- Helper ----------------
def neo4j_env():
    cfg = get_neo4j_config() or {}
    try:
        from auto_ingest_config import get_neo4j_password
        pw = get_neo4j_password(cfg.get("password"))
    except Exception:
        pw = os.environ.get("NEO4J_PASSWORD") or cfg.get("password") or os.environ.get("NEO4J_PASSWORD_DEFAULT") or "knowledge_graph_2026"
    return {
        "uri": os.environ.get("NEO4J_URI") or cfg.get("uri") or "bolt://localhost:7687",
        "user": os.environ.get("NEO4J_USER") or cfg.get("user") or "neo4j",
        "password": pw,
        "db": os.environ.get("NEO4J_DB") or "neo4j",
    }

# --------------- Config ----------------
_NEO4J_CFG = get_neo4j_config()
NEO4J_URI  = os.getenv("NEO4J_URI", _NEO4J_CFG["uri"])
NEO4J_USER = os.getenv("NEO4J_USER", _NEO4J_CFG["user"])
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", _NEO4J_CFG["password"])

BATCH_LIMIT   = 1000             # per run; tune for your box
MODEL_DIR     = os.getenv("ZS_MODEL_DIR") or "facebook/bart-large-mnli"
CANDIDATES    = ["song lyrics", "conversation"]

# Prefer NOT misclassifying speech -> conservative thresholds
W_AUDIO = 0.50
W_TEXT  = 0.40
W_RULES = 0.10
THR_LYRICS = 0.80     # require high evidence to call lyrics
THR_SPEECH = 0.35     # be generous labeling speech

# Where to fetch music segments from
SEGMENTS_SOURCE = os.getenv("SEGMENTS_SOURCE", "neo4j")  # "neo4j" or "sidecar"
SIDECAR_SUFFIX  = ".music.json"
AUDIO_ROOTS     = [
    get_fileserver_path("audio"),
    get_fileserver_path("dashcam/audio"),
    get_fileserver_path("bodycam/audio"),
]

# --------------- Helpers ---------------
@dataclass
class Utt:
    uid: str
    text: str
    start: Optional[float]
    end: Optional[float]
    audio_key: Optional[str]

def _fetch_utterances(tx, lim: int, only_unclassified: bool = False) -> List[Utt]:
    where = "u.lyrics_score IS NULL" if only_unclassified else "u.lyrics_score IS NULL OR u.review_needed = true"
    q = f"""
    MATCH (u:Utterance)
    WHERE {where}
    OPTIONAL MATCH (t:Transcription)-[:HAS_UTTERANCE]->(u)
    RETURN u.id AS uid, u.text AS text, u.start AS start, u.end AS end,
           coalesce(u.audio_key, t.key) AS audio_key
    LIMIT $lim
    """
    rows = tx.run(q, lim=lim).data()
    return [Utt(r["uid"], r.get("text") or "", r.get("start"), r.get("end"), r.get("audio_key")) for r in rows]


def _fetch_flagged_utterances(tx, lim: int) -> List[Utt]:
    """Utterances already classified as lyrics/music (for --mark-only backfill)."""
    q = """
    MATCH (u:Utterance)
    WHERE (u.is_lyrics = true OR u.music_overlap >= 0.5)
      AND u.lyrics_score IS NOT NULL
    RETURN u.id AS uid, u.text AS text, u.start AS start, u.end AS end,
           u.audio_key AS audio_key
    LIMIT $lim
    """
    rows = tx.run(q, lim=lim).data()
    return [Utt(r["uid"], r.get("text") or "", r.get("start"), r.get("end"), r.get("audio_key")) for r in rows]

def _write_back(tx, uid: str, payload: Dict[str, Any]):
    q = """
    MATCH (u:Utterance {id:$uid})
    SET u.is_lyrics      = $is_lyrics,
        u.lyrics_score   = $lyrics_score,
        u.lyrics_evidence= $lyrics_evidence,
        u.music_overlap  = $music_overlap,
        u.review_needed  = $review_needed
    """
    tx.run(q,
           uid=uid,
           is_lyrics=payload["is_lyrics"],
           lyrics_score=payload["lyrics_score"],
           lyrics_evidence=json.dumps(payload["lyrics_evidence"]),
           music_overlap=payload["music_overlap"],
           review_needed=payload["review_needed"])

# UNIFICATION §3.2: propagate a non-speech label onto the Segment nodes that the
# linker embeds. The linker (auto_ingest.diarize.link_global_speakers) gates on
# Segment.segment_type, so we map any flagged Utterance -> overlapping Segment on
# the same Transcription and tag it music_or_media. This is the only new property
# we add; it requires no schema change and piggybacks on the flags above.
def _mark_overlapping_segments(tx, uid: str, segment_type: str):
    q = """
    MATCH (t:Transcription)-[:HAS_UTTERANCE]->(u:Utterance {id:$uid})
    MATCH (t)-[:HAS_SEGMENT]->(s:Segment)
    WHERE u.start IS NOT NULL AND u.end IS NOT NULL
      AND s.start IS NOT NULL AND s.end IS NOT NULL
      AND coalesce(s.end,0) > coalesce(u.start,0)
      AND coalesce(u.end,0) > coalesce(s.start,0)
    SET s.segment_type = $st
    """
    tx.run(q, uid=uid, st=segment_type)

# Load all music segments for a given audio_key (from Neo4j or sidecar JSON)
def _load_segments_from_neo4j(tx, key: str) -> List[Tuple[float,float]]:
    q = """
    MATCH (af:AudioFile {key:$key})-[:HAS_SEGMENT]->(s:AudioSegment {kind:"music"})
    RETURN s.start AS start, s.end AS end
    ORDER BY start
    """
    return [(float(r["start"]), float(r["end"])) for r in tx.run(q, key=key).data()]

def _sidecar_paths_for_key(key: str) -> List[str]:
    """Resolve sidecar candidate paths for `key` in O(1) via the audio index.

    Walks the audio tree only when the key is not present in the index (rare,
    e.g. audio added after the last index build). Never lists a missing root.
    """
    audio_index: Dict[str, List[str]] = {}
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_index.json")) as fh:
            audio_index = json.load(fh)
    except Exception:
        audio_index = {}
    sidecars = []
    for path in audio_index.get(key, []):
        sc = path + SIDECAR_SUFFIX
        if os.path.exists(sc):
            sidecars.append(sc)
    if sidecars:
        return sidecars
    # fallback: guarded walk for keys missing from the index
    for root in AUDIO_ROOTS:
        if not os.path.isdir(root):
            continue
        for dirpath, _, files in os.walk(root):
            for f in files:
                stem, ext = os.path.splitext(f)
                if ext.lower() in (".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".mp4"):
                    if key == stem or key in stem:
                        sc = os.path.join(dirpath, f) + SIDECAR_SUFFIX
                        if os.path.exists(sc):
                            sidecars.append(sc)
    return sidecars

def _load_segments_from_sidecar(key: str) -> List[Tuple[float,float]]:
    for sc in _sidecar_paths_for_key(key):
        try:
            with open(sc) as f:
                data = json.load(f)
            return [(float(s), float(e)) for s, e in data.get("music", [])]
        except Exception:
            continue
    return []

def interval_overlap(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    s = max(a[0], b[0]); e = min(a[1], b[1])
    return max(0.0, e - s)

def music_overlap_fraction(segments: List[Tuple[float,float]], ustart: Optional[float], uend: Optional[float]) -> float:
    if ustart is None or uend is None or uend <= ustart or not segments:
        return 0.0
    dur = float(uend - ustart)
    inter = 0.0
    for s, e in segments:
        inter += interval_overlap((ustart, uend), (s, e))
    return max(0.0, min(1.0, inter / dur))

# --------- Text scoring ----------
ZS_BATCH = 32  # zero-shot texts per pipeline call (CPU throughput sweet spot)

def build_zs(model_dir: str):
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else -1
    except Exception:
        dev = -1
    return pipeline("zero-shot-classification", model=model_dir, device=dev, batch_size=ZS_BATCH)

def text_lyrics_prob(zs, text: str) -> float:
    t = (text or "").strip()
    if not t:
        return 0.0
    out = zs(t, candidate_labels=CANDIDATES, multi_label=False)
    labs = out["labels"]; scores = out["scores"]
    d = dict(zip(labs, scores))
    return float(d.get("song lyrics", 0.0))

def text_lyrics_prob_batch(zs, texts: List[str]) -> List[float]:
    """Batched zero-shot scoring. HF pipeline preserves input order in its
    list output, so one call per N texts is far faster than per-text on CPU.
    Empty/whitespace texts are returned as 0.0 (zero-shot rejects blanks)."""
    probs = [0.0] * len(texts)
    idxs = [i for i, t in enumerate(texts) if (t or "").strip()]
    if not idxs:
        return probs
    batch = [texts[i] for i in idxs]
    try:
        results = zs(batch, candidate_labels=CANDIDATES, multi_label=False)
        for slot, i in enumerate(idxs):
            res = results[slot]
            d = dict(zip(res["labels"], res["scores"]))
            probs[i] = float(d.get("song lyrics", 0.0))
    except Exception as e:
        # Fall back to per-item scoring if the batch call rejects the input.
        logging.warning(f"zero-shot batch call failed ({e!r}); falling back to per-item")
        for i in idxs:
            probs[i] = text_lyrics_prob(zs, texts[i])
    return probs

def rules_features_prob(text: str) -> Tuple[float, Dict[str,float]]:
    t = (text or "").lower().strip()
    if not t:
        return 0.0, {"repetition":0.0,"rhyme":0.0,"stopwords":0.0,"lala":0.0,"linebreak":0.0}

    lines = [ln.strip() for ln in re.split(r"[\r\n]+", t) if ln.strip()]
    linebreak_rate = min(1.0, (len(lines)-1) / max(1, len(t)/80))

    tokens = re.findall(r"[a-zA-Z']+", t)
    bigrams = list(zip(tokens, tokens[1:]))
    rep = 0.0
    if bigrams:
        from collections import Counter
        c = Counter(bigrams)
        rep = sum(v for _,v in c.items() if v>1) / max(1,len(bigrams))
        rep = min(1.0, rep*3)

    endings = [ln[-3:] for ln in lines if len(ln)>=3]
    rhyme = 0.0
    if endings:
        from collections import Counter
        c = Counter(endings)
        rhyme = max(c.values())/max(1,len(lines))
        rhyme = min(1.0, rhyme)

    lala = 1.0 if re.search(r"\b(la+|na+|yeah+|ooh+|mmm+)\b", t) else 0.0

    stop = {"the","a","an","and","or","but","to","of","in","on","for","with","is","are","am","be","was","were","it","that","this","you","i"}
    sw_ratio = sum(1 for tok in tokens if tok in stop)/max(1,len(tokens))
    stop_inv = max(0.0, 0.4 - sw_ratio) / 0.4

    vals = {"repetition":float(rep),"rhyme":float(rhyme),"stopwords":float(stop_inv),
            "lala":float(lala),"linebreak":float(linebreak_rate)}
    rules_prob = float(np.clip(0.25*rep + 0.25*rhyme + 0.20*lala + 0.15*linebreak_rate + 0.15*stop_inv, 0, 1))
    return rules_prob, vals

def ensemble(audio_p: float, text_p: float, rules_p: float) -> Tuple[float, bool, bool]:
    score = float(W_AUDIO*audio_p + W_TEXT*text_p + W_RULES*rules_p)
    if score >= THR_LYRICS:
        return score, True, False
    if score <= THR_SPEECH:
        return score, False, False
    return score, False, True

# --------------- Main ---------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=BATCH_LIMIT)
    ap.add_argument("--model", type=str, default=MODEL_DIR)
    ap.add_argument("--segments-source", choices=["neo4j","sidecar"], default=SEGMENTS_SOURCE)
    ap.add_argument("--all", action="store_true",
                    help="Loop in batches until every Utterance with lyrics_score IS NULL is "
                         "classified (review_needed=true re-reviews are excluded so the loop "
                         "terminates; re-run manually to revisit borderline cases).")
    ap.add_argument("--mark-only", action="store_true",
                    help="Skip classification; only backfill Segment.segment_type from "
                         "existing flags (is_lyrics=true OR music_overlap>=0.5). One-time "
                         "migration for utterances classified before segment marking existed.")
    args = ap.parse_args()

    zs = build_zs(args.model)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    seg_cache: Dict[str, List[Tuple[float,float]]] = {}

    with driver.session() as sess:
        while True:
            if args.mark_only:
                utts = sess.execute_read(_fetch_flagged_utterances, args.limit)
                if not utts:
                    logging.info("No more flagged utterances to backfill; done.")
                    break
                for u in tqdm(utts, desc="mark"):
                    sess.execute_write(_mark_overlapping_segments, u.uid, "music_or_media")
                break
            utts = sess.execute_read(_fetch_utterances, args.limit, only_unclassified=args.all)
            if not utts:
                logging.info("No more unclassified utterances; done.")
                break
            # Preload segments for all keys in this batch (1 round trip per key if Neo4j)
            keys = sorted({u.audio_key for u in utts if u.audio_key})
            if args.segments_source == "neo4j":
                for k in keys:
                    segs = sess.execute_read(_load_segments_from_neo4j, k)
                    seg_cache[k] = segs
            else:
                for k in keys:
                    seg_cache[k] = _load_segments_from_sidecar(k)

            # Batch the zero-shot text calls: the model is the bottleneck and HF's
            # pipeline accepts a list of texts (output follows input order), so one
            # call per ZS_BATCH texts is far faster than one call per text on CPU.
            for i in range(0, len(utts), ZS_BATCH):
                chunk = utts[i:i + ZS_BATCH]
                text_probs = text_lyrics_prob_batch(zs, [u.text for u in chunk])
                for u, text_prob in zip(tqdm(chunk, desc="classify"), text_probs):
                    # audio overlap
                    segs = seg_cache.get(u.audio_key or "", [])
                    overlap = music_overlap_fraction(segs, u.start, u.end)
                    # map overlap -> audio probability (simple identity works well)
                    audio_prob = overlap

                    rules_prob, rules_vals = rules_features_prob(u.text)

                    score, is_lyrics, needs_review = ensemble(audio_prob, text_prob, rules_prob)

                    evidence = {
                        "audio_prob": audio_prob,
                        "music_overlap": overlap,
                        "text_prob": text_prob,
                        "rules_prob": rules_prob,
                        "rules": rules_vals,
                        "weights": {"audio":W_AUDIO,"text":W_TEXT,"rules":W_RULES},
                        "model": os.path.basename(args.model.rstrip("/")),
                        "segments_source": args.segments_source
                    }
                    payload = {
                        "is_lyrics": bool(is_lyrics),
                        "lyrics_score": float(score),
                        "lyrics_evidence": evidence,
                        "music_overlap": float(overlap),
                        "review_needed": bool(needs_review)
                    }
                    sess.execute_write(_write_back, u.uid, payload)
                    # Flag overlapping Segment nodes as non-speech so the global linker
                    # excludes them from ECAPA/pyannote embedding + linking.
                    if is_lyrics or overlap >= 0.5:
                        sess.execute_write(_mark_overlapping_segments, u.uid, "music_or_media")

            if not args.all:
                break

    driver.close()

if __name__ == "__main__":
    main()
