#!/usr/bin/env python3
import os, re, json, math, argparse
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from collections import defaultdict

import numpy as np
from neo4j import GraphDatabase
from transformers import pipeline
from tqdm import tqdm

# --------------- Config ----------------
NEO4J_URI  = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = os.getenv("NEO4J_PASS") or "livelongandprosper"

BATCH_LIMIT   = 1000             # per run; tune for your box
MODEL_DIR     = os.getenv("ZS_MODEL_DIR") or "models/roberta-large-mnli"  # or models/deberta-mnli
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
    "/mnt/8TB_2025/fileserver/audio",
    "/mnt/8TB_2025/fileserver/dashcam/audio",
    "/mnt/8TB_2025/fileserver/bodycam/audio",
]

# --------------- Helpers ---------------
@dataclass
class Utt:
    uid: str
    text: str
    start: Optional[float]
    end: Optional[float]
    audio_key: Optional[str]

def _fetch_utterances(tx, lim: int) -> List[Utt]:
    q = """
    MATCH (u:Utterance)
    WHERE u.lyrics_score IS NULL OR u.review_needed = true
    OPTIONAL MATCH (t:Transcription)-[:HAS_UTTERANCE]->(u)
    RETURN u.id AS uid, u.text AS text, u.start AS start, u.end AS end,
           coalesce(u.audio_key, t.key) AS audio_key
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

# Load all music segments for a given audio_key (from Neo4j or sidecar JSON)
def _load_segments_from_neo4j(tx, key: str) -> List[Tuple[float,float]]:
    q = """
    MATCH (af:AudioFile {key:$key})-[:HAS_SEGMENT]->(s:AudioSegment {kind:"music"})
    RETURN s.start AS start, s.end AS end
    ORDER BY start
    """
    return [(float(r["start"]), float(r["end"])) for r in tx.run(q, key=key).data()]

def _sidecar_paths_for_key(key: str) -> List[str]:
    cands = []
    for root in AUDIO_ROOTS:
        # try exact stem match
        cands.extend([os.path.join(root, p) for p in os.listdir(root) if False]) # placeholder to avoid os error
    # robust scan:
    sidecars = []
    for root in AUDIO_ROOTS:
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
def build_zs(model_dir: str):
    return pipeline("zero-shot-classification", model=model_dir, device=-1)

def text_lyrics_prob(zs, text: str) -> float:
    t = (text or "").strip()
    if not t:
        return 0.0
    out = zs(t, candidate_labels=CANDIDATES, multi_label=False)
    labs = out["labels"]; scores = out["scores"]
    d = dict(zip(labs, scores))
    return float(d.get("song lyrics", 0.0))

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=BATCH_LIMIT)
    ap.add_argument("--model", type=str, default=MODEL_DIR)
    ap.add_argument("--segments-source", choices=["neo4j","sidecar"], default=SEGMENTS_SOURCE)
    args = ap.parse_args()

    zs = build_zs(args.model)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    seg_cache: Dict[str, List[Tuple[float,float]]] = {}

    with driver.session() as sess:
        utts = sess.execute_read(_fetch_utterances, args.limit)

        # Preload segments for all keys in this batch (1 round trip per key if Neo4j)
        keys = sorted({u.audio_key for u in utts if u.audio_key})
        if args.segments_source == "neo4j":
            for k in keys:
                segs = sess.execute_read(_load_segments_from_neo4j, k)
                seg_cache[k] = segs
        else:
            for k in keys:
                seg_cache[k] = _load_segments_from_sidecar(k)

        for u in tqdm(utts, desc="classify"):
            # audio overlap
            segs = seg_cache.get(u.audio_key or "", [])
            overlap = music_overlap_fraction(segs, u.start, u.end)
            # map overlap -> audio probability (simple identity works well)
            audio_prob = overlap

            # text
            text_prob = text_lyrics_prob(zs, u.text)
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

    driver.close()

if __name__ == "__main__":
    main()
