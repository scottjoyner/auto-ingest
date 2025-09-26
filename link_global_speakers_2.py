#!/usr/bin/env python3
"""
Global speaker linking with:
  - Segment-level diarization selection (SPOKEN_BY proportion) with auto fallback to Utterances
  - SNR/RMS gating to drop noisy snips
  - Weighted centroids (duration * proportion * quality factor)
  - SQLite per-snippet embedding cache
  - GlobalSpeaker embeddings stored & incrementally updated in Neo4j (running mean with weight_sum)
  - Vector index on GlobalSpeaker.embedding for similarity search
  - Quarantine of uncertain clusters + hold-out validation
  - Per-file snip caps, audio path cache
  - FAISS prefilter for local clustering (optional)
  - **NEW: Neo4j→FAISS Global prefilter** — assign locals directly to existing GlobalSpeakers before local clustering

Requires: neo4j, torch, torchaudio, soundfile, numpy, speechbrain (or pyannote)
Optional: faiss (faiss-cpu)
"""

import os
import re
import json
import random
import logging
import hashlib
import math
import sqlite3
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torchaudio
import soundfile as sf
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

# Optional FAISS
try:
    import faiss  # pip install faiss-cpu
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False

# -----------------------
# Config
# -----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

AUDIO_BASE = Path(os.getenv("AUDIO_BASE", "/mnt/8TB_2025/fileserver/audio"))
ALT_AUDIO_BASES = [
    Path("/mnt/8TB_2025/fileserver/dashcam/audio"),
    Path("/mnt/8TB_2025/fileserver/dashcam"),
]
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg"}

# Neo4j
NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "livelongandprosper")
NEO4J_DB       = os.getenv("NEO4J_DB", "neo4j")

# Embedding backend
SPK_MODEL  = os.getenv("SPK_MODEL", "speechbrain")  # "speechbrain" | "pyannote"
ECAPA_NAME = os.getenv("ECAPA_NAME", "speechbrain/spkrec-ecapa-voxceleb")

# Defaults
DEFAULT_MIN_SEG = 0.7     # min segment/utterance duration (s)
DEFAULT_MAX_SNIPS = 8     # target #snips per local Speaker
DEFAULT_MAX_PER_FILE = 3  # cap #snips per speaker per file_key
DEFAULT_PAD = 0.15        # seconds padding around snip window
DEFAULT_SR = 16000
DEFAULT_THRESH = 0.72     # cosine threshold to cluster new groups
DEFAULT_MIN_PROP = 0.5    # min SPOKEN_BY proportion to accept a segment
DEFAULT_SNIP_LEN = 1.6    # fixed snip length (s)
TEXT_ALPHA_MIN = 2        # text heuristic (ignore music/noise)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hold-out validation
DEFAULT_HOLDOUT = True
DEFAULT_HOLDOUT_MIN = 0.60
DEFAULT_HOLDOUT_ACTION = "drop-members"  # "drop-members" | "skip-cluster" | "ignore"

# Audio path cache
DEFAULT_AUDIO_CACHE = "./audio_path_cache.json"
DEFAULT_CACHE_REFRESH = False

# Per-snippet embedding cache (SQLite)
DEFAULT_EMB_CACHE = "./emb_cache.sqlite"
DEFAULT_EMB_REFRESH = False
EMB_CACHE_SCHEMA_VER = 2  # bumped to add rms/snr columns

# Quarantine thresholds
DEFAULT_QUARANTINE_MIN = 0.70  # min internal pairwise score for "confirmed"
DEFAULT_SINGLETON_TENTATIVE = True

# Gating
DEFAULT_MIN_RMS = 0.005
DEFAULT_MIN_SNR_DB = 6.0
DEFAULT_WEIGHT_QUALITY = True  # include quality factor in weights

# FAISS (local clustering)
DEFAULT_FAISS_PREFILTER = False
DEFAULT_FAISS_K = 64
DEFAULT_FAISS_INDEX = "hnsw"  # "hnsw" | "flatip"
DEFAULT_FAISS_M = 32
DEFAULT_FAISS_EF = 128

# Global prefilter to existing GS
DEFAULT_GLOBAL_PREFILTER = True
DEFAULT_GLOBAL_K = 8
DEFAULT_GLOBAL_INDEX = "hnsw"  # "hnsw" | "flatip"
DEFAULT_GLOBAL_M = 32
DEFAULT_GLOBAL_EF = 128
DEFAULT_GLOBAL_THRESH = 0.78
DEFAULT_GLOBAL_INCLUDE_TENTATIVE = False
DEFAULT_SKIP_ALREADY_LINKED = True

# ---- config (keep your existing sets, but add this) ----
AUDIO_ONLY_DIR_HINTS = ("audio", "dashcam", "recorder", "wav", "mp3")  # directories likely to contain audio
SIDECAR_MARKERS = (
    "_transcription", "_speakers", "_entities", "_subjects", "_summary",
    "_tasks", "_chunks", "_medium", "_large", "_tiny", "_base", "_small",
)

def _norm_stem(key: str) -> str:
    """
    Normalize a file_key into a plain audio stem by stripping common sidecar suffixes.
    Example: 2025_0202_171732_medium_transcription_speakers -> 2025_0202_171732
    """
    s = key
    # drop extension if present
    s = re.sub(r"\.[A-Za-z0-9]+$", "", s)
    # remove repeated sidecar tokens
    for m in SIDECAR_MARKERS:
        s = re.sub(fr"{re.escape(m)}($|_)", "_", s)
    s = re.sub(r"_+$", "", s)
    # dashcam R-suffix normalize
    s = re.sub(r"_R$", "", s)
    return s

def _is_audio_path(p: Path) -> bool:
    return p.suffix.lower() in AUDIO_EXTS and p.is_file()

def _looks_like_audio_dir(p: Path) -> bool:
    return any(tok in p.as_posix().lower().split("/") for tok in AUDIO_ONLY_DIR_HINTS)

def discover_audio_for_key(key: str) -> Optional[Path]:
    """
    Strictly return a valid audio file for a given key.
    - Only consider AUDIO_EXTS
    - Prefer files in 'audio-like' directories
    - Avoid sidecar/transcription folders by name
    """
    stem = _norm_stem(key)
    # 1) strict filename match stem + ext
    for base in [AUDIO_BASE, *ALT_AUDIO_BASES]:
        for ext in AUDIO_EXTS:
            hits = sorted(base.rglob(f"**/{stem}{ext}"))
            hits = [h for h in hits if _is_audio_path(h)]
            if hits:
                # prefer 'real' audio directories
                hits.sort(key=lambda h: (not _looks_like_audio_dir(h.parent), len(h.as_posix())))
                return hits[0]

    # 2) relaxed: allow stem-* with audio ext, but reject known sidecar patterns
    bad_fragments = ("transcription", "speakers", "entities", "subjects", "summary", "chunks")
    for base in [AUDIO_BASE, *ALT_AUDIO_BASES]:
        candidates = []
        for ext in AUDIO_EXTS:
            candidates.extend(base.rglob(f"**/{stem}_*{ext}"))
        candidates = [c for c in candidates if _is_audio_path(c)]
        candidates = [c for c in candidates if not any(b in c.name.lower() for b in bad_fragments)]
        if candidates:
            candidates.sort(key=lambda h: (not _looks_like_audio_dir(h.parent), len(h.as_posix())))
            return candidates[0]

    return None

def load_audio(path: Path, target_sr: int = DEFAULT_SR) -> Tuple[torch.Tensor, int]:
    """
    Robust loader:
      - Torchaudio primary
      - SoundFile fallback for WAV/FLAC
    Returns mono float32 tensor [T], sr
    """
    if not _is_audio_path(path):
        raise RuntimeError(f"Not an audio file: {path}")

    try:
        wav, sr = torchaudio.load(str(path))
    except Exception:
        # fallback for typical PCM containers
        if path.suffix.lower() in {".wav", ".flac", ".ogg"}:
            import soundfile as _sf
            data, sr = _sf.read(str(path), dtype="float32", always_2d=False)
            if data.ndim == 2:
                data = data.mean(axis=1)
            wav = torch.from_numpy(data)
        else:
            raise

    if wav.ndim == 2:
        wav = wav.mean(dim=0)  # mono
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return wav.contiguous(), sr


# -----------------------
# Utils
# -----------------------
def stable_id(*parts: str) -> str:
    h = hashlib.md5()
    for p in parts:
        h.update((p or "").encode("utf-8", errors="ignore"))
        h.update(b"|")
    return h.hexdigest()

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x)
    return x / n if n > 0 else x


# -----------------------
# Neo4j helpers
# -----------------------
def driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def ensure_schema(drv):
    cy = [
        "CREATE CONSTRAINT global_speaker_id IF NOT EXISTS FOR (g:GlobalSpeaker) REQUIRE g.id IS UNIQUE",
        "CREATE INDEX speaker_label IF NOT EXISTS FOR (s:Speaker) ON (s.label)",
    ]
    with drv.session(database=NEO4J_DB) as sess:
        for q in cy:
            sess.run(q)

def ensure_global_vector_index(sess, dim: int):
    try:
        sess.run(f"""
        CREATE VECTOR INDEX global_speaker_embedding_index IF NOT EXISTS
        FOR (g:GlobalSpeaker) ON (g.embedding)
        OPTIONS {{
          indexConfig: {{
            `vector.dimensions`: {dim},
            `vector.similarity_function`: 'cosine'
          }}
        }}
        """)
    except Neo4jError as e:
        logging.warning(f"Vector index creation warning: {e}")


def _regex_ok(s: Optional[str], rx: Optional[re.Pattern]) -> bool:
    return bool(s and rx and rx.search(s))

def ensure_priority_gs(sess, name_regex: Optional[str]) -> Optional[str]:
    """
    Find (or create) the priority GlobalSpeaker by matching name/aliases with regex.
    Returns gid or None.
    """
    if not name_regex:
        return None
    rx = re.compile(name_regex, re.I)

    # Try to locate an existing GS by name or aliases
    recs = sess.run("""
      MATCH (g:GlobalSpeaker)
      RETURN g.id AS gid, g.name AS name, coalesce(g.aliases, []) AS aliases, g.embedding AS emb
    """)
    best_gid, best_hit = None, -1
    for r in recs:
        name = r["name"] or ""
        aliases = r["aliases"] or []
        if _regex_ok(name, rx) or any(_regex_ok(a, rx) for a in aliases):
            # Prefer GS with embedding (already accumulated)
            hit = 1 if (r["emb"] is not None) else 0
            if hit > best_hit:
                best_gid, best_hit = r["gid"], hit

    if best_gid:
        return best_gid

    # Create a fresh priority GS if not found
    gid = hashlib.md5(("priority|" + name_regex).encode("utf-8")).hexdigest()
    sess.run("""
      MERGE (g:GlobalSpeaker {id:$gid})
      ON CREATE SET g.created_at=datetime(), g.status='tentative', g.name=$name
      ON MATCH  SET g.updated_at=datetime()
    """, gid=gid, name=name_regex)
    return gid

def assign_to_priority_first(
    sess,
    priority_gid: str,
    local_centroids: Dict[str, np.ndarray],
    local_weights: Dict[str, float],
    skip_map: Dict[str, str],
    thresh: float,
    max_attach: int
) -> Tuple[Dict[str, Tuple[str, float]], List[str]]:
    """
    Greedy, stricter, single-target assignment to priority_gid.
    Returns:
      best_map: sid -> (gid, score)  (only for attached)
      attached_sids: list of sids attached
    """
    if not priority_gid or not local_centroids:
        return {}, []

    # Fetch priority embedding (may be None on first run)
    r = sess.run("MATCH (g:GlobalSpeaker {id:$gid}) RETURN g.embedding AS emb, g.weight_sum AS w",
                 gid=priority_gid).single()
    e_prev = None if not r or r["emb"] is None else unit(np.asarray(r["emb"], dtype=np.float32))
    # If no prior emb, we bootstrap by taking top-N densest locals (by weight) as seed
    items = sorted(((sid, float(local_weights.get(sid,1.0))) for sid in local_centroids if sid not in skip_map),
                   key=lambda x: x[1], reverse=True)
    best_map: Dict[str, Tuple[str, float]] = {}
    attached = []

    # If we have an existing embedding, do a strict dot check; else take first few to build it
    seed_take = min(20, len(items))
    if e_prev is None and seed_take:
        sum_vec = np.zeros_like(next(iter(local_centroids.values())), dtype=np.float32)
        wsum = 0.0
        for sid, w in items[:seed_take]:
            sum_vec += unit(local_centroids[sid]) * w
            wsum += w
        e_prev = unit(sum_vec) if wsum > 0 else None

    if e_prev is None:
        # nothing to do; no basis to compare
        return {}, []

    # Now attach by strict threshold
    for sid, _w in items:
        if len(attached) >= max_attach:
            break
        sc = float(np.dot(unit(local_centroids[sid]), e_prev))
        if sc >= thresh:
            best_map[sid] = (priority_gid, sc)
            attached.append(sid)

    # Write the incremental update with only these assignments
    if attached:
        update_existing_gs_with_assignments(
            sess,
            {priority_gid: attached},
            local_centroids,
            local_weights,
            method="priority-attach",
            quarantine_min=0.75  # keep it a bit conservative
        )
    return best_map, attached

def _seek_page_speaker_ids(sess, min_seg_sec: float, min_prop: float,
                           after_id: Optional[str], limit: int) -> List[Tuple[str, str]]:
    """
    Returns list[(speaker_id, label)], ordered by speaker_id, strictly greater than after_id (seek pagination).
    Only speakers that HAVE qualifying segments are returned (duration & proportion gates).
    """
    cy = """
    MATCH (t:Transcription)-[:HAS_SEGMENT]->(s:Segment)-[r:SPOKEN_BY]->(sp:Speaker)
    WHERE coalesce(s.end,0) - coalesce(s.start,0) >= $min_seg
      AND coalesce(r.proportion,1.0) >= $min_prop
      AND ($after IS NULL OR sp.id > $after)
    WITH sp.id AS sid, max(sp.label) AS label
    RETURN sid AS speaker_id, label AS label
    ORDER BY speaker_id
    LIMIT $limit
    """
    rows = sess.run(cy, min_seg=min_seg_sec, min_prop=min_prop, after=after_id, limit=limit)
    return [(r["speaker_id"], r["label"]) for r in rows]

def _fetch_items_for_speaker(sess, sid: str, min_seg_sec: float, min_prop: float, per_speaker_limit: int):
    """
    Returns list of tuples: (file_key, start, end, text, prop), already sorted by quality.
    """
    cy = """
    MATCH (sp:Speaker {id:$sid})<- [r:SPOKEN_BY] - (s:Segment) <-[:HAS_SEGMENT]- (t:Transcription)
    WHERE coalesce(s.end,0) - coalesce(s.start,0) >= $min_seg
      AND coalesce(r.proportion,1.0) >= $min_prop
    RETURN t.key AS file_key, s.start AS start, s.end AS end, s.text AS text, coalesce(r.proportion,1.0) AS prop
    ORDER BY prop DESC, (coalesce(s.end,0) - coalesce(s.start,0)) DESC
    LIMIT $lim
    """
    rows = sess.run(cy, sid=sid, min_seg=min_seg_sec, min_prop=min_prop, lim=per_speaker_limit)
    out = []
    for r in rows:
        out.append( (r["file_key"], float(r["start"] or 0.0), float(r["end"] or 0.0),
                     (r["text"] or ""), float(r["prop"] or 1.0)) )
    return out


def compute_dominance_and_label_transcriptions(sess):
    """
    - Compute dominance metrics for GlobalSpeakers (duration, files, days).
    - Label each Transcription with its primary GlobalSpeaker (by sum proportion).
    """
    # Aggregate per-transcription proportions to GlobalSpeaker
    sess.run("""
    // Link Transcription -> GlobalSpeaker with proportion sum
    MATCH (t:Transcription)-[:HAS_SEGMENT]->(s:Segment)-[r:SPOKEN_BY]->(sp:Speaker)-[:SAME_PERSON]->(g:GlobalSpeaker)
    WITH t, g, sum(coalesce(r.proportion,1.0) * (coalesce(s.end,0)-coalesce(s.start,0))) AS dur
    MERGE (t)-[rg:PRIMARY_SPEAKER_OF]->(g)
    ON CREATE SET rg.created_at=datetime()
    SET rg.updated_at=datetime(), rg.duration=dur
    """)
    sess.run("""
    // Keep only the max PRIMARY_SPEAKER_OF per Transcription
    MATCH (t:Transcription)-[r:PRIMARY_SPEAKER_OF]->(g:GlobalSpeaker)
    WITH t, g, r, r.duration AS d
    ORDER BY t.key, d DESC
    WITH t, collect({g:g, r:r}) AS lst
    UNWIND range(0, size(lst)-1) AS i
    WITH t, lst[i].g AS g, lst[i].r AS r, i
    WITH t, g, r, i
    FOREACH (_ IN CASE WHEN i=0 THEN [1] ELSE [] END | SET t.primary_global_speaker = g.id)
    FOREACH (_ IN CASE WHEN i>0 THEN [1] ELSE [] END | DELETE r)
    """)
    # Dominance metrics per GlobalSpeaker
    sess.run("""
    MATCH (t:Transcription)-[:HAS_SEGMENT]->(s:Segment)-[r:SPOKEN_BY]->(:Speaker)-[:SAME_PERSON]->(g:GlobalSpeaker)
    WITH g,
         sum(coalesce(r.proportion,1.0) * (coalesce(s.end,0)-coalesce(s.start,0))) AS total_speech_s,
         count(DISTINCT t.key) AS files,
         count(DISTINCT date(datetime(coalesce(t.started_at, t.created_at)))) AS active_days
    SET g.total_speech_s = total_speech_s,
        g.file_count     = files,
        g.active_days    = active_days,
        g.dominance      = total_speech_s + 30*active_days + 5*files,
        g.updated_at     = datetime()
    """)


# -----------------------
# Fetch (Segment first, fallback to Utterance)
# -----------------------
def _fetch_from_segments(sess, min_seg_sec: float, min_prop: float) -> Dict[str, Dict]:
    cy = """
    MATCH (t:Transcription)-[:HAS_SEGMENT]->(s:Segment)-[r:SPOKEN_BY]->(sp:Speaker)
    WHERE coalesce(s.end,0) - coalesce(s.start,0) >= $min_seg
      AND coalesce(r.proportion, 1.0) >= $min_prop
    RETURN sp.id AS speaker_id, sp.label AS label, t.key AS file_key,
           s.start AS start, s.end AS end, s.text AS text,
           coalesce(r.proportion,1.0) AS prop
    ORDER BY speaker_id, file_key, start
    """
    out: Dict[str, Dict] = {}
    for rec in sess.run(cy, min_seg=min_seg_sec, min_prop=min_prop):
        sid = rec["speaker_id"]
        d = out.setdefault(sid, {"label": rec["label"], "items": []})
        d["items"].append((rec["file_key"], float(rec["start"] or 0.0),
                           float(rec["end"] or 0.0), (rec["text"] or ""),
                           float(rec["prop"] or 1.0)))
    return out

def _fetch_from_utterances(sess, min_seg_sec: float) -> Dict[str, Dict]:
    cy = """
    MATCH (t:Transcription)-[:HAS_UTTERANCE]->(u:Utterance)-[:SPOKEN_BY]->(sp:Speaker)
    WHERE coalesce(u.end,0) - coalesce(u.start,0) >= $min_seg
    RETURN sp.id AS speaker_id, sp.label AS label, t.key AS file_key,
           u.start AS start, u.end AS end, u.text AS text,
           1.0 AS prop
    ORDER BY speaker_id, file_key, start
    """
    out: Dict[str, Dict] = {}
    for rec in sess.run(cy, min_seg=min_seg_sec):
        sid = rec["speaker_id"]
        d = out.setdefault(sid, {"label": rec["label"], "items": []})
        d["items"].append((rec["file_key"], float(rec["start"] or 0.0),
                           float(rec["end"] or 0.0), (rec["text"] or ""),
                           1.0))
    return out

def fetch_speakers_and_segments(drv, min_seg_sec: float, min_prop: float,
                                include_unknown: bool, limit_speakers: int = 0,
                                source_level: str = "auto",
                                speaker_batch: int = 1500,
                                items_per_speaker: int = 64,
                                max_speakers: int = 0) -> Dict[str, Dict]:
    """
    Batched, seek-paginated pull to avoid server transaction OOMs.
    Returns { speaker_id: {label, items:[(file_key,start,end,text,prop), ...]} }
    """
    out: Dict[str, Dict] = {}
    total = 0
    after = None
    with drv.session(database=NEO4J_DB) as sess:
        while True:
            ids = _seek_page_speaker_ids(sess, min_seg_sec, min_prop, after, speaker_batch)
            if not ids:
                break
            for sid, label in ids:
                items = _fetch_items_for_speaker(sess, sid, min_seg_sec, min_prop, items_per_speaker)
                if items:
                    out[sid] = {"label": label, "items": items}
                    total += 1
                    if (limit_speakers and total >= limit_speakers) or (max_speakers and total >= max_speakers):
                        return out
            after = ids[-1][0]
    return out



# -----------------------
# Audio path cache
# -----------------------
def load_audio_cache(path: Optional[Path], refresh: bool) -> Dict[str, str]:
    if refresh:
        return {}
    if not path or not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_audio_cache(path: Optional[Path], cache: Dict[str, str]):
    if not path:
        return
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as ex:
        logging.warning(f"Failed to write audio cache {path}: {ex}")

def _norm_stem(key: str) -> str:
    return re.sub(r"_R$", "", key)

def discover_audio_for_key(key: str) -> Optional[Path]:
    stem = _norm_stem(key)
    for ext in AUDIO_EXTS:
        hits = list(AUDIO_BASE.rglob(f"**/{stem}{ext}"))
        if hits:
            return hits[0]
    for base in ALT_AUDIO_BASES:
        for ext in AUDIO_EXTS:
            hits = list(base.rglob(f"**/{stem}{ext}"))
            if hits:
                return hits[0]
    hits = list(AUDIO_BASE.rglob(f"**/{stem}*"))
    return hits[0] if hits else None


# -----------------------
# Audio loading + snips + gating
# -----------------------
def load_audio(path: Path, target_sr: int = DEFAULT_SR) -> Tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return wav.squeeze(0), sr

def fixed_snip(wav: torch.Tensor, sr: int, start: float, end: float, pad: float, snip_len: float) -> torch.Tensor:
    seg_mid = (start + end) / 2.0
    half = snip_len / 2.0
    s = max(0, int(round((seg_mid - half - pad) * sr)))
    e = min(wav.numel(), int(round((seg_mid + half + pad) * sr)))
    if e <= s:
        return torch.zeros(0)
    return wav[s:e].clone()

def rms_and_snr_db(x: torch.Tensor) -> Tuple[float, float]:
    if x.numel() == 0:
        return 0.0, 0.0
    x_np = x.detach().cpu().numpy().astype(np.float32)
    rms = float(np.sqrt(np.mean(x_np**2)))
    noise = float(np.percentile(np.abs(x_np), 20.0))
    snr = 20.0 * math.log10(max(rms, 1e-8) / max(noise, 1e-8))
    return rms, snr


# -----------------------
# Embedding backends
# -----------------------
class SpkEmbedder:
    def __init__(self, backend: str = SPK_MODEL):
        self.backend = backend
        self._ecapa = None
        self._pyannote = None

    def _ensure_ecapa(self):
        if self._ecapa is None:
            from speechbrain.pretrained import EncoderClassifier
            logging.info("Loading SpeechBrain ECAPA encoder…")
            self._ecapa = EncoderClassifier.from_hparams(
                source=ECAPA_NAME, run_opts={"device": DEVICE}
            )

    def _ensure_pyannote(self):
        if self._pyannote is None:
            from pyannote.audio import Inference
            logging.info("Loading pyannote speaker embedding…")
            self._pyannote = Inference("pyannote/embedding", device=DEVICE)

    @torch.inference_mode()
    def embed(self, wav: torch.Tensor, sr: int) -> np.ndarray:
        if wav.numel() == 0:
            return np.zeros((192,), dtype=np.float32)
        if self.backend == "speechbrain":
            self._ensure_ecapa()
            emb = self._ecapa.encode_batch(wav.unsqueeze(0).to(DEVICE)).squeeze(0).squeeze(0).cpu().numpy()
            return emb.astype(np.float32)
        else:
            self._ensure_pyannote()
            arr = wav.cpu().numpy().astype(np.float32)
            emb = self._pyannote(arr, sample_rate=sr)
            return np.asarray(emb, dtype=np.float32)


# -----------------------
# SQLite per-snippet embedding cache
# -----------------------
class EmbCache:
    def __init__(self, path: Optional[Path], refresh: bool, backend: str, model_name: str, snip_len: float, sr: int):
        self.path = Path(path) if path else None
        self.refresh = refresh
        self.backend = backend
        self.model_name = model_name
        self.snip_len = snip_len
        self.sr = sr
        self.conn = None
        if self.path:
            self._open()

    def _open(self):
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)")
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS emb_cache (
          key TEXT PRIMARY KEY,
          speaker_id TEXT,
          file_key TEXT,
          start REAL,
          end REAL,
          backend TEXT,
          model TEXT,
          snip_len REAL,
          sr INTEGER,
          dim INTEGER,
          emb BLOB,
          rms REAL,
          snr_db REAL,
          created_at REAL
        )
        """)
        cur = self.conn.execute("SELECT v FROM meta WHERE k='schema_ver'")
        row = cur.fetchone()
        if not row or int(row[0]) != EMB_CACHE_SCHEMA_VER:
            self.conn.execute("INSERT OR REPLACE INTO meta(k,v) VALUES('schema_ver',?)", (str(EMB_CACHE_SCHEMA_VER),))
            self.conn.commit()

    def _mk_key(self, speaker_id: str, file_key: str, start: float, end: float) -> str:
        base = f"{speaker_id}|{file_key}|{start:.3f}|{end:.3f}|{self.backend}|{self.model_name}|{self.snip_len:.2f}|{self.sr}"
        return hashlib.md5(base.encode("utf-8")).hexdigest()

    def get(self, speaker_id: str, file_key: str, start: float, end: float) -> Optional[Tuple[np.ndarray, float, float]]:
        if not self.conn or self.refresh:
            return None
        k = self._mk_key(speaker_id, file_key, start, end)
        cur = self.conn.execute("SELECT dim, emb, rms, snr_db FROM emb_cache WHERE key=?", (k,))
        row = cur.fetchone()
        if not row:
            return None
        dim = int(row[0]); arr = np.frombuffer(row[1], dtype=np.float32)
        if arr.size != dim:
            return None
        rms = float(row[2]) if row[2] is not None else 0.0
        snr_db = float(row[3]) if row[3] is not None else 0.0
        return arr, rms, snr_db

    def set(self, speaker_id: str, file_key: str, start: float, end: float, emb: np.ndarray, rms: float, snr_db: float):
        if not self.conn:
            return
        k = self._mk_key(speaker_id, file_key, start, end)
        emb = np.asarray(emb, dtype=np.float32)
        self.conn.execute(
            "INSERT OR REPLACE INTO emb_cache(key,speaker_id,file_key,start,end,backend,model,snip_len,sr,dim,emb,rms,snr_db,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (k, speaker_id, file_key, float(start), float(end), self.backend, self.model_name, float(self.snip_len), int(self.sr),
             int(emb.size), emb.tobytes(), float(rms), float(snr_db), time.time())
        )
        self.conn.commit()


# -----------------------
# Selection utilities
# -----------------------
def cap_snips_per_file(items: List[Tuple[str, float, float, str, float]],
                       max_per_file: int, max_total: int, rng: random.Random):
    items_sorted = sorted(items, key=lambda x: (x[4], x[2] - x[1]), reverse=True)
    buckets: Dict[str, List] = {}
    for it in items_sorted:
        buckets.setdefault(it[0], []).append(it)
    selected = []
    for fk, lst in buckets.items():
        selected.extend(lst[:max_per_file])
        if len(selected) >= max_total:
            break
    if len(selected) > max_total:
        rng.shuffle(selected)
        selected = selected[:max_total]
    return selected


# -----------------------
# FAISS approx clustering (locals)
# -----------------------
def cluster_by_threshold_faiss(
    centroids: Dict[str, np.ndarray],
    thresh: float,
    k: int,
    index_type: str = "hnsw",
    hnsw_m: int = 32,
    hnsw_ef: int = 128
):
    ids = list(centroids.keys())
    if len(ids) <= 1:
        return [ids] if ids else [], {}

    arr = np.stack([unit(np.asarray(centroids[i], dtype=np.float32)) for i in ids], axis=0).astype(np.float32)
    d = arr.shape[1]
    k = max(1, min(k, len(ids) - 1)) + 1  # include self

    if index_type == "flatip":
        index = faiss.IndexFlatIP(d)
    else:
        index = faiss.IndexHNSWFlat(d, hnsw_m)
        index.hnsw.efSearch = hnsw_ef

    index.add(arr)
    sims, nbrs = index.search(arr, k)

    parent = {i: i for i in range(len(ids))}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    scores: Dict[Tuple[str,str], float] = {}
    seen = set()
    candidates = 0
    for i in range(len(ids)):
        for jj in range(1, k):
            j = int(nbrs[i, jj])
            if j < 0 or j == i:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            candidates += 1
            s = float(np.dot(arr[i], arr[j]))  # cosine
            if s >= thresh:
                union(i, j)
                scores[(ids[a], ids[b])] = s

    comps: Dict[int, List[str]] = {}
    for i, sid in enumerate(ids):
        r = find(i)
        comps.setdefault(r, []).append(sid)
    groups = list(comps.values())
    groups.sort(key=len, reverse=True)

    total_pairs = len(ids) * (len(ids) - 1) // 2
    logging.info(f"FAISS local prefilter ~{candidates} pairs vs {total_pairs} exact.")
    return groups, scores


# -----------------------
# Global prefilter helpers
# -----------------------
def fetch_global_speaker_embs(sess, include_tentative: bool) -> Dict[str, np.ndarray]:
    if include_tentative:
        cy = "MATCH (g:GlobalSpeaker) WHERE g.embedding IS NOT NULL RETURN g.id AS gid, g.embedding AS emb"
    else:
        cy = "MATCH (g:GlobalSpeaker {status:'confirmed'}) WHERE g.embedding IS NOT NULL RETURN g.id AS gid, g.embedding AS emb"
    out = {}
    for rec in sess.run(cy):
        arr = np.asarray(rec["emb"], dtype=np.float32)
        if arr.size:
            out[rec["gid"]] = unit(arr)
    return out

def fetch_already_linked_speakers(sess) -> Dict[str, str]:
    cy = "MATCH (s:Speaker)-[:SAME_PERSON]->(g:GlobalSpeaker) RETURN s.id AS sid, g.id AS gid"
    m = {}
    for rec in sess.run(cy):
        m[rec["sid"]] = rec["gid"]
    return m

def assign_locals_to_globals(
    local_centroids: Dict[str, np.ndarray],
    global_embs: Dict[str, np.ndarray],
    thresh: float,
    k: int,
    use_faiss: bool,
    index_type: str,
    hnsw_m: int,
    hnsw_ef: int
) -> Tuple[Dict[str, List[str]], Dict[str, Tuple[str, float]]]:
    """
    Returns:
      assignments: gid -> [local_speaker_id, ...]
      best_map: local_speaker_id -> (gid, score)
    """
    assignments: Dict[str, List[str]] = {}
    best_map: Dict[str, Tuple[str, float]] = {}
    if not global_embs or not local_centroids:
        return assignments, best_map

    # Prepare matrices
    gids = list(global_embs.keys())
    G = np.stack([unit(global_embs[g]) for g in gids], axis=0).astype(np.float32)
    l_ids = list(local_centroids.keys())
    L = np.stack([unit(local_centroids[s]) for s in l_ids], axis=0).astype(np.float32)

    if use_faiss and HAVE_FAISS:
        d = G.shape[1]
        if index_type == "flatip":
            index = faiss.IndexFlatIP(d)
        else:
            index = faiss.IndexHNSWFlat(d, hnsw_m)
            index.hnsw.efSearch = hnsw_ef
        index.add(G)
        kq = min(k, len(gids))
        sims, nbrs = index.search(L, kq)
        for i, sid in enumerate(l_ids):
            # pick top-1 over threshold
            s_best = -1.0
            g_best = None
            for jj in range(kq):
                j = int(nbrs[i, jj])
                if j < 0:
                    continue
                s = float(np.dot(L[i], G[j]))  # cosine (unit vectors)
                if s > s_best:
                    s_best = s
                    g_best = gids[j]
            if g_best is not None and s_best >= thresh:
                assignments.setdefault(g_best, []).append(sid)
                best_map[sid] = (g_best, s_best)
    else:
        # exact
        Gt = G.T  # (d, Ng)
        sims = L @ Gt  # (Nl, Ng)
        for i, sid in enumerate(l_ids):
            j = int(np.argmax(sims[i]))
            s_best = float(sims[i, j])
            if s_best >= thresh:
                gid = gids[j]
                assignments.setdefault(gid, []).append(sid)
                best_map[sid] = (gid, s_best)
    return assignments, best_map

def update_existing_gs_with_assignments(
    sess,
    assignments: Dict[str, List[str]],
    local_centroids: Dict[str, np.ndarray],
    local_weights: Dict[str, float],
    method: str,
    quarantine_min: float
):
    """
    For each gid:
      - fetch g.embedding and g.weight_sum
      - compute new embedding by running-mean with local weighted sum
      - set status=confirmed if min(new edge scores) >= quarantine_min else tentative
      - create SAME_PERSON edges for members with score=cos(local, e_new)
    """
    for gid, sids in assignments.items():
        prev = sess.run("MATCH (g:GlobalSpeaker {id:$gid}) RETURN g.embedding AS emb, g.weight_sum AS w", gid=gid).single()
        if not prev:
            continue
        e_prev = np.asarray(prev["emb"], dtype=np.float32) if prev["emb"] is not None else None
        w_prev = float(prev["w"]) if prev["w"] is not None else 0.0

        # local contribution
        sum_vec = np.zeros_like(next(iter(local_centroids.values())), dtype=np.float32)
        wsum = 0.0
        for sid in sids:
            w = float(local_weights.get(sid, 1.0))
            sum_vec += w * unit(np.asarray(local_centroids[sid], dtype=np.float32))
            wsum += w

        if e_prev is not None and e_prev.size:
            sum_combined = unit(e_prev) * w_prev + sum_vec
            w_combined = w_prev + wsum
        else:
            sum_combined = sum_vec
            w_combined = wsum if wsum > 0 else 1.0

        e_new = unit(sum_combined)
        status = "confirmed"
        # compute edge scores & tentative flags
        edge_payload = []
        min_edge = 1.0
        for sid in sids:
            sc = float(np.dot(unit(local_centroids[sid]), e_new))
            min_edge = min(min_edge, sc)
            tentative = (sc < quarantine_min)
            if tentative:
                status = "tentative"
            edge_payload.append((sid, sc, tentative, float(local_weights.get(sid, 1.0))))

        # write GS
        sess.run("""
          MERGE (g:GlobalSpeaker {id:$gid})
          ON CREATE SET g.created_at=datetime()
          ON MATCH  SET g.updated_at=datetime()
          SET g.embedding=$emb, g.weight_sum=$w, g.status=$status, g.method=$method
        """, gid=gid, emb=list(map(float, e_new)), w=float(w_combined), status=status, method=method)

        # write edges
        for sid, sc, tentative, wloc in edge_payload:
            sess.run("""
              MATCH (s:Speaker {id:$sid}), (g:GlobalSpeaker {id:$gid})
              MERGE (s)-[r:SAME_PERSON]->(g)
              ON CREATE SET r.created_at=datetime()
              SET r.updated_at=datetime(), r.method=$method, r.score=$score, r.tentative=$tentative, r.weight=$w
            """, sid=sid, gid=gid, method=method, score=float(sc), tentative=bool(tentative), w=float(wloc))


# -----------------------
# Write clusters with incremental update in Neo4j (new groups)
# -----------------------
def write_clusters_incremental(
    drv,
    groups: List[List[str]],
    pair_scores: Dict[Tuple[str,str], float],
    centroids: Dict[str, np.ndarray],
    spk_weights: Dict[str, float],
    method: str,
    quarantine_min: float,
    singleton_tentative: bool,
    store_embeddings: bool,
    dry_run: bool
):
    cluster_payload = []
    for g in groups:
        sum_vec = np.zeros_like(next(iter(centroids.values())), dtype=np.float32)
        wsum = 0.0
        for sid in g:
            w = float(spk_weights.get(sid, 1.0))
            sum_vec += w * np.asarray(centroids[sid], dtype=np.float32)
            wsum += w
        gcen_unit = unit(sum_vec)
        conf = 1.0
        if len(g) >= 2:
            vals = []
            for i in range(len(g)):
                for j in range(i+1, len(g)):
                    a, b = g[i], g[j]
                    vals.append(pair_scores.get((min(a,b), max(a,b)), 0.0))
            conf = min(vals) if vals else 0.0
        tentative = ((len(g) == 1) and singleton_tentative) or (conf < quarantine_min)
        cluster_payload.append((g, sum_vec, wsum, gcen_unit, conf, tentative))

    with drv.session(database=NEO4J_DB) as sess:
        if store_embeddings and cluster_payload:
            dim = len(cluster_payload[0][3])
            ensure_global_vector_index(sess, dim)

        for g, sum_vec, wsum, gcen_unit, conf, tentative in cluster_payload:
            cluster_sorted = sorted(g)
            gid = hashlib.md5("|".join(cluster_sorted).encode("utf-8")).hexdigest()
            status = "tentative" if tentative else "confirmed"

            if dry_run:
                logging.info(f"DRY-RUN cluster<{len(cluster_sorted)}> gs={gid} status={status} conf={conf:.3f} members={cluster_sorted}")
            else:
                prev = sess.run(
                    "MATCH (g:GlobalSpeaker {id:$gid}) RETURN g.embedding AS emb, g.weight_sum AS w",
                    gid=gid
                ).single()
                if store_embeddings:
                    if prev and prev["emb"] is not None and prev["w"] is not None:
                        e_prev = np.asarray(prev["emb"], dtype=np.float32)
                        w_prev = float(prev["w"])
                        sum_combined = e_prev * w_prev + sum_vec
                        w_combined = w_prev + wsum
                        e_new = unit(sum_combined)
                        w_new = w_combined
                    else:
                        e_new = gcen_unit
                        w_new = wsum if wsum > 0 else 1.0

                    sess.run("""
                      MERGE (g:GlobalSpeaker {id:$gid})
                      ON CREATE SET g.created_at=datetime()
                      ON MATCH  SET g.updated_at=datetime()
                      SET g.status=$status, g.method=$method, g.confidence=$conf,
                          g.embedding=$emb, g.weight_sum=$w
                    """, gid=gid, status=status, method=method, conf=float(conf),
                         emb=list(map(float, e_new)), w=float(w_new))
                else:
                    sess.run("""
                      MERGE (g:GlobalSpeaker {id:$gid})
                      ON CREATE SET g.created_at=datetime()
                      ON MATCH  SET g.updated_at=datetime()
                      SET g.status=$status, g.method=$method, g.confidence=$conf
                    """, gid=gid, status=status, method=method, conf=float(conf))

                for sid in cluster_sorted:
                    best = 1.0 if len(cluster_sorted) == 1 else max(
                        pair_scores.get((min(sid, x), max(sid, x)), 0.0)
                        for x in cluster_sorted if x != sid
                    )
                    sess.run("""
                      MATCH (s:Speaker {id:$sid}), (g:GlobalSpeaker {id:$gid})
                      MERGE (s)-[r:SAME_PERSON]->(g)
                      ON CREATE SET r.method=$method, r.score=$score, r.weight=$w, r.tentative=$tentative, r.created_at=datetime()
                      ON MATCH  SET r.method=$method, r.score=$score, r.weight=$w, r.tentative=$tentative, r.updated_at=datetime()
                    """, sid=sid, gid=gid, method=method, score=float(best),
                         w=float(spk_weights.get(sid, 1.0)), tentative=bool(tentative))


# -----------------------
# CLI + main
# -----------------------
@dataclass
class Args:
    # selection / clipping
    min_seg: float
    max_snips: int
    max_per_file: int
    pad: float
    snip_len: float
    thresh: float
    min_prop: float
    include_unknown: bool
    limit_speakers: int
    backend: str
    write_snips: bool
    dry_run: bool
    snips_dir: Optional[Path]
    source_level: str
    speaker_batch: int
    items_per_speaker: int
    max_speakers: int
    # holdout
    holdout: bool
    holdout_min: float
    holdout_action: str

    # caches
    audio_cache: Optional[Path]
    cache_refresh: bool
    emb_cache: Optional[Path]
    emb_refresh: bool

    # embeddings / quarantine
    store_embeddings: bool
    quarantine_min: float
    singleton_tentative: bool

    # gating & weighting
    min_rms: float
    min_snr_db: float
    weight_quality: bool

    # FAISS (local)
    faiss_prefilter: bool
    faiss_k: int
    faiss_index: str
    faiss_m: int
    faiss_ef: int

    # Global prefilter
    global_prefilter: bool
    global_k: int
    global_index: str
    global_m: int
    global_ef: int
    global_thresh: float
    global_include_tentative: bool
    skip_already_linked: bool

    # 🔹 NEW: priority identity + ranking
    priority_name: Optional[str]          # regex like "Scott|Kipnerter"
    priority_thresh: float                # stricter attach threshold
    priority_max_attach: int              # safety cap
    rank_and_label: bool                  # compute dominance + label transcriptions


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Link local Speaker nodes to GlobalSpeaker identities (Segment-based, FAISS global prefilter, caches, quarantine, incremental update).")
    p.add_argument("--min-seg", type=float, default=DEFAULT_MIN_SEG)
    p.add_argument("--max-snips", type=int, default=DEFAULT_MAX_SNIPS)
    p.add_argument("--max-per-file", type=int, default=DEFAULT_MAX_PER_FILE)
    p.add_argument("--pad", type=float, default=DEFAULT_PAD)
    p.add_argument("--snip-len", type=float, default=DEFAULT_SNIP_LEN)
    p.add_argument("--thresh", type=float, default=DEFAULT_THRESH)
    p.add_argument("--min-proportion", dest="min_prop", type=float, default=DEFAULT_MIN_PROP)
    p.add_argument("--include-unknown", action="store_true")
    p.add_argument("--limit-speakers", type=int, default=0)
    p.add_argument("--backend", type=str, choices=["speechbrain","pyannote"], default=SPK_MODEL)
    p.add_argument("--write-snips", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--snips-dir", type=str, default=None)
    p.add_argument("--source-level", type=str, choices=["auto","segment","utterance"], default="auto")

    # holdout
    p.add_argument("--holdout", action="store_true", default=DEFAULT_HOLDOUT)
    p.add_argument("--holdout-min", type=float, default=DEFAULT_HOLDOUT_MIN)
    p.add_argument("--holdout-action", type=str, choices=["drop-members","skip-cluster","ignore"], default=DEFAULT_HOLDOUT_ACTION)

    # caches
    p.add_argument("--audio-cache", type=str, default=DEFAULT_AUDIO_CACHE)
    p.add_argument("--cache-refresh", action="store_true", default=DEFAULT_CACHE_REFRESH)
    p.add_argument("--emb-cache", type=str, default=DEFAULT_EMB_CACHE)
    p.add_argument("--emb-refresh", action="store_true", default=DEFAULT_EMB_REFRESH)

    # embeddings/quarantine
    p.add_argument("--store-embeddings", dest="store_embeddings", action="store_true", default=True)
    p.add_argument("--no-store-embeddings", dest="store_embeddings", action="store_false")
    p.add_argument("--quarantine-min", type=float, default=DEFAULT_QUARANTINE_MIN)
    p.add_argument("--singleton-tentative", dest="singleton_tentative", action="store_true", default=DEFAULT_SINGLETON_TENTATIVE)
    p.add_argument("--no-singleton-tentative", dest="singleton_tentative", action="store_false")

    # gating & weighting
    p.add_argument("--min-rms", type=float, default=DEFAULT_MIN_RMS)
    p.add_argument("--min-snr-db", type=float, default=DEFAULT_MIN_SNR_DB)
    p.add_argument("--no-weight-quality", dest="weight_quality", action="store_false", default=DEFAULT_WEIGHT_QUALITY)

    # FAISS (local)
    p.add_argument("--faiss-prefilter", action="store_true", default=DEFAULT_FAISS_PREFILTER)
    p.add_argument("--faiss-k", type=int, default=DEFAULT_FAISS_K)
    p.add_argument("--faiss-index", type=str, choices=["hnsw","flatip"], default=DEFAULT_FAISS_INDEX)
    p.add_argument("--faiss-m", type=int, default=DEFAULT_FAISS_M)
    p.add_argument("--faiss-ef", type=int, default=DEFAULT_FAISS_EF)

    # NEW: Global prefilter
    p.add_argument("--global-prefilter", action="store_true", default=DEFAULT_GLOBAL_PREFILTER, help="Assign locals to existing GlobalSpeakers before local clustering.")
    p.add_argument("--global-k", type=int, default=DEFAULT_GLOBAL_K)
    p.add_argument("--global-index", type=str, choices=["hnsw","flatip"], default=DEFAULT_GLOBAL_INDEX)
    p.add_argument("--global-m", type=int, default=DEFAULT_GLOBAL_M)
    p.add_argument("--global-ef", type=int, default=DEFAULT_GLOBAL_EF)
    p.add_argument("--global-thresh", type=float, default=DEFAULT_GLOBAL_THRESH, help="Cosine threshold to attach a local speaker to an existing GlobalSpeaker.")
    p.add_argument("--global-include-tentative", action="store_true", default=DEFAULT_GLOBAL_INCLUDE_TENTATIVE, help="Include tentative GlobalSpeakers in the assignment index.")
    p.add_argument("--skip-already-linked", action="store_true", default=DEFAULT_SKIP_ALREADY_LINKED, help="Skip local speakers that already have SAME_PERSON to any GlobalSpeaker.")
    p.add_argument("--priority-name", type=str, default=None, help="Regex (| separated) to match GlobalSpeaker.name/aliases for the priority identity (e.g., 'Scott|Kipnerter').")
    p.add_argument("--priority-thresh", type=float, default=0.82, help="Cosine threshold to attach locals to the priority GlobalSpeaker (stricter to avoid contamination).")
    p.add_argument("--priority-max-attach", type=int, default=20000, help="Safety cap for number of local speakers to attach to the priority GS this run.")
    p.add_argument("--rank-and-label", action="store_true", default=False, help="Compute dominance metrics and label Transcriptions' primary GlobalSpeaker.")
    p.add_argument("--speaker-batch", type=int, default=1500,
               help="How many distinct Speaker IDs to page per DB roundtrip.")
    p.add_argument("--items-per-speaker", type=int, default=64,
                help="Upper bound of (Segment, proportion) rows fetched per speaker before local caps.")
    p.add_argument("--max-speakers", type=int, default=0,
                help="Optional global cap of speakers processed this run (0 = no cap).")

    return Args(**vars(p.parse_args()))


# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()
    drv = driver()
    ensure_schema(drv)

    logging.info("Querying Segment-level diarization (with auto fallback)…")
    speakers = fetch_speakers_and_segments(
        drv,
        min_seg_sec=args.min_seg,
        min_prop=args.min_prop,
        include_unknown=args.include_unknown,
        limit_speakers=args.limit_speakers,
        source_level=args.source_level,
    )
    logging.info(f"Found {len(speakers)} speakers with qualifying items.")

    # audio cache
    cache_path = Path(args.audio_cache) if args.audio_cache else None
    audio_cache: Dict[str, str] = load_audio_cache(cache_path, args.cache_refresh)

    def get_audio_for_key(key: str) -> Optional[Tuple[torch.Tensor, int, Path]]:
        # 1) Try cache
        p_str = audio_cache.get(key)
        if p_str:
            p = Path(p_str)
            try:
                if _is_audio_path(p):
                    wav, sr = load_audio(p, DEFAULT_SR)
                    return wav, sr, p
                else:
                    # cached path is not audio; drop it
                    audio_cache.pop(key, None)
            except Exception as ex:
                logging.warning(f"Cached audio failed for key={key}: {p} ({ex}); removing from cache.")
                audio_cache.pop(key, None)

        # 2) Discover afresh
        p = discover_audio_for_key(key)
        if not p or not p.exists():
            logging.warning(f"Audio missing for key={key}")
            return None

        # Validate + load
        try:
            wav, sr = load_audio(p, DEFAULT_SR)
        except Exception as ex:
            logging.warning(f"Discovered audio failed for key={key}: {p} ({ex})")
            return None

        audio_cache[key] = str(p)
        return wav, sr, p


    # per-snippet embedding cache
    model_name = ECAPA_NAME if args.backend == "speechbrain" else "pyannote/embedding"
    emb_cache = EmbCache(Path(args.emb_cache) if args.emb_cache else None,
                         refresh=args.emb_refresh,
                         backend=args.backend,
                         model_name=model_name,
                         snip_len=args.snip_len,
                         sr=DEFAULT_SR)

    embedder = SpkEmbedder(args.backend)
    rng = random.Random(1337)

    # outputs
    spk_centroids: Dict[str, np.ndarray] = {}
    spk_weights: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    holdout_embs: Dict[str, Optional[np.ndarray]] = {}

    out_dir = Path(args.snips_dir) if args.write_snips and args.snips_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Selecting, gating, clipping & embedding per Speaker…")
    for sid, meta in speakers.items():
        items = meta["items"]  # (key, start, end, text, prop)
        items.sort(key=lambda x: (x[4], x[2]-x[1]), reverse=True)

        # basic text filter to avoid non-speech
        items_text = [it for it in items if sum(c.isalpha() for c in (it[3] or "")) >= TEXT_ALPHA_MIN] or items

        # cap per file and max total
        chosen = cap_snips_per_file(items_text, args.max_per_file, args.max_snips, rng)
        if not chosen:
            logging.warning(f"No eligible segments for speaker {sid}")
            continue

        # hold-out candidate
        hold_idx = (len(chosen) - 1) if (args.holdout and len(chosen) >= 2) else None

        # accumulate weighted sum
        sum_vec = None
        wsum = 0.0
        used = 0
        held: Optional[np.ndarray] = None

        for idx, (key, s, e, text, prop) in enumerate(chosen):
            pack = get_audio_for_key(key)
            if pack is None:
                continue
            wav, sr, _ = pack
            snip = fixed_snip(wav, sr, s, e, args.pad, args.snip_len)
            if snip.numel() == 0:
                continue

            # gating
            rms, snr_db = rms_and_snr_db(snip)
            if rms < args.min_rms or snr_db < args.min_snr_db:
                continue

            if out_dir:
                safe_key = re.sub(r"[^A-Za-z0-9_-]", "", key)
                name = f"{sid}_{safe_key}_{s:.2f}-{e:.2f}_p{prop:.2f}_r{rms:.3f}_snr{snr_db:.1f}{'_hold' if hold_idx==idx else ''}"
                sf.write(str(out_dir / f"{name}.wav"), snip.cpu().numpy(), sr, subtype="PCM_16")

            cached = emb_cache.get(sid, key, s, e)
            if cached is None:
                emb = embedder.embed(snip, sr)
                emb_cache.set(sid, key, s, e, emb, rms, snr_db)
            else:
                emb, _, _ = cached

            # per-snippet weight
            duration = max(float(e - s), 0.0)
            qscale = 1.0
            if args.weight_quality:
                qscale = max(0.5, min(2.0, (snr_db - args.min_snr_db) / 10.0 + 1.0))
            w = float(prop) * duration * qscale

            emb_u = unit(np.asarray(emb, dtype=np.float32))
            sum_vec = (emb_u * w) if sum_vec is None else (sum_vec + emb_u * w)
            wsum += w
            used += 1

            if hold_idx is not None and idx == hold_idx:
                held = emb_u

        if used == 0:
            logging.warning(f"All snips gated out for speaker {sid}")
            continue

        cen = unit(sum_vec if sum_vec is not None else np.zeros_like(emb_u))
        spk_centroids[sid] = cen
        spk_weights[sid] = max(wsum, 1e-6)
        counts[sid] = used
        holdout_embs[sid] = held

    if cache_path:
        save_audio_cache(cache_path, audio_cache)

    if not spk_centroids:
        logging.warning("No centroids built; nothing to cluster.")
        return

    med = int(np.median(list(counts.values()))) if counts else 0
    dim = len(next(iter(spk_centroids.values())))
    logging.info(f"Built {len(spk_centroids)} speaker centroids (dim={dim}, median snips per speaker used={med}).")

    method = "ecapa" if args.backend == "speechbrain" else "pyannote"

    # -----------------------
    # Global prefilter: assign locals directly to existing GlobalSpeakers
    # -----------------------
    assigned_locals: Dict[str, Tuple[str, float]] = {}  # sid -> (gid, score)
    if args.global_prefilter:
        with drv.session(database=NEO4J_DB) as sess:
            gs_embs = fetch_global_speaker_embs(sess, include_tentative=args.global_include_tentative)
            if not gs_embs:
                logging.info("No existing GlobalSpeaker embeddings found; skipping global prefilter.")
            else:
                already = fetch_already_linked_speakers(sess) if args.skip_already_linked else {}
                # remove already linked from consideration (optional)
                local_space = {sid: spk_centroids[sid] for sid in spk_centroids if sid not in already}
                if not local_space:
                    logging.info("All local speakers already linked; nothing to do.")
                else:
                    use_faiss = HAVE_FAISS
                    if not HAVE_FAISS and args.global_index in ("hnsw","flatip"):
                        logging.warning("FAISS not installed; using exact assignment to globals.")
                    assignments, best_map = assign_locals_to_globals(
                        local_space,
                        gs_embs,
                        thresh=args.global_thresh,
                        k=args.global_k,
                        use_faiss=use_faiss,
                        index_type=args.global_index,
                        hnsw_m=args.global_m,
                        hnsw_ef=args.global_ef
                    )
                    if assignments:
                        logging.info(f"Assigning {sum(len(v) for v in assignments.values())} local speakers to {len(assignments)} existing GlobalSpeakers.")
                        if not args.dry_run:
                            update_existing_gs_with_assignments(sess, assignments, spk_centroids, spk_weights, method, args.quarantine_min)
                        else:
                            logging.info("DRY-RUN: skipping GS update writes.")
                        assigned_locals = best_map
                    else:
                        logging.info("Global prefilter produced no assignments ≥ threshold.")

    # Remove assigned locals from the pool before local clustering
    remaining_centroids = {sid: vec for sid, vec in spk_centroids.items() if sid not in assigned_locals}
        # --- Priority attach to Scott/Kipnerter first (if requested) ---
    if args.priority_name:
        with drv.session(database=NEO4J_DB) as sess:
            pr_gid = ensure_priority_gs(sess, args.priority_name)
            if pr_gid:
                already = fetch_already_linked_speakers(sess) if args.skip_already_linked else {}
                best_map_pr, attached = assign_to_priority_first(
                    sess,
                    pr_gid,
                    remaining_centroids,
                    remaining_weights,
                    already,
                    thresh=args.priority_thresh,
                    max_attach=args.priority_max_attach
                )
                # remove those from remaining pool
                for sid in attached:
                    remaining_centroids.pop(sid, None)
                    remaining_weights.pop(sid, None)
                    remaining_holdouts.pop(sid, None)
                logging.info(f"Priority attach: {len(attached)} local speakers → GS<{pr_gid}>.")

    remaining_weights   = {sid: spk_weights[sid] for sid in remaining_centroids.keys()}
    remaining_holdouts  = {sid: holdout_embs[sid] for sid in remaining_centroids.keys()}

    logging.info(f"Remaining locals after global assignment: {len(remaining_centroids)}")

    # -----------------------
    # Cluster remaining locals (FAISS optional)
    # -----------------------
    if len(remaining_centroids) == 0:
        logging.info("Nothing left to cluster locally; done.")
        return

    if args.faiss_prefilter:
        if not HAVE_FAISS:
            logging.warning("FAISS not installed; falling back to exact clustering. pip install faiss-cpu")
            use_faiss_local = False
        else:
            use_faiss_local = True
    else:
        use_faiss_local = False

    if use_faiss_local:
        logging.info(f"Clustering (locals) with FAISS prefilter (k={args.faiss_k}, index={args.faiss_index}, M={args.faiss_m}, ef={args.faiss_ef}) at threshold {args.thresh:.3f} …")
        groups, pair_scores = cluster_by_threshold_faiss(
            remaining_centroids,
            thresh=args.thresh,
            k=args.faiss_k,
            index_type=args.faiss_index,
            hnsw_m=args.faiss_m,
            hnsw_ef=args.faiss_ef
        )
    else:
        logging.info(f"Clustering (locals) exactly at threshold {args.thresh:.3f} …")
        ids = list(remaining_centroids.keys())
        parent = {i: i for i in range(len(ids))}
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
        arr = {sid: unit(remaining_centroids[sid]) for sid in ids}
        pair_scores: Dict[Tuple[str,str], float] = {}
        n = len(ids)
        for i in range(n):
            ai = arr[ids[i]]
            for j in range(i+1, n):
                s = float(np.dot(ai, arr[ids[j]]))
                if s >= args.thresh:
                    union(i, j)
                    pair_scores[(ids[i], ids[j])] = s
        comps: Dict[int, List[str]] = {}
        for i, sid in enumerate(ids):
            r = find(i)
            comps.setdefault(r, []).append(sid)
        groups = list(comps.values())
        groups.sort(key=len, reverse=True)

    # ---- Hold-out validation on remaining clusters ----
    if args.holdout and groups:
        logging.info(f"Hold-out validation (min={args.holdout_min:.2f}, action={args.holdout_action}) …")
        filtered_groups = []
        dim = len(next(iter(remaining_centroids.values())))
        for g in groups:
            sum_vec = np.zeros(dim, dtype=np.float32)
            wsum = 0.0
            for s in g:
                w = remaining_weights.get(s, 1.0)
                sum_vec += w * remaining_centroids[s]
                wsum += w
            gcen = unit(sum_vec)
            fails = []
            for s in g:
                h = remaining_holdouts.get(s)
                if h is None:
                    continue
                score = cosine(h, gcen)
                if score < args.holdout_min:
                    fails.append((s, score))
            if not fails:
                filtered_groups.append(g)
            elif args.holdout_action == "ignore":
                filtered_groups.append(g)
            elif args.holdout_action == "skip-cluster":
                logging.info(f"Cluster skipped due to holdout fails: {[(s,round(sc,3)) for s,sc in fails]}")
            else:
                survivors = [s for s in g if all(s != f[0] for f in fails)]
                logging.info(f"Cluster pruned: dropped {len(fails)}; kept {len(survivors)}.")
                if survivors:
                    filtered_groups.append(survivors)
        groups = [sorted(list(dict.fromkeys(grp))) for grp in filtered_groups if grp]

    for i, g in enumerate(groups, 1):
        logging.info(f"New cluster {i}: size={len(g)} members={g[:10]}{' …' if len(g)>10 else ''}")

    write_clusters_incremental(
        drv,
        groups,
        pair_scores,
        remaining_centroids,
        remaining_weights,
        method=method,
        quarantine_min=args.quarantine_min,
        singleton_tentative=args.singleton_tentative,
        store_embeddings=args.store_embeddings,
        dry_run=args.dry_run,
    )
    if args.rank_and_label:
        with drv.session(database=NEO4J_DB) as sess:
            compute_dominance_and_label_transcriptions(sess)
        logging.info("Dominance computed and Transcriptions labeled with primary_global_speaker.")


    logging.info("Done.")


if __name__ == "__main__":
    main()

# python link_global_speakers.py \
#   --global-prefilter --global-thresh 0.78 --global-k 8 --global-index hnsw --global-m 32 --global-ef 128 \
#   --skip-already-linked \
#   --faiss-prefilter --faiss-k 64 --faiss-index hnsw --faiss-m 32 --faiss-ef 128 \
#   --source-level auto \
#   --min-seg 0.7 --max-snips 8 --max-per-file 3 --snip-len 1.6 \
#   --min-proportion 0.5 --min-rms 0.005 --min-snr-db 6.0 \
#   --thresh 0.72 \
#   --holdout --holdout-min 0.62 --holdout-action drop-members \
#   --audio-cache ./audio_path_cache.json \
#   --emb-cache ./emb_cache.sqlite

# More Agressive Command
# ./.venv/bin/python3 link_global_speakers.py \
#   --global-prefilter \
#   --global-thresh 0.74 --global-k 16 --global-index hnsw --global-m 48 --global-ef 256 \
#   --global-include-tentative \
#   --skip-already-linked \
#   --faiss-prefilter --faiss-k 128 --faiss-index hnsw --faiss-m 48 --faiss-ef 256 \
#   --source-level auto \
#   --min-seg 0.6 --max-snips 12 --max-per-file 4 --snip-len 2.0 \
#   --min-proportion 0.4 --min-rms 0.004 --min-snr-db 5.0 \
#   --thresh 0.68 \
#   --holdout --holdout-min 0.58 --holdout-action drop-members \
#   --quarantine-min 0.72 \
#   --priority-name "Scott|Kipnerter|Kipnerter Scott" \
#   --priority-thresh 0.82 \
#   --priority-max-attach 20000 \
#   --rank-and-label \
#   --audio-cache ./audio_path_cache.json \
#   --emb-cache ./emb_cache.sqlite || echo "❌ Failed: link_global_speakers.py"
