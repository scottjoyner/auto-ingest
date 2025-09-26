#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ingest_transcriptsv5_3.py

Neo4j-safe (chunked) ingestion of transcripts + embeddings + speakers/utterances,
plus robust dashcam metadata ingestion with data-quality detection/repair.

Highlights vs v5.1:
- FIX: MODEL_PREF defined and used safely
- FIX: chunked writes (segments / utterances / edges / entities) to avoid tx OOM
- NEW: robust metadata parser w/ bbox, speed filter, lon flip, lat/lon swap, flags & quality
- NEW: CLI for tx batch sizes/timeouts/fetch size and metadata quality controls
"""

import os, re, uuid, csv, json, time, hashlib, logging, math, itertools
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

try:
    from zoneinfo import ZoneInfo  # py>=3.9
except Exception:
    ZoneInfo = None

# =========================
# Config & Logging
# =========================
LOG_LEVEL_DEFAULT = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL_DEFAULT, logging.INFO),
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ingest_transcripts")

EMA_ALPHA = float(os.getenv("EMA_ALPHA", "0.2"))

DEFAULT_SCAN_ROOTS = [
    "/mnt/8TB_2025/fileserver/dashcam/audio",
    "/mnt/8TB_2025/fileserver/dashcam/transcriptions",
    "/mnt/8TB_2025/fileserver/audio",
    "/mnt/8TB_2025/fileserver/audio/transcriptions",
    "/mnt/8TB_2025/fileserver/bodycam",
    "/mnt/8TB_2025/fileserver/dashcam",
    "/mnt/8TBHDD/fileserver/dashcam",
]
SCAN_ROOTS = [p.strip() for p in os.getenv("SCAN_ROOTS", ",".join(DEFAULT_SCAN_ROOTS)).split(",") if p.strip()]

# extra scan for metadata under dashcam
DASHCAM_ROOT = os.getenv("DASHCAM_ROOT", "/mnt/8TB_2025/fileserver/dashcam")
OLD_DASHCAM_ROOT = os.getenv("OLD_DASHCAM_ROOT", "/mnt/8TBHDD/fileserver/dashcam")


LOCAL_TZ = os.getenv("LOCAL_TZ", "America/New_York")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")  # 384 dims
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "livelongandprosper")
NEO4J_DB = os.getenv("NEO4J_DB", "neo4j")
NEO4J_ENABLED = bool(NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD)

DEFAULT_BATCH_SIZE = int(os.getenv("EMBED_BATCH", "32"))

AUDIO_BASE = Path("/mnt/8TB_2025/fileserver/audio")
PAT_TRANS_JSON_TXT = re.compile(r"_([A-Za-z0-9\-\._]+)_transcription\.txt$", re.IGNORECASE)
PAT_TRANS_CSV      = re.compile(r"_transcription\.csv$", re.IGNORECASE)
PAT_ENTITIES       = re.compile(r"_transcription_(entites|entities)\.csv$", re.IGNORECASE)
PAT_RTTM           = re.compile(r"_speakers\.rttm$", re.IGNORECASE)
PAT_MEDIA          = re.compile(r"\.(wav|mp3|m4a|flac|mp4|mov|mkv|MP4|MOV|MKV)$", re.IGNORECASE)
PAT_META_CSV       = re.compile(r"_metadata\.csv$", re.IGNORECASE)

# Model quality preference
DEFAULT_MODEL_PREF = [
    "large-v3", "large-v2", "large", "turbo",
    "medium.en", "medium",
    "small.en", "small", "base.en", "base", "tiny.en", "tiny",
    "faster-whisper:large-v3", "faster-whisper:large-v2", "faster-whisper:large",
    "faster-whisper:medium", "faster-whisper:small", "faster-whisper:base", "faster-whisper:tiny",
]
MODEL_PREF = [s.strip() for s in os.getenv("MODEL_PREF", ",".join(DEFAULT_MODEL_PREF)).split(",") if s.strip()]

# RTTM discovery
DEFAULT_RTTM_DIRS = [
    "/mnt/8TB_2025/fileserver/audio",
    "/mnt/8TB_2025/fileserver/dashcam/audio",
]
RTTM_DIRS = [p.strip() for p in os.getenv("RTTM_DIRS", ",".join(DEFAULT_RTTM_DIRS)).split(",") if p.strip()]
PAT_RTTM_FILE = re.compile(r"_speakers\.rttm$", re.IGNORECASE)

# =========================
# Stage timing helpers
# =========================
class StageStats:
    def __init__(self, name: str, alpha: float = 0.2):
        self.name = name
        self.alpha = alpha
        self.count = 0
        self.total = 0.0
        self.ema = None
    def update(self, dt: float):
        self.count += 1
        self.total += dt
        self.ema = dt if self.ema is None else self.alpha * dt + (1 - self.alpha) * self.ema
    @property
    def avg(self) -> float:
        return (self.total / self.count) if self.count else 0.0
    def summary(self) -> str:
        ema = f"{self.ema:.2f}s" if self.ema is not None else "n/a"
        avg = f"{self.avg:.2f}s"
        return f"{self.name}: n={self.count}, avg={avg}, ema={ema}, total={self.total:.2f}s"

class TimedStage:
    def __init__(self, stats: StageStats, detail: str = ""):
        self.stats = stats; self.detail = detail; self.start = 0.0; self.dt = 0.0
    def __enter__(self):
        self.start = time.perf_counter(); return self
    def __exit__(self, exc_type, exc, tb):
        self.dt = time.perf_counter() - self.start
        if exc is None:
            self.stats.update(self.dt)
            log.info(f"[stage:{self.stats.name}] {self.dt:.2f}s | {self.stats.summary()} | {self.detail}")
        else:
            log.error(f"[stage:{self.stats.name}] FAILED after {self.dt:.2f}s | {self.detail}")
        return False

st_discover  = StageStats("discover", EMA_ALPHA)
st_load      = StageStats("load", EMA_ALPHA)
st_validate  = StageStats("validate", EMA_ALPHA)
st_embed     = StageStats("embed", EMA_ALPHA)
st_ingest    = StageStats("ingest", EMA_ALPHA)
GLOBAL_START = time.perf_counter()

# =========================
# Model load (once)
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    try: torch.backends.cuda.matmul.allow_tf32 = True
    except Exception: pass
log.info(f"Loading embedding model on {DEVICE}…")
emb_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
emb_model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(DEVICE).eval()

entity_recognition_classifier = None  # lazy

# =========================
# Utils
# =========================
def _chunks(seq, size):
    size = max(1, int(size))
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

def stable_id(*parts: str) -> str:
    h = hashlib.md5()
    for p in parts:
        h.update((p or "").encode("utf-8", errors="ignore")); h.update(b"|")
    return h.hexdigest()

def _to_localized(dt: datetime) -> datetime:
    if ZoneInfo: return dt.replace(tzinfo=ZoneInfo(LOCAL_TZ))
    return dt.replace(tzinfo=timezone.utc)

def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None: dt = _to_localized(dt)
    return dt.astimezone(timezone.utc)

def iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.astimezone(timezone.utc).isoformat() if dt else None

def _parse_any_iso_or_epoch(v: Any) -> Optional[datetime]:
    if v is None: return None
    try:
        if isinstance(v, (int, float)):
            val = int(v); val = int(val/1000) if val > 1e12 else val
            return datetime.fromtimestamp(val, tz=timezone.utc)
        s = str(v).strip()
        if re.fullmatch(r"\d{10,13}", s):
            val = int(s); val = int(val/1000) if val > 1e12 else val
            return datetime.fromtimestamp(val, tz=timezone.utc)
        return _to_utc(datetime.fromisoformat(s.replace("Z","+00:00")))
    except Exception:
        return None

def parse_key_datetime_utc_from_string(s: str) -> Optional[datetime]:
    s = s.strip()
    m = re.search(r"(?P<dt14>\d{14})", s)
    if m:
        try: return _to_utc(_to_localized(datetime.strptime(m.group("dt14"), "%Y%m%d%H%M%S")))
        except Exception: pass
    for pat, fmt in [
        (r"(\d{4})_(\d{4})_(\d{6})", "%Y_%m%d_%H%M%S"),
        (r"(\d{8})_(\d{6})", "%Y%m%d_%H%M%S"),
        (r"(\d{8})(\d{6})", "%Y%m%d%H%M%S"),
        (r"(\d{4})-(\d{2})-(\d{2})[_\-](\d{2})-(\d{2})-(\d{2})", "%Y-%m-%d_%H-%M-%S"),
        (r"(\d{4})_(\d{2})_(\d{2})[_\-](\d{2})_(\d{2})_(\d{2})", "%Y_%m_%d_%H_%M_%S"),
    ]:
        m2 = re.search(pat, s)
        if m2:
            try: return _to_utc(_to_localized(datetime.strptime(m2.group(0), fmt)))
            except Exception: pass
    m = re.search(r"/(?P<Y>\d{4})/(?P<M>\d{2})/(?P<D>\d{2})/", s)
    if m:
        Y, M, D = m.group("Y"), m.group("M"), m.group("D")
        m2 = re.search(r"(?<!\d)(\d{6})(?!\d)", os.path.basename(s))
        if m2:
            try: return _to_utc(_to_localized(datetime.strptime(f"{Y}{M}{D}{m2.group(1)}", "%Y%m%d%H%M%S")))
            except Exception: pass
    return None

def canonicalize_key(name_without_suffix: str, full_path: str) -> str:
    dt = parse_key_datetime_utc_from_string(name_without_suffix) or parse_key_datetime_utc_from_string(full_path)
    if dt: return dt.astimezone(timezone.utc).strftime("%Y_%m%d_%H%M%S")
    base = re.sub(r"[^\w\-]+", "_", name_without_suffix).strip("_")
    return base or stable_id(full_path)

def file_key_from_name(name: str) -> str:
    base = os.path.basename(name)
    base = re.sub(r"\.(json|txt|csv|rttm)$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"_([A-Za-z0-9\-\._]+)_transcription(_(entites|entities))?$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"_transcription(_(entites|entities))?$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"_speakers$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"_metadata$", "", base, flags=re.IGNORECASE)
    return base

# =========================
# Embeddings
# =========================
def normalize(v: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(v, p=2, dim=1)

def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = torch.sum(masked, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts

def embed_texts(texts: List[str], batch_size: int, max_length: int = 512) -> List[List[float]]:
    if not texts: return []
    vectors = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = emb_tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            enc = {k: v.to(DEVICE) for k,v in enc.items()}
            out = emb_model(**enc)
            pooled = mean_pooling(out.last_hidden_state, enc["attention_mask"])
            pooled = normalize(pooled)
            vectors.extend(pooled.cpu().numpy().tolist())
    return vectors

def chunk_by_tokens(text: str, tokenizer, max_tokens: int = 256, overlap: int = 64) -> List[str]:
    if not text.strip(): return []
    words = text.split()
    if not words: return []
    chunks, i = [], 0
    while i < len(words):
        lo, hi = i, min(len(words), i + max_tokens * 2)
        pivot = lo
        while lo <= hi:
            mid = (lo + hi) // 2
            toks = tokenizer(" ".join(words[i:mid]), add_special_tokens=False).input_ids
            if len(toks) <= max_tokens:
                pivot = mid; lo = mid + 1
            else:
                hi = mid - 1
        if pivot == i: pivot = min(i + max_tokens, len(words))
        chunk = " ".join(words[i:pivot]).strip()
        if chunk: chunks.append(chunk)
        i = pivot - overlap if pivot - overlap > i else pivot
    return chunks

def embed_long_text_via_segments(segments: List[str], batch_size: int) -> List[float]:
    if not segments: return [0.0] * EMBED_DIM
    seg_vecs = embed_texts(segments, batch_size=batch_size)
    arr = np.array(seg_vecs, dtype=np.float32)
    vec = arr.mean(axis=0)
    n = np.linalg.norm(vec)
    if n > 0: vec = vec / n
    return vec.tolist()

def transcript_embedding_v2_from_chunks(text: str, batch_size: int, max_tokens: int = 256, overlap: int = 64) -> List[float]:
    parts = chunk_by_tokens(text, emb_tokenizer, max_tokens=max_tokens, overlap=overlap)
    if not parts: return [0.0] * EMBED_DIM
    vecs = embed_texts(parts, batch_size=batch_size, max_length=max_tokens)
    V = np.asarray(vecs, dtype=np.float32)
    weights = []
    for p in parts:
        weights.append(len(emb_tokenizer(p, add_special_tokens=False).input_ids))
    w = np.asarray(weights, dtype=np.float32); w = w / (w.sum() + 1e-6)
    out = (V * w[:, None]).sum(axis=0)
    n = np.linalg.norm(out)
    if n > 0: out = out / n
    return out.astype(np.float32).tolist()

def transcript_embedding_v2_from_segments(segments: List[Dict[str, Any]]) -> List[float]:
    vecs, weights = [], []
    for s in segments:
        v = s.get("embedding")
        if not v: continue
        dur = max(0.0, float(s.get("end", 0.0)) - float(s.get("start", 0.0)))
        vecs.append(np.asarray(v, dtype=np.float32)); weights.append(dur if np.isfinite(dur) else 0.0)
    if not vecs: return [0.0] * EMBED_DIM
    V = np.stack(vecs, axis=0); w = np.asarray(weights, dtype=np.float32)
    if not np.isfinite(w).any() or w.sum() <= 1e-6: w = np.ones_like(w, dtype=np.float32)
    w = w / (w.sum() + 1e-6); out = (V * w[:, None]).sum(axis=0)
    n = np.linalg.norm(out)
    if n > 0: out = out / n
    return out.astype(np.float32).tolist()

# =========================
# GLiNER (lazy)
# =========================
def gliner_extract_chunked(text: str, max_tokens: int = 256, overlap: int = 32) -> List[Dict[str, Any]]:
    global entity_recognition_classifier
    try:
        if entity_recognition_classifier is None:
            from gliner import GLiNER
            log.info("Loading GLiNER (chunked)…")
            entity_recognition_classifier = GLiNER.from_pretrained("urchade/gliner_large-v2", device=DEVICE)
        tok = getattr(entity_recognition_classifier, "tokenizer", None) or emb_tokenizer
        parts = chunk_by_tokens(text, tok, max_tokens=max_tokens, overlap=overlap)
        ents_all = []
        for chunk in parts:
            ents = entity_recognition_classifier.predict_entities(
                chunk, ["Person","Place","Event","Date","Subject"]
            )
            ents_all.extend(ents)
        return [{
            "text": e.get("text",""),
            "label": e.get("label",""),
            "score": float(e.get("score", 0.0) or 0.0),
            "start": float(e.get("start", -1) or -1),
            "end": float(e.get("end", -1) or -1),
        } for e in ents_all]
    except Exception as ex:
        log.warning(f"GLiNER chunked fallback failed: {ex}")
        return []

def aggregate_entities(ents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    bucket: Dict[Tuple[str,str], Dict[str, Any]] = {}
    for e in ents:
        key = (e["text"].strip(), e["label"].strip())
        b = bucket.setdefault(key, {"text": key[0], "label": key[1], "count": 0, "starts": [], "ends": [], "scores": []})
        b["count"] += 1
        if "start" in e: b["starts"].append(float(e["start"]))
        if "end" in e:   b["ends"].append(float(e["end"]))
        if "score" in e: b["scores"].append(float(e["score"]))
    out = []
    for (txt, lbl), b in bucket.items():
        eid = stable_id(txt, lbl)
        avg = float(np.mean(b["scores"])) if b["scores"] else 0.0
        out.append({"id": eid, "text": txt, "label": lbl, "count": b["count"], "starts": b["starts"], "ends": b["ends"], "avg_score": avg})
    return out

# =========================
# Speaker/utterance helpers
# =========================
def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))

def words_from_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    words = []
    for seg in segments:
        for w in seg.get("words", []) or []:
            wt = (w.get("word") or "").strip()
            ws = float(w.get("start", seg.get("start", 0.0)) or seg.get("start", 0.0))
            we = float(w.get("end", seg.get("end", ws)) or seg.get("end", ws))
            if wt:
                words.append({"text": wt, "start": ws, "end": we})
    words.sort(key=lambda x: x["start"])
    return words

def utterances_from_rttm_with_words(rttm: List[Tuple[float,float,str]], segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    words = words_from_segments(segments)
    if not words: return []
    utterances = []
    for i, (start, end, spk) in enumerate(rttm):
        mid = lambda w: (w["start"] + w["end"]) / 2.0
        chunk = [w for w in words if mid(w) >= start and mid(w) <= end]
        if not chunk: continue
        text = " ".join(w["text"] for w in chunk).strip()
        u_start = chunk[0]["start"]; u_end = chunk[-1]["end"]
        u_id = stable_id("utt", spk, f"{u_start:.3f}", f"{u_end:.3f}")
        utterances.append({
            "id": u_id, "speaker_label": spk, "start": u_start, "end": u_end,
            "text": text, "segment_id": None, "idx": i
        })
    return utterances

def utterances_from_rttm_dominant_segment(rttm: List[Tuple[float,float,str]], segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    utterances = []
    for i, seg in enumerate(segments):
        s_start = float(seg.get("start", 0.0) or 0.0)
        s_end   = float(seg.get("end", s_start) or s_start)
        best_overlap = 0.0; best_spk = None
        for (a, b, spk) in rttm:
            ov = overlap(s_start, s_end, a, b)
            if ov > best_overlap: best_overlap = ov; best_spk = spk
        spk_label = best_spk or "UNKNOWN"
        u_id = stable_id("utt", spk_label, str(seg.get("id")), f"{s_start:.3f}", f"{s_end:.3f}")
        utterances.append({
            "id": u_id, "speaker_label": spk_label, "start": s_start, "end": s_end,
            "text": seg.get("text", ""), "segment_id": seg.get("id"), "idx": i
        })
    return utterances

def _sum_overlap(seg_start: float, seg_end: float, rttm: List[Tuple[float,float,str]]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for a,b,spk in rttm:
        ov = overlap(seg_start, seg_end, a, b)
        if ov > 0: totals[spk] = totals.get(spk, 0.0) + ov
    return totals

def compute_segment_speaker_overlaps(rttm, segments, speaker_map, min_proportion: float = 0.05):
    seg_best: Dict[str, Dict[str, Any]] = {}; seg_edges: List[Dict[str, Any]] = []
    for s in segments:
        s_start = float(s.get("start", 0.0) or 0.0)
        s_end   = float(s.get("end", s_start) or s_start)
        sid     = s["id"]
        totals = _sum_overlap(s_start, s_end, rttm)
        total_ov = sum(totals.values()) or 0.0
        best_lbl = max(totals.items(), key=lambda kv: kv[1])[0] if totals else "UNKNOWN"
        sp = speaker_map.get(best_lbl)
        best_id = sp["id"] if sp else speaker_map.get("UNKNOWN", {}).get("id")
        seg_best[sid] = {"speaker_label": best_lbl, "speaker_id": best_id}
        if total_ov > 0.0:
            for lbl, ov in totals.items():
                prop = ov / total_ov
                if prop >= min_proportion:
                    spx = speaker_map.get(lbl)
                    spx_id = spx["id"] if spx else speaker_map.get("UNKNOWN", {}).get("id")
                    seg_edges.append({"segment_id": sid, "speaker_id": spx_id, "label": lbl, "overlap": float(ov), "proportion": float(prop)})
    return seg_best, seg_edges

# =========================
# Neo4j plumbing
# =========================
def neo4j_driver():
    if not NEO4J_ENABLED: return None
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

SCHEMA_QUERIES = [
    "CREATE CONSTRAINT transcription_id IF NOT EXISTS FOR (t:Transcription) REQUIRE t.id IS UNIQUE",
    "CREATE CONSTRAINT segment_id IF NOT EXISTS FOR (s:Segment) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
    "CREATE CONSTRAINT utterance_id IF NOT EXISTS FOR (u:Utterance) REQUIRE u.id IS UNIQUE",
    "CREATE CONSTRAINT speaker_id IF NOT EXISTS FOR (sp:Speaker) REQUIRE sp.id IS UNIQUE",
    "CREATE INDEX transcription_key IF NOT EXISTS FOR (t:Transcription) ON (t.key)",
    "CREATE INDEX frame_key IF NOT EXISTS FOR (f:Frame) ON (f.key)",
]

def _create_vector_index(sess, label: str, prop: str, name: str, dim: int):
    q = f"""
    CREATE VECTOR INDEX {name} IF NOT EXISTS
    FOR (n:{label}) ON (n.{prop})
    OPTIONS {{
      indexConfig: {{
        `vector.dimensions`: {dim},
        `vector.similarity_function`: 'cosine'
      }}
    }}
    """
    sess.run(q)

def ensure_schema(driver):
    if not driver:
        log.info("Neo4j not configured; skipping schema setup."); return
    with driver.session(database=NEO4J_DB) as sess:
        for q in SCHEMA_QUERIES: sess.run(q)
        try:
            _create_vector_index(sess, "Segment", "embedding", "segment_embedding_index", EMBED_DIM)
            _create_vector_index(sess, "Transcription", "embedding", "transcription_embedding_index", EMBED_DIM)
            _create_vector_index(sess, "Utterance", "embedding", "utterance_embedding_index", EMBED_DIM)
            _create_vector_index(sess, "Transcription", "embedding_v2", "transcription_embedding_v2_index", EMBED_DIM)
        except Neo4jError as e:
            msg = str(e)
            if "Invalid input 'VECTOR'" in msg or "Unrecognized command" in msg:
                raise RuntimeError("This Neo4j server does not support VECTOR indexes (need 5.11+).") from e
            raise
    log.info("Neo4j schema ensured.")

def get_ingestion_status(driver, t_id: str) -> Dict[str, Any]:
    if not driver: return {"exists": False}
    cy = """
    OPTIONAL MATCH (t:Transcription {id:$id})
    WITH t
    OPTIONAL MATCH (t)-[:HAS_SEGMENT]->(s:Segment)
    WITH t, count(s) AS seg_count, sum(CASE WHEN s.embedding IS NULL THEN 1 ELSE 0 END) AS seg_no_emb
    OPTIONAL MATCH (t)-[:HAS_UTTERANCE]->(u:Utterance)
    RETURN
      t IS NOT NULL AS exists,
      seg_count,
      seg_no_emb,
      count(u) AS utt_count,
      sum(CASE WHEN u.embedding IS NULL THEN 1 ELSE 0 END) AS utt_no_emb,
      t.started_at AS started_at,
      t.source_json AS source_json,
      t.source_csv AS source_csv,
      t.source_rttm AS source_rttm,
      t.source_media AS source_media
    """
    with driver.session(database=NEO4J_DB) as sess:
        rec = sess.run(cy, id=t_id).single()
        return rec.data() if rec else {"exists": False}

# =========================
# Loaders
# =========================
def load_transcription_json_txt(path: str) -> Optional[Dict[str, Any]]:
    with TimedStage(st_load, detail=f"json_txt={path}"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
            data = json.loads(raw)
            text = (data.get("text") or data.get("transcript") or "").strip()
            segments = data.get("segments") or []
            candidates = [data.get("started_at"), data.get("start_time"), data.get("start"), (data.get("metadata") or {}).get("started_at")]
            enders = [data.get("ended_at"), data.get("end_time"), data.get("end"), (data.get("metadata") or {}).get("ended_at")]
            file_started_at = None
            for c in candidates:
                file_started_at = _parse_any_iso_or_epoch(c)
                if file_started_at: break
            file_ended_at = None
            for e in enders:
                file_ended_at = _parse_any_iso_or_epoch(e)
                if file_ended_at: break
            return {"text": text, "segments": segments, "language": data.get("language"),
                    "file_started_at": file_started_at, "file_ended_at": file_ended_at}
        except json.JSONDecodeError as ex:
            log.warning(f"JSON parse error (expected JSON in {path}): {ex}")
            return None
        except Exception as ex:
            log.warning(f"Failed to parse JSON TXT {path}: {ex}")
            return None

def load_transcription_csv(path: str) -> Optional[Dict[str, Any]]:
    with TimedStage(st_load, detail=f"csv={path}"):
        try:
            segments, full_text = [], []
            file_start_abs: Optional[datetime] = None
            file_end_abs: Optional[datetime] = None
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row_idx, row in enumerate(reader):
                    def ffloat(k: str, default: float = 0.0) -> float:
                        try: return float(row.get(k, default) or default)
                        except Exception: return default
                    text  = (row.get("Text") or row.get("text") or "").strip()
                    start = ffloat("StartTime", ffloat("start", 0.0))
                    end   = ffloat("EndTime",   ffloat("end", start))
                    abs_start = None; abs_end = None
                    for k in ("AbsoluteStart","AbsStart","absolute_start","StartISO","start_iso","StartEpochMillis","start_epoch_ms"):
                        if row.get(k): abs_start = _parse_any_iso_or_epoch(row.get(k)); break
                    for k in ("AbsoluteEnd","AbsEnd","absolute_end","EndISO","end_iso","EndEpochMillis","end_epoch_ms"):
                        if row.get(k): abs_end = _parse_any_iso_or_epoch(row.get(k)); break
                    seg = {
                        "id": row.get("SegmentId") or stable_id(path,"seg",str(row_idx),text[:64]),
                        "seek": None, "start": start, "end": end, "text": text, "tokens": [], "words": [],
                        "abs_start": iso(abs_start) if abs_start else None, "abs_end": iso(abs_end) if abs_end else None
                    }
                    segments.append(seg)
                    if text: full_text.append(text)
                    if abs_start: file_start_abs = min(file_start_abs, abs_start) if file_start_abs else abs_start
                    if abs_end:   file_end_abs = max(file_end_abs, abs_end) if file_end_abs else abs_end
            return {"text": " ".join(full_text), "segments": segments, "language": "en",
                    "file_started_at": file_start_abs, "file_ended_at": file_end_abs}
        except Exception as ex:
            log.warning(f"Failed to parse CSV transcript {path}: {ex}"); return None

# Entities CSV (if exists)
def load_entities_csv(path: str) -> List[Dict[str, Any]]:
    ents = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                def ffloat(k: str, default: float = 0.0) -> float:
                    try: return float(row.get(k, default) or default)
                    except Exception: return default
                ents.append({
                    "text": (row.get("text") or row.get("Text") or "").strip(),
                    "label": (row.get("label") or row.get("Label") or "").strip(),
                    "score": ffloat("score", ffloat("Score", 0.0)),
                    "start": ffloat("start", ffloat("StartTime", 0.0)),
                    "end":   ffloat("end",   ffloat("EndTime", 0.0)),
                })
    except Exception as ex:
        log.warning(f"Failed to parse entities CSV {path}: {ex}")
    return ents

# RTTM
def load_rttm(path: str) -> List[Tuple[float,float,str]]:
    intervals = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"): continue
                parts = line.split()
                if len(parts) < 5 or parts[0].upper() != "SPEAKER": continue
                try:
                    start = float(parts[3]); dur = float(parts[4])
                    if dur <= 0: continue
                    end = start + dur
                except Exception:
                    continue
                spk = parts[7] if len(parts) >= 8 else parts[-1]
                spk = "UNKNOWN" if str(spk).upper() in ("UNK","UNKNOWN","<UNK>") else spk
                intervals.append((start, end, spk))
        intervals.sort(key=lambda x: x[0])
    except Exception as ex:
        log.warning(f"Failed to read RTTM {path}: {ex}")
    return intervals

# =========================
# Discovery
# =========================
def extract_model_tag_from_json_txt(p: str) -> str:
    m = PAT_TRANS_JSON_TXT.search(os.path.basename(p)); return m.group(1) if m else ""

def model_rank(tag: str) -> int:
    if not tag: return 10_000
    t = tag.lower()
    try: return MODEL_PREF.index(t)
    except ValueError:
        if "large" in t: return 100
        if "medium" in t: return 200
        if "small" in t: return 300
        if "base" in t: return 400
        if "tiny" in t: return 500
        return 9999

def is_in_audio_base(p: str) -> bool:
    try: return str(Path(p).resolve()).startswith(str(AUDIO_BASE.resolve()))
    except Exception: return False

def select_best_json(json_paths: List[str], csv_paths: List[str]) -> Optional[str]:
    if not json_paths: return None
    scored: List[Tuple[int,int,float,float,str]] = []
    for p in json_paths:
        tag = extract_model_tag_from_json_txt(p); rank = model_rank(tag)
        segs = -1.0
        try:
            with open(p, "r", encoding="utf-8") as f: j = json.load(f)
            segs = float(len(j.get("segments") or []))
        except Exception: segs = -1.0
        mtime = 0.0
        try: mtime = Path(p).stat().st_mtime
        except Exception: pass
        scored.append((0 if is_in_audio_base(p) else 1, rank, -segs, -mtime, p))
    scored.sort()
    best = scored[0][-1] if scored else None
    if best: log.info(f"[select] best JSON = {best}")
    else: log.warning("[select] no JSON candidate could be selected")
    return best

def _rttm_key_from_name(name: str) -> str:
    base = re.sub(r"\.rttm$", "", name, flags=re.IGNORECASE)
    base = re.sub(r"_speakers$", "", base, flags=re.IGNORECASE)
    return base

def index_rttm_dirs(search_dirs: List[str]) -> Dict[str, List[str]]:
    idx: Dict[str, List[str]] = {}; total = 0
    for root_dir in search_dirs:
        if not root_dir or not os.path.isdir(root_dir): continue
        for r, _, files in os.walk(root_dir):
            for name in files:
                if not PAT_RTTM_FILE.search(name): continue
                p = os.path.join(r, name); naive = _rttm_key_from_name(name)
                key = canonicalize_key(naive, p)
                idx.setdefault(key, []).append(os.path.abspath(p)); total += 1
    log.info(f"[rttm-index] indexed {total} RTTM files across {len(idx)} keys from {len(search_dirs)} dir(s)")
    return idx

def pick_rttm_for_key(key: str, idx: Dict[str, List[str]]) -> Optional[str]:
    paths = idx.get(key)
    if paths: return sorted(paths, key=lambda p: (p.count(os.sep), len(p)))[0]
    candidates = [(k, v) for k, v in idx.items() if (key in k) or (k in key)]
    if not candidates: return None
    candidates.sort(key=lambda kv: (abs(len(kv[0]) - len(key)), len(kv[0])))
    return sorted(candidates[0][1], key=lambda p: (p.count(os.sep), len(p)))[0]

def discover_keys() -> Dict[str, Dict[str, Any]]:
    with TimedStage(st_discover, detail=f"roots={len(SCAN_ROOTS)}"):
        mapping: Dict[str, Dict[str, Any]] = {}
        def add(kind: str, path: str):
            base = file_key_from_name(os.path.basename(path))
            key = canonicalize_key(base, path)
            mapping.setdefault(key, {"json_all": [], "csv_all": [], "media_all": [], "meta_all": []})
            if kind == "json": mapping[key]["json_all"].append(path)
            elif kind == "csv": mapping[key]["csv_all"].append(path)
            elif kind == "entities": mapping[key]["entities"] = path
            elif kind == "rttm": mapping[key]["rttm"] = path
            elif kind == "media": mapping[key]["media_all"].append(path)
            elif kind == "meta": mapping[key]["meta_all"].append(path)

        for root in SCAN_ROOTS:
            if not os.path.isdir(root): continue
            for r, _, files in os.walk(root):
                for name in files:
                    p = os.path.join(r, name)
                    if PAT_TRANS_JSON_TXT.search(name): add("json", p)
                    elif PAT_TRANS_CSV.search(name):    add("csv", p)
                    elif PAT_ENTITIES.search(name):     add("entities", p)
                    elif PAT_RTTM.search(name):         add("rttm", p)
                    elif PAT_MEDIA.search(name):        add("media", p)
                    elif PAT_META_CSV.search(name):     add("meta", p)

        # Dedicated pass to catch dashcam *_metadata.csv anywhere
        if os.path.isdir(DASHCAM_ROOT):
            for r, _, files in os.walk(DASHCAM_ROOT):
                for name in files:
                    if PAT_META_CSV.search(name):
                        add("meta", os.path.join(r, name))
        if os.path.isdir(OLD_DASHCAM_ROOT):
            for r, _, files in os.walk(OLD_DASHCAM_ROOT):
                for name in files:
                    if PAT_META_CSV.search(name):
                        add("meta", os.path.join(r, name))
        # RTTM attach
        rttm_idx = index_rttm_dirs(RTTM_DIRS)
        attached = 0
        for key, rec in mapping.items():
            if rec.get("rttm") and os.path.isfile(rec["rttm"]): continue
            cand = pick_rttm_for_key(key, rttm_idx)
            if cand: rec["rttm"] = cand; attached += 1
        log.info(f"[rttm-attach] attached RTTM to {attached} of {len(mapping)} keys")
        return mapping

# =========================
# Validation / cleaning
# =========================
def validate_and_clean_segments(key: str, segs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str,int]]:
    fixed = []; stats = {"nonfinite":0,"neg_dur":0,"reordered":0,"empty_txt":0,"kept":0}
    last_end = -math.inf
    for i, s in enumerate(segs):
        try:
            start = float(s.get("start", 0.0)); end = float(s.get("end", start))
            if not math.isfinite(start) or not math.isfinite(end):
                stats["nonfinite"] += 1; continue
            if end < start: end = start; stats["neg_dur"] += 1
            txt = (s.get("text") or "").strip()
            if not txt: stats["empty_txt"] += 1
            sid = s.get("id") or stable_id(key,"seg",str(i),txt[:64])
            if start < last_end: stats["reordered"] += 1
            last_end = max(last_end, end)
            fixed.append({
                "id": sid, "idx": s.get("idx", i), "start": start, "end": end, "text": txt,
                "tokens": s.get("tokens", []), "words": s.get("words", []),
                "abs_start": s.get("abs_start"), "abs_end": s.get("abs_end"),
            })
        except Exception:
            stats["nonfinite"] += 1; continue
    stats["kept"] = len(fixed); return fixed, stats

# =========================
# Dashcam metadata (robust)
# =========================
def _parse_bbox(bbox_str: str) -> Optional[Tuple[float,float,float,float]]:
    if not bbox_str: return None
    try:
        lat_min, lat_max, lon_min, lon_max = [float(x.strip()) for x in bbox_str.split(",")]
        if lat_min > lat_max: lat_min, lat_max = lat_max, lat_min
        if lon_min > lon_max: lon_min, lon_max = lon_max, lon_min
        return (lat_min, lat_max, lon_min, lon_max)
    except Exception:
        log.warning(f"[meta] Invalid --geo-bbox={bbox_str}; disabling bbox."); return None

def _in_bbox(lat: float, lon: float, bbox: Tuple[float,float,float,float]) -> bool:
    lat_min, lat_max, lon_min, lon_max = bbox
    return (lat_min <= lat <= lat_max) and (lon_min <= lon <= lon_max)

def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    from math import radians, sin, cos, atan2, sqrt
    φ1, λ1, φ2, λ2 = map(radians, [lat1, lon1, lat2, lon2])
    dφ = φ2 - φ1; dλ = λ2 - λ1
    a = sin(dφ/2)**2 + cos(φ1)*cos(φ2)*sin(dλ/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1-a))

def _norm_lat_lon(lat, lon) -> Tuple[Optional[float], Optional[float]]:
    try:
        lat = float(lat); lon = float(lon)
    except Exception:
        return None, None
    if not (-90.0 <= lat <= 90.0):  lat = None
    if not (-180.0 <= lon <= 180.0): lon = None
    return lat, lon

def _repair_latlon(lat: Optional[float], lon: Optional[float], *, bbox, lon_auto_west: bool, allow_swap: bool):
    flags = []
    if lat is None or lon is None: return lat, lon, flags
    if lon_auto_west and lon is not None and lon > 0:
        flipped = -abs(lon)
        if bbox:
            if _in_bbox(lat, flipped, bbox): lon = flipped; flags.append("FIXED_WEST")
        else:
            if 20.0 <= lon <= 170.0: lon = flipped; flags.append("FIXED_WEST")
    if allow_swap and bbox and not _in_bbox(lat, lon, bbox):
        if _in_bbox(lon, lat, bbox): lat, lon = lon, lat; flags.append("SWAPPED")
    return lat, lon, flags

def parse_dashcam_metadata_csv(path: str, fps: float, downsample_sec: int,
                               lon_auto_west: bool, allow_swap: bool, bbox_str: str,
                               max_speed_mph: float, max_rows: int = 0):
    bbox = _parse_bbox(bbox_str)
    seen = 0; per_sec_points: Dict[int, List[Tuple[float,float,Optional[float],List[str]]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if max_rows and seen >= max_rows: break
            seen += 1
            try: frame = int(str(row.get("Frame","")).strip())
            except Exception: continue
            if frame < 0: continue
            sec_raw = int(math.floor(frame / max(1e-9, fps)))
            sec = (sec_raw // max(1, downsample_sec)) * max(1, downsample_sec)
            mph = None; mph_raw = str(row.get("MPH","")).strip()
            if mph_raw:
                try: mph = float(int(mph_raw)) if mph_raw.isdigit() else float(mph_raw)
                except Exception: mph = None
            lat_raw = row.get("Lat"); lon_raw = row.get("Long")
            lat, lon = _norm_lat_lon(lat_raw, lon_raw)
            if lat is None or lon is None: continue
            lat, lon, rep_flags = _repair_latlon(lat, lon, bbox=bbox, lon_auto_west=lon_auto_west, allow_swap=allow_swap)
            if lat is None or lon is None: continue
            flags = list(rep_flags)
            if bbox and not _in_bbox(lat, lon, bbox): flags.append("OUT_OF_BOUNDS")
            per_sec_points.setdefault(sec, []).append((lat, lon, mph, flags))
    from statistics import median
    aggregated = []
    for sec, pts in per_sec_points.items():
        if not pts: continue
        lats = [p[0] for p in pts]; lons = [p[1] for p in pts]
        mphs = [p[2] for p in pts if p[2] is not None]
        flags = list(itertools.chain.from_iterable(p[3] for p in pts))
        lat_m = median(lats); lon_m = median(lons); mph_m = median(mphs) if mphs else None
        frame_max = max(int(sec * fps), 0)
        aggregated.append({"sec": int(sec), "frame": frame_max, "mph": mph_m, "lat": float(lat_m), "lon": float(lon_m), "flags": flags})
    aggregated.sort(key=lambda r: r["sec"])
    kept = []; prev = None; max_v_mps = max_speed_mph * 0.44704
    for r in aggregated:
        quality = 100
        if prev is not None:
            dt = r["sec"] - prev["sec"]
            if dt <= 0: r["flags"].append("NON_MONOTONIC_TIME"); quality -= 20
            else:
                d_m = _haversine_m(prev["lat"], prev["lon"], r["lat"], r["lon"])
                v = d_m / dt
                if v > max_v_mps: r["flags"].append("TOO_FAST"); quality -= 60
        if bbox and not _in_bbox(r["lat"], r["lon"], bbox): quality -= 20
        r["quality"] = max(0, min(100, quality))
        if r["quality"] >= 40 and "TOO_FAST" not in r["flags"]:
            kept.append(r); prev = r
    good_ratio = (len(kept) / max(1, len(aggregated))) if aggregated else 0.0
    stats = {"file": path, "seen": seen, "kept": len(kept), "good_ratio": good_ratio}
    return kept, stats

# =========================
# Chunked ingestion
# =========================
def ingest_transcription_header(driver, t_id, key, started_at_iso, ended_at_iso, source_paths, text, transcript_emb, transcript_emb_v2):
    cy = """
    MERGE (t:Transcription {id: $t_id})
    ON CREATE SET t.key=$key, t.created_at=datetime()
    ON MATCH  SET t.key=$key, t.updated_at=datetime()
    SET t.text=$text,
        t.embedding=$transcript_emb,
        t.embedding_v2=CASE WHEN $transcript_emb_v2 IS NULL THEN t.embedding_v2 ELSE $transcript_emb_v2 END,
        t.source_json=$source_json,
        t.source_csv=$source_csv,
        t.source_rttm=$source_rttm,
        t.source_media=$source_media,
        t.started_at=CASE WHEN $started_at IS NULL THEN t.started_at ELSE datetime($started_at) END,
        t.ended_at  =CASE WHEN $ended_at   IS NULL THEN t.ended_at   ELSE datetime($ended_at)   END
    """
    with driver.session(database=NEO4J_DB) as sess:
        sess.run(cy, t_id=t_id, key=key, text=text, transcript_emb=transcript_emb,
                 transcript_emb_v2=transcript_emb_v2, source_json=source_paths.get("json"),
                 source_csv=source_paths.get("csv"), source_rttm=source_paths.get("rttm"),
                 source_media=source_paths.get("media"),
                 started_at=started_at_iso, ended_at=ended_at_iso)

def ingest_segments_chunked(driver, t_id, segments, batch=200, timeout=120):
    if not segments: return 0
    cy = """
    MATCH (t:Transcription {id:$t_id})
    WITH t
    UNWIND $segments AS seg
    MERGE (s:Segment {id: seg.id})
      ON CREATE SET s.idx=seg.idx, s.start=seg.start, s.end=seg.end, s.created_at=datetime()
      ON MATCH  SET s.idx=seg.idx, s.start=seg.start, s.end=seg.end, s.updated_at=datetime()
    SET s.text=seg.text,
        s.tokens_count=seg.tokens_count,
        s.embedding=seg.embedding,
        s.absolute_start=CASE WHEN seg.abs_start IS NULL THEN s.absolute_start ELSE datetime(seg.abs_start) END,
        s.absolute_end  =CASE WHEN seg.abs_end   IS NULL THEN s.absolute_end   ELSE datetime(seg.abs_end)   END,
        s.speaker_label=seg.speaker_label,
        s.speaker_id=seg.speaker_id
    MERGE (t)-[:HAS_SEGMENT]->(s)
    """
    written = 0
    with driver.session(database=NEO4J_DB) as sess:
        for batch_list in _chunks(segments, batch):
            sess.run(cy, t_id=t_id, segments=batch_list, timeout=timeout)
            written += len(batch_list)
    return written

def ingest_speakers_chunked(driver, speakers, batch=200, timeout=120):
    if not speakers: return 0
    cy = """
    UNWIND $speakers AS sp
    MERGE (spk:Speaker {id: sp.id})
      ON CREATE SET spk.label=sp.label, spk.key=sp.key, spk.created_at=datetime()
      ON MATCH  SET spk.label=sp.label, spk.key=sp.key, spk.updated_at=datetime()
    """
    written = 0
    with driver.session(database=NEO4J_DB) as sess:
        for batch_list in _chunks(speakers, batch):
            sess.run(cy, speakers=batch_list, timeout=timeout)
            written += len(batch_list)
    return written

def ingest_utterances_chunked(driver, t_id, utterances, batch=200, timeout=120):
    if not utterances: return 0
    cy = """
    MATCH (t:Transcription {id:$t_id})
    WITH t
    UNWIND $utterances AS u
    MERGE (uNode:Utterance {id: u.id})
      ON CREATE SET uNode.idx=u.idx, uNode.created_at=datetime()
      ON MATCH  SET uNode.idx=u.idx, uNode.updated_at=datetime()
    SET uNode.start=u.start, uNode.end=u.end, uNode.text=u.text, uNode.embedding=u.embedding,
        uNode.absolute_start=CASE WHEN u.abs_start IS NULL THEN uNode.absolute_start ELSE datetime(u.abs_start) END,
        uNode.absolute_end  =CASE WHEN u.abs_end   IS NULL THEN uNode.absolute_end   ELSE datetime(u.abs_end) END
    MERGE (t)-[:HAS_UTTERANCE]->(uNode)
    """
    written = 0
    with driver.session(database=NEO4J_DB) as sess:
        for batch_list in _chunks(utterances, batch):
            sess.run(cy, t_id=t_id, utterances=batch_list, timeout=timeout)
            written += len(batch_list)
    return written

def link_utterances_to_segments(driver, utterances, batch=300, timeout=120):
    # requires u.segment_id present
    pairs = [{"uid": u["id"], "sid": u.get("segment_id")} for u in utterances if u.get("segment_id")]
    if not pairs: return 0
    cy = """
    UNWIND $pairs AS p
    MATCH (u:Utterance {id:p.uid})
    MATCH (s:Segment {id:p.sid})
    MERGE (u)-[:OF_SEGMENT]->(s)
    """
    written = 0
    with driver.session(database=NEO4J_DB) as sess:
        for batch_list in _chunks(pairs, batch):
            sess.run(cy, pairs=batch_list, timeout=timeout)
            written += len(batch_list)
    return written

def link_utterances_to_speakers(driver, utterances, batch=300, timeout=120):
    pairs = [{"uid": u["id"], "spid": u.get("speaker_id")} for u in utterances if u.get("speaker_id")]
    if not pairs: return 0
    cy = """
    UNWIND $pairs AS p
    MATCH (u:Utterance {id:p.uid})
    MATCH (spk:Speaker {id:p.spid})
    MERGE (u)-[:SPOKEN_BY]->(spk)
    """
    written = 0
    with driver.session(database=NEO4J_DB) as sess:
        for batch_list in _chunks(pairs, batch):
            sess.run(cy, pairs=batch_list, timeout=timeout)
            written += len(batch_list)
    return written

def ingest_segment_speaker_edges(driver, seg_edges, batch=500, timeout=120):
    if not seg_edges: return 0
    cy = """
    UNWIND $rows AS ss
    MATCH (s:Segment {id:ss.segment_id})
    MATCH (spk:Speaker {id:ss.speaker_id})
    MERGE (s)-[r:SPOKEN_BY]->(spk)
    SET r.overlap=coalesce(ss.overlap,0.0),
        r.proportion=coalesce(ss.proportion,0.0)
    """
    written = 0
    with driver.session(database=NEO4J_DB) as sess:
        for batch_list in _chunks(seg_edges, batch):
            sess.run(cy, rows=batch_list, timeout=timeout)
            written += len(batch_list)
    return written

def ingest_entities_chunked(driver, t_id, entities, batch=300, timeout=120):
    if not entities: return 0
    cy = """
    MATCH (t:Transcription {id:$t_id})
    WITH t
    UNWIND $entities AS ent
      MERGE (e:Entity {id: ent.id})
        ON CREATE SET e.text=ent.text, e.label=ent.label, e.created_at=datetime()
        ON MATCH  SET e.text=ent.text, e.label=ent.label, e.updated_at=datetime()
      MERGE (t)-[m:MENTIONS]->(e)
      SET m.count=ent.count, m.starts=ent.starts, m.ends=ent.ends, m.avg_score=ent.avg_score
    """
    written = 0
    with driver.session(database=NEO4J_DB) as sess:
        for batch_list in _chunks(entities, batch):
            sess.run(cy, t_id=t_id, entities=batch_list, timeout=timeout)
            written += len(batch_list)
    return written

def ingest_location_samples_chunked(driver, t_id: str, key: str, started_at_iso: Optional[str],
                                    rows: List[Dict[str, Any]], batch: int = 500, timeout: int = 120):
    if not rows: return 0
    t0 = None
    if started_at_iso:
        try: t0 = datetime.fromisoformat(started_at_iso.replace("Z","+00:00")).astimezone(timezone.utc)
        except Exception: t0 = None
    prepared = []
    for i, r in enumerate(rows):
        abs_iso = None
        if t0 is not None:
            try: abs_iso = (t0 + timedelta(seconds=int(r["sec"]))).isoformat()
            except Exception: abs_iso = None
        rid = stable_id("loc", key, str(r["sec"]), f"{r.get('lat')}", f"{r.get('lon')}")
        prepared.append({
            "id": rid, "key": key, "idx": i, "sec": int(r["sec"]), "frame": int(r["frame"]),
            "mph": float(r["mph"]) if r.get("mph") is not None else None,
            "lat": float(r["lat"]), "lon": float(r["lon"]), "abs_time": abs_iso,
            "flags": list(r.get("flags") or []), "quality": int(r.get("quality", 100)),
        })
    cy = """
    MATCH (t:Transcription {id: $t_id})
    WITH t
    UNWIND $rows AS r
    MERGE (loc:LocationSample {id: r.id})
      ON CREATE SET loc.created_at = datetime()
      ON MATCH  SET loc.updated_at = datetime()
    SET loc.key = r.key,
        loc.idx = r.idx,
        loc.second = r.sec,
        loc.frame = r.frame,
        loc.mph = r.mph,
        loc.lat = r.lat,
        loc.lon = r.lon,
        loc.flags = r.flags,
        loc.quality = r.quality,
        loc.timestamp = CASE WHEN r.abs_time IS NULL THEN loc.timestamp ELSE datetime(r.abs_time) END
    MERGE (t)-[:HAS_LOCATION]->(loc)
    """
    written = 0
    with driver.session(database=NEO4J_DB) as sess:
        for batch_list in _chunks(prepared, batch):
            sess.run(cy, t_id=t_id, rows=batch_list, timeout=timeout)
            written += len(batch_list)
    return written

# =========================
# Processing one key
# =========================
def process_key(key: str, paths: Dict[str, Any], driver,
                batch_size: int, dry_run: bool = False, emb_v2: bool = True,
                tx_sizes: Dict[str,int]=None, tx_timeout_seconds: int=120, fetch_size: int=100,
                meta_opts: Dict[str, Any] = None):
    tx_sizes = tx_sizes or {}
    # Select best JSON; fallback CSV
    json_paths: List[str] = paths.get("json_all") or []
    csv_paths: List[str] = paths.get("csv_all") or []
    best_json = select_best_json(json_paths, csv_paths)
    data = None
    src_dir_for_sidecar = None

    if best_json:
        data = load_transcription_json_txt(best_json)
        src_dir_for_sidecar = os.path.dirname(best_json)
    if data is None and csv_paths:
        csv_sorted = sorted(csv_paths, key=lambda p: (0 if is_in_audio_base(p) else 1, -Path(p).stat().st_mtime if Path(p).exists() else 0))
        best_csv = None
        for c in csv_sorted:
            data = load_transcription_csv(c)
            if data:
                src_dir_for_sidecar = os.path.dirname(c); best_csv = c; break
    else:
        best_csv = None
        if best_json:
            stem = os.path.splitext(best_json)[0]
            candidate_csv = stem.rsplit("_", 1)[0] + "_transcription.csv"
            if os.path.isfile(candidate_csv):
                best_csv = candidate_csv
            else:
                dirp = Path(best_json).parent
                choices = list(dirp.glob(f"{Path(best_json).stem.rsplit('_',1)[0]}*_transcription.csv"))
                if choices: best_csv = str(choices[0])

    if data is None:
        log.warning(f"[{key}] No transcription data found; skipping."); return None

    text = data.get("text","") or ""
    segments_raw = data.get("segments", []) or []

    # Validate & clean segments
    with TimedStage(st_validate, detail=f"key={key}"):
        cleaned_segments, vstats = validate_and_clean_segments(key, segments_raw)
        log.info(f"[validate] key={key} segs_in={len(segments_raw)} segs_kept={vstats['kept']} "
                 f"nonfinite={vstats['nonfinite']} neg_dur={vstats['neg_dur']} empty_txt={vstats['empty_txt']} reordered={vstats['reordered']}")

    # Prepare enriched segments (embedding later)
    seg_texts = [s["text"] for s in cleaned_segments]
    enriched_segments = []
    max_end = 0.0
    for idx, seg in enumerate(cleaned_segments):
        s_tok_count = len(seg.get("tokens", [])) if isinstance(seg.get("tokens", []), (list,tuple)) else int(seg.get("tokens") or 0)
        enriched_segments.append({
            "id": seg["id"], "idx": idx, "start": seg["start"], "end": seg["end"], "text": seg["text"],
            "tokens_count": s_tok_count, "embedding": None,
            "abs_start": seg.get("abs_start") or None, "abs_end": seg.get("abs_end") or None,
        })
        max_end = max(max_end, seg["end"])

    file_start_dt = data.get("file_started_at"); file_end_dt = data.get("file_ended_at")
    dt0_utc = file_start_dt.astimezone(timezone.utc) if isinstance(file_start_dt, datetime) else parse_key_datetime_utc_from_string(key)
    started_at_iso = iso(dt0_utc) if dt0_utc else None
    ended_at_iso = iso(file_end_dt) if isinstance(file_end_dt, datetime) else (iso(dt0_utc + timedelta(seconds=max_end)) if dt0_utc else None)
    if dt0_utc:
        for s in enriched_segments:
            if not s["abs_start"]: s["abs_start"] = iso(dt0_utc + timedelta(seconds=float(s["start"])))
            if not s["abs_end"]:   s["abs_end"]   = iso(dt0_utc + timedelta(seconds=float(s["end"])))

    # Embeddings
    with TimedStage(st_embed, detail=f"key={key} type=transcript+segments"):
        if seg_texts:
            seg_vecs = embed_texts(seg_texts, batch_size=batch_size)
            for i,v in enumerate(seg_vecs): enriched_segments[i]["embedding"] = v
            transcript_vec = embed_long_text_via_segments(seg_texts, batch_size=batch_size)
            transcript_vec_v2 = transcript_embedding_v2_from_segments(enriched_segments) if emb_v2 else None
        else:
            transcript_vec = embed_texts([text], batch_size=batch_size)[0]
            transcript_vec_v2 = None
    if emb_v2 and transcript_vec_v2 is None:
        transcript_vec_v2 = transcript_embedding_v2_from_chunks(text, batch_size=batch_size, max_tokens=256, overlap=64)

    # Entities
    ents_raw = []
    entities_path = paths.get("entities")
    if entities_path and os.path.isfile(entities_path):
        ents_raw = load_entities_csv(entities_path)
    elif text.strip():
        ents_raw = gliner_extract_chunked(text, max_tokens=256, overlap=32)
    entities = aggregate_entities(ents_raw) if ents_raw else []

    # Speakers/utterances
    utterances = []; speakers = []; speaker_map: Dict[str, Dict[str,str]] = {}; seg_edges: List[Dict[str, Any]] = []
    if paths.get("rttm"): log.info(f"[{key}] RTTM => {paths['rttm']}")
    else: log.warning(f"[{key}] RTTM missing")
    if "rttm" in paths and paths["rttm"] and os.path.isfile(paths["rttm"]):
        rttm = load_rttm(paths["rttm"])
        labels = sorted(set([spk for _,_,spk in rttm]))
        for lbl in labels:
            sp_id = stable_id(key,"spk",lbl); speakers.append({"id": sp_id, "label": lbl, "key": key})
            speaker_map[lbl] = {"id": sp_id, "label": lbl}
        if "UNKNOWN" not in speaker_map:
            unknown_id = stable_id(key,"spk","UNKNOWN")
            speakers.append({"id": unknown_id, "label": "UNKNOWN", "key": key})
            speaker_map["UNKNOWN"] = {"id": unknown_id, "label": "UNKNOWN"}
        seg_best, seg_edges = compute_segment_speaker_overlaps(rttm, enriched_segments, speaker_map, min_proportion=0.05)
        for s in enriched_segments:
            best = seg_best.get(s["id"], {})
            s["speaker_label"] = best.get("speaker_label","UNKNOWN")
            s["speaker_id"] = best.get("speaker_id", speaker_map["UNKNOWN"]["id"])
        utts = utterances_from_rttm_with_words(rttm, cleaned_segments)
        if not utts: utts = utterances_from_rttm_dominant_segment(rttm, enriched_segments)
        for u in utts:
            best, best_id = 0.0, None
            for s in enriched_segments:
                ov = overlap(u["start"], u["end"], s["start"], s["end"])
                if ov > best: best = ov; best_id = s["id"]
            u["segment_id"] = best_id
        for u in utts:
            sp_map = speaker_map.get(u.get("speaker_label","UNKNOWN")) or speaker_map["UNKNOWN"]
            u["speaker_id"] = sp_map["id"]
        utt_texts = [u["text"] for u in utts]
        if utt_texts:
            utt_vecs = embed_texts(utt_texts, batch_size=batch_size)
            for i,v in enumerate(utt_vecs): utts[i]["embedding"] = v
        if dt0_utc:
            for u in utts:
                u["abs_start"] = iso(dt0_utc + timedelta(seconds=float(u["start"])))
                u["abs_end"]   = iso(dt0_utc + timedelta(seconds=float(u["end"])))
        utterances = utts

    # Sidecar subjects
    if entities and src_dir_for_sidecar:
        subj_model = "best"
        subjects_path = os.path.join(src_dir_for_sidecar, f"{key}_{subj_model}_transcription_subjects.json")
        try:
            with open(subjects_path, "w", encoding="utf-8") as f:
                json.dump({"entities": entities}, f, ensure_ascii=False, indent=2)
            log.info(f"[{key}] Subjects saved -> {subjects_path}")
        except Exception as ex:
            log.warning(f"[{key}] Subjects save failed: {ex}")

    t_id = stable_id(key,"transcription")
    source_paths = {"json": best_json, "csv": best_csv, "rttm": paths.get("rttm"), "media": (paths.get("media_best") or (paths.get("media_all") or [None])[0])}

    if dry_run:
        log.info(f"[DRY-RUN] key={key} segs={len(enriched_segments)} utts={len(utterances)} ents={len(entities)} speakers={len(speakers)}")
        return {"key": key, "segments": len(enriched_segments), "utterances": len(utterances),
                "speakers": len(speakers), "entities": len(entities)}

    # --- CHUNKED INGEST ---
    with TimedStage(st_ingest, detail=f"key={key}"):
        ingest_transcription_header(driver, t_id, key, started_at_iso, ended_at_iso, source_paths, text, transcript_vec, transcript_vec_v2)
        ingest_speakers_chunked(driver, speakers, batch=tx_sizes.get("spk", 200), timeout=tx_timeout_seconds)
        ingest_segments_chunked(driver, t_id, enriched_segments, batch=tx_sizes.get("seg", 200), timeout=tx_timeout_seconds)
        ingest_utterances_chunked(driver, t_id, utterances, batch=tx_sizes.get("utt", 200), timeout=tx_timeout_seconds)
        link_utterances_to_segments(driver, utterances, batch=tx_sizes.get("utt_seg", 300), timeout=tx_timeout_seconds)
        link_utterances_to_speakers(driver, utterances, batch=tx_sizes.get("utt_spk", 300), timeout=tx_timeout_seconds)
        ingest_segment_speaker_edges(driver, seg_edges, batch=tx_sizes.get("seg_spk", 500), timeout=tx_timeout_seconds)
        ingest_entities_chunked(driver, t_id, entities, batch=tx_sizes.get("ent", 300), timeout=tx_timeout_seconds)

    # --- OPTIONAL: robust dashcam metadata ingestion ---
    meta_written = 0
    if (meta_opts or {}).get("enabled") and (paths.get("meta_all")):
        fps = float(meta_opts.get("fps", 30.0))
        dsec = int(meta_opts.get("downsample_sec", 1))
        max_rows = int(meta_opts.get("max_rows", 0))
        lon_auto_west = bool(meta_opts.get("lon_auto_west", False))
        allow_swap = bool(meta_opts.get("allow_swap", False))
        bbox_str = str(meta_opts.get("bbox") or "")
        max_speed_mph = float(meta_opts.get("max_speed_mph", 120.0))
        min_keep_ratio = float(meta_opts.get("min_keep_ratio", 0.5))
        skip_when_bad = bool(meta_opts.get("skip_when_bad", False))
        meta_files = [p for p in paths.get("meta_all") or [] if p.lower().endswith("_metadata.csv")]
        for mp in meta_files:
            try:
                rows, stats = parse_dashcam_metadata_csv(
                    mp, fps=fps, downsample_sec=dsec,
                    lon_auto_west=lon_auto_west, allow_swap=allow_swap,
                    bbox_str=bbox_str, max_speed_mph=max_speed_mph, max_rows=max_rows
                )
                if stats["seen"] == 0:
                    log.warning(f"[{key}] metadata had 0 readable rows: {mp}")
                    continue
                if stats["good_ratio"] < min_keep_ratio:
                    msg = (f"[{key}] metadata quality low: kept={stats['kept']}/{stats['seen']} "
                           f"({stats['good_ratio']:.2%}) min={min_keep_ratio:.2%} file={mp}")
                    if skip_when_bad:
                        log.warning(msg + " -> SKIP ingest")
                        continue
                    else:
                        log.warning(msg + " -> INGEST with flags/quality")
                if rows:
                    w = ingest_location_samples_chunked(driver, t_id, key, started_at_iso, rows, batch=tx_sizes.get("loc", 500), timeout=tx_timeout_seconds)
                    meta_written += w
                    log.info(f"[{key}] metadata ingested: file={mp} kept={len(rows)} written={w} (good_ratio={stats['good_ratio']:.2%})")
                else:
                    log.warning(f"[{key}] metadata parse yielded 0 kept rows: {mp}")
            except Exception as ex:
                log.warning(f"[{key}] metadata parse failed: {mp} :: {ex}")
        if meta_written:
            log.info(f"[{key}] total LocationSample written: {meta_written}")

    return {
        "key": key, "segments": len(enriched_segments), "utterances": len(utterances),
        "speakers": len(speakers), "entities": len(entities), "loc_samples": meta_written,
    }

# =========================
# Helpers for re-ingest decision
# =========================
def probe_expected(paths: Dict[str, Any]) -> Tuple[int, bool]:
    expected_segments = 0
    data = None
    best_json = select_best_json(paths.get("json_all") or [], paths.get("csv_all") or [])
    if best_json and os.path.isfile(best_json): data = load_transcription_json_txt(best_json)
    if data is None:
        for c in (paths.get("csv_all") or []):
            if os.path.isfile(c):
                data = load_transcription_csv(c)
                if data: break
    if data is not None: expected_segments = len(data.get("segments") or [])
    has_rttm = bool(paths.get("rttm") and os.path.isfile(paths["rttm"]))
    return expected_segments, has_rttm

def should_reingest(status: Dict[str, Any], expected_segments: int, has_rttm: bool, paths: Dict[str, Any]) -> bool:
    if not status.get("exists"): return True
    for k in ("json","csv","rttm","media"):
        if paths.get(k) and not status.get(f"source_{k}"): return True
    if status.get("started_at") is None: return True
    seg_count = int(status.get("seg_count") or 0)
    seg_no_emb = int(status.get("seg_no_emb") or 0)
    utt_count = int(status.get("utt_count") or 0)
    if expected_segments > 0 and seg_count == 0: return True
    if has_rttm and utt_count == 0: return True
    if seg_count > 0 and seg_no_emb > max(1, seg_count // 2): return True
    return False

# =========================
# CLI
# =========================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ingest transcripts with batching, RTTM discovery, robust metadata, and logging.")
    # Search stub (not implemented here)
    parser.add_argument("--search", type=str, default=None)
    parser.add_argument("--target", type=str, default="utterance", choices=["utterance","segment","transcription"])
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--win-mins", type=int, default=10)
    parser.add_argument("--include-emb", action="store_true")

    # Ingest controls
    parser.add_argument("--limit", type=int, default=None, help="Process at most N keys.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Embedding batch size.")
    parser.add_argument("--force", action="store_true", help="Force re-ingest even if transcription already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be ingested (no writes).")

    parser.add_argument("--transcript-emb-v2", dest="emb_v2", action="store_true", default=True)
    parser.add_argument("--no-transcript-emb-v2", dest="emb_v2", action="store_false")

    # Neo4j tx tuning
    parser.add_argument("--tx-seg-batch", type=int, default=int(os.getenv("TX_SEG_BATCH","200")))
    parser.add_argument("--tx-utt-batch", type=int, default=int(os.getenv("TX_UTT_BATCH","200")))
    parser.add_argument("--tx-edge-batch", type=int, default=int(os.getenv("TX_EDGE_BATCH","300")))
    parser.add_argument("--tx-ent-batch", type=int, default=int(os.getenv("TX_ENT_BATCH","300")))
    parser.add_argument("--tx-loc-batch", type=int, default=int(os.getenv("TX_LOC_BATCH","500")))
    parser.add_argument("--tx-timeout-sec", type=int, default=int(os.getenv("TX_TIMEOUT_SEC","120")))
    parser.add_argument("--fetch-size", type=int, default=int(os.getenv("FETCH_SIZE","100")))
    parser.add_argument("--log-level", type=str, default=LOG_LEVEL_DEFAULT, help="DEBUG/INFO/WARNING/ERROR")

    # Dashcam metadata ingestion
    parser.add_argument("--ingest-dashcam-meta", action="store_true",
                        help="Parse and ingest dashcam *_metadata.csv into LocationSample nodes.")
    parser.add_argument("--meta-fps", type=float, default=float(os.getenv("META_FPS","30")))
    parser.add_argument("--meta-downsample-sec", type=int, default=int(os.getenv("META_DOWNSAMPLE_SEC","1")))
    parser.add_argument("--meta-max-rows", type=int, default=int(os.getenv("META_MAX_ROWS","0")))

    # Data quality controls
    parser.add_argument("--geo-bbox", type=str, default=os.getenv("GEO_BBOX",""),
                        help="Bounding box 'lat_min,lat_max,lon_min,lon_max' (e.g. '34.0,36.5,-83.0,-77.0').")
    parser.add_argument("--meta-max-speed-mph", type=float, default=float(os.getenv("META_MAX_SPEED_MPH","120")))
    parser.add_argument("--lon-auto-west", action="store_true")
    parser.add_argument("--allow-latlon-swap", action="store_true")
    parser.add_argument("--meta-min-keep-ratio", type=float, default=float(os.getenv("META_MIN_KEEP_RATIO","0.5")))
    parser.add_argument("--meta-skip-when-bad", action="store_true")

    args = parser.parse_args()
    log.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    driver = neo4j_driver()
    ensure_schema(driver)

    # Search mode placeholder
    if args.search:
        log.error("Vector search body not included in v5.3.")
        return

    mapping = discover_keys()
    keys = sorted(mapping.keys())
    if args.limit: keys = keys[:args.limit]
    log.info(f"Found {len(keys)} key(s) to consider for ingest/re-ingest.")

    processed = 0; skipped = 0
    tx_sizes = {
        "seg": args.tx_seg_batch, "utt": args.tx_utt_batch, "utt_seg": args.tx_edge_batch,
        "utt_spk": args.tx_edge_batch, "seg_spk": args.tx_edge_batch, "ent": args.tx_ent_batch, "loc": args.tx_loc_batch
    }
    meta_opts = {
        "enabled": args.ingest_dashcam_meta, "fps": args.meta_fps, "downsample_sec": args.meta_downsample_sec,
        "max_rows": args.meta_max_rows, "lon_auto_west": args.lon_auto_west, "allow_swap": args.allow_latlon_swap,
        "bbox": args.geo_bbox, "max_speed_mph": args.meta_max_speed_mph,
        "min_keep_ratio": args.meta_min_keep_ratio, "skip_when_bad": args.meta_skip_when_bad,
    }

    for i, key in enumerate(keys, 1):
        paths = mapping[key]
        # Choose a best media for convenience
        media_all = paths.get("media_all") or []
        media_best = None
        if media_all:
            # Prefer MP4 then MP3 in same dir depth, shortest path
            ranked = sorted(media_all, key=lambda p: (0 if str(p).lower().endswith(".mp4") else 1, p.count(os.sep), len(p)))
            media_best = ranked[0]
        paths["media_best"] = media_best

        log.info("===== BEGIN INGEST =====\n"
                 f"  idx={i}/{len(keys)}\n"
                 f"  key={key}\n"
                 f"  json_candidates={len(paths.get('json_all') or [])}\n"
                 f"  csv_candidates={len(paths.get('csv_all') or [])}\n"
                 f"  media_all={len(paths.get('media_all') or [])}\n"
                 f"  meta_all={len(paths.get('meta_all') or [])}\n"
                 f"  rttm={paths.get('rttm')}\n"
                 f"  media_best={paths.get('media_best')}")

        try:
            t_id = stable_id(key,"transcription")
            if not args.force:
                status = get_ingestion_status(driver, t_id)
                expected_segments, has_rttm = probe_expected(paths)
                do_ingest = should_reingest(status, expected_segments, has_rttm, paths)
                if not do_ingest:
                    skipped += 1
                    log.info(f"[{i}/{len(keys)}] SKIP (up-to-date): {key}")
                    log.info("===== END INGEST ===== key=%s | %s | %s | %s | %s",
                             key, st_discover.summary(), st_load.summary(), st_validate.summary(), st_ingest.summary())
                    continue

            info = process_key(
                key, paths, driver,
                batch_size=args.batch_size, dry_run=args.dry_run, emb_v2=args.emb_v2,
                tx_sizes=tx_sizes, tx_timeout_seconds=args.tx_timeout_sec, fetch_size=args.fetch_size,
                meta_opts=meta_opts
            )
            if info:
                processed += 1
                log.info(f"[{i}/{len(keys)}] OK: {key} "
                         f"(segments={info['segments']}, utterances={info['utterances']}, "
                         f"speakers={info['speakers']}, entities={info['entities']}, "
                         f"loc_samples={info.get('loc_samples', 0)})")
            else:
                log.warning(f"[{i}/{len(keys)}] SKIP: {key} (no data)")
        except Exception as ex:
            log.exception(f"[{i}/{len(keys)}] FAILED: {key} :: {ex}")
        finally:
            log.info("===== END INGEST ===== key=%s | %s | %s | %s | %s",
                     key, st_discover.summary(), st_load.summary(), st_validate.summary(), st_ingest.summary())

    total_elapsed = time.perf_counter() - GLOBAL_START
    log.info("ALL DONE | elapsed=%.2fs | processed=%d skipped=%d total=%d | %s | %s | %s | %s",
             total_elapsed, processed, skipped, len(keys),
             st_discover.summary(), st_load.summary(), st_validate.summary(), st_ingest.summary())

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        log.exception("Fatal error: %s", e)
        print("❌ Failed: ingest_transcriptsv5_3.py", flush=True)
        raise
