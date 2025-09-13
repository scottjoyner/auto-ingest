#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ingest transcripts into Neo4j with embeddings, GLiNER subjects (with fan-out),
speaker/utterance mapping, and optional graph-pack exports for LOAD CSV.

Key Features
------------
- Best-transcript selection (prefers AUDIO_BASE + higher-quality model tags)
- Robust loaders (JSON-in-.txt, CSV/TSV fallback), RTTM → Utterances
- Embeddings (MiniLM default), vector-ready nodes
- GLiNER-based Subjects at Transcription level + fan-out to Segments/Utterances
  using prototype-embedding similarity + keyword signals
- Optional Graph Pack CSV emit for bulk loading (no DB required)
- Idempotent Neo4j writes with constraints and vector indexes
- Enhanced logging & stage timing for validation and debugging

Usage examples
--------------
./.venv/bin/python3 ingest_transcripts.py
./.venv/bin/python3 ingest_transcripts.py --dry-run --log-level DEBUG --limit 20
./.venv/bin/python3 ingest_transcripts.py --limit 200 --batch-size 64

Env (common)
------------
AUDIO_BASE=/mnt/8TB_2025/fileserver/audio
EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
NEO4J_URI=bolt://localhost:7687  NEO4J_USER=neo4j  NEO4J_PASSWORD=...
GLINER_MODEL=urchade/gliner_large-v2
GRAPH_PACK_DIR=/mnt/8TB_2025/fileserver/audio/_graphpack
"""

import os, re, uuid, csv, json, time, hashlib, logging, math, sqlite3, itertools
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tupled

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
# Config
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
]
SCAN_ROOTS = [p.strip() for p in os.getenv("SCAN_ROOTS", ",".join(DEFAULT_SCAN_ROOTS)).split(",") if p.strip()]

LOCAL_TZ = os.getenv("LOCAL_TZ", "America/New_York")

# Embeddings
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")  # 384 dims
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))
DEFAULT_BATCH_SIZE = int(os.getenv("EMBED_BATCH", "32"))

# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")
NEO4J_DB = os.getenv("NEO4J_DB", "neo4j")
NEO4J_ENABLED = bool(NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD)

# GLiNER Subjects
GLINER_MODEL = os.getenv("GLINER_MODEL", "urchade/gliner_large-v2")
SUBJECT_LABELS = [s.strip() for s in os.getenv("SUBJECT_LABELS", """
Driving, Errand, PhoneCall, Meeting, Work, Travel, Music, Vehicle, Finance, Insurance, Health, Shopping, DevOps, Models, Data, Legal, Personal, Planning, Family
""").split(",") if s.strip()]

# Subject scoring weights
SUBJ_W_EMB = float(os.getenv("SUBJ_W_EMB", "0.7"))
SUBJ_W_RX  = float(os.getenv("SUBJ_W_RX",  "0.3"))
SUBJ_THRESHOLD_TRANSCRIPTION = float(os.getenv("SUBJ_THRESHOLD_T", "0.25"))
SUBJ_THRESHOLD_SEGMENT       = float(os.getenv("SUBJ_THRESHOLD_S", "0.30"))
SUBJ_THRESHOLD_UTTERANCE     = float(os.getenv("SUBJ_THRESHOLD_U", "0.30"))

# Graph Pack (optional)
GRAPH_PACK_DIR = os.getenv("GRAPH_PACK_DIR", "")  # e.g., /mnt/8TB_2025/fileserver/audio/_graphpack
GRAPH_PACK_APPEND = os.getenv("GRAPH_PACK_APPEND", "1") not in {"0","false","False",""}

# Filenames and patterns
AUDIO_BASE = Path("/mnt/8TB_2025/fileserver/audio")
PAT_TRANS_JSON_TXT = re.compile(r"_([A-Za-z0-9\-\._]+)_transcription\.txt$", re.IGNORECASE)
PAT_TRANS_CSV      = re.compile(r"_transcription\.csv$", re.IGNORECASE)
PAT_ENTITIES       = re.compile(r"_transcription_(entites|entities)\.csv$", re.IGNORECASE)
PAT_RTTM           = re.compile(r"_speakers\.rttm$", re.IGNORECASE)
PAT_MEDIA          = re.compile(r"\.(wav|mp3|m4a|flac|mp4|mov|mkv)$", re.IGNORECASE)
PAT_MUSIC_JSON     = re.compile(r"\.mp3\.music\.json$", re.IGNORECASE)

# Model quality preference
DEFAULT_MODEL_PREF = [
    "large-v3", "large-v2", "large", "turbo",
    "medium.en", "medium",
    "small.en", "small", "base.en", "base", "tiny.en", "tiny",
    "faster-whisper:large-v3", "faster-whisper:large-v2", "faster-whisper:large",
    "faster-whisper:medium", "faster-whisper:small", "faster-whisper:base", "faster-whisper:tiny",
]
MODEL_PREF = [s.strip() for s in os.getenv("MODEL_PREF", ",".join(DEFAULT_MODEL_PREF)).split(",") if s.strip()]

# Heuristic keyword patterns to assist subjects fan-out
SUBJECT_REGEX = {
    "Driving":   re.compile(r"\b(exit|merge|turn|traffic|speed|mile|highway|lane|gps|intersection)\b", re.I),
    "Errand":    re.compile(r"\b(store|checkout|grocery|receipt|errand|pharmacy|pickup|drop[- ]off)\b", re.I),
    "PhoneCall": re.compile(r"\b(call|dial|voicemail|ring|phone|speakerphone)\b", re.I),
    "Meeting":   re.compile(r"\b(meeting|agenda|minutes|action items|follow[- ]up|notes)\b", re.I),
    "Work":      re.compile(r"\b(project|deadline|ticket|deploy|git|kubernetes|model|dataset|pipeline)\b", re.I),
    "Travel":    re.compile(r"\b(flight|hotel|booking|reservation|check[- ]in|boarding|gate|airbnb)\b", re.I),
    "Music":     re.compile(r"\b(song|music|album|playlist|spotify|sirius|radio)\b", re.I),
    "Vehicle":   re.compile(r"\b(oil|tire|engine|battery|charge|maintenance|diagnostic|sensor)\b", re.I),
    "Finance":   re.compile(r"\b(invoice|payment|balance|account|transfer|bank|budget|expense)\b", re.I),
    "Insurance": re.compile(r"\b(policy|claim|deductible|coverage|premium|copay)\b", re.I),
    "Health":    re.compile(r"\b(appointment|doctor|prescription|pharmacy|clinic|therap\w+)\b", re.I),
    "Shopping":  re.compile(r"\b(cart|checkout|order|price|discount|coupon)\b", re.I),
    "DevOps":    re.compile(r"\b(kubernetes|helm|ingress|pod|deployment|cluster|ci/cd|terraform)\b", re.I),
    "Models":    re.compile(r"\b(model|weights|embeddings|llm|quantization|checkpoint|safetensors)\b", re.I),
    "Data":      re.compile(r"\b(dataset|csv|sqlite|neo4j|index|query|etl|ingest)\b", re.I),
    "Legal":     re.compile(r"\b(nda|contract|agreement|terms|liability|warranty|compliance)\b", re.I),
    "Personal":  re.compile(r"\b(birthday|anniversary|family|friend|party|gift)\b", re.I),
    "Planning":  re.compile(r"\b(plan|schedule|timeline|milestone|roadmap|todo|task)\b", re.I),
    "Family":    re.compile(r"\b(mom|dad|sister|brother|kids|child|family)\b", re.I),
}

# =========================
# Stage timing helpers
# =========================
class StageStats:
    def __init__(self, name: str, alpha: float = 0.2):
        self.name = name
        self.alpha = alpha
        self.count = 0
        self.total = 0.0
        self.ema = None  # type: Optional[float]
    def update(self, dt: float):
        self.count += 1
        self.total += dt
        self.ema = dt if self.ema is None else self.alpha * dt + (1 - self.alpha) * self.ema
    @property
    def avg(self) -> float: return (self.total / self.count) if self.count else 0.0
    def summary(self) -> str:
        ema = f"{self.ema:.2f}s" if self.ema is not None else "n/a"
        avg = f"{self.avg:.2f}s"
        return f"{self.name}: n={self.count}, avg={avg}, ema={ema}, total={self.total:.2f}s"

class TimedStage:
    def __init__(self, stats: StageStats, detail: str = ""):
        self.stats = stats
        self.detail = detail
        self.start = 0.0
        self.dt = 0.0
    def __enter__(self):
        self.start = time.perf_counter()
        return self
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
st_subjects  = StageStats("subjects", EMA_ALPHA)
st_ingest    = StageStats("ingest", EMA_ALPHA)
GLOBAL_START = time.perf_counter()

# =========================
# Models (loaded once)
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
log.info(f"Loading embedding model on {DEVICE}…")
emb_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
emb_model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(DEVICE).eval()

# GLiNER lazy
_gliner_entities = None
_gliner_subjects = None

# =========================
# Utilities (hash/id/embeddings)
# =========================
def stable_id(*parts: str) -> str:
    h = hashlib.md5()
    for p in parts:
        h.update((p or "").encode("utf-8", errors="ignore"))
        h.update(b"|")
    return h.hexdigest()

def normalize(v: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(v, p=2, dim=1)

def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = torch.sum(masked, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts

def embed_texts(texts: List[str], batch_size: int, max_length: int = 512) -> List[List[float]]:
    if not texts:
        return []
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

def embed_long_text_via_segments(segments: List[str], batch_size: int) -> List[float]:
    if not segments:
        return [0.0] * EMBED_DIM
    seg_vecs = embed_texts(segments, batch_size=batch_size)
    arr = np.array(seg_vecs, dtype=np.float32)
    vec = arr.mean(axis=0)
    n = np.linalg.norm(vec)
    if n > 0:
        vec = vec / n
    return vec.tolist()

# ---------- Key / timestamp helpers ----------
def _to_localized(dt: datetime) -> datetime:
    if ZoneInfo:
        return dt.replace(tzinfo=ZoneInfo(LOCAL_TZ))
    return dt.replace(tzinfo=timezone.utc)

def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = _to_localized(dt)
    return dt.astimezone(timezone.utc)

def parse_key_datetime_utc_from_string(s: str) -> Optional[datetime]:
    s = s.strip()
    m = re.search(r"(?P<dt14>\d{14})", s)
    if m:
        try:
            dt = datetime.strptime(m.group("dt14"), "%Y%m%d%H%M%S")
            return _to_utc(_to_localized(dt))
        except Exception:
            pass
    for pat, fmt in [
        (r"(\d{4})_(\d{4})_(\d{6})", "%Y_%m%d_%H%M%S"),
        (r"(\d{8})_(\d{6})", "%Y%m%d_%H%M%S"),
        (r"(\d{8})(\d{6})", "%Y%m%d%H%M%S"),
        (r"(\d{4})-(\d{2})-(\d{2})[_\-](\d{2})-(\d{2})-(\d{2})", "%Y-%m-%d_%H-%M-%S"),
        (r"(\d{4})_(\d{2})_(\d{2})[_\-](\d{2})_(\d{2})_(\d{2})", "%Y_%m_%d_%H_%M_%S"),
    ]:
        m2 = re.search(pat, s)
        if m2:
            try:
                dt = datetime.strptime(m2.group(0), fmt)
                return _to_utc(_to_localized(dt))
            except Exception:
                pass
    m = re.search(r"/(?P<Y>\d{4})/(?P<M>\d{2})/(?P<D>\d{2})/", s)
    if m:
        Y, M, D = m.group("Y"), m.group("M"), m.group("D")
        m2 = re.search(r"(?<!\d)(\d{6})(?!\d)", os.path.basename(s))
        if m2:
            hhmmss = m2.group(1)
            try:
                dt = datetime.strptime(f"{Y}{M}{D}{hhmmss}", "%Y%m%d%H%M%S")
                return _to_utc(_to_localized(dt))
            except Exception:
                pass
    return None

def canonicalize_key(name_without_suffix: str, full_path: str) -> str:
    dt = parse_key_datetime_utc_from_string(name_without_suffix) or parse_key_datetime_utc_from_string(full_path)
    if dt:
        return dt.astimezone(timezone.utc).strftime("%Y_%m%d_%H%M%S")
    base = re.sub(r"[^\w\-]+", "_", name_without_suffix).strip("_")
    return base or stable_id(full_path)

def file_key_from_name(name: str) -> str:
    base = os.path.basename(name)
    base = re.sub(r"\.(json|txt|csv|rttm)$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"_([A-Za-z0-9\-\._]+)_transcription(_(entites|entities))?$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"_transcription(_(entites|entities))?$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"_speakers$", "", base, flags=re.IGNORECASE)
    return base

def iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.astimezone(timezone.utc).isoformat() if dt else None

def _parse_any_iso_or_epoch(v: Any) -> Optional[datetime]:
    if v is None:
        return None
    try:
        if isinstance(v, (int, float)):
            val = int(v)
            if val > 1e12:
                val = int(val / 1000)
            return datetime.fromtimestamp(val, tz=timezone.utc)
        s = str(v).strip()
        if re.fullmatch(r"\d{10,13}", s):
            val = int(s)
            if val > 1e12:
                val = int(val / 1000)
            return datetime.fromtimestamp(val, tz=timezone.utc)
        return _to_utc(datetime.fromisoformat(s.replace("Z", "+00:00")))
    except Exception:
        return None

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
            if not text and segments:
                try:
                    text = " ".join([(s.get("text") or "").strip() for s in segments if (s.get("text") or "").strip()])
                except Exception:
                    pass
            # attempt start/end
            candidates = [data.get("started_at"), data.get("start_time"), data.get("start"),
                          (data.get("metadata") or {}).get("started_at")]
            enders = [data.get("ended_at"), data.get("end_time"), data.get("end"),
                      (data.get("metadata") or {}).get("ended_at")]
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
                if reader.fieldnames is None or len(reader.fieldnames) <= 1:
                    f.seek(0)
                    reader = csv.DictReader(f, delimiter="\t")
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
                        "id": row.get("SegmentId") or row.get("id") or stable_id(path, "seg", str(row_idx), text[:64]),
                        "seek": None, "start": start, "end": end, "text": text,
                        "tokens": [], "words": [], "abs_start": iso(abs_start) if abs_start else None, "abs_end": iso(abs_end) if abs_end else None,
                    }
                    segments.append(seg)
                    if text: full_text.append(text)
                    if abs_start: file_start_abs = min(file_start_abs, abs_start) if file_start_abs else abs_start
                    if abs_end:   file_end_abs   = max(file_end_abs, abs_end)     if file_end_abs   else abs_end
            return {"text": " ".join(full_text), "segments": segments, "language": "en",
                    "file_started_at": file_start_abs, "file_ended_at": file_end_abs}
        except Exception as ex:
            log.warning(f"Failed to parse CSV transcript {path}: {ex}")
            return None

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

def load_rttm(path: str) -> List[Tuple[float,float,str]]:
    intervals = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"): continue
                parts = line.split()
                if len(parts) < 9: continue
                try:
                    start = float(parts[3]); dur = float(parts[4]); end = start + dur
                    spk = parts[7]
                    intervals.append((start, end, spk))
                except Exception:
                    continue
        intervals.sort(key=lambda x: x[0])
    except Exception as ex:
        log.warning(f"Failed to read RTTM {path}: {ex}")
    return intervals

def load_music_flag_from_sidecar(music_json_path: Optional[str]) -> Optional[bool]:
    if not music_json_path or not os.path.isfile(music_json_path):
        return None
    try:
        with open(music_json_path, "r", encoding="utf-8") as f:
            j = json.load(f)
        for k in ("is_music", "music", "has_music"):
            if k in j:
                return bool(j[k])
    except Exception as ex:
        log.warning(f"Music sidecar parse failed ({music_json_path}): {ex}")
    return None

# =========================
# Model/version selection
# =========================
def extract_model_tag_from_json_txt(p: str) -> str:
    m = PAT_TRANS_JSON_TXT.search(os.path.basename(p))
    return m.group(1) if m else ""

def model_rank(tag: str) -> int:
    if not tag: return 10_000
    t = tag.lower()
    try:
        return MODEL_PREF.index(t)
    except ValueError:
        if "large" in t: return 100
        if "medium" in t: return 200
        if "small" in t: return 300
        if "base" in t: return 400
        if "tiny" in t: return 500
        return 9999

def is_in_audio_base(p: str) -> bool:
    try:
        return str(Path(p).resolve()).startswith(str(AUDIO_BASE.resolve()))
    except Exception:
        return False

def select_best_json(json_paths: List[str], csv_paths: List[str]) -> Optional[str]:
    if not json_paths:
        return None
    scored: List[Tuple[int,int,float,float,str]] = []  # (not_audio_base, model_rank, -segments, -mtime, path)
    for p in json_paths:
        tag = extract_model_tag_from_json_txt(p)
        rank = model_rank(tag)
        segs = -1.0
        try:
            with open(p, "r", encoding="utf-8") as f:
                j = json.load(f)
            segs = float(len(j.get("segments") or []))
        except Exception:
            segs = -1.0
        mtime = 0.0
        try:
            mtime = Path(p).stat().st_mtime
        except Exception:
            pass
        scored.append((0 if is_in_audio_base(p) else 1, rank, -segs, -mtime, p))
    scored.sort()
    best = scored[0][-1] if scored else None
    if best:
        log.info(f"[select] best JSON = {best}")
    else:
        log.warning("[select] no JSON candidate could be selected")
    return best

# =========================
# Entities & GLiNER subjects
# =========================
def gliner_extract_entities(text: str) -> List[Dict[str, Any]]:
    global _gliner_entities
    try:
        if _gliner_entities is None:
            from gliner import GLiNER
            log.info(f"Loading GLiNER (entities): {GLINER_MODEL}")
            _gliner_entities = GLiNER.from_pretrained(GLINER_MODEL, device=DEVICE)
        ents = _gliner_entities.predict_entities(text, ["Person","Place","Event","Date","Organization","Product"])
        return [{
            "text": e.get("text",""),
            "label": e.get("label",""),
            "score": float(e.get("score", 0.0) or 0.0),
            "start": float(e.get("start", -1) or -1),
            "end": float(e.get("end", -1) or -1),
        } for e in ents]
    except Exception as ex:
        log.warning(f"GLiNER entities failed: {ex}")
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

# Subjects (overall)
def gliner_subject_topics(text: str, labels: List[str]) -> List[Tuple[str, float]]:
    global _gliner_subjects
    if not text.strip():
        return []
    try:
        if _gliner_subjects is None:
            from gliner import GLiNER
            log.info(f"Loading GLiNER (subjects): {GLINER_MODEL}")
            _gliner_subjects = GLiNER.from_pretrained(GLINER_MODEL, device=DEVICE)
        preds = _gliner_subjects.predict_entities(text, labels)
        scores: Dict[str, List[float]] = {}
        for p in preds:
            lbl = (p.get("label") or "").strip()
            sc  = float(p.get("score") or 0.0)
            if lbl:
                scores.setdefault(lbl, []).append(sc)
        ranked = []
        for lbl, arr in scores.items():
            ranked.append((lbl, float(np.mean(arr)) + 0.02 * min(len(arr), 5)))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return [r for r in ranked if r[1] >= SUBJ_THRESHOLD_TRANSCRIPTION][:8]
    except Exception as ex:
        log.warning(f"GLiNER subject extraction failed: {ex}")
        return []

# =========================
# Validation / cleaning
# =========================
def validate_and_clean_segments(key: str, segs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str,int]]:
    fixed = []; stats = {"nonfinite":0, "neg_dur":0, "reordered":0, "empty_txt":0, "kept":0}
    last_end = -math.inf
    for i, s in enumerate(segs):
        try:
            start = float(s.get("start", 0.0)); end = float(s.get("end", start))
            if not math.isfinite(start) or not math.isfinite(end):
                stats["nonfinite"] += 1; continue
            if end < start:
                end = start; stats["neg_dur"] += 1
            txt = (s.get("text") or "").strip()
            if not txt: stats["empty_txt"] += 1
            sid = s.get("id") or stable_id(key, "seg", str(i), txt[:64])
            if start < last_end: stats["reordered"] += 1
            last_end = max(last_end, end)
            fixed.append({
                "id": sid, "idx": s.get("idx", i),
                "start": start, "end": end, "text": txt,
                "tokens": s.get("tokens", []), "words": s.get("words", []),
                "abs_start": s.get("abs_start"), "abs_end": s.get("abs_end"),
            })
        except Exception:
            stats["nonfinite"] += 1
            continue
    stats["kept"] = len(fixed)
    return fixed, stats

# =========================
# Speaker/utterance utils
# =========================
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

def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))

def utterances_from_rttm_with_words(rttm: List[Tuple[float,float,str]], segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    words = words_from_segments(segments)
    if not words:
        return []
    utterances = []
    for i, (start, end, spk) in enumerate(rttm):
        mid = lambda w: (w["start"] + w["end"]) / 2.0
        chunk = [w for w in words if mid(w) >= start and mid(w) <= end]
        if not chunk:
            continue
        text = " ".join(w["text"] for w in chunk).strip()
        u_start = chunk[0]["start"]; u_end = chunk[-1]["end"]
        u_id = stable_id("utt", spk, f"{u_start:.3f}", f"{u_end:.3f}")
        utterances.append({"id": u_id, "speaker_label": spk, "start": u_start, "end": u_end,
                           "text": text, "segment_id": None, "idx": i})
    return utterances

def utterances_from_rttm_dominant_segment(rttm: List[Tuple[float,float,str]], segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    utterances = []
    for i, seg in enumerate(segments):
        s_start = float(seg.get("start", 0.0) or 0.0); s_end = float(seg.get("end", s_start) or s_start)
        best_overlap = 0.0; best_spk = None
        for (a, b, spk) in rttm:
            ov = overlap(s_start, s_end, a, b)
            if ov > best_overlap:
                best_overlap = ov; best_spk = spk
        spk_label = best_spk or "UNKNOWN"
        u_id = stable_id("utt", spk_label, str(seg.get("id")), f"{s_start:.3f}", f"{s_end:.3f}")
        utterances.append({"id": u_id, "speaker_label": spk_label, "start": s_start, "end": s_end,
                           "text": seg.get("text", ""), "segment_id": seg.get("id"), "idx": i})
    return utterances

def _sum_overlap(seg_start: float, seg_end: float, rttm: List[Tuple[float,float,str]]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for a,b,spk in rttm:
        ov = overlap(seg_start, seg_end, a, b)
        if ov > 0:
            totals[spk] = totals.get(spk, 0.0) + ov
    return totals

def compute_segment_speaker_overlaps(
    rttm: List[Tuple[float,float,str]],
    segments: List[Dict[str, Any]],
    speaker_map: Dict[str, Dict[str,str]],
    min_proportion: float = 0.05
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    seg_best: Dict[str, Dict[str, Any]] = {}
    seg_edges: List[Dict[str, Any]] = []
    for s in segments:
        s_start = float(s.get("start", 0.0) or 0.0); s_end = float(s.get("end", s_start) or s_start)
        sid = s["id"]
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
    if not NEO4J_ENABLED:
        return None
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

SCHEMA_QUERIES = [
    "CREATE CONSTRAINT transcription_id IF NOT EXISTS FOR (t:Transcription) REQUIRE t.id IS UNIQUE",
    "CREATE CONSTRAINT segment_id IF NOT EXISTS FOR (s:Segment) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
    "CREATE CONSTRAINT utterance_id IF NOT EXISTS FOR (u:Utterance) REQUIRE u.id IS UNIQUE",
    "CREATE CONSTRAINT speaker_id IF NOT EXISTS FOR (sp:Speaker) REQUIRE sp.id IS UNIQUE",
    "CREATE CONSTRAINT subject_id IF NOT EXISTS FOR (s:Subject) REQUIRE s.id IS UNIQUE",
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
        log.info("Neo4j not configured; skipping schema setup.")
        return
    with driver.session(database=NEO4J_DB) as sess:
        for q in SCHEMA_QUERIES:
            sess.run(q)
        try:
            _create_vector_index(sess, "Segment", "embedding", "segment_embedding_index", EMBED_DIM)
            _create_vector_index(sess, "Transcription", "embedding", "transcription_embedding_index", EMBED_DIM)
            _create_vector_index(sess, "Utterance", "embedding", "utterance_embedding_index", EMBED_DIM)
        except Neo4jError as e:
            msg = str(e)
            if "Invalid input 'VECTOR'" in msg or "Unrecognized command" in msg:
                raise RuntimeError("This Neo4j server does not support VECTOR indexes (need 5.11+).") from e
            raise
    log.info("Neo4j schema ensured (constraints + vector indexes + btree indexes).")

def already_ingested(driver, t_id: str) -> bool:
    if not driver: return False
    with driver.session(database=NEO4J_DB) as sess:
        rec = sess.run("MATCH (t:Transcription {id:$id}) RETURN t.id AS id LIMIT 1", id=t_id).single()
        return bool(rec)

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

def should_reingest(status: Dict[str, Any], expected_segments: int, has_rttm: bool, paths: Dict[str, Any]) -> bool:
    if not status.get("exists"):
        return True
    for k in ("json", "csv", "rttm", "media"):
        if paths.get(k) and not status.get(f"source_{k}"):
            return True
    if status.get("started_at") is None:
        return True
    seg_count = int(status.get("seg_count") or 0)
    seg_no_emb = int(status.get("seg_no_emb") or 0)
    utt_count = int(status.get("utt_count") or 0)
    if expected_segments > 0 and seg_count == 0:
        return True
    if has_rttm and utt_count == 0:
        return True
    if seg_count > 0 and seg_no_emb > max(1, seg_count // 2):
        return True
    return False

# =========================
# Graph Pack writers (optional)
# =========================
def _csv_init_once(path: Path, header: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(header)

def gp_write_rows(path: Path, rows: List[List[Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow(r)

def graph_pack_emit_all(base_dir: str,
                        t_id: str, key: str, activity_type: str,
                        segments: List[Dict[str, Any]],
                        utterances: List[Dict[str, Any]],
                        speakers: List[Dict[str, Any]],
                        entities: List[Dict[str, Any]],
                        t_subjects: List[Tuple[str,float]],
                        seg_subjects: Dict[str, List[Tuple[str,float]]],
                        utt_subjects: Dict[str, List[Tuple[str,float]]]):
    if not base_dir: return
    b = Path(base_dir)

    # nodes
    n_subj = b / "nodes_subjects.csv"; _csv_init_once(n_subj, ["subject_id","label"])
    n_spk  = b / "nodes_speakers.csv"; _csv_init_once(n_spk,  ["speaker_id","label"])
    n_ent  = b / "nodes_entities.csv"; _csv_init_once(n_ent,  ["entity_id","text","label"])
    n_tr   = b / "nodes_transcriptions.csv"; _csv_init_once(n_tr, ["transcription_id","key","activity_type"])
    n_seg  = b / "nodes_segments.csv"; _csv_init_once(n_seg, ["segment_id","idx","start","end"])
    n_utt  = b / "nodes_utterances.csv"; _csv_init_once(n_utt, ["utterance_id","idx","start","end"])

    gp_write_rows(n_tr, [[t_id, key, activity_type]])
    gp_write_rows(n_seg, [[s["id"], s["idx"], f'{s["start"]:.3f}', f'{s["end"]:.3f}'] for s in segments])
    gp_write_rows(n_utt, [[u["id"], u["idx"], f'{u["start"]:.3f}', f'{u["end"]:.3f}'] for u in utterances])
    gp_write_rows(n_spk, [[sp["id"], sp["label"]] for sp in speakers])
    gp_write_rows(n_ent, [[e["id"], e["text"], e["label"]] for e in entities])

    # relations
    r_t_has_s = b / "rel_transcription_has_segment.csv"; _csv_init_once(r_t_has_s, ["transcription_id","segment_id"])
    r_t_has_u = b / "rel_transcription_has_utterance.csv"; _csv_init_once(r_t_has_u, ["transcription_id","utterance_id"])
    gp_write_rows(r_t_has_s, [[t_id, s["id"]] for s in segments])
    gp_write_rows(r_t_has_u, [[t_id, u["id"]] for u in utterances])

    r_seg_spk = b / "rel_segment_spoken_by.csv"; _csv_init_once(r_seg_spk, ["segment_id","speaker_id"])
    rows = []
    for s in segments:
        sid = s["id"]; spk_id = s.get("speaker_id")
        if spk_id: rows.append([sid, spk_id])
    gp_write_rows(r_seg_spk, rows)

    r_u_spk = b / "rel_utterance_spoken_by.csv"; _csv_init_once(r_u_spk, ["utterance_id","speaker_id"])
    rows = []
    for u in utterances:
        spk_id = u.get("speaker_id")
        if spk_id: rows.append([u["id"], spk_id])
    gp_write_rows(r_u_spk, rows)

    r_t_about = b / "rel_transcription_about_subject.csv"; _csv_init_once(r_t_about, ["transcription_id","subject_id","score"])
    r_s_about = b / "rel_segment_about_subject.csv";       _csv_init_once(r_s_about, ["segment_id","subject_id","score"])
    r_u_about = b / "rel_utterance_about_subject.csv";     _csv_init_once(r_u_about, ["utterance_id","subject_id","score"])
    # Ensure subject nodes
    gp_write_rows(n_subj, [[stable_id("subject", lbl.lower()), lbl] for (lbl, _) in t_subjects])
    for sid, subjs in seg_subjects.items():
        for (lbl, _) in subjs:
            gp_write_rows(n_subj, [[stable_id("subject", lbl.lower()), lbl]])
    for uid, subjs in utt_subjects.items():
        for (lbl, _) in subjs:
            gp_write_rows(n_subj, [[stable_id("subject", lbl.lower()), lbl]])

    gp_write_rows(r_t_about, [[t_id, stable_id("subject", lbl.lower()), f"{float(sc):.4f}"] for (lbl, sc) in t_subjects])
    rows = []
    for sid, subjs in seg_subjects.items():
        for (lbl, sc) in subjs:
            rows.append([sid, stable_id("subject", lbl.lower()), f"{float(sc):.4f}"])
    gp_write_rows(r_s_about, rows)
    rows = []
    for uid, subjs in utt_subjects.items():
        for (lbl, sc) in subjs:
            rows.append([uid, stable_id("subject", lbl.lower()), f"{float(sc):.4f}"])
    gp_write_rows(r_u_about, rows)

    r_t_mentions = b / "rel_transcription_mentions_entity.csv"; _csv_init_once(r_t_mentions, ["transcription_id","entity_id","count","avg_score"])
    gp_write_rows(r_t_mentions, [[t_id, e["id"], e.get("count",0), f'{float(e.get("avg_score",0.0)):.4f}'] for e in entities])

# =========================
# Ingest (Cypher)
# =========================
def ingest_record(
    driver,
    t_id: str,
    key: str,
    started_at_iso: Optional[str],
    ended_at_iso: Optional[str],
    source_paths: Dict[str,str],
    text: str,
    transcript_emb: List[float],
    segments: List[Dict[str, Any]],
    entities: List[Dict[str, Any]],
    utterances: List[Dict[str, Any]],
    speakers: List[Dict[str, Any]],
    segment_speakers: List[Dict[str, Any]],
    subjects_ranked: List[Tuple[str,float]],
    activity_type: str,
    seg_subjects: Dict[str, List[Tuple[str,float]]],
    utt_subjects: Dict[str, List[Tuple[str,float]]],
):
    if not driver:
        log.info("Neo4j not configured; skipping ingestion.")
        return
    with TimedStage(st_ingest, detail=f"key={key}"):
        cypher = """
        MERGE (t:Transcription {id: $t_id})
        ON CREATE SET t.key = $key, t.created_at = datetime()
        ON MATCH  SET t.key = $key, t.updated_at = datetime()
        SET t.text = $text, t.embedding = $transcript_emb,
            t.source_json = $source_json, t.source_csv = $source_csv, t.source_rttm = $source_rttm, t.source_media = $source_media,
            t.started_at = CASE WHEN $started_at IS NULL THEN t.started_at ELSE datetime($started_at) END,
            t.ended_at   = CASE WHEN $ended_at   IS NULL THEN t.ended_at   ELSE datetime($ended_at)   END,
            t.activity_type = $activity_type

        WITH t, $segments AS segs
        UNWIND segs AS seg
        MERGE (s:Segment {id: seg.id})
          ON CREATE SET s.idx = seg.idx, s.start = seg.start, s.end = seg.end, s.created_at = datetime()
          ON MATCH  SET s.idx = seg.idx, s.start = seg.start, s.end = seg.end, s.updated_at = datetime()
        SET s.text = seg.text,
            s.tokens_count = seg.tokens_count,
            s.embedding = seg.embedding,
            s.absolute_start = CASE WHEN seg.abs_start IS NULL THEN s.absolute_start ELSE datetime(seg.abs_start) END,
            s.absolute_end   = CASE WHEN seg.abs_end   IS NULL THEN s.absolute_end   ELSE datetime(seg.abs_end)   END,
            s.speaker_label  = seg.speaker_label,
            s.speaker_id     = seg.speaker_id
        MERGE (t)-[:HAS_SEGMENT]->(s)

        WITH t, $speakers AS spks
        UNWIND spks AS sp
        MERGE (spk:Speaker {id: sp.id})
          ON CREATE SET spk.label = sp.label, spk.key = sp.key, spk.created_at = datetime()
          ON MATCH  SET spk.label = sp.label, spk.key = sp.key, spk.updated_at = datetime()

        WITH t, $utterances AS uts
        UNWIND uts AS u
        MERGE (uNode:Utterance {id: u.id})
          ON CREATE SET uNode.idx = u.idx, uNode.created_at = datetime()
          ON MATCH  SET uNode.idx = u.idx, uNode.updated_at = datetime()
        SET uNode.start = u.start, uNode.end = u.end, uNode.text = u.text, uNode.embedding = u.embedding,
            uNode.absolute_start = CASE WHEN u.abs_start IS NULL THEN uNode.absolute_start ELSE datetime(u.abs_start) END,
            uNode.absolute_end   = CASE WHEN u.abs_end   IS NULL THEN uNode.absolute_end   ELSE datetime(u.abs_end)   END
        MERGE (t)-[:HAS_UTTERANCE]->(uNode)

        WITH $utterances AS uts
        UNWIND uts AS u
        OPTIONAL MATCH (uNode:Utterance {id: u.id})
        OPTIONAL MATCH (s:Segment {id: u.segment_id})
        FOREACH (_ IN CASE WHEN s IS NULL OR uNode IS NULL THEN [] ELSE [1] END |
            MERGE (uNode)-[:OF_SEGMENT]->(s)
        )
        WITH $utterances AS uts
        UNWIND uts AS u
        OPTIONAL MATCH (uNode:Utterance {id: u.id})
        OPTIONAL MATCH (spk:Speaker {id: u.speaker_id})
        FOREACH (_ IN CASE WHEN spk IS NULL OR uNode IS NULL THEN [] ELSE [1] END |
            MERGE (uNode)-[:SPOKEN_BY]->(spk)
        )

        WITH $segment_speakers AS ss
        UNWIND ss AS x
        MATCH (s:Segment {id: x.segment_id})
        MATCH (spk:Speaker {id: x.speaker_id})
        MERGE (s)-[r:SPOKEN_BY]->(spk)
        SET r.overlap = coalesce(x.overlap, 0.0),
            r.proportion = coalesce(x.proportion, 0.0)
        """
        cypher_entities = """
        MATCH (t:Transcription {id: $t_id})
        WITH t
        UNWIND $entities AS ent
          MERGE (e:Entity {id: ent.id})
            ON CREATE SET e.text=ent.text, e.label=ent.label, e.created_at=datetime()
            ON MATCH  SET e.text=ent.text, e.label=ent.label, e.updated_at=datetime()
          MERGE (t)-[m:MENTIONS]->(e)
          SET m.count=ent.count, m.starts=ent.starts, m.ends=ent.ends, m.avg_score=ent.avg_score
        """
        cypher_subjects_t = """
        MATCH (t:Transcription {id: $t_id})
        WITH t, $subjects AS subs
        UNWIND subs AS s
          MERGE (sub:Subject {id: s.subject_id})
            ON CREATE SET sub.label = s.label, sub.created_at = datetime()
            ON MATCH  SET sub.label = s.label, sub.updated_at = datetime()
          MERGE (t)-[r:ABOUT]->(sub)
          SET r.score = s.score
        """
        cypher_subjects_s = """
        UNWIND $seg_subjects AS item
        MATCH (s:Segment {id: item.segment_id})
        MERGE (sub:Subject {id: item.subject_id})
          ON CREATE SET sub.label = item.label, sub.created_at = datetime()
          ON MATCH  SET sub.label = item.label, sub.updated_at = datetime()
        MERGE (s)-[r:ABOUT]->(sub)
        SET r.score = item.score
        """
        cypher_subjects_u = """
        UNWIND $utt_subjects AS item
        MATCH (u:Utterance {id: item.utterance_id})
        MERGE (sub:Subject {id: item.subject_id})
          ON CREATE SET sub.label = item.label, sub.created_at = datetime()
          ON MATCH  SET sub.label = item.label, sub.updated_at = datetime()
        MERGE (u)-[r:ABOUT]->(sub)
        SET r.score = item.score
        """

        # Prepare params
        params = {
            "t_id": t_id,
            "key": key,
            "text": text,
            "transcript_emb": transcript_emb,
            "source_json": source_paths.get("json"),
            "source_csv":  source_paths.get("csv"),
            "source_rttm": source_paths.get("rttm"),
            "source_media": source_paths.get("media"),
            "segments": segments,
            "utterances": utterances,
            "speakers": speakers,
            "entities": entities,
            "started_at": started_at_iso,
            "ended_at": ended_at_iso,
            "segment_speakers": segment_speakers,
            "subjects": [{"subject_id": stable_id("subject", lbl.lower()), "label": lbl, "score": float(sc)} for (lbl, sc) in subjects_ranked],
            "activity_type": activity_type,
            "seg_subjects": [
                {"segment_id": sid, "subject_id": stable_id("subject", lbl.lower()), "label": lbl, "score": float(sc)}
                for sid, subjs in seg_subjects.items() for (lbl, sc) in subjs
            ],
            "utt_subjects": [
                {"utterance_id": uid, "subject_id": stable_id("subject", lbl.lower()), "label": lbl, "score": float(sc)}
                for uid, subjs in utt_subjects.items() for (lbl, sc) in subjs
            ],
        }
        with driver.session(database=NEO4J_DB) as sess:
            sess.run(cypher, **params)
            if entities:
                sess.run(cypher_entities, **params)
            if subjects_ranked:
                sess.run(cypher_subjects_t, **params)
            if params["seg_subjects"]:
                sess.run(cypher_subjects_s, **params)
            if params["utt_subjects"]:
                sess.run(cypher_subjects_u, **params)

# =========================
# Discovery
# =========================
def discover_keys() -> Dict[str, Dict[str, Any]]:
    with TimedStage(st_discover, detail=f"roots={len(SCAN_ROOTS)}"):
        mapping: Dict[str, Dict[str, Any]] = {}
        def add(kind: str, path: str):
            base = file_key_from_name(os.path.basename(path))
            key = canonicalize_key(base, path)
            mapping.setdefault(key, {"json_all": [], "csv_all": []})
            if kind == "json": mapping[key]["json_all"].append(path)
            elif kind == "csv": mapping[key]["csv_all"].append(path)
            else: mapping[key][kind] = path

        for root in SCAN_ROOTS:
            if not os.path.isdir(root): continue
            for r, _, files in os.walk(root):
                for name in files:
                    p = os.path.join(r, name)
                    if PAT_TRANS_JSON_TXT.search(name): add("json", p)
                    elif PAT_TRANS_CSV.search(name):   add("csv", p)
                    elif PAT_ENTITIES.search(name):    add("entities", p)
                    elif PAT_RTTM.search(name):        add("rttm", p)
                    elif PAT_MUSIC_JSON.search(name):  add("music_json", p)
                    elif PAT_MEDIA.search(name):       add("media", p)
        return mapping

# =========================
# Subjects fan-out scoring
# =========================
def infer_activity_type(subject_tags: List[str], is_music: Optional[bool], text_snippet: str) -> str:
    tags = {t.lower() for t in subject_tags}
    t = text_snippet.lower()
    if "driving" in tags or re.search(SUBJECT_REGEX["Driving"], t):    return "Driving"
    if "phonecall" in tags or re.search(SUBJECT_REGEX["PhoneCall"], t): return "PhoneCall"
    if "shopping" in tags or "errand" in tags or re.search(SUBJECT_REGEX["Errand"], t): return "Errand"
    if "meeting" in tags or re.search(SUBJECT_REGEX["Meeting"], t):    return "Meeting"
    if is_music is True:                                              return "MusicListening"
    if "work" in tags or "devops" in tags or "models" in tags or "data" in tags: return "Work"
    if "travel" in tags:                                              return "Travel"
    if "vehicle" in tags:                                             return "Vehicle"
    return "General"

def score_subjects_for_texts(texts: List[str],
                             vecs: List[List[float]],
                             subject_labels: List[str],
                             subj_proto_vecs: Dict[str, List[float]],
                             threshold: float) -> List[List[Tuple[str, float]]]:
    """
    For each text+vec, compute blended score per subject:
       score = W_emb * cosine(vec, proto[label]) + W_rx * 1_{regex hit}
    Return list parallel to texts: [(label, score), ...] filtered by threshold.
    """
    out: List[List[Tuple[str,float]]] = []
    for i, t in enumerate(texts):
        v = np.array(vecs[i], dtype=np.float32) if i < len(vecs) and vecs[i] is not None else None
        pairs: List[Tuple[str,float]] = []
        for lbl in subject_labels:
            rx = SUBJECT_REGEX.get(lbl)  # may be None
            hit = 1.0 if (rx and rx.search(t or "")) else 0.0
            emb = 0.0
            if v is not None and lbl in subj_proto_vecs:
                pv = np.array(subj_proto_vecs[lbl], dtype=np.float32)
                denom = (np.linalg.norm(v) * np.linalg.norm(pv)) or 1.0
                emb = float(np.dot(v, pv) / denom)
            score = SUBJ_W_EMB * emb + SUBJ_W_RX * hit
            if score >= threshold:
                pairs.append((lbl, score))
        pairs.sort(key=lambda x: x[1], reverse=True)
        out.append(pairs[:8])
    return out

# =========================
# Helpers for re-ingest decisions
# =========================
def probe_expected(paths: Dict[str, Any]) -> Tuple[int, bool]:
    expected_segments = 0
    data = None
    best_json = select_best_json(paths.get("json_all") or [], paths.get("csv_all") or [])
    if best_json and os.path.isfile(best_json):
        data = load_transcription_json_txt(best_json)
    if data is None:
        for c in (paths.get("csv_all") or []):
            if os.path.isfile(c):
                data = load_transcription_csv(c)
                if data: break
    if data is not None:
        expected_segments = len(data.get("segments") or [])
    has_rttm = bool(paths.get("rttm") and os.path.isfile(paths["rttm"]))
    return expected_segments, has_rttm

# =========================
# Processing one key
# =========================
def process_key(key: str, paths: Dict[str, Any], driver, batch_size: int, dry_run: bool = False):
    # Choose best JSON; fallback to CSV if needed
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
                src_dir_for_sidecar = os.path.dirname(c)
                best_csv = c
                break
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
                if choices:
                    best_csv = str(choices[0])

    if data is None:
        log.warning(f"[{key}] No transcription data found; skipping.")
        return None

    text = data.get("text","") or ""
    segments_raw = data.get("segments", []) or []

    # ---- Validate & clean segments
    with TimedStage(st_validate, detail=f"key={key}"):
        cleaned_segments, vstats = validate_and_clean_segments(key, segments_raw)
        log.info(f"[validate] key={key} segs_in={len(segments_raw)} segs_kept={vstats['kept']} "
                 f"nonfinite={vstats['nonfinite']} neg_dur={vstats['neg_dur']} empty_txt={vstats['empty_txt']} reordered={vstats['reordered']}")

    # ---- Build enriched segments & embeddings
    seg_texts = [s["text"] for s in cleaned_segments]
    enriched_segments = []
    max_end = 0.0
    for idx, seg in enumerate(cleaned_segments):
        s_tokens = seg.get("tokens", [])
        s_tok_count = len(s_tokens) if isinstance(s_tokens, (list,tuple)) else int(s_tokens or 0)
        enriched_segments.append({
            "id": seg["id"], "idx": idx, "start": seg["start"], "end": seg["end"],
            "text": seg["text"], "tokens_count": s_tok_count, "embedding": None,
            "abs_start": seg.get("abs_start") or None, "abs_end": seg.get("abs_end") or None,
        })
        if seg["end"] > max_end: max_end = seg["end"]

    file_start_dt = data.get("file_started_at"); file_end_dt = data.get("file_ended_at")
    dt0_utc = file_start_dt.astimezone(timezone.utc) if isinstance(file_start_dt, datetime) else parse_key_datetime_utc_from_string(key)
    started_at_iso = iso(dt0_utc) if dt0_utc else None
    ended_at_iso = iso(file_end_dt) if isinstance(file_end_dt, datetime) else (iso(dt0_utc + timedelta(seconds=max_end)) if dt0_utc else None)

    if dt0_utc:
        for s in enriched_segments:
            if not s["abs_start"]: s["abs_start"] = iso(dt0_utc + timedelta(seconds=float(s["start"])))
            if not s["abs_end"]:   s["abs_end"]   = iso(dt0_utc + timedelta(seconds=float(s["end"])))

    with TimedStage(st_embed, detail=f"key={key} type=transcript+segments(+utts if any)"):
        if seg_texts:
            seg_vecs = embed_texts(seg_texts, batch_size=batch_size)
            for i,v in enumerate(seg_vecs):
                enriched_segments[i]["embedding"] = v
            transcript_vec = embed_long_text_via_segments(seg_texts, batch_size=batch_size)
        else:
            transcript_vec = embed_texts([text], batch_size=batch_size)[0]
            seg_vecs = []

    # ---- Entities
    ents_raw = []
    entities_path = paths.get("entities")
    if entities_path and os.path.isfile(entities_path):
        ents_raw = load_entities_csv(entities_path)
    elif text.strip():
        ents_raw = gliner_extract_entities(text)
    entities = aggregate_entities(ents_raw) if ents_raw else []

    # ---- Speakers/utterances
    utterances = []
    speakers = []
    speaker_map: Dict[str, Dict[str,str]] = {}
    seg_edges: List[Dict[str, Any]] = []

    if "rttm" in paths and paths["rttm"] and os.path.isfile(paths["rttm"]):
        rttm = load_rttm(paths["rttm"])
        labels = sorted(set([spk for _,_,spk in rttm]))
        for lbl in labels:
            sp_id = stable_id(key, "spk", lbl)
            speakers.append({"id": sp_id, "label": lbl, "key": key})
            speaker_map[lbl] = {"id": sp_id, "label": lbl}
        if "UNKNOWN" not in speaker_map:
            unknown_id = stable_id(key, "spk", "UNKNOWN")
            speakers.append({"id": unknown_id, "label": "UNKNOWN", "key": key})
            speaker_map["UNKNOWN"] = {"id": unknown_id, "label": "UNKNOWN"}

        seg_best, seg_edges = compute_segment_speaker_overlaps(
            rttm=rttm, segments=enriched_segments, speaker_map=speaker_map, min_proportion=0.05,
        )
        for s in enriched_segments:
            best = seg_best.get(s["id"], {})
            s["speaker_label"] = best.get("speaker_label", "UNKNOWN")
            s["speaker_id"] = best.get("speaker_id", speaker_map["UNKNOWN"]["id"])

        utts = utterances_from_rttm_with_words(rttm, cleaned_segments)
        if not utts:
            utts = utterances_from_rttm_dominant_segment(rttm, enriched_segments)

        for u in utts:
            best, best_id = 0.0, None
            for s in enriched_segments:
                ov = overlap(u["start"], u["end"], s["start"], s["end"])
                if ov > best: best = ov; best_id = s["id"]
            u["segment_id"] = best_id
        for u in utts:
            sp_map = speaker_map.get(u.get("speaker_label", "UNKNOWN")) or speaker_map["UNKNOWN"]
            u["speaker_id"] = sp_map["id"]

        utt_texts = [u["text"] for u in utts]
        if utt_texts:
            utt_vecs = embed_texts(utt_texts, batch_size=batch_size)
            for i,v in enumerate(utt_vecs):
                utts[i]["embedding"] = v
        else:
            utt_vecs = []

        if dt0_utc:
            for u in utts:
                u["abs_start"] = iso(dt0_utc + timedelta(seconds=float(u["start"])))
                u["abs_end"]   = iso(dt0_utc + timedelta(seconds=float(u["end"])))
        utterances = utts
    else:
        utt_vecs = []

    # ---- Subjects (Transcription level) + Activity Type + Sidecar
    with TimedStage(st_subjects, detail=f"key={key} subjects+fanout"):
        subjects_ranked = gliner_subject_topics(text, SUBJECT_LABELS)
        is_music = load_music_flag_from_sidecar(paths.get("music_json"))
        activity_type = infer_activity_type([lbl for (lbl,_) in subjects_ranked], is_music, text[:512])

        # Prepare subject prototypes (embedding of "topic: Label")
        subj_proto_vecs = {}
        proto_texts = [f"topic: {lbl}" for lbl in SUBJECT_LABELS]
        proto_vecs = embed_texts(proto_texts, batch_size=min(32, DEFAULT_BATCH_SIZE))
        for i, lbl in enumerate(SUBJECT_LABELS):
            if i < len(proto_vecs):
                subj_proto_vecs[lbl] = proto_vecs[i]

        # Fan-out to Segments
        seg_subject_scores = score_subjects_for_texts(seg_texts, seg_vecs, SUBJECT_LABELS, subj_proto_vecs, SUBJ_THRESHOLD_SEGMENT)
        seg_subjects: Dict[str, List[Tuple[str,float]]] = {}
        for i, pairs in enumerate(seg_subject_scores):
            if pairs:
                seg_subjects[enriched_segments[i]["id"]] = pairs

        # Fan-out to Utterances (if present)
        utt_subjects: Dict[str, List[Tuple[str,float]]] = {}
        if utterances:
            utt_texts = [u["text"] for u in utterances]
            utt_subject_scores = score_subjects_for_texts(utt_texts, utt_vecs, SUBJECT_LABELS, subj_proto_vecs, SUBJ_THRESHOLD_UTTERANCE)
            for i, pairs in enumerate(utt_subject_scores):
                if pairs:
                    utt_subjects[utterances[i]["id"]] = pairs

        # Optional sidecar: subjects JSON
        if subjects_ranked and src_dir_for_sidecar:
            subjects_path = os.path.join(src_dir_for_sidecar, f"{key}_best_transcription_subjects.json")
            try:
                with open(subjects_path, "w", encoding="utf-8") as f:
                    json.dump({"subjects": [{"label": l, "score": float(s)} for (l,s) in subjects_ranked],
                               "activity_type": activity_type}, f, ensure_ascii=False, indent=2)
                log.info(f"[{key}] Subjects sidecar -> {subjects_path}")
            except Exception as ex:
                log.warning(f"[{key}] Subjects sidecar failed: {ex}")

    t_id = stable_id(key, "transcription")
    source_paths = {
        "json": best_json,
        "csv":  best_csv,
        "rttm": paths.get("rttm"),
        "media": paths.get("media"),
    }

    # Dry-run reporting (rich)
    if dry_run:
        log.info(f"[DRY-RUN] key={key} segs={len(enriched_segments)} utts={len(utterances)} "
                 f"ents={len(entities)} speakers={len(speakers)} "
                 f"subjectsT={[lbl for (lbl,_) in subjects_ranked]} activity={activity_type} "
                 f"json={best_json} csv={best_csv} rttm={paths.get('rttm')} media={paths.get('media')}")
        return {
            "key": key, "segments": len(enriched_segments),
            "utterances": len(utterances), "speakers": len(speakers),
            "entities": len(entities), "subjectsT": [lbl for (lbl,_) in subjects_ranked],
            "activity_type": activity_type
        }

    # Graph Pack export (optional)
    if GRAPH_PACK_DIR:
        try:
            graph_pack_emit_all(GRAPH_PACK_DIR, t_id, key, activity_type,
                                enriched_segments, utterances, speakers, entities,
                                subjects_ranked, seg_subjects, utt_subjects)
        except Exception as ex:
            log.warning(f"[{key}] Graph pack emit failed: {ex}")

    # Ingest
    ingest_record(
        driver=driver,
        t_id=t_id,
        key=key,
        started_at_iso=started_at_iso,
        ended_at_iso=ended_at_iso,
        source_paths=source_paths,
        text=text,
        transcript_emb=transcript_vec,
        segments=enriched_segments,
        entities=entities,
        utterances=utterances,
        speakers=speakers,
        segment_speakers=seg_edges,
        subjects_ranked=subjects_ranked,
        activity_type=activity_type,
        seg_subjects=seg_subjects,
        utt_subjects=utt_subjects,
    )

    return {
        "key": key,
        "segments": len(enriched_segments),
        "utterances": len(utterances),
        "speakers": len(speakers),
        "entities": len(entities),
        "subjectsT": [lbl for (lbl,_) in subjects_ranked],
        "activity_type": activity_type,
    }

# =========================
# Vector search (placeholder to keep CLI stable)
# =========================
def vector_search(driver, text_query: str, target: str = "utterance", top_k: int = 5, include_embedding: bool = False, win_minutes: int = 10):
    raise NotImplementedError("Vector search body unchanged — paste your previous implementation here.")

# =========================
# CLI
# =========================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ingest transcripts with GLiNER subjects, fan-out, and graph-pack export.")
    parser.add_argument("--search", type=str, default=None, help="Run vector search (no ingest).")
    parser.add_argument("--target", type=str, default="utterance", choices=["utterance","segment","transcription"], help="Index to search.")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--win-mins", type=int, default=10)
    parser.add_argument("--include-emb", action="store_true")

    parser.add_argument("--limit", type=int, default=None, help="Process at most N keys on ingest.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Embedding batch size.")
    parser.add_argument("--force", action="store_true", help="Force re-ingest even if transcription already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be ingested (no writes).")
    parser.add_argument("--log-level", type=str, default=LOG_LEVEL_DEFAULT, help="DEBUG/INFO/WARNING/ERROR")
    args = parser.parse_args()

    log.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    driver = neo4j_driver()
    ensure_schema(driver)

    if args.search:
        log.error("Vector search body omitted here for brevity. Paste prior implementation into vector_search().")
        return

    mapping = discover_keys()
    keys = sorted(mapping.keys())
    if args.limit: keys = keys[:args.limit]
    log.info(f"Found {len(keys)} key(s) to consider for ingest/re-ingest.")

    processed = 0; skipped = 0
    for i, key in enumerate(keys, 1):
        paths = mapping[key]
        log.info("===== BEGIN INGEST =====\n"
                 f"  idx={i}/{len(keys)}\n"
                 f"  key={key}\n"
                 f"  json_candidates={len(paths.get('json_all') or [])}\n"
                 f"  csv_candidates={len(paths.get('csv_all') or [])}\n"
                 f"  rttm={paths.get('rttm')}\n"
                 f"  media={paths.get('media')}")
        try:
            t_id = stable_id(key, "transcription")
            if not args.force:
                status = get_ingestion_status(driver, t_id)
                expected_segments, has_rttm = probe_expected(paths)
                do_ingest = should_reingest(status, expected_segments, has_rttm, paths)
                if not do_ingest:
                    skipped += 1
                    log.info(f"[{i}/{len(keys)}] SKIP (up-to-date): {key}")
                    log.info("===== END INGEST ===== key=%s | %s | %s | %s | %s | %s",
                             key, st_discover.summary(), st_load.summary(), st_validate.summary(), st_subjects.summary(), st_ingest.summary())
                    continue

            info = process_key(key, paths, driver, batch_size=args.batch_size, dry_run=args.dry_run)
            if info:
                processed += 1
                log.info(f"[{i}/{len(keys)}] OK: {key} "
                         f"(segments={info['segments']}, utterances={info['utterances']}, "
                         f"speakers={info['speakers']}, entities={info['entities']}, "
                         f"subjectsT={','.join(info.get('subjectsT', []))}, activity={info.get('activity_type')})")
            else:
                log.warning(f"[{i}/{len(keys)}] SKIP: {key} (no data)")
        except Exception as ex:
            log.exception(f"[{i}/{len(keys)}] FAILED: {key} :: {ex}")
        finally:
            log.info("===== END INGEST ===== key=%s | %s | %s | %s | %s | %s",
                     key, st_discover.summary(), st_load.summary(), st_validate.summary(), st_subjects.summary(), st_ingest.summary())

    total_elapsed = time.perf_counter() - GLOBAL_START
    log.info("ALL DONE | elapsed=%.2fs | processed=%d skipped=%d total=%d | %s | %s | %s | %s | %s",
             total_elapsed, processed, skipped, len(keys),
             st_discover.summary(), st_load.summary(), st_validate.summary(), st_subjects.summary(), st_ingest.summary())

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        log.exception("Fatal error: %s", e)
        print("❌ Failed: ingest_transcripts.py", flush=True)
        raise
