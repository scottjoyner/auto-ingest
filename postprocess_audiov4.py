#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ingest transcripts (dashcam/bodycam/audio) into Neo4j with:
- Best-model transcript selection (prefers <model>_transcription.txt JSON)
- Robust CSV fallback
- GLiNER entity extraction (uses existing sidecars if present; else runs GLiNER)
- Subject taxonomy & classification (embeddings + regex blend)
- Speaker/utterance graph wiring when RTTM is present
- Confusion matrices (segment/utterance subjects) written to CSV + logged
- Vector indexes (Neo4j 5.11+) for Segment/Utterance/Transcription embeddings
- Sidecars for all processed data (segments, utterances, speakers, entities, subjects, edges, summary, narrative)

Usage
-----
# Ingest everything with info logs, read taxonomy, write confmats & sidecars:
./.venv/bin/python3 ingest_transcripts.py \
  --taxonomy /path/to/taxonomy.yml \
  --confmat-out /mnt/8TB_2025/fileserver/audio/_reports \
  --sidecar-pretty --narrative --log-level INFO

# Dry run (no writes), show decisions
./.venv/bin/python3 ingest_transcripts.py --dry-run --log-level INFO

# Force re-ingest even if graph looks complete
./.venv/bin/python3 ingest_transcripts.py --force

# Limit to 50 keys
./.venv/bin/python3 ingest_transcripts.py --limit 50
"""
import os, re, uuid, csv, json, time, hashlib, logging, math, itertools, copy
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

# Optional YAML taxonomy
try:
    import yaml  # pip install pyyaml
    HAVE_YAML = True
except Exception:
    HAVE_YAML = False

# =========================
# Config
# =========================
LOG_LEVEL_DEFAULT = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL_DEFAULT, logging.INFO),
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ingest_transcripts")

# Stage EMA alpha
EMA_ALPHA = float(os.getenv("EMA_ALPHA", "0.2"))

# Default scan roots (env override: SCAN_ROOTS="a,b,c")
DEFAULT_SCAN_ROOTS = [
    "/mnt/8TB_2025/fileserver/dashcam/audio",
    "/mnt/8TB_2025/fileserver/dashcam/transcriptions",
    "/mnt/8TB_2025/fileserver/audio",
    "/mnt/8TB_2025/fileserver/audio/transcriptions",
    "/mnt/8TB_2025/fileserver/bodycam",
]
SCAN_ROOTS = [p.strip() for p in os.getenv("SCAN_ROOTS", ",".join(DEFAULT_SCAN_ROOTS)).split(",") if p.strip()]

# Local timezone used to interpret file keys, then converted to UTC
LOCAL_TZ = os.getenv("LOCAL_TZ", "America/New_York")

# Embedding model
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")  # 384 dims
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))

# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "livelongandprosper")
NEO4J_DB = os.getenv("NEO4J_DB", "neo4j")
NEO4J_ENABLED = bool(NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD)

# Defaults
DEFAULT_BATCH_SIZE = int(os.getenv("EMBED_BATCH", "32"))

# Filenames and patterns (JSON lives in <model>_transcription.txt per new pipeline)
AUDIO_BASE = Path("/mnt/8TB_2025/fileserver/audio")
PAT_TRANS_JSON_TXT   = re.compile(r"_([A-Za-z0-9\-\._]+)_transcription\.txt$", re.IGNORECASE)
PAT_TRANS_CSV        = re.compile(r"_transcription\.csv$", re.IGNORECASE)
PAT_ENTITIES         = re.compile(r"_transcription_(entites|entities)\.csv$", re.IGNORECASE)
PAT_RTTM             = re.compile(r"_speakers\.rttm$", re.IGNORECASE)
PAT_MEDIA            = re.compile(r"\.(wav|mp3|m4a|flac|mp4|mov|mkv)$", re.IGNORECASE)
PAT_SUBJECTS_SIDECAR = re.compile(r"_transcription_subjects\.json$", re.IGNORECASE)

# Model quality preference (left = best). You can override via env MODEL_PREF (comma-separated).
DEFAULT_MODEL_PREF = [
    "large-v3", "large-v2", "large", "turbo",
    "medium.en", "medium",
    "small.en", "small", "base.en", "base", "tiny.en", "tiny",
    "faster-whisper:large-v3", "faster-whisper:large-v2", "faster-whisper:large",
    "faster-whisper:medium", "faster-whisper:small", "faster-whisper:base", "faster-whisper:tiny",
]
MODEL_PREF = [s.strip() for s in os.getenv("MODEL_PREF", ",".join(DEFAULT_MODEL_PREF)).split(",") if s.strip()]

# ========= Subject taxonomy defaults (overridable via --taxonomy) =========
SUBJECT_LABELS: List[str] = [
    "Driving", "PhoneCall", "Errand", "Meeting", "Work", "Travel", "Music",
    "Vehicle", "Finance", "Insurance", "Health", "Shopping", "DevOps",
    "Models", "Data", "Legal", "Personal", "Planning", "Family",
]
SUBJECT_REGEX: Dict[str, re.Pattern] = {
    "Driving":   re.compile(r"\b(exit|merge|turn|traffic|speed|mile|highway|lane|gps|intersection)\b", re.I),
    "PhoneCall": re.compile(r"\b(call|dial|voicemail|ring|phone|speakerphone)\b", re.I),
    "Errand":    re.compile(r"\b(store|checkout|grocery|receipt|errand|pharmacy|pickup|drop[- ]off)\b", re.I),
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
SUBJECT_SYNONYMS: Dict[str, List[str]] = {}
SUBJECT_PROMPTS: Dict[str, str] = {}

# subject scoring weights/thresholds (can be overridden by taxonomy)
SUBJ_W_EMB = float(os.getenv("SUBJ_W_EMB", "0.72"))  # embedding similarity weight
SUBJ_W_RX  = float(os.getenv("SUBJ_W_RX",  "0.28"))  # regex hit weight
SUBJ_THRESHOLD_TRANSCRIPTION = float(os.getenv("SUBJ_THRESHOLD_T", "0.25"))
SUBJ_THRESHOLD_SEGMENT       = float(os.getenv("SUBJ_THRESHOLD_S", "0.30"))
SUBJ_THRESHOLD_UTTERANCE     = float(os.getenv("SUBJ_THRESHOLD_U", "0.30"))

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
    def avg(self) -> float:
        return (self.total / self.count) if self.count else 0.0
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

# Lazy: only load GLiNER if we actually need it
entity_recognition_classifier = None

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
    base = re.sub(r"_transcription_subjects$", "", base, flags=re.IGNORECASE)
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
            candidates = [
                data.get("started_at"), data.get("start_time"), data.get("start"),
                (data.get("metadata") or {}).get("started_at"),
            ]
            enders = [
                data.get("ended_at"), data.get("end_time"), data.get("end"),
                (data.get("metadata") or {}).get("ended_at"),
            ]
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

                    abs_start = None
                    abs_end   = None
                    for k in ("AbsoluteStart","AbsStart","absolute_start","StartISO","start_iso","StartEpochMillis","start_epoch_ms"):
                        if row.get(k):
                            abs_start = _parse_any_iso_or_epoch(row.get(k)); break
                    for k in ("AbsoluteEnd","AbsEnd","absolute_end","EndISO","end_iso","EndEpochMillis","end_epoch_ms"):
                        if row.get(k):
                            abs_end = _parse_any_iso_or_epoch(row.get(k)); break

                    seg = {
                        "id": row.get("SegmentId") or row.get("id") or stable_id(path, "seg", str(row_idx), text[:64]),
                        "seek": None,
                        "start": start,
                        "end": end,
                        "text": text,
                        "tokens": [],
                        "words": [],
                        "abs_start": iso(abs_start) if abs_start else None,
                        "abs_end":   iso(abs_end)   if abs_end   else None,
                    }
                    segments.append(seg)
                    if text: full_text.append(text)

                    if abs_start:
                        file_start_abs = min(file_start_abs, abs_start) if file_start_abs else abs_start
                    if abs_end:
                        file_end_abs = max(file_end_abs, abs_end) if file_end_abs else abs_end

            return {
                "text": " ".join(full_text),
                "segments": segments,
                "language": "en",
                "file_started_at": file_start_abs,
                "file_ended_at": file_end_abs
            }
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

def load_subjects_sidecar(path: str) -> List[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)
        ents = j.get("entities") or []
        if isinstance(ents, list):
            out = []
            for e in ents:
                out.append({
                    "id": e.get("id") or stable_id(path, e.get("text",""), e.get("label","")),
                    "text": e.get("text",""),
                    "label": e.get("label",""),
                    "count": int(e.get("count") or 0),
                    "starts": e.get("starts") or [],
                    "ends": e.get("ends") or [],
                    "avg_score": float(e.get("avg_score") or 0.0),
                })
            return out
    except Exception as ex:
        log.warning(f"Failed to parse subjects sidecar {path}: {ex}")
    return []

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
    scored: List[Tuple[int,int,float,float,str]] = []
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
# Entity aggregation / fallback
# =========================
def gliner_extract(text: str) -> List[Dict[str, Any]]:
    global entity_recognition_classifier
    try:
        if entity_recognition_classifier is None:
            from gliner import GLiNER
            log.info("Loading GLiNER (fallback only)…")
            entity_recognition_classifier = GLiNER.from_pretrained("urchade/gliner_large-v2", device=DEVICE)
        ents = entity_recognition_classifier.predict_entities(
            text, ["Person","Place","Event","Date","Subject","Organization","Product","Location"]
        )
        return [{
            "text": e.get("text",""),
            "label": e.get("label",""),
            "score": float(e.get("score", 0.0) or 0.0),
            "start": float(e.get("start", -1) or -1),
            "end": float(e.get("end", -1) or -1),
        } for e in ents]
    except Exception as ex:
        log.warning(f"GLiNER fallback failed: {ex}")
        return []

def aggregate_entities(ents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    bucket: Dict[Tuple[str,str], Dict[str, Any]] = {}
    for e in ents:
        key = (e.get("text","").strip(), e.get("label","").strip())
        if not key[0]:
            continue
        b = bucket.setdefault(key, {"text": key[0], "label": key[1], "count": 0, "starts": [], "ends": [], "scores": []})
        b["count"] += 1
        if "start" in e and e["start"] is not None: b["starts"].append(float(e["start"]))
        if "end"   in e and e["end"]   is not None: b["ends"].append(float(e["end"]))
        if "score" in e and e["score"] is not None: b["scores"].append(float(e["score"]))
    out = []
    for (txt, lbl), b in bucket.items():
        eid = stable_id(txt, lbl)
        avg = float(np.mean(b["scores"])) if b["scores"] else 0.0
        out.append({"id": eid, "text": txt, "label": lbl, "count": b["count"], "starts": b["starts"], "ends": b["ends"], "avg_score": avg})
    return out

# =========================
# Validation / cleaning
# =========================
def validate_and_clean_segments(key: str, segs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str,int]]:
    fixed = []
    stats = {"nonfinite":0, "neg_dur":0, "reordered":0, "empty_txt":0, "kept":0}
    last_end = -math.inf
    for i, s in enumerate(segs):
        try:
            start = float(s.get("start", 0.0))
            end   = float(s.get("end", start))
            if not math.isfinite(start) or not math.isfinite(end):
                stats["nonfinite"] += 1
                continue
            if end < start:
                end = start
                stats["neg_dur"] += 1
            txt = (s.get("text") or "").strip()
            if not txt:
                stats["empty_txt"] += 1
            sid = s.get("id") or stable_id(key, "seg", str(i), txt[:64])
            if start < last_end:
                stats["reordered"] += 1
            last_end = max(last_end, end)
            fixed.append({
                "id": sid,
                "idx": s.get("idx", i),
                "start": start,
                "end": end,
                "text": txt,
                "tokens": s.get("tokens", []),
                "words":  s.get("words", []),
                "abs_start": s.get("abs_start"),
                "abs_end":   s.get("abs_end"),
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
        u_start = chunk[0]["start"]
        u_end   = chunk[-1]["end"]
        u_id = stable_id("utt", spk, f"{u_start:.3f}", f"{u_end:.3f}")
        utterances.append({
            "id": u_id,
            "speaker_label": spk,
            "start": u_start,
            "end": u_end,
            "text": text,
            "segment_id": None,
            "idx": i
        })
    return utterances

def utterances_from_rttm_dominant_segment(rttm: List[Tuple[float,float,str]], segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    utterances = []
    for i, seg in enumerate(segments):
        s_start = float(seg.get("start", 0.0) or 0.0)
        s_end   = float(seg.get("end", s_start) or s_start)
        best_overlap = 0.0
        best_spk = None
        for (a, b, spk) in rttm:
            ov = overlap(s_start, s_end, a, b)
            if ov > best_overlap:
                best_overlap = ov
                best_spk = spk
        spk_label = best_spk or "UNKNOWN"
        u_id = stable_id("utt", spk_label, str(seg.get("id")), f"{s_start:.3f}", f"{s_end:.3f}")
        utterances.append({
            "id": u_id,
            "speaker_label": spk_label,
            "start": s_start,
            "end": s_end,
            "text": seg.get("text", ""),
            "segment_id": seg.get("id"),
            "idx": i
        })
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
# Taxonomy loader & confusion matrix
# =========================
def _load_json_or_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        if path.lower().endswith((".yml", ".yaml")):
            if not HAVE_YAML:
                raise RuntimeError("PyYAML not installed; use JSON or `pip install pyyaml`.")
            return yaml.safe_load(f)
        return json.load(f)

def _compile_patterns(patterns: List[str]) -> re.Pattern:
    if not patterns:
        return re.compile(r"$^", re.I)
    joined = "|".join(f"(?:{p})" for p in patterns)
    return re.compile(joined, re.I)

def apply_taxonomy(tax: Dict[str, Any]) -> None:
    global SUBJECT_LABELS, SUBJECT_REGEX, SUBJECT_SYNONYMS
    global SUBJ_W_EMB, SUBJ_W_RX
    global SUBJ_THRESHOLD_TRANSCRIPTION, SUBJ_THRESHOLD_SEGMENT, SUBJ_THRESHOLD_UTTERANCE

    settings = (tax.get("settings") or {})
    weights  = (settings.get("weights") or {})
    th       = (settings.get("thresholds") or {})

    SUBJ_W_EMB = float(weights.get("emb", SUBJ_W_EMB))
    SUBJ_W_RX  = float(weights.get("rx",  SUBJ_W_RX))

    SUBJ_THRESHOLD_TRANSCRIPTION = float(th.get("transcription", SUBJ_THRESHOLD_TRANSCRIPTION))
    SUBJ_THRESHOLD_SEGMENT       = float(th.get("segment",       SUBJ_THRESHOLD_SEGMENT))
    SUBJ_THRESHOLD_UTTERANCE     = float(th.get("utterance",     SUBJ_THRESHOLD_UTTERANCE))

    subjects = tax.get("subjects") or []
    labels: List[str] = []
    regex_map: Dict[str, re.Pattern] = {}
    synonyms_map: Dict[str, List[str]] = {}

    for item in subjects:
        lbl = str(item.get("label") or "").strip()
        if not lbl:
            continue
        labels.append(lbl)
        pats = item.get("patterns") or []
        regex_map[lbl] = _compile_patterns(pats) if pats else re.compile(r"$^", re.I)
        syns = item.get("synonyms") or []
        synonyms_map[lbl] = [str(s).strip() for s in syns if str(s).strip()]

    if labels:
        SUBJECT_LABELS = labels
        for k in list(SUBJECT_REGEX.keys()):
            if k not in labels:
                SUBJECT_REGEX.pop(k, None)
        for lbl in labels:
            SUBJECT_REGEX[lbl] = regex_map.get(lbl, SUBJECT_REGEX.get(lbl, re.compile(r"$^", re.I)))

    SUBJECT_SYNONYMS = synonyms_map

    log.info(
        "[taxonomy] Loaded: labels=%s | weights={emb=%.2f, rx=%.2f} | thresholds={T=%.2f,S=%.2f,U=%.2f}",
        ",".join(SUBJECT_LABELS), SUBJ_W_EMB, SUBJ_W_RX,
        SUBJ_THRESHOLD_TRANSCRIPTION, SUBJ_THRESHOLD_SEGMENT, SUBJ_THRESHOLD_UTTERANCE
    )

class ConfusionMatrix:
    def __init__(self, labels: List[str], name: str):
        self.labels = labels
        self.index = {lbl: i for i, lbl in enumerate(labels)}
        n = len(labels)
        self.mat = np.zeros((n, n), dtype=np.int64)
        self.name = name

    def update_from_lists(self, label_lists: List[List[str]]):
        for labels in label_lists:
            uniq = sorted(set(l for l in labels if l in self.index))
            for a, b in itertools.combinations_with_replacement(uniq, 2):
                i = self.index[a]; j = self.index[b]
                self.mat[i, j] += 1
                if i != j:
                    self.mat[j, i] += 1

    def to_csv(self, out_path: str):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow([""] + self.labels)
            for i, lbl in enumerate(self.labels):
                row = [lbl] + [int(x) for x in self.mat[i].tolist()]
                w.writerow(row)

    def log_top_pairs(self, top_k: int = 15):
        pairs = []
        n = len(self.labels)
        for i in range(n):
            for j in range(i, n):
                c = int(self.mat[i, j])
                if c > 0:
                    pairs.append((c, self.labels[i], self.labels[j]))
        pairs.sort(reverse=True)
        head = pairs[:top_k]
        pretty = ", ".join([f"{a}&{b}:{c}" for (c, a, b) in head])
        log.info("[confmat:%s] top co-occurrences: %s", self.name, pretty or "(none)")

# =========================
# Subject classification helpers
# =========================
_subject_label_vecs: Optional[List[List[float]]] = None

def _ensure_subject_label_vecs(batch_size: int) -> List[List[float]]:
    global _subject_label_vecs
    if _subject_label_vecs is not None:
        return _subject_label_vecs
    prompts = []
    for lbl in SUBJECT_LABELS:
        syns = SUBJECT_SYNONYMS.get(lbl, [])
        prompt = f"Topic: {lbl}. Synonyms: {', '.join(syns)}." if syns else f"Topic: {lbl}."
        prompts.append(prompt)
    _subject_label_vecs = embed_texts(prompts, batch_size=batch_size)
    return _subject_label_vecs

def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b: return 0.0
    av = np.array(a, dtype=np.float32); bv = np.array(b, dtype=np.float32)
    na = np.linalg.norm(av); nb = np.linalg.norm(bv)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(av, bv) / (na * nb))

def classify_subjects_for_text(
    unit_text: str,
    unit_vec: List[float],
    batch_size: int,
    threshold: float
) -> List[Tuple[str, float]]:
    label_vecs = _ensure_subject_label_vecs(batch_size)
    results: List[Tuple[str, float]] = []
    for i, lbl in enumerate(SUBJECT_LABELS):
        sim = _cosine(unit_vec, label_vecs[i])
        rx_hit = 1.0 if SUBJECT_REGEX.get(lbl, re.compile(r"$^")).search(unit_text or "") else 0.0
        score = SUBJ_W_EMB * sim + SUBJ_W_RX * rx_hit
        if score >= threshold:
            results.append((lbl, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

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
    "CREATE CONSTRAINT subject_id IF NOT EXISTS FOR (sb:Subject) REQUIRE sb.id IS UNIQUE",
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
    if not driver:
        return False
    with driver.session(database=NEO4J_DB) as sess:
        rec = sess.run("MATCH (t:Transcription {id:$id}) RETURN t.id AS id LIMIT 1", id=t_id).single()
        return bool(rec)

def get_ingestion_status(driver, t_id: str) -> Dict[str, Any]:
    if not driver:
        return {"exists": False}
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
    subjectsT: List[Tuple[str,float]],
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
            t.ended_at   = CASE WHEN $ended_at   IS NULL THEN t.ended_at   ELSE datetime($ended_at)   END

        WITH t
        UNWIND $segments AS seg
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

        WITH t
        UNWIND $speakers AS sp
        MERGE (spk:Speaker {id: sp.id})
            ON CREATE SET spk.label = sp.label, spk.key = sp.key, spk.created_at = datetime()
            ON MATCH  SET spk.label = sp.label, spk.key = sp.key, spk.updated_at = datetime()

        WITH t
        UNWIND $utterances AS u
        MERGE (uNode:Utterance {id: u.id})
            ON CREATE SET uNode.idx = u.idx, uNode.created_at = datetime()
            ON MATCH  SET uNode.idx = u.idx, uNode.updated_at = datetime()
        SET uNode.start = u.start, uNode.end = u.end, uNode.text = u.text, uNode.embedding = u.embedding,
            uNode.absolute_start = CASE WHEN u.abs_start IS NULL THEN uNode.absolute_start ELSE datetime(u.abs_start) END,
            uNode.absolute_end   = CASE WHEN u.abs_end   IS NULL THEN uNode.absolute_end   ELSE datetime(u.abs_end) END
        MERGE (t)-[:HAS_UTTERANCE]->(uNode)

        WITH uNode, u
        OPTIONAL MATCH (s:Segment {id: u.segment_id})
        FOREACH (_ IN CASE WHEN s IS NULL THEN [] ELSE [1] END |
            MERGE (uNode)-[:OF_SEGMENT]->(s)
        )

        WITH uNode, u
        OPTIONAL MATCH (spk:Speaker {id: u.speaker_id})
        FOREACH (_ IN CASE WHEN spk IS NULL THEN [] ELSE [1] END |
            MERGE (uNode)-[:SPOKEN_BY]->(spk)
        )

        WITH t
        UNWIND $segment_speakers AS ss
        MATCH (s:Segment {id: ss.segment_id})
        MATCH (spk:Speaker {id: ss.speaker_id})
        MERGE (s)-[r:SPOKEN_BY]->(spk)
        SET r.overlap = coalesce(ss.overlap, 0.0),
            r.proportion = coalesce(ss.proportion, 0.0)
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
        cypher_subjects = """
        MATCH (t:Transcription {id: $t_id})
        WITH t
        UNWIND $subjectsT AS st
          MERGE (sb:Subject {id: st.id})
            ON CREATE SET sb.label = st.label, sb.created_at = datetime()
            ON MATCH  SET sb.label = st.label, sb.updated_at = datetime()
          MERGE (t)-[abt:ABOUT]->(sb)
          SET abt.score = st.score

        WITH t
        UNWIND $seg_subjects AS srec
          MATCH (s:Segment {id: srec.segment_id})
          UNWIND srec.labels AS sbl
            MERGE (sb:Subject {id: sbl.id})
              ON CREATE SET sb.label = sbl.label, sb.created_at = datetime()
              ON MATCH  SET sb.label = sbl.label, sb.updated_at = datetime()
            MERGE (s)-[ab:ABOUT]->(sb)
            SET ab.score = sbl.score

        WITH t
        UNWIND $utt_subjects AS urec
          MATCH (u:Utterance {id: urec.utt_id})
          UNWIND urec.labels AS ubl
            MERGE (sb:Subject {id: ubl.id})
              ON CREATE SET sb.label = ubl.label, sb.created_at = datetime()
              ON MATCH  SET sb.label = ubl.label, sb.updated_at = datetime()
            MERGE (u)-[ab2:ABOUT]->(sb)
            SET ab2.score = ubl.score
        """

        subjectsT_payload = [{"id": stable_id("subject", lbl), "label": lbl, "score": float(sc)} for (lbl,sc) in subjectsT]
        seg_subjects_payload = []
        for sid, pairs in seg_subjects.items():
            seg_subjects_payload.append({
                "segment_id": sid,
                "labels": [{"id": stable_id("subject", lbl), "label": lbl, "score": float(sc)} for (lbl,sc) in pairs]
            })
        utt_subjects_payload = []
        for uid, pairs in utt_subjects.items():
            utt_subjects_payload.append({
                "utt_id": uid,
                "labels": [{"id": stable_id("subject", lbl), "label": lbl, "score": float(sc)} for (lbl,sc) in pairs]
            })

        params = {
            "t_id": t_id,
            "key": key,
            "text": text,
            "transcript_emb": transcript_emb,
            "source_json": source_paths.get("json"),
            "source_csv": source_paths.get("csv"),
            "source_rttm": source_paths.get("rttm"),
            "source_media": source_paths.get("media"),
            "segments": segments,
            "utterances": utterances,
            "speakers": speakers,
            "entities": entities,
            "started_at": started_at_iso,
            "ended_at": ended_at_iso,
            "segment_speakers": segment_speakers,
            "subjectsT": subjectsT_payload,
            "seg_subjects": seg_subjects_payload,
            "utt_subjects": utt_subjects_payload,
        }
        with driver.session(database=NEO4J_DB) as sess:
            sess.run(cypher, **params)
            if entities:
                sess.run(cypher_entities, **params)
            if subjectsT_payload or seg_subjects_payload or utt_subjects_payload:
                sess.run(cypher_subjects, **params)

# =========================
# Sidecar helpers
# =========================
def _choose_sidecar_dir(best_json: Optional[str], best_csv: Optional[str], paths: Dict[str, Any], override: Optional[str]) -> str:
    if override:
        return override
    for p in [best_json, best_csv, paths.get("media"), paths.get("rttm")]:
        if p and os.path.isfile(p):
            return os.path.dirname(p)
    # fallback to AUDIO_BASE/<YYYY>/<MM>/<DD> if key-like info exists
    return None  # caller will handle None

def _write_json(path: str, obj: Any, pretty: bool, overwrite: bool) -> Optional[str]:
    try:
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        if os.path.exists(path) and not overwrite:
            log.info("[sidecar] SKIP exists: %s", path)
            return path
        with open(path, "w", encoding="utf-8") as f:
            if pretty:
                json.dump(obj, f, ensure_ascii=False, indent=2)
            else:
                json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
        log.info("[sidecar] wrote: %s (%.1f KB)", path, os.path.getsize(path)/1024.0)
        return path
    except Exception as ex:
        log.warning("[sidecar] failed: %s :: %s", path, ex)
        return None

def _format_hms(sec: float) -> str:
    try:
        ms = int(round((sec - int(sec)) * 1000))
        h = int(sec) // 3600
        m = (int(sec) % 3600) // 60
        s = int(sec) % 60
        return f"{h:02}:{m:02}:{s:02}.{ms:03}"
    except Exception:
        return "00:00:00.000"

def _build_speaker_narrative(utterances: List[Dict[str, Any]], speakers: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_spk: Dict[str, List[Dict[str, Any]]] = {}
    spk_label_by_id = {s["id"]: s.get("label","") for s in speakers}
    for u in sorted(utterances, key=lambda x: (float(x.get("start",0.0)), x.get("idx",0))):
        sid = u.get("speaker_id")
        lbl = spk_label_by_id.get(sid, u.get("speaker_label","UNKNOWN"))
        by_spk.setdefault(lbl, []).append({
            "t": _format_hms(float(u.get("start",0.0))),
            "text": u.get("text",""),
            "segment_id": u.get("segment_id"),
            "speaker_id": sid,
        })
    timeline = []
    for u in sorted(utterances, key=lambda x: float(x.get("start",0.0))):
        timeline.append({
            "t": _format_hms(float(u.get("start",0.0))),
            "speaker_label": u.get("speaker_label","UNKNOWN"),
            "speaker_id": u.get("speaker_id"),
            "text": u.get("text",""),
            "segment_id": u.get("segment_id"),
        })
    return {"by_speaker": by_spk, "timeline": timeline}

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
            if kind == "json":
                mapping[key]["json_all"].append(path)
            elif kind == "csv":
                mapping[key]["csv_all"].append(path)
            else:
                mapping[key][kind] = path

        for root in SCAN_ROOTS:
            if not os.path.isdir(root):
                continue
            for r, _, files in os.walk(root):
                for name in files:
                    p = os.path.join(r, name)
                    if PAT_TRANS_JSON_TXT.search(name):
                        add("json", p)
                    elif PAT_TRANS_CSV.search(name):
                        add("csv", p)
                    elif PAT_ENTITIES.search(name):
                        add("entities", p)
                    elif PAT_SUBJECTS_SIDECAR.search(name):
                        add("subjects_json", p)
                    elif PAT_RTTM.search(name):
                        add("rttm", p)
                    elif PAT_MEDIA.search(name):
                        add("media", p)
        return mapping

# =========================
# Processing one key
# =========================
def process_key(key: str, paths: Dict[str, Any], driver, batch_size: int, dry_run: bool = False,
                sidecars: bool = True, sidecar_dir_override: Optional[str] = None,
                overwrite_sidecars: bool = True, sidecar_embeddings: bool = False,
                sidecar_pretty: bool = False, write_narrative: bool = False):
    # Choose best JSON; fallback to CSV if needed
    json_paths: List[str] = paths.get("json_all") or []
    csv_paths: List[str] = paths.get("csv_all") or []
    best_json = select_best_json(json_paths, csv_paths)
    data = None
    src_dir_for_sidecar = None
    model_tag = "best"

    if best_json:
        data = load_transcription_json_txt(best_json)
        src_dir_for_sidecar = os.path.dirname(best_json)
        model_tag = extract_model_tag_from_json_txt(best_json) or "best"
    if data is None and csv_paths:
        csv_sorted = sorted(csv_paths, key=lambda p: (0 if is_in_audio_base(p) else 1, -Path(p).stat().st_mtime if Path(p).exists() else 0))
        for c in csv_sorted:
            data = load_transcription_csv(c)
            if data:
                src_dir_for_sidecar = os.path.dirname(c)
                best_csv = c
                break
        else:
            best_csv = None
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
            "id": seg["id"],
            "idx": idx,
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "tokens_count": s_tok_count,
            "embedding": None,
            "abs_start": seg.get("abs_start") or None,
            "abs_end":   seg.get("abs_end")   or None,
        })
        if seg["end"] > max_end:
            max_end = seg["end"]

    file_start_dt = data.get("file_started_at")
    file_end_dt   = data.get("file_ended_at")

    dt0_utc = file_start_dt.astimezone(timezone.utc) if isinstance(file_start_dt, datetime) else parse_key_datetime_utc_from_string(key)
    started_at_iso = iso(dt0_utc) if dt0_utc else None
    ended_at_iso = iso(file_end_dt) if isinstance(file_end_dt, datetime) else (iso(dt0_utc + timedelta(seconds=max_end)) if dt0_utc else None)

    if dt0_utc:
        for s in enriched_segments:
            if not s["abs_start"]:
                s["abs_start"] = iso(dt0_utc + timedelta(seconds=float(s["start"])))
            if not s["abs_end"]:
                s["abs_end"]   = iso(dt0_utc + timedelta(seconds=float(s["end"])))

    # Embeddings
    with TimedStage(st_embed, detail=f"key={key} type=transcript+segments"):
        if seg_texts:
            seg_vecs = embed_texts(seg_texts, batch_size=batch_size)
            for i,v in enumerate(seg_vecs):
                enriched_segments[i]["embedding"] = v
            transcript_vec = embed_long_text_via_segments(seg_texts, batch_size=batch_size)
        else:
            transcript_vec = embed_texts([text], batch_size=batch_size)[0]

    # Entities / Subjects sidecar (existing) or GLiNER fallback
    ents_raw = []
    entities_path = paths.get("entities")
    subjects_sidecar_path = paths.get("subjects_json")

    aggregated_entities = []
    used_existing_subjects = False
    if subjects_sidecar_path and os.path.isfile(subjects_sidecar_path):
        aggregated_entities = load_subjects_sidecar(subjects_sidecar_path)
        used_existing_subjects = True
        log.info(f"[{key}] using subjects sidecar: {subjects_sidecar_path} (entities={len(aggregated_entities)})")
    else:
        if entities_path and os.path.isfile(entities_path):
            ents_raw = load_entities_csv(entities_path)
            log.info(f"[{key}] using entities CSV: {entities_path} (rows={len(ents_raw)})")
        elif text.strip():
            ents_raw = gliner_extract(text)
            log.info(f"[{key}] GLiNER extracted entities: {len(ents_raw)}")
        aggregated_entities = aggregate_entities(ents_raw) if ents_raw else []

    # Speakers/utterances
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
            rttm=rttm,
            segments=enriched_segments,
            speaker_map=speaker_map,
            min_proportion=0.05,
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
                if ov > best:
                    best = ov; best_id = s["id"]
            u["segment_id"] = best_id

        for u in utts:
            sp_map = speaker_map.get(u.get("speaker_label", "UNKNOWN")) or speaker_map["UNKNOWN"]
            u["speaker_id"] = sp_map["id"]

        utt_texts = [u["text"] for u in utts]
        if utt_texts:
            utt_vecs = embed_texts(utt_texts, batch_size=batch_size)
            for i,v in enumerate(utt_vecs):
                utts[i]["embedding"] = v

        if dt0_utc:
            for u in utts:
                u["abs_start"] = iso(dt0_utc + timedelta(seconds=float(u["start"])))
                u["abs_end"]   = iso(dt0_utc + timedelta(seconds=float(u["end"])))
        utterances = utts

    # ---------- Subject classification (embedding + regex) ----------
    subjects_ranked = classify_subjects_for_text(
        unit_text=text,
        unit_vec=transcript_vec,
        batch_size=batch_size,
        threshold=SUBJ_THRESHOLD_TRANSCRIPTION,
    )

    seg_subjects: Dict[str, List[Tuple[str,float]]] = {}
    for s in enriched_segments:
        pairs = classify_subjects_for_text(
            unit_text=s["text"],
            unit_vec=s.get("embedding") or [0.0]*EMBED_DIM,
            batch_size=batch_size,
            threshold=SUBJ_THRESHOLD_SEGMENT,
        )
        if pairs:
            seg_subjects[s["id"]] = pairs

    utt_subjects: Dict[str, List[Tuple[str,float]]] = {}
    for u in utterances:
        pairs = classify_subjects_for_text(
            unit_text=u["text"],
            unit_vec=u.get("embedding") or [0.0]*EMBED_DIM,
            batch_size=batch_size,
            threshold=SUBJ_THRESHOLD_UTTERANCE,
        )
        if pairs:
            utt_subjects[u["id"]] = pairs

    # ---------- Confusion-matrix collectors ----------
    seg_subjects_lists = []
    for s in enriched_segments:
        pairs = seg_subjects.get(s["id"], [])
        seg_subjects_lists.append([lbl for (lbl, _sc) in pairs])

    utt_subjects_lists = []
    for u in utterances:
        pairs = utt_subjects.get(u["id"], [])
        utt_subjects_lists.append([lbl for (lbl, _sc) in pairs])

    t_id = stable_id(key, "transcription")
    source_paths = {
        "json": best_json,
        "csv":  best_csv,
        "rttm": paths.get("rttm"),
        "media": paths.get("media"),
    }

    # ---------- Sidecars ----------
    written_sidecars = {}
    if sidecars and not dry_run:
        target_dir = _choose_sidecar_dir(best_json, best_csv, paths, sidecar_dir_override)
        if not target_dir:
            # fallback: use AUDIO_BASE/YYYY/MM/DD if we can infer
            dt_guess = parse_key_datetime_utc_from_string(key)
            if dt_guess:
                target_dir = str(AUDIO_BASE / dt_guess.strftime("%Y") / dt_guess.strftime("%m") / dt_guess.strftime("%d"))
            else:
                target_dir = str(AUDIO_BASE)

        base = f"{key}_{model_tag}_transcription"

        # Copy-safe dumps (omit embeddings unless requested)
        seg_dump = copy.deepcopy(enriched_segments)
        utt_dump = copy.deepcopy(utterances)
        if not sidecar_embeddings:
            for s in seg_dump: s.pop("embedding", None)
            for u in utt_dump: u.pop("embedding", None)

        # Raw entities sidecar (if we actually computed/loaded them)
        if ents_raw:
            written_sidecars["entities_raw"] = _write_json(
                os.path.join(target_dir, f"{base}_entities_raw.json"),
                {"entities": ents_raw}, sidecar_pretty, overwrite_sidecars
            )

        # Aggregated entities sidecar: write if we computed OR if no existing subjects sidecar was present
        if aggregated_entities and not used_existing_subjects:
            written_sidecars["entities_agg_subjects"] = _write_json(
                os.path.join(target_dir, f"{base}_subjects.json"),
                {"entities": aggregated_entities}, sidecar_pretty, overwrite_sidecars
            )
        elif aggregated_entities and used_existing_subjects:
            log.info("[sidecar] using existing subjects sidecar; not overwriting.")

        # Always write normalized aggregated entities JSON too (handy)
        if aggregated_entities:
            written_sidecars["entities_agg"] = _write_json(
                os.path.join(target_dir, f"{base}_entities_agg.json"),
                {"entities": aggregated_entities}, sidecar_pretty, overwrite_sidecars
            )

        written_sidecars["segments"] = _write_json(
            os.path.join(target_dir, f"{base}_segments.json"),
            {"segments": seg_dump}, sidecar_pretty, overwrite_sidecars
        )
        written_sidecars["utterances"] = _write_json(
            os.path.join(target_dir, f"{base}_utterances.json"),
            {"utterances": utt_dump}, sidecar_pretty, overwrite_sidecars
        )
        if speakers:
            written_sidecars["speakers"] = _write_json(
                os.path.join(target_dir, f"{base}_speakers.json"),
                {"speakers": speakers}, sidecar_pretty, overwrite_sidecars
            )
        if seg_edges:
            written_sidecars["segment_speakers"] = _write_json(
                os.path.join(target_dir, f"{base}_segment_speakers.json"),
                {"segment_speakers": seg_edges}, sidecar_pretty, overwrite_sidecars
            )

        # Subjects: transcription / segment / utterance
        written_sidecars["subjectsT"] = _write_json(
            os.path.join(target_dir, f"{base}_seg0_transcript_subjects.json"),
            {"subjects": [{"label": l, "score": float(s)} for (l,s) in subjects_ranked]},
            sidecar_pretty, overwrite_sidecars
        )
        written_sidecars["seg_subjects"] = _write_json(
            os.path.join(target_dir, f"{base}_seg_subjects.json"),
            {"by_segment": {sid: [{"label": l, "score": float(s)} for (l,s) in pairs] for sid, pairs in seg_subjects.items()}},
            sidecar_pretty, overwrite_sidecars
        )
        written_sidecars["utt_subjects"] = _write_json(
            os.path.join(target_dir, f"{base}_utt_subjects.json"),
            {"by_utterance": {uid: [{"label": l, "score": float(s)} for (l,s) in pairs] for uid, pairs in utt_subjects.items()}},
            sidecar_pretty, overwrite_sidecars
        )

        # Summary
        summary = {
            "key": key,
            "model_tag": model_tag,
            "counts": {
                "segments": len(enriched_segments),
                "utterances": len(utterances),
                "speakers": len(speakers),
                "entities_agg": len(aggregated_entities),
            },
            "top_subjects": [{"label": l, "score": float(s)} for (l,s) in subjects_ranked[:5]],
            "sources": source_paths,
            "started_at": started_at_iso,
            "ended_at": ended_at_iso,
        }
        written_sidecars["summary"] = _write_json(
            os.path.join(target_dir, f"{base}_summary.json"),
            summary, sidecar_pretty, overwrite_sidecars
        )

        # Narrative (optional)
        if write_narrative and utterances:
            narrative = _build_speaker_narrative(utterances, speakers)
            written_sidecars["narrative"] = _write_json(
                os.path.join(target_dir, f"{base}_narrative.json"),
                narrative, sidecar_pretty, overwrite_sidecars
            )

        # Index manifest
        manifest = {
            "dir": target_dir,
            "base": base,
            "written": {k: v for k, v in written_sidecars.items() if v}
        }
        written_sidecars["index"] = _write_json(
            os.path.join(target_dir, f"{base}_index.json"),
            manifest, sidecar_pretty, overwrite_sidecars
        )

    # Dry-run reporting
    if dry_run:
        log.info(
            f"[DRY-RUN] key={key} segs={len(enriched_segments)} utts={len(utterances)} "
            f"ents={len(aggregated_entities)} speakers={len(speakers)} "
            f"subjectsT={[lbl for (lbl,_) in subjects_ranked]}"
        )
        return {
            "key": key,
            "segments": len(enriched_segments),
            "utterances": len(utterances),
            "speakers": len(speakers),
            "entities": len(aggregated_entities),
            "subjectsT": [lbl for (lbl,_) in subjects_ranked],
            "activity_type": subjects_ranked[0][0] if subjects_ranked else None,
            "seg_subjects_lists": seg_subjects_lists,
            "utt_subjects_lists": utt_subjects_lists,
        }

    # Ingest to Neo4j
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
        entities=aggregated_entities,
        utterances=utterances,
        speakers=speakers,
        segment_speakers=seg_edges,
        subjectsT=subjects_ranked,
        seg_subjects=seg_subjects,
        utt_subjects=utt_subjects,
    )

    return {
        "key": key,
        "segments": len(enriched_segments),
        "utterances": len(utterances),
        "speakers": len(speakers),
        "entities": len(aggregated_entities),
        "subjectsT": [lbl for (lbl,_) in subjects_ranked],
        "activity_type": subjects_ranked[0][0] if subjects_ranked else None,
        "seg_subjects_lists": seg_subjects_lists,
        "utt_subjects_lists": utt_subjects_lists,
    }

# =========================
# Vector search (placeholder)
# =========================
def vector_search(driver, text_query: str, target: str = "utterance", top_k: int = 5, include_embedding: bool = False, win_minutes: int = 10):
    raise NotImplementedError("Vector search body unchanged — add your previous implementation here if needed.")

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
# CLI
# =========================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ingest transcripts with best-model selection, taxonomy subjects, GLiNER, validation, sidecars, and rich logging.")
    # Search mode (optional)
    parser.add_argument("--search", type=str, default=None, help="Run vector search (no ingest).")
    parser.add_argument("--target", type=str, default="utterance", choices=["utterance","segment","transcription"], help="Index to search.")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--win-mins", type=int, default=10)
    parser.add_argument("--include-emb", action="store_true")

    # Ingest mode
    parser.add_argument("--limit", type=int, default=None, help="Process at most N keys on ingest.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Embedding batch size.")
    parser.add_argument("--force", action="store_true", help="Force re-ingest even if transcription already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be ingested (no writes).")

    # Logging & taxonomy/confusion matrices
    parser.add_argument("--log-level", type=str, default=LOG_LEVEL_DEFAULT, help="DEBUG/INFO/WARNING/ERROR")
    parser.add_argument("--taxonomy", type=str, default=None, help="Path to taxonomy YAML/JSON to override subject labels/regex/weights/thresholds.")
    parser.add_argument("--confmat-out", type=str, default="", help="Directory to write confusion matrices as CSV (segment/utterance).")
    parser.add_argument("--confmat-topk", type=int, default=15, help="Top co-occurring subject pairs to log.")

    # Sidecars
    parser.add_argument("--no-sidecars", action="store_true", help="Disable writing sidecars.")
    parser.add_argument("--sidecar-dir", type=str, default=None, help="Override directory for sidecars (default: alongside best JSON/CSV).")
    parser.add_argument("--overwrite-sidecars", action="store_true", help="Overwrite existing sidecars (default: overwrite).")
    parser.add_argument("--sidecar-embeddings", action="store_true", help="Include embeddings inside sidecars (large).")
    parser.add_argument("--sidecar-pretty", action="store_true", help="Pretty-print sidecar JSON.")
    parser.add_argument("--narrative", action="store_true", help="Emit speaker narrative sidecar.")

    args = parser.parse_args()
    log.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    # Load taxonomy (optional)
    if args.taxonomy:
        try:
            tax = _load_json_or_yaml(args.taxonomy)
            apply_taxonomy(tax)
        except Exception as ex:
            log.exception("Failed to load taxonomy '%s': %s", args.taxonomy, ex)
            return

    # Init confusion matrices
    confmat_seg = ConfusionMatrix(SUBJECT_LABELS, name="segment")
    confmat_utt = ConfusionMatrix(SUBJECT_LABELS, name="utterance")

    driver = neo4j_driver()
    ensure_schema(driver)

    if args.search:
        log.error("Vector search body omitted here. Paste your previous implementation into vector_search().")
        return

    mapping = discover_keys()
    keys = sorted(mapping.keys())
    if args.limit:
        keys = keys[:args.limit]
    log.info(f"Found {len(keys)} key(s) to consider for ingest/re-ingest.")

    processed = 0
    skipped = 0
    for i, key in enumerate(keys, 1):
        paths = mapping[key]
        log.info("===== BEGIN INGEST =====\n"
                 f"  idx={i}/{len(keys)}\n"
                 f"  key={key}\n"
                 f"  json_candidates={len(paths.get('json_all') or [])}\n"
                 f"  csv_candidates={len(paths.get('csv_all') or [])}\n"
                 f"  rttm={paths.get('rttm')}\n"
                 f"  media={paths.get('media')}\n"
                 f"  entities_csv={paths.get('entities')}\n"
                 f"  subjects_json={paths.get('subjects_json')}")
        try:
            t_id = stable_id(key, "transcription")
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
                key, paths, driver, batch_size=args.batch_size, dry_run=args.dry_run,
                sidecars=(not args.no_sidecars), sidecar_dir_override=args.sidecar_dir,
                overwrite_sidecars=(True if args.overwrite_sidecars else True),  # default overwrite True
                sidecar_embeddings=args.sidecar_embeddings, sidecar_pretty=args.sidecar_pretty,
                write_narrative=args.narrative
            )
            if info:
                processed += 1
                log.info(f"[{i}/{len(keys)}] OK: {key} "
                         f"(segments={info['segments']}, utterances={info['utterances']}, "
                         f"speakers={info['speakers']}, entities={info['entities']}, "
                         f"subjectsT={info.get('subjectsT')})")
                confmat_seg.update_from_lists(info.get("seg_subjects_lists") or [])
                confmat_utt.update_from_lists(info.get("utt_subjects_lists") or [])
            else:
                log.warning(f"[{i}/{len(keys)}] SKIP: {key} (no data)")
        except Exception as ex:
            log.exception(f"[{i}/{len(keys)}] FAILED: {key} :: {ex}")
        finally:
            log.info("===== END INGEST ===== key=%s | %s | %s | %s | %s",
                     key, st_discover.summary(), st_load.summary(), st_validate.summary(), st_ingest.summary())

    if args.confmat_out:
        try:
            seg_csv = str(Path(args.confmat_out) / "confusion_matrix_segments.csv")
            utt_csv = str(Path(args.confmat_out) / "confusion_matrix_utterances.csv")
            confmat_seg.to_csv(seg_csv)
            confmat_utt.to_csv(utt_csv)
            log.info("[confmat] CSVs written: %s , %s", seg_csv, utt_csv)
        except Exception as ex:
            log.warning("[confmat] failed to write CSVs: %s", ex)

    confmat_seg.log_top_pairs(top_k=args.confmat_topk)
    confmat_utt.log_top_pairs(top_k=args.confmat_topk)

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
        print("❌ Failed: ingest_transcripts.py", flush=True)
        raise
