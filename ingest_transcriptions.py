#!/usr/bin/env python3
import os, re, uuid, csv, json, time, hashlib, logging
from datetime import datetime, timedelta, timezone
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
# Config
# =========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

AUDIO_RTTM_DIR = "/mnt/8TB_2025/fileserver/dashcam/audio"
TRANSCRIPTION_DIR = "/mnt/8TB_2025/fileserver/dashcam/transcriptions"

# Filenames we consider as sources
PAT_TRANS_JSON = re.compile(r"_medium_transcription\.(json|txt)$", re.IGNORECASE)
PAT_TRANS_CSV  = re.compile(r"_medium_transcription\.csv$", re.IGNORECASE)
PAT_ENTITIES   = re.compile(r"_medium_transcription_(entites|entities)\.csv$", re.IGNORECASE)
PAT_RTTM       = re.compile(r"_speakers\.rttm$", re.IGNORECASE)

# Local timezone used to interpret file keys, then converted to UTC
LOCAL_TZ = os.getenv("LOCAL_TZ", "America/New_York")

# Embedding model
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dims
EMBED_DIM = 384

# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "livelongandprosper")
NEO4J_DB = os.getenv("NEO4J_DB", "neo4j")
NEO4J_ENABLED = bool(NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD)

# Defaults
DEFAULT_BATCH_SIZE = int(os.getenv("EMBED_BATCH", "32"))

# Filenames we consider as sources
# Accept any model name for JSON/TXT (e.g., _large-v3_transcription.txt)
PAT_TRANS_JSON = re.compile(r"_([A-Za-z0-9\-\._]+)_transcription\.(json|txt)$", re.IGNORECASE)
# Merged CSV produced by whisper_audio_chunked: <stem>_transcription.csv
PAT_TRANS_CSV  = re.compile(r"_transcription\.csv$", re.IGNORECASE)
# Entities may be produced with or without the model tag
PAT_ENTITIES   = re.compile(r"_transcription_(entites|entities)\.csv$", re.IGNORECASE)
PAT_RTTM       = re.compile(r"_speakers\.rttm$", re.IGNORECASE)

# =========================
# Models (loaded once)
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass

logging.info(f"Loading embedding model on {DEVICE}…")
emb_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
emb_model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(DEVICE).eval()

# Lazy: only load GLiNER if we actually need it
entity_recognition_classifier = None

# =========================
# Utilities
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

def embed_texts(texts: List[str], batch_size: int = DEFAULT_BATCH_SIZE, max_length: int = 512) -> List[List[float]]:
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

def embed_long_text_via_segments(segments: List[str]) -> List[float]:
    if not segments:
        return [0.0] * EMBED_DIM
    seg_vecs = embed_texts(segments)
    arr = np.array(seg_vecs, dtype=np.float32)
    vec = arr.mean(axis=0)
    n = np.linalg.norm(vec)
    if n > 0:
        vec = vec / n
    return vec.tolist()

def file_key_from_name(name: str) -> str:
    base = os.path.basename(name)
    base = re.sub(r"\.(json|txt|csv|rttm)$", "", base, flags=re.IGNORECASE)
    # Strip trailing tags in flexible order:
    # _<model>_transcription(_entities|_entites)?
    base = re.sub(r"_([A-Za-z0-9\-\._]+)_transcription(_(entites|entities))?$", "", base, flags=re.IGNORECASE)
    # merged CSV naming: _transcription
    base = re.sub(r"_transcription(_(entites|entities))?$", "", base, flags=re.IGNORECASE)
    # diarization
    base = re.sub(r"_speakers$", "", base, flags=re.IGNORECASE)
    return base


def parse_key_datetime_utc(key: str) -> Optional[datetime]:
    """
    Expect keys like 2025_0314_171532  (YYYY_MMDD_HHMMSS)
    Interpret as LOCAL_TZ and convert to UTC.
    """
    try:
        dt_local = datetime.strptime(key, "%Y_%m%d_%H%M%S")
        if ZoneInfo:
            tz = ZoneInfo(LOCAL_TZ)
            dt_local = dt_local.replace(tzinfo=tz)
            return dt_local.astimezone(timezone.utc)
        else:
            # Fallback: treat as UTC if zoneinfo not available
            return dt_local.replace(tzinfo=timezone.utc)
    except Exception:
        return None

def iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.astimezone(timezone.utc).isoformat() if dt else None

# =========================
# Loaders
# =========================
def load_transcription_json_or_txt(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"text": raw, "segments": []}
        return data
    except Exception as ex:
        logging.warning(f"Failed to parse JSON/TXT transcript {path}: {ex}")
        return None

def load_transcription_csv(path: str) -> Optional[Dict[str, Any]]:
    try:
        segments, full_text = [], []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    idx = int(row.get("SegmentIndex", len(segments)))
                except:
                    idx = len(segments)
                start = float(row.get("StartTime", 0.0) or 0.0)
                end   = float(row.get("EndTime",   0.0) or 0.0)
                text  = (row.get("Text") or "").strip()
                segments.append({"id": idx, "seek": None, "start": start, "end": end, "text": text, "tokens": [], "words": []})
                if text: full_text.append(text)
        return {"text": " ".join(full_text), "segments": segments, "language": "en"}
    except Exception as ex:
        logging.warning(f"Failed to parse CSV transcript {path}: {ex}")
        return None

def load_entities_csv(path: str) -> List[Dict[str, Any]]:
    ents = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    start = float(row.get("start", 0.0) or 0.0)
                except:
                    start = 0.0
                try:
                    end = float(row.get("end", 0.0) or 0.0)
                except:
                    end = 0.0
                ents.append({
                    "text": (row.get("text") or "").strip(),
                    "label": (row.get("label") or "").strip(),
                    "score": float(row.get("score", 0.0) or 0.0),
                    "start": start,
                    "end": end,
                })
    except Exception as ex:
        logging.warning(f"Failed to parse entities CSV {path}: {ex}")
    return ents

def load_rttm(path: str) -> List[Tuple[float,float,str]]:
    intervals = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 9:
                    continue
                try:
                    start = float(parts[3]); dur = float(parts[4]); end = start + dur
                    spk = parts[7]
                    intervals.append((start, end, spk))
                except Exception:
                    continue
        intervals.sort(key=lambda x: x[0])
    except Exception as ex:
        logging.warning(f"Failed to read RTTM {path}: {ex}")
    return intervals

# =========================
# Entity aggregation / fallback
# =========================
def gliner_extract(text: str) -> List[Dict[str, Any]]:
    global entity_recognition_classifier
    try:
        if entity_recognition_classifier is None:
            from gliner import GLiNER
            logging.info("Loading GLiNER (fallback only)…")
            entity_recognition_classifier = GLiNER.from_pretrained("urchade/gliner_large-v2", device=DEVICE)
        ents = entity_recognition_classifier.predict_entities(text, ["Person","Place","Event","Date","Subject"])
        return [{
            "text": e.get("text",""),
            "label": e.get("label",""),
            "score": float(e.get("score", 0.0) or 0.0),
            "start": float(e.get("start", -1) or -1),
            "end": float(e.get("end", -1) or -1),
        } for e in ents]
    except Exception as ex:
        logging.warning(f"GLiNER fallback failed: {ex}")
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

def save_subjects_file(subjects_path: str, entities: List[Dict[str, Any]]):
    with open(subjects_path, "w", encoding="utf-8") as f:
        json.dump({"entities": entities}, f, ensure_ascii=False, indent=2)

# =========================
# Speaker overlap → Utterances
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
    for seg in segments:
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
            "idx": int(seg.get("idx", len(utterances)))
        })
    return utterances

def _sum_overlap(seg_start: float, seg_end: float, rttm: List[Tuple[float,float,str]]) -> Dict[str, float]:
    """Total overlapped seconds with each speaker for this segment window."""
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
    """
    For each segment, compute:
      - dominant speaker (by max overlap) → {'speaker_label','speaker_id'}
      - mixture edges for speakers with >= min_proportion of overlap
    Returns:
      seg_best: { seg_id: {'speaker_label': ..., 'speaker_id': ...} }
      seg_edges: [ {'segment_id','speaker_id','label','overlap','proportion'} ... ]
    """
    seg_best: Dict[str, Dict[str, Any]] = {}
    seg_edges: List[Dict[str, Any]] = []

    for s in segments:
        s_start = float(s.get("start", 0.0) or 0.0)
        s_end   = float(s.get("end", s_start) or s_start)
        sid     = s["id"]

        totals = _sum_overlap(s_start, s_end, rttm)
        total_ov = sum(totals.values()) or 0.0

        # dominant
        if totals:
            best_lbl = max(totals.items(), key=lambda kv: kv[1])[0]
        else:
            best_lbl = "UNKNOWN"

        sp = speaker_map.get(best_lbl)
        best_id = sp["id"] if sp else speaker_map.get("UNKNOWN", {}).get("id")
        seg_best[sid] = {"speaker_label": best_lbl, "speaker_id": best_id}

        # mixture edges
        if total_ov > 0.0:
            for lbl, ov in totals.items():
                prop = ov / total_ov
                if prop >= min_proportion:
                    spx = speaker_map.get(lbl)
                    spx_id = spx["id"] if spx else speaker_map.get("UNKNOWN", {}).get("id")
                    seg_edges.append({
                        "segment_id": sid,
                        "speaker_id": spx_id,
                        "label": lbl,
                        "overlap": float(ov),
                        "proportion": float(prop),
                    })

    return seg_best, seg_edges

# =========================
# Neo4j
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
    "CREATE INDEX transcription_key IF NOT EXISTS FOR (t:Transcription) ON (t.key)",
    "CREATE INDEX phonelog_timestamp IF NOT EXISTS FOR (p:PhoneLog) ON (p.timestamp)",
    "CREATE INDEX phonelog_user IF NOT EXISTS FOR (p:PhoneLog) ON (p.user_id)",
    "CREATE INDEX phonelog_epoch IF NOT EXISTS FOR (p:PhoneLog) ON (p.epoch_millis)",
    "CREATE INDEX frame_key IF NOT EXISTS FOR (f:Frame) ON (f.key)",
    "CREATE INDEX frame_key_frame IF NOT EXISTS FOR (f:Frame) ON (f.key, f.frame)"
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
        logging.info("Neo4j not configured; skipping schema setup.")
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
    logging.info("Neo4j schema ensured (constraints + vector indexes + btree indexes).")

def already_ingested(driver, t_id: str) -> bool:
    if not driver:
        return False
    with driver.session(database=NEO4J_DB) as sess:
        rec = sess.run("MATCH (t:Transcription {id:$id}) RETURN t.id AS id LIMIT 1", id=t_id).single()
        return bool(rec)

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
    segment_speakers: List[Dict[str, Any]]
):
    if not driver:
        logging.info("Neo4j not configured; skipping ingestion.")
        return

    cypher = """
    MERGE (t:Transcription {id: $t_id})
    ON CREATE SET t.key = $key, t.created_at = datetime()
    ON MATCH  SET t.key = $key, t.updated_at = datetime()
    SET t.text = $text, t.embedding = $transcript_emb,
        t.source_json = $source_json, t.source_csv = $source_csv, t.source_rttm = $source_rttm,
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
        s.speaker_label  = seg.speaker_label,               // NEW: dominant speaker label on Segment
        s.speaker_id     = seg.speaker_id                   // NEW: dominant speaker id on Segment
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
        uNode.absolute_end   = CASE WHEN u.abs_end   IS NULL THEN uNode.absolute_end   ELSE datetime(u.abs_end)   END
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

    // ---- NEW: Segment speaker mixture edges ----
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

    params = {
        "t_id": t_id,
        "key": key,
        "text": text,
        "transcript_emb": transcript_emb,
        "source_json": source_paths.get("json"),
        "source_csv": source_paths.get("csv"),
        "source_rttm": source_paths.get("rttm"),
        "segments": segments,
        "utterances": utterances,
        "speakers": speakers,
        "entities": entities,
        "started_at": started_at_iso,
        "ended_at": ended_at_iso,
        "segment_speakers": segment_speakers,
    }
    with driver.session(database=NEO4J_DB) as sess:
        sess.run(cypher, **params)
        if entities:
            sess.run(cypher_entities, **params)

# =========================
# Discovery
# =========================
def discover_keys() -> Dict[str, Dict[str,str]]:
    mapping: Dict[str, Dict[str,str]] = {}

    # Recurse transcription tree
    if os.path.isdir(TRANSCRIPTION_DIR):
        for root, _, files in os.walk(TRANSCRIPTION_DIR):
            for name in files:
                p = os.path.join(root, name)
                if PAT_TRANS_JSON.search(name):
                    key = file_key_from_name(name); mapping.setdefault(key, {})["json"] = p
                elif PAT_TRANS_CSV.search(name):
                    key = file_key_from_name(name); mapping.setdefault(key, {})["csv"] = p
                elif PAT_ENTITIES.search(name):
                    key = file_key_from_name(name); mapping.setdefault(key, {})["entities"] = p

    # Recurse diarization tree (rttm lives under AUDIO_RTTM_DIR, possibly flat or mirrored)
    if os.path.isdir(AUDIO_RTTM_DIR):
        for root, _, files in os.walk(AUDIO_RTTM_DIR):
            for name in files:
                if not PAT_RTTM.search(name): 
                    continue
                p = os.path.join(root, name)
                key = file_key_from_name(name)
                mapping.setdefault(key, {})["rttm"] = p

    return mapping

# =========================
# Processing one key
# =========================
def process_key(key: str, paths: Dict[str,str], driver, batch_size: int):
    # Load transcription
    data = None
    if "json" in paths:
        data = load_transcription_json_or_txt(paths["json"])
    if data is None and "csv" in paths:
        data = load_transcription_csv(paths["csv"])
    if data is None:
        logging.warning(f"[{key}] No transcription data found; skipping.")
        return None

    text = data.get("text","") or ""
    segments_raw = data.get("segments", []) or []

    # Prepare segments
    seg_texts = []
    enriched_segments = []
    max_end = 0.0
    for idx, seg in enumerate(segments_raw):
        s_text = seg.get("text","") or ""
        s_start = float(seg.get("start", 0.0) or 0.0)
        s_end   = float(seg.get("end", s_start) or s_start)
        s_tokens = seg.get("tokens", [])
        s_tok_count = len(s_tokens) if isinstance(s_tokens, (list,tuple)) else int(s_tokens or 0)
        seg_id = stable_id(key, "seg", str(idx), s_text[:64])
        seg_texts.append(s_text)
        enriched_segments.append({
            "id": seg_id,
            "idx": idx,
            "start": s_start,
            "end": s_end,
            "text": s_text,
            "tokens_count": s_tok_count,
            "embedding": None,  # to fill later
            "abs_start": None,  # to fill later
            "abs_end":   None,
        })
        if s_end > max_end:
            max_end = s_end

    # Absolute time anchors from key
    dt0_utc = parse_key_datetime_utc(key)
    started_at_iso = iso(dt0_utc)
    ended_at_iso = iso(dt0_utc + timedelta(seconds=max_end)) if dt0_utc else None

    # Compute absolute start/end for segments
    if dt0_utc:
        for s in enriched_segments:
            s["abs_start"] = iso(dt0_utc + timedelta(seconds=float(s["start"])))
            s["abs_end"]   = iso(dt0_utc + timedelta(seconds=float(s["end"])))

    # Embeddings
    transcript_vec = embed_long_text_via_segments(seg_texts) if seg_texts else embed_texts([text], batch_size=batch_size)[0]
    if seg_texts:
        seg_vecs = embed_texts(seg_texts, batch_size=batch_size)
        for i,v in enumerate(seg_vecs):
            enriched_segments[i]["embedding"] = v

    # Entities (prefer CSV; fallback to GLiNER)
    ents_raw = []
    if "entities" in paths and os.path.isfile(paths["entities"]):
        ents_raw = load_entities_csv(paths["entities"])
    elif text.strip():
        ents_raw = gliner_extract(text)  # lazy load

    entities = aggregate_entities(ents_raw) if ents_raw else []

    # Speakers → Utterances
    utterances = []
    speakers = []
    speaker_map: Dict[str, Dict[str,str]] = {}

    if "rttm" in paths and os.path.isfile(paths["rttm"]):
        rttm = load_rttm(paths["rttm"])
        labels = sorted(set([spk for _,_,spk in rttm]))

        for lbl in labels:
            sp_id = stable_id(key, "spk", lbl)
            speakers.append({"id": sp_id, "label": lbl, "key": key})
            speaker_map[lbl] = {"id": sp_id, "label": lbl}

        # Ensure UNKNOWN exists
        if "UNKNOWN" not in speaker_map:
            unknown_id = stable_id(key, "spk", "UNKNOWN")
            speakers.append({"id": unknown_id, "label": "UNKNOWN", "key": key})
            speaker_map["UNKNOWN"] = {"id": unknown_id, "label": "UNKNOWN"}

        # ---- NEW: assign speakers to SEGMENTS (dominant + mixture) ----
        seg_best, seg_edges = compute_segment_speaker_overlaps(
            rttm=rttm,
            segments=enriched_segments,
            speaker_map=speaker_map,
            min_proportion=0.05,   # keep edges with >=5% overlap
        )

        # Annotate segments with dominant speaker
        for s in enriched_segments:
            best = seg_best.get(s["id"], {})
            s["speaker_label"] = best.get("speaker_label", "UNKNOWN")
            s["speaker_id"] = best.get("speaker_id", speaker_map["UNKNOWN"]["id"])

        # Utterances (kept from your original logic)
        utts = utterances_from_rttm_with_words(rttm, segments_raw)
        if not utts:
            utts = utterances_from_rttm_dominant_segment(rttm, enriched_segments)

        # attach best-overlap segment_id
        for u in utts:
            best, best_id = 0.0, None
            for s in enriched_segments:
                ov = overlap(u["start"], u["end"], s["start"], s["end"])
                if ov > best:
                    best = ov; best_id = s["id"]
            u["segment_id"] = best_id

        # speaker_id resolution for utterances
        for u in utts:
            sp_map = speaker_map.get(u.get("speaker_label", "UNKNOWN")) or speaker_map["UNKNOWN"]
            u["speaker_id"] = sp_map["id"]

        # embeddings for utterances
        utt_texts = [u["text"] for u in utts]
        if utt_texts:
            utt_vecs = embed_texts(utt_texts, batch_size=batch_size)
            for i,v in enumerate(utt_vecs):
                utts[i]["embedding"] = v

        # absolute times for utterances
        if dt0_utc:
            for u in utts:
                u["abs_start"] = iso(dt0_utc + timedelta(seconds=float(u["start"])))
                u["abs_end"]   = iso(dt0_utc + timedelta(seconds=float(u["end"])))

        utterances = utts
    else:
        seg_edges = []  # no RTTM → no segment speaker edges


    # subjects sidecar
    if entities:
        subjects_path = os.path.join(TRANSCRIPTION_DIR, f"{key}_medium_transcription_subjects.json")
        save_subjects_file(subjects_path, entities)
        logging.info(f"[{key}] Subjects saved -> {subjects_path}")

    # Transcription id
    t_id = stable_id(key, "transcription")

    # Ingest
    source_paths = {
        "json": paths.get("json"),
        "csv":  paths.get("csv"),
        "rttm": paths.get("rttm"),
    }
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
    )

    return {
        "key": key,
        "segments": len(enriched_segments),
        "utterances": len(utterances),
        "speakers": len(speakers),
        "entities": len(entities),
    }

# =========================
# Vector search (with location join)
# =========================
def vector_search(
    driver,
    text_query: str,
    target: str = "utterance",
    top_k: int = 5,
    include_embedding: bool = False,
    win_minutes: int = 10,
):
    """
    target ∈ {"utterance","segment","transcription"}
    Returns enriched rows with nearest PhoneLog or (fallback) Frame metadata.

    Priority:
      1) PhoneLog within a numeric epoch window (fast).
      2) Frame chosen by relative position in the video (no timestamps needed on Frame).
      Timestamp is always estimated when possible (never None in most cases).
    """
    qvec = embed_texts([text_query])[0]

    if target == "utterance":
        index_name = "utterance_embedding_index"
        cypher = f"""
        CALL db.index.vector.queryNodes('{index_name}', $k, $qvec)
          YIELD node, score
        WHERE 'Utterance' IN labels(node)
        MATCH (t:Transcription)-[:HAS_UTTERANCE]->(u:Utterance)
        WHERE u = node
        WITH node, score, t, u,
             coalesce(u.absolute_start, t.started_at) AS ts_start,
             coalesce(u.absolute_end,   t.ended_at)   AS ts_end,
             toInteger($win) * 60000 AS win_ms
        WITH node, score, t, u, win_ms,
             CASE WHEN ts_start IS NULL THEN NULL ELSE ts_start.epochMillis END AS ts_start_ms,
             CASE WHEN ts_end   IS NULL THEN NULL ELSE ts_end.epochMillis   END AS ts_end_ms,
             CASE WHEN t.started_at IS NULL THEN NULL ELSE t.started_at.epochMillis END AS t0_ms,
             CASE WHEN t.ended_at   IS NULL THEN NULL ELSE t.ended_at.epochMillis   END AS t1_ms
        WITH node, score, t, u, win_ms, ts_start_ms, ts_end_ms, t0_ms, t1_ms,
             CASE
               WHEN ts_start_ms IS NOT NULL AND ts_end_ms IS NOT NULL
                 THEN toInteger((ts_start_ms + ts_end_ms) / 2)
               WHEN t0_ms IS NOT NULL
                 THEN t0_ms + toInteger( ((coalesce(toFloat(u.start), toFloat(u.end), 0.0) + coalesce(toFloat(u.end), toFloat(u.start), 0.0)) / 2.0) * 1000 )
               WHEN t0_ms IS NOT NULL AND t1_ms IS NOT NULL
                 THEN toInteger((t0_ms + t1_ms) / 2)
               ELSE NULL
             END AS ts_mid_ms
        WITH node, score, t, u, win_ms, ts_start_ms, ts_end_ms, ts_mid_ms, t0_ms, t1_ms,
             coalesce(ts_start_ms, ts_mid_ms) AS left_base,
             coalesce(ts_end_ms,   ts_mid_ms) AS right_base
        OPTIONAL MATCH (pl:PhoneLog)
          WHERE ts_mid_ms IS NOT NULL
            AND pl.epoch_millis >= left_base  - win_ms
            AND pl.epoch_millis <= right_base + win_ms
        WITH node, score, t, u, ts_mid_ms, t0_ms, t1_ms, pl
        ORDER BY CASE
                   WHEN pl IS NULL OR ts_mid_ms IS NULL THEN 9223372036854775807
                   ELSE abs(pl.epoch_millis - ts_mid_ms)
                 END ASC
        WITH node, score, t, u, ts_mid_ms, t0_ms, t1_ms, collect(pl)[0] AS nearest_pl
        OPTIONAL MATCH (t)-[:HAS_SEGMENT]->(allSeg)
        WITH node, score, t, u, ts_mid_ms, t0_ms, t1_ms, nearest_pl, max(allSeg.end) AS t_dur_s
        WITH node, score, t, u, ts_mid_ms, nearest_pl, t0_ms, t1_ms, t_dur_s,
             CASE
               WHEN t0_ms IS NOT NULL AND t1_ms IS NOT NULL AND t1_ms <> t0_ms AND ts_mid_ms IS NOT NULL
                 THEN (toFloat(ts_mid_ms - t0_ms) / toFloat(t1_ms - t0_ms))
               ELSE NULL
             END AS pos_time,
             CASE
               WHEN t_dur_s IS NOT NULL AND t_dur_s > 0
                 THEN ((coalesce(toFloat(u.start),0.0) + coalesce(toFloat(u.end),0.0)) / 2.0) / toFloat(t_dur_s)
               ELSE NULL
             END AS pos_seg
        WITH node, score, t, u, ts_mid_ms, nearest_pl,
             CASE
               WHEN pos_time IS NOT NULL THEN pos_time
               WHEN pos_seg  IS NOT NULL THEN pos_seg
               ELSE 0.5
             END AS pos_raw
        WITH node, score, t, u, ts_mid_ms, nearest_pl,
             CASE
               WHEN pos_raw < 0 THEN 0.0
               WHEN pos_raw > 1 THEN 1.0
               ELSE pos_raw
             END AS pos_clamped
        OPTIONAL MATCH (fmax:Frame {{key: t.key}})
        WITH node, score, t, u, ts_mid_ms, nearest_pl, pos_clamped, max(fmax.frame) AS max_frame
        WITH node, score, t, u, ts_mid_ms, nearest_pl,
             CASE
               WHEN max_frame IS NULL THEN NULL
               ELSE toInteger(round(pos_clamped * toFloat(max_frame)))
             END AS target_frame
        OPTIONAL MATCH (f:Frame {{key: t.key}})
        WHERE target_frame IS NOT NULL
        WITH node, score, t, u, ts_mid_ms, nearest_pl, target_frame, f
        ORDER BY abs(f.frame - target_frame) ASC
        WITH node, score, t, u, ts_mid_ms, nearest_pl, collect(f)[0] AS nearest_frame
        WITH node, score, t, u, ts_mid_ms, nearest_pl, nearest_frame,
             coalesce(nearest_pl.latitude,  nearest_frame.lat)  AS latitude,
             coalesce(nearest_pl.longitude, nearest_frame.long) AS longitude,
             nearest_pl.user_id AS user_id,
             CASE
               WHEN nearest_pl.timestamp IS NOT NULL THEN nearest_pl.timestamp
               WHEN nearest_pl.epoch_millis IS NOT NULL THEN datetime({{epochMillis: toInteger(nearest_pl.epoch_millis)}})
               WHEN ts_mid_ms IS NOT NULL THEN datetime({{epochMillis: toInteger(ts_mid_ms)}})
               ELSE NULL
             END AS location_ts,
             CASE
               WHEN nearest_pl IS NOT NULL THEN 'PhoneLog'
               WHEN nearest_frame IS NOT NULL THEN 'Frame'
               WHEN ts_mid_ms IS NOT NULL THEN 'TranscriptionMid'
               ELSE NULL
             END AS location_source,
             CASE WHEN nearest_frame IS NOT NULL THEN nearest_frame.mph ELSE NULL END AS speed_mph
        RETURN
            node.id AS id,
            score,
            node.text AS text,
            node.start AS start,
            node.end AS end,
            t.id AS transcription_id,
            t.key AS file_key,
            t.started_at AS started_at,
            CASE WHEN $include_embedding THEN node.embedding ELSE NULL END AS embedding,
            latitude,
            longitude,
            location_ts,
            user_id,
            location_source,
            speed_mph
        ORDER BY score DESC
        """

    elif target == "segment":
        index_name = "segment_embedding_index"
        cypher = f"""
        CALL db.index.vector.queryNodes('{index_name}', $k, $qvec)
          YIELD node, score
        WHERE 'Segment' IN labels(node)
        MATCH (t:Transcription)-[:HAS_SEGMENT]->(s:Segment)
        WHERE s = node
        WITH node, score, t, s,
             coalesce(s.absolute_start, t.started_at) AS ts_start,
             coalesce(s.absolute_end,   t.ended_at)   AS ts_end,
             toInteger($win) * 60000 AS win_ms
        WITH node, score, t, s, win_ms,
             CASE WHEN ts_start IS NULL THEN NULL ELSE ts_start.epochMillis END AS ts_start_ms,
             CASE WHEN ts_end   IS NULL THEN NULL ELSE ts_end.epochMillis   END AS ts_end_ms,
             CASE WHEN t.started_at IS NULL THEN NULL ELSE t.started_at.epochMillis END AS t0_ms,
             CASE WHEN t.ended_at   IS NULL THEN NULL ELSE t.ended_at.epochMillis   END AS t1_ms
        WITH node, score, t, s, win_ms, ts_start_ms, ts_end_ms, t0_ms, t1_ms,
             CASE
               WHEN ts_start_ms IS NOT NULL AND ts_end_ms IS NOT NULL
                 THEN toInteger((ts_start_ms + ts_end_ms) / 2)
               WHEN t0_ms IS NOT NULL
                 THEN t0_ms + toInteger( ((coalesce(toFloat(s.start), toFloat(s.end), 0.0) + coalesce(toFloat(s.end), toFloat(s.start), 0.0)) / 2.0) * 1000 )
               WHEN t0_ms IS NOT NULL AND t1_ms IS NOT NULL
                 THEN toInteger((t0_ms + t1_ms) / 2)
               ELSE NULL
             END AS ts_mid_ms
        WITH node, score, t, s, win_ms, ts_start_ms, ts_end_ms, ts_mid_ms,
             coalesce(ts_start_ms, ts_mid_ms) AS left_base,
             coalesce(ts_end_ms,   ts_mid_ms) AS right_base
        OPTIONAL MATCH (pl:PhoneLog)
          WHERE ts_mid_ms IS NOT NULL
            AND pl.epoch_millis >= left_base  - win_ms
            AND pl.epoch_millis <= right_base + win_ms
        WITH node, score, t, s, ts_mid_ms, pl
        ORDER BY CASE
                   WHEN pl IS NULL OR ts_mid_ms IS NULL THEN 9223372036854775807
                   ELSE abs(pl.epoch_millis - ts_mid_ms)
                 END ASC
        WITH node, score, t, s, ts_mid_ms, collect(pl)[0] AS nearest_pl
        OPTIONAL MATCH (t)-[:HAS_SEGMENT]->(allSeg)
        WITH node, score, t, s, ts_mid_ms, nearest_pl, max(allSeg.end) AS t_dur_s
        WITH node, score, t, s, ts_mid_ms, nearest_pl, t_dur_s,
             CASE
               WHEN t_dur_s IS NOT NULL AND t_dur_s > 0
                 THEN ((coalesce(toFloat(s.start),0.0) + coalesce(toFloat(s.end),0.0)) / 2.0) / toFloat(t_dur_s)
               ELSE NULL
             END AS pos_seg
        WITH node, score, t, s, ts_mid_ms, nearest_pl,
             coalesce(pos_seg, 0.5) AS pos_clamped
        OPTIONAL MATCH (fmax:Frame {{key: t.key}})
        WITH node, score, t, s, ts_mid_ms, nearest_pl, pos_clamped, max(fmax.frame) AS max_frame
        WITH node, score, t, s, ts_mid_ms, nearest_pl,
             CASE
               WHEN max_frame IS NULL THEN NULL
               ELSE toInteger(round(pos_clamped * toFloat(max_frame)))
             END AS target_frame
        OPTIONAL MATCH (f:Frame {{key: t.key}})
        WHERE target_frame IS NOT NULL
        WITH node, score, t, s, ts_mid_ms, nearest_pl, target_frame, f
        ORDER BY abs(f.frame - target_frame) ASC
        WITH node, score, t, s, ts_mid_ms, nearest_pl, collect(f)[0] AS nearest_frame
        WITH node, score, t, s, ts_mid_ms, nearest_pl, nearest_frame,
             coalesce(nearest_pl.latitude,  nearest_frame.lat)  AS latitude,
             coalesce(nearest_pl.longitude, nearest_frame.long) AS longitude,
             nearest_pl.user_id AS user_id,
             CASE
               WHEN nearest_pl.timestamp IS NOT NULL THEN nearest_pl.timestamp
               WHEN nearest_pl.epoch_millis IS NOT NULL THEN datetime({{epochMillis: toInteger(nearest_pl.epoch_millis)}})
               WHEN ts_mid_ms IS NOT NULL THEN datetime({{epochMillis: toInteger(ts_mid_ms)}})
               ELSE NULL
             END AS location_ts,
             CASE
               WHEN nearest_pl IS NOT NULL THEN 'PhoneLog'
               WHEN nearest_frame IS NOT NULL THEN 'Frame'
               WHEN ts_mid_ms IS NOT NULL THEN 'TranscriptionMid'
               ELSE NULL
             END AS location_source,
             CASE WHEN nearest_frame IS NOT NULL THEN nearest_frame.mph ELSE NULL END AS speed_mph
        RETURN
            node.id AS id,
            score,
            node.text AS text,
            node.start AS start,
            node.end AS end,
            t.id AS transcription_id,
            t.key AS file_key,
            t.started_at AS started_at,
            CASE WHEN $include_embedding THEN node.embedding ELSE NULL END AS embedding,
            latitude,
            longitude,
            location_ts,
            user_id,
            location_source,
            speed_mph
        ORDER BY score DESC
        """

    else:  # "transcription"
        index_name = "transcription_embedding_index"
        cypher = f"""
        CALL db.index.vector.queryNodes('{index_name}', $k, $qvec)
          YIELD node, score
        WHERE 'Transcription' IN labels(node)
        WITH node, score,
             toInteger($win) * 60000 AS win_ms,
             CASE WHEN node.started_at IS NULL THEN NULL ELSE node.started_at.epochMillis END AS ts_start_ms,
             CASE WHEN node.ended_at   IS NULL THEN NULL ELSE node.ended_at.epochMillis   END AS ts_end_ms
        WITH node AS t, score, win_ms, ts_start_ms, ts_end_ms
        WITH t, score, win_ms, ts_start_ms, ts_end_ms,
             CASE
               WHEN ts_start_ms IS NOT NULL AND ts_end_ms IS NOT NULL
                 THEN toInteger((ts_start_ms + ts_end_ms) / 2)
               WHEN ts_start_ms IS NOT NULL
                 THEN ts_start_ms
               WHEN ts_end_ms IS NOT NULL
                 THEN ts_end_ms
               ELSE NULL
             END AS ts_mid_ms
        WITH t, score, win_ms, ts_start_ms, ts_end_ms, ts_mid_ms,
             coalesce(ts_start_ms, ts_mid_ms) AS left_base,
             coalesce(ts_end_ms,   ts_mid_ms) AS right_base
        OPTIONAL MATCH (pl:PhoneLog)
          WHERE ts_mid_ms IS NOT NULL
            AND pl.epoch_millis >= left_base  - win_ms
            AND pl.epoch_millis <= right_base + win_ms
        WITH t, score, ts_mid_ms, pl
        ORDER BY CASE
                   WHEN pl IS NULL OR ts_mid_ms IS NULL THEN 9223372036854775807
                   ELSE abs(pl.epoch_millis - ts_mid_ms)
                 END ASC
        WITH t, score, ts_mid_ms, collect(pl)[0] AS nearest_pl
        OPTIONAL MATCH (fmax:Frame {{key: t.key}})
        WITH t, score, ts_mid_ms, nearest_pl, max(fmax.frame) AS max_frame
        WITH t, score, ts_mid_ms, nearest_pl,
             CASE
               WHEN max_frame IS NULL THEN NULL
               ELSE toInteger(round(0.5 * toFloat(max_frame)))
             END AS target_frame
        OPTIONAL MATCH (f:Frame {{key: t.key}})
        WHERE target_frame IS NOT NULL
        WITH t, score, ts_mid_ms, nearest_pl, target_frame, f
        ORDER BY abs(f.frame - target_frame) ASC
        WITH t, score, ts_mid_ms, nearest_pl, collect(f)[0] AS nearest_frame
        WITH t, score, ts_mid_ms, nearest_pl, nearest_frame,
             coalesce(nearest_pl.latitude,  nearest_frame.lat)  AS latitude,
             coalesce(nearest_pl.longitude, nearest_frame.long) AS longitude,
             nearest_pl.user_id AS user_id,
             CASE
               WHEN nearest_pl.timestamp IS NOT NULL THEN nearest_pl.timestamp
               WHEN nearest_pl.epoch_millis IS NOT NULL THEN datetime({{epochMillis: toInteger(nearest_pl.epoch_millis)}})
               WHEN ts_mid_ms IS NOT NULL THEN datetime({{epochMillis: toInteger(ts_mid_ms)}})
               ELSE NULL
             END AS location_ts,
             CASE
               WHEN nearest_pl IS NOT NULL THEN 'PhoneLog'
               WHEN nearest_frame IS NOT NULL THEN 'Frame'
               WHEN ts_mid_ms IS NOT NULL THEN 'TranscriptionMid'
               ELSE NULL
             END AS location_source,
             CASE WHEN nearest_frame IS NOT NULL THEN nearest_frame.mph ELSE NULL END AS speed_mph
        RETURN
            t.id AS id,
            score,
            t.text AS text,
            null AS start,
            null AS end,
            t.id AS transcription_id,
            t.key AS file_key,
            t.started_at AS started_at,
            CASE WHEN $include_embedding THEN t.embedding ELSE NULL END AS embedding,
            latitude,
            longitude,
            location_ts,
            user_id,
            location_source,
            speed_mph
        ORDER BY score DESC
        """

    with driver.session(database=NEO4J_DB) as sess:
        res = sess.run(
            cypher,
            k=top_k,
            qvec=qvec,
            include_embedding=include_embedding,
            win=win_minutes,
        )
        return [r.data() for r in res]




# =========================
# CLI
# =========================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Dashcam: RTTM + Transcripts (+Entities) → Embeddings → Neo4j (enriched vector search)")
    parser.add_argument("--search", type=str, default=None, help="Run a vector search over nodes by text query")
    parser.add_argument("--target", type=str, default="utterance", choices=["utterance","segment","transcription"],
                        help="Which index to search")
    parser.add_argument("--topk", type=int, default=5, help="Top-K results for vector search")
    parser.add_argument("--win-mins", type=int, default=10, help="± minutes window to snap to nearest PhoneLog")
    parser.add_argument("--include-emb", action="store_true", help="Include node embedding in search results")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N keys on ingest when not searching")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Embedding batch size")
    parser.add_argument("--force", action="store_true", help="Force re-ingest even if transcription already exists")
    args = parser.parse_args()

    driver = neo4j_driver()
    ensure_schema(driver)

    if args.search:
        if not driver:
            print("Neo4j not configured (set NEO4J_URI/USER/PASSWORD).")
            return
        hits = vector_search(
            driver,
            args.search,
            target=args.target,
            top_k=args.topk,
            include_embedding=args.include_emb,
            win_minutes=args.win_mins,
        )
        print(f"\nTop-{args.topk} {args.target} matches for: {args.search!r}\n")
        for h in hits:
            text = (h.get("text") or "").strip().replace("\n", " ")
            if len(text) > 180: text = text[:177] + "..."
            loc = ""
            if h.get("latitude") is not None and h.get("longitude") is not None:
                loc = f" @({h['latitude']:.6f},{h['longitude']:.6f})"
            meta = f" file={h.get('file_key')}  t={h.get('transcription_id')}  start={h.get('start')} end={h.get('end')}  user={h.get('user_id')}  loc_ts={h.get('location_ts')}"
            print(f"- [{h['score']:.4f}]{loc} :: {text}\n    {meta}")
            if args.include_emb and h.get("embedding") is not None:
                print(f"    emb_dim={len(h['embedding'])}")
        return

    mapping = discover_keys()
    keys = sorted(mapping.keys())
    if args.limit:
        keys = keys[:args.limit]
    logging.info(f"Found {len(keys)} key(s) to process.")

    processed = 0
    for i, key in enumerate(keys, 1):
        try:
            t_id = stable_id(key, "transcription")
            if not args.force and already_ingested(driver, t_id):
                logging.info(f"[{i}/{len(keys)}] SKIP (exists): {key}")
                continue
            info = process_key(key, mapping[key], driver, batch_size=args.batch_size)
            if info:
                processed += 1
                logging.info(f"[{i}/{len(keys)}] OK: {key} "
                             f"(segments={info['segments']}, utterances={info['utterances']}, "
                             f"speakers={info['speakers']}, entities={info['entities']})")
            else:
                logging.warning(f"[{i}/{len(keys)}] SKIP: {key}")
        except Exception as ex:
            logging.exception(f"[{i}/{len(keys)}] FAILED: {key} :: {ex}")

    logging.info(f"Done. Processed {processed}/{len(keys)} key(s).")

if __name__ == "__main__":
    main()
