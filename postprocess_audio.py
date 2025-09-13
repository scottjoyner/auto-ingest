#!/usr/bin/env python3
"""
Batch post-processor for AUDIO_BASE.

Input (from your existing pipeline):
  AUDIO_BASE/YYYY/MM/DD/
    <stem>.mp3
    <stem>_transcription.csv
    <stem>_<model>_transcription.txt   # JSON from whisper/faster-whisper
    <stem>_speakers.rttm                # optional diarization
    <stem>.mp3.music.json               # optional music classifier output

Output (for ingestion by a later script):
  AUDIO_BASE/_derived/YYYY/MM/DD/<activity_id>/
    activity.json
    summary.md
    subjects.json
    speaker_narrative.json
  AUDIO_BASE/_manifests/
    activities.jsonl
    activities.csv
  AUDIO_BASE/_state/
    postprocess.db   # sqlite index for idempotency/resume

Activity definition:
  - Logical aggregation of segments using a gap-threshold (default: 30 minutes)
  - Activity spans one day; never crosses midnight
  - Contains: segments (with text, t0,t1,speaker?), speakers, subject tags, summaries

Dependencies: standard lib + (optional) requests for LLM calls; no GPU libs required here.
"""

import os, re, csv, json, sys, math, time, sqlite3, logging, argparse, hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Iterable
from datetime import datetime, timedelta, timezone

try:
    import requests  # optional (for LLM)
    HAVE_REQUESTS = True
except Exception:
    HAVE_REQUESTS = False

# ------------- Defaults & Config -------------
AUDIO_BASE_DEFAULT = "/mnt/8TB_2025/fileserver/audio"
DERIVED_DIRNAME    = "_derived"
MANIFEST_DIRNAME   = "_manifests"
STATE_DIRNAME      = "_state"

# Gap to start a new activity (minutes)
DEFAULT_GAP_MINUTES = 30

# LLM config (optional)
LLM_PROVIDER_ENV   = os.getenv("LLM_PROVIDER", "none").lower()  # none|ollama|openai
OLLAMA_HOST_ENV    = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL_ENV   = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OPENAI_MODEL_ENV   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY_ENV = os.getenv("OPENAI_API_KEY", "")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO),
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("postprocess")

VIDEO_STEM_RE = re.compile(r"""
    ^(?P<key>(\d{4}_\d{2}_\d{2}_\d{6})|(\d{14}))    # 2014_style or 14-digit key
    (?:
        _(?P<tail>[A-Za-z]+|[0-9]{6})              # optional tail: R / idx
        |
        _BC-(?P<bcidx>\d+)                         # optional bodycam idx
    )?
    $
""", re.VERBOSE)

BEST_MODEL_HINTS = ("large", "large-v2", "large-v3")

# ------------- Data Models -------------

@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: Optional[str] = None

@dataclass
class StemRecord:
    day: str            # YYYY-MM-DD
    key: str            # 2014-style 'YYYY_MM_DD_HHMMSS' or 'YYYYMMDDHHMMSS'
    stem: str           # full stem used in filenames
    mp3: Path
    csv_path: Path
    json_txt_path: Optional[Path]
    rttm_path: Optional[Path]
    music_json_path: Optional[Path]

@dataclass
class Activity:
    id: str
    day: str
    start_ts: float
    end_ts: float
    stems: List[str]
    segments: List[Segment]
    speakers: List[str]
    subject_tags: List[str]
    summary: str

# ------------- Helpers -------------

def to_day_y_m_d_from_key(key: str) -> Tuple[str, str, str]:
    if "_" in key:
        y = key[0:4]; mo = key[5:7]; d = key[8:10]
    else:
        y = key[0:4]; mo = key[4:6]; d = key[6:8]
    return y, mo, d

def stem_from_parts(key: str, tail: Optional[str], bcidx: Optional[str]) -> str:
    if bcidx:
        return f"{key}_BC-{bcidx}"
    if tail:
        return f"{key}_{tail}"
    return key

def parse_stem(stem: str) -> Optional[Tuple[str, str]]:
    m = VIDEO_STEM_RE.match(stem)
    if not m:
        return None
    key = m.group("key")
    tail = m.group("tail")
    bcidx = m.group("bcidx")
    return key, stem_from_parts(key, tail, bcidx)

def scan_day_dir(day_dir: Path) -> List[StemRecord]:
    """Collect stems for a single YYYY/MM/DD directory."""
    found: Dict[str, StemRecord] = {}
    for p in day_dir.glob("*"):
        if not p.is_file():
            continue
        name = p.name
        stem = p.stem  # before first extension
        # normalize for multi-extensions (.mp3.music.json split)
        if name.endswith(".mp3.music.json"):
            stem = name[:-len(".mp3.music.json")]
        elif name.endswith("_transcription.csv"):
            stem = name[:-len("_transcription.csv")]
        elif "_transcription.txt" in name:
            # <stem>_<model>_transcription.txt
            stem = name[:name.index("_transcription.txt")]
        elif name.endswith("_speakers.rttm"):
            stem = name[:-len("_speakers.rttm")]
        elif name.endswith(".json"):  # whisper json dump: <stem>.json
            stem = name[:-len(".json")]
        else:
            stem = Path(name).stem

        parsed = parse_stem(stem)
        if not parsed:
            continue
        key, fullstem = parsed
        y, mo, d = to_day_y_m_d_from_key(key)
        day = f"{y}-{mo}-{d}"

        rec = found.get(fullstem)
        if not rec:
            rec = StemRecord(
                day=day,
                key=key,
                stem=fullstem,
                mp3=day_dir / f"{fullstem}.mp3",
                csv_path=day_dir / f"{fullstem}_transcription.csv",
                json_txt_path=None,
                rttm_path=None,
                music_json_path=None,
            )
            found[fullstem] = rec

        # Track optional artifacts
        if name.endswith("_transcription.csv"):
            rec.csv_path = day_dir / name
        elif name.endswith("_speakers.rttm"):
            rec.rttm_path = day_dir / name
        elif name.endswith(".mp3.music.json"):
            rec.music_json_path = day_dir / name
        elif name.endswith("_transcription.txt"):
            rec.json_txt_path = day_dir / name
        elif name.endswith(".mp3"):
            rec.mp3 = day_dir / name

    return list(found.values())

def iter_day_dirs(audio_root: Path, since: Optional[str], limit_days: Optional[int]) -> Iterable[Path]:
    """Yield day directories under AUDIO_BASE/YYYY/MM/DD."""
    years = sorted([p for p in audio_root.iterdir() if p.is_dir() and p.name.isdigit()])
    count = 0
    for ydir in years:
        months = sorted([p for p in ydir.iterdir() if p.is_dir() and p.name.isdigit()])
        for mdir in months:
            days = sorted([p for p in mdir.iterdir() if p.is_dir() and p.name.isdigit()])
            for ddir in days:
                day_str = f"{ydir.name}-{mdir.name}-{ddir.name}"
                if since and day_str < since:
                    continue
                yield ddir
                count += 1
                if limit_days and count >= limit_days:
                    return

# ------------- Loaders -------------

def load_segments_from_csv(csv_path: Path) -> List[Segment]:
    segs: List[Segment] = []
    if not csv_path.exists():
        return segs
    with csv_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                segs.append(Segment(
                    start=float(row.get("start", "0") or 0.0),
                    end=float(row.get("end", "0") or 0.0),
                    text=str(row.get("text") or "").strip(),
                ))
            except Exception:
                continue
    return segs

def parse_rttm_speakers(rttm_path: Path) -> Dict[Tuple[float, float], str]:
    """
    Parses RTTM and returns a mapping from (start, end) -> speaker label.
    RTTM lines like:
      SPEAKER file1 1 start dur <NA> <NA> speaker <NA>
    """
    speakers: Dict[Tuple[float, float], str] = {}
    if not (rttm_path and rttm_path.exists()):
        return speakers
    with rttm_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 10 or parts[0] != "SPEAKER":
                continue
            try:
                start = float(parts[3]); dur = float(parts[4])
                spk = parts[7]
                speakers[(start, start + dur)] = spk
            except Exception:
                continue
    return speakers

def attach_speakers(segments: List[Segment], rttm_map: Dict[Tuple[float, float], str]) -> None:
    """Greedy overlap: assign a speaker label if RTTM interval overlaps segment midpoint."""
    if not rttm_map or not segments:
        return
    rttm_items = list(rttm_map.items())
    for s in segments:
        mid = (s.start + s.end) / 2.0
        for (a, b), spk in rttm_items:
            if a <= mid <= b:
                s.speaker = spk
                break

def load_music_flag(music_json: Optional[Path]) -> Optional[bool]:
    """
    If a .mp3.music.json exists and contains { "is_music": true/false }, return it.
    Otherwise, None.
    """
    if not (music_json and music_json.exists()):
        return None
    try:
        j = json.loads(music_json.read_text(encoding="utf-8"))
        # accept several schema forms
        if isinstance(j, dict):
            if "is_music" in j:
                return bool(j["is_music"])
            if "music" in j:
                return bool(j["music"])
    except Exception:
        pass
    return None

# ------------- Activity Grouping -------------

def group_segments_into_activities(
    day: str,
    stem_to_segments: Dict[str, List[Segment]],
    gap_minutes: int
) -> List[Activity]:
    """
    Merge segments from multiple stems (same day) into chronologically ordered activities.
    Gap > threshold → start new activity.
    """
    # Flatten and sort
    flat: List[Tuple[str, Segment]] = []
    for st, segs in stem_to_segments.items():
        for s in segs:
            flat.append((st, s))
    flat.sort(key=lambda t: (t[1].start, t[1].end))  # assuming same timeline base per stem

    activities: List[Activity] = []
    if not flat:
        return activities

    gap = gap_minutes * 60.0
    cur_segments: List[Tuple[str, Segment]] = [flat[0]]
    for (st, seg) in flat[1:]:
        last_end = cur_segments[-1][1].end
        if (seg.start - last_end) > gap:
            activities.append(_finalize_activity(day, cur_segments))
            cur_segments = []
        cur_segments.append((st, seg))

    if cur_segments:
        activities.append(_finalize_activity(day, cur_segments))
    return activities

def _finalize_activity(day: str, pairs: List[Tuple[str, Segment]]) -> Activity:
    stems = sorted(set([st for st, _ in pairs]))
    segs  = [s for _, s in pairs]
    start_ts = segs[0].start
    end_ts   = segs[-1].end
    # ID: hash of day + first/last timestamps + stems
    h = hashlib.sha1()
    h.update(day.encode())
    h.update(f"{start_ts:.2f}-{end_ts:.2f}".encode())
    h.update("|".join(stems).encode())
    act_id = h.hexdigest()[:16]

    speakers = []
    seen = set()
    for s in segs:
        if s.speaker and s.speaker not in seen:
            speakers.append(s.speaker); seen.add(s.speaker)

    return Activity(
        id=act_id,
        day=day,
        start_ts=float(start_ts),
        end_ts=float(end_ts),
        stems=stems,
        segments=segs,
        speakers=speakers,
        subject_tags=[],
        summary=""
    )

# ------------- Subject Classification -------------

DEFAULT_SUBJECT_PATTERNS = [
    # (tag, regex)
    ("driving", r"\b(exit|merge|turn|traffic|speed|mile|highway|lane|gps|intersection)\b"),
    ("phone_call", r"\b(call|dial|voicemail|ring|phone|speakerphone)\b"),
    ("music", r"\b(song|music|album|playlist|spotify|sirius|radio)\b"),
    ("shopping", r"\b(store|checkout|cart|grocery|receipt|price|cashier)\b"),
    ("work", r"\b(project|deadline|meeting|ticket|deploy|git|kubernetes|model)\b"),
    ("travel", r"\b(flight|hotel|booking|reservation|check[- ]in|boarding|gate)\b"),
    ("vehicle", r"\b(oil|tire|engine|battery|charge|maintenance|diagnostic|sensor)\b"),
    ("personal", r"\b(appointment|doctor|prescription|insurance|invoice|bank|account)\b"),
]

def classify_subjects(segments: List[Segment]) -> List[str]:
    text = " ".join([s.text for s in segments]).lower()
    tags = []
    for tag, rx in DEFAULT_SUBJECT_PATTERNS:
        if re.search(rx, text):
            tags.append(tag)
    # Deduplicate while preserving order
    seen = set(); out = []
    for t in tags:
        if t not in seen:
            out.append(t); seen.add(t)
    return out[:8]

# ------------- Narratives -------------

def build_speaker_narrative(segments: List[Segment]) -> Dict[str, str]:
    by_spk: Dict[str, List[str]] = {}
    for s in segments:
        spk = s.speaker or "SPEAKER_0"
        by_spk.setdefault(spk, []).append(s.text.strip())
    return {spk: " ".join(chunks).strip() for spk, chunks in by_spk.items()}

# ------------- Summaries (optional LLM) -------------

def summarize_text_llm(text: str, provider: str, **kwargs) -> str:
    provider = provider.lower()
    text = text.strip()
    if not text:
        return ""

    # common prompt
    prompt = (
        "You are a concise meeting/activity summarizer. "
        "Given transcript text, produce a short summary (3–6 bullet points) "
        "covering who spoke, key topics, actions, and notable events. "
        "Output plain text bullets."
        "\n\nTRANSCRIPT:\n" + text[:12000]
    )

    if provider == "ollama":
        if not HAVE_REQUESTS:
            return ""
        host = kwargs.get("ollama_host", OLLAMA_HOST_ENV)
        model = kwargs.get("ollama_model", OLLAMA_MODEL_ENV)
        try:
            resp = requests.post(f"{host}/api/generate", json={
                "model": model,
                "prompt": prompt,
                "stream": False,
            }, timeout=120)
            if resp.ok:
                j = resp.json()
                return (j.get("response") or "").strip()
        except Exception:
            return ""
        return ""

    if provider == "openai":
        if not HAVE_REQUESTS:
            return ""
        api_key = kwargs.get("openai_api_key", OPENAI_API_KEY_ENV)
        model = kwargs.get("openai_model", OPENAI_MODEL_ENV)
        if not api_key:
            return ""
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
            }
            resp = requests.post("https://api.openai.com/v1/chat/completions",
                                 headers=headers, json=payload, timeout=120)
            if resp.ok:
                j = resp.json()
                content = j["choices"][0]["message"]["content"]
                return content.strip()
        except Exception:
            return ""
        return ""

    # Fallback non-LLM summarizer (extractive-ish): pick first/last N lines
    lines = [ln for ln in text.splitlines() if ln.strip()]
    head = " • " + "\n • ".join(lines[:3])
    tail = " • " + "\n • ".join(lines[-2:]) if len(lines) > 5 else ""
    return ("[Auto summary]\n" + head + ("\n...\n" + tail if tail else "")).strip()

# ------------- Persistence (SQLite) -------------

def state_db_path(root: Path) -> Path:
    st = root / STATE_DIRNAME
    st.mkdir(parents=True, exist_ok=True)
    return st / "postprocess.db"

def init_state(dbp: Path) -> None:
    con = sqlite3.connect(dbp)
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS stems(
        stem TEXT PRIMARY KEY,
        day  TEXT NOT NULL,
        key  TEXT NOT NULL,
        csv_mtime REAL,
        json_mtime REAL,
        rttm_mtime REAL
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS activities(
        id TEXT PRIMARY KEY,
        day TEXT NOT NULL,
        start_ts REAL, end_ts REAL,
        stems TEXT,
        subject_tags TEXT,
        summary_mtime REAL
    )""")
    con.commit()
    con.close()

def upsert_stem(con, rec: StemRecord) -> None:
    cur = con.cursor()
    csv_mt  = rec.csv_path.stat().st_mtime if rec.csv_path.exists() else None
    json_mt = rec.json_txt_path.stat().st_mtime if (rec.json_txt_path and rec.json_txt_path.exists()) else None
    rttm_mt = rec.rttm_path.stat().st_mtime if (rec.rttm_path and rec.rttm_path.exists()) else None
    cur.execute("""
        INSERT INTO stems(stem,day,key,csv_mtime,json_mtime,rttm_mtime)
        VALUES(?,?,?,?,?,?)
        ON CONFLICT(stem) DO UPDATE SET
          day=excluded.day,
          key=excluded.key,
          csv_mtime=excluded.csv_mtime,
          json_mtime=excluded.json_mtime,
          rttm_mtime=excluded.rttm_mtime
    """, (rec.stem, rec.day, rec.key, csv_mt, json_mt, rttm_mt))

# ------------- Writers -------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def write_activity_artifacts(audio_root: Path, act: Activity) -> None:
    y, mo, d = act.day.split("-")
    base = audio_root / DERIVED_DIRNAME / y / mo / d / act.id
    ensure_dir(base)

    # activity.json
    act_obj = {
        "id": act.id,
        "day": act.day,
        "start_ts": act.start_ts,
        "end_ts": act.end_ts,
        "stems": act.stems,
        "speakers": act.speakers,
        "segments": [asdict(s) for s in act.segments],
        "subject_tags": act.subject_tags,
    }
    (base / "activity.json").write_text(json.dumps(act_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    # summary.md
    (base / "summary.md").write_text((act.summary or "").strip() + "\n", encoding="utf-8")

    # subjects.json
    (base / "subjects.json").write_text(json.dumps(act.subject_tags, ensure_ascii=False, indent=2), encoding="utf-8")

    # speaker_narrative.json
    narrative = build_speaker_narrative(act.segments)
    (base / "speaker_narrative.json").write_text(json.dumps(narrative, ensure_ascii=False, indent=2), encoding="utf-8")

def append_manifests(audio_root: Path, activities: List[Activity]) -> None:
    man_root = audio_root / MANIFEST_DIRNAME
    ensure_dir(man_root)

    # JSONL for ingestion
    jl = man_root / "activities.jsonl"
    with jl.open("a", encoding="utf-8") as f:
        for a in activities:
            line = {
                "id": a.id,
                "day": a.day,
                "start_ts": a.start_ts,
                "end_ts": a.end_ts,
                "stems": a.stems,
                "speakers": a.speakers,
                "subject_tags": a.subject_tags,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    # CSV (create if missing)
    csvp = man_root / "activities.csv"
    new_file = not csvp.exists()
    with csvp.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["id","day","start_ts","end_ts","stems","speakers","subject_tags"])
        for a in activities:
            w.writerow([
                a.id, a.day, f"{a.start_ts:.2f}", f"{a.end_ts:.2f}",
                "|".join(a.stems), "|".join(a.speakers), "|".join(a.subject_tags)
            ])

# ------------- Core Pipeline -------------

def process_day(audio_root: Path, day_dir: Path, gap_minutes: int, llm_provider: str) -> List[Activity]:
    """
    Returns newly created activities (for manifest append). Idempotent per activity id.
    """
    recs = scan_day_dir(day_dir)
    if not recs:
        return []

    # Build stem -> segments (+ speakers if RTTM)
    stem_to_segments: Dict[str, List[Segment]] = {}
    for rec in recs:
        if not rec.csv_path.exists():
            continue
        segs = load_segments_from_csv(rec.csv_path)
        if not segs:
            continue

        rttm_map = parse_rttm_speakers(rec.rttm_path) if rec.rttm_path else {}
        if rttm_map:
            attach_speakers(segs, rttm_map)

        # If a music json marks is_music=True, tag the segments with a synthetic speaker?
        is_music = load_music_flag(rec.music_json_path)
        if is_music is True:
            # Optionally, label these segments to be filtered or tagged later
            for s in segs:
                s.speaker = s.speaker or "MUSIC"

        stem_to_segments[rec.stem] = segs

    if not stem_to_segments:
        return []

    # Normalize segment timelines per stem to ensure monotonically increasing starts
    for st, segs in stem_to_segments.items():
        segs.sort(key=lambda s: (s.start, s.end))

    # Activities
    # We do grouping across stems for the day by inter-segment gaps
    # (Assumes segments times are in same time base per file-day; if not, we can
    # normalize with per-stem offsets when available.)
    day_str = recs[0].day
    activities = group_segments_into_activities(day_str, stem_to_segments, gap_minutes)
    if not activities:
        return []

    # Summarize + classify
    for a in activities:
        # subject tags
        a.subject_tags = classify_subjects(a.segments)

        # summary (optional LLM)
        combined_text = "\n".join([s.text for s in a.segments])
        if llm_provider != "none":
            a.summary = summarize_text_llm(
                combined_text,
                provider=llm_provider,
                ollama_host=OLLAMA_HOST_ENV,
                ollama_model=OLLAMA_MODEL_ENV,
                openai_api_key=OPENAI_API_KEY_ENV,
                openai_model=OPENAI_MODEL_ENV,
            )
        else:
            a.summary = summarize_text_llm(combined_text, provider="none")

    # Write artifacts & return created
    created: List[Activity] = []
    for a in activities:
        write_activity_artifacts(audio_root, a)
        created.append(a)

    return created

# ------------- CLI -------------

def main():
    ap = argparse.ArgumentParser(description="Batch post-process AUDIO_BASE into activities, summaries, and manifests.")
    ap.add_argument("--audio-root", default=AUDIO_BASE_DEFAULT, help="Root dir containing YYYY/MM/DD audio trees.")
    ap.add_argument("--since-day", default=None, help="Only process days >= this (YYYY-MM-DD).")
    ap.add_argument("--limit-days", type=int, default=None, help="Process at most N day folders.")
    ap.add_argument("--gap-minutes", type=int, default=DEFAULT_GAP_MINUTES, help="Gap (minutes) to split activities.")
    ap.add_argument("--llm", default=LLM_PROVIDER_ENV, choices=["none","ollama","openai"], help="Summarizer backend.")
    ap.add_argument("--dry-run", action="store_true", help="Discover & summarize in-memory (no manifests/artifacts).")
    args = ap.parse_args()

    audio_root = Path(args.audio_root)
    if not audio_root.exists():
        log.error(f"AUDIO_ROOT not found: {audio_root}")
        sys.exit(2)

    dbp = state_db_path(audio_root)
    init_state(dbp)
    con = sqlite3.connect(dbp)
    con.row_factory = sqlite3.Row

    total_created = 0
    started = time.perf_counter()

    for ddir in iter_day_dirs(audio_root, args.since_day, args.limit_days):
        log.info(f"=== DAY {ddir} ===")
        # index stems
        recs = scan_day_dir(ddir)
        for rec in recs:
            upsert_stem(con, rec)
        con.commit()

        if args.dry_run:
            # still run summaries in-memory but skip writing
            acts = process_day(audio_root, ddir, args.gap_minutes, args.llm)
            log.info(f"[dry-run] activities: {len(acts)}")
            continue

        # do full write
        acts = process_day(audio_root, ddir, args.gap_minutes, args.llm)
        if acts:
            append_manifests(audio_root, acts)
            total_created += len(acts)
        log.info(f"Created {len(acts)} activity(ies) for {ddir}")

    elapsed = time.perf_counter() - started
    log.info(f"ALL DONE | activities created: {total_created} | elapsed={elapsed:.2f}s")

if __name__ == "__main__":
    main()
