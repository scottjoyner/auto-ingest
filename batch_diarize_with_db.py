#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch diarization + transcription for YYYY/MM/DD/<Contents> directories (CPU-optimized).
Also writes a SQLite index with tables: files, speakers, segments (+ optional FTS5).

Artifacts per file (mirrored under --outdir):
  - <file>.rttm
  - <file>_speaker_transcript.{json,csv,srt}
  - speakers/<SPEAKER_X>.wav (concatenated)
  - segments/*.wav (optional)

SQLite schema (created if missing):
  files(id, rel_path UNIQUE, abs_path, date_str, year, month, day, samplerate, duration, rttm_path, json_path, csv_path, srt_path, processed_at)
  speakers(id, file_id, label, UNIQUE(file_id,label))
  segments(id, file_id, speaker_id, start, end, text, FOREIGN KEYs)
  segments_fts(text)  -- optional FTS5 mirror with triggers

CPU defaults tuned for i7-7th gen:
  faster-whisper: small / int8_float16 / beam=1 / vad_filter=True
  min_segment_len = 0.8s
  multiprocessing across files (workers = CPU-1)
"""

import os
import re
import sys
import io
import json
import math
import time
import sqlite3
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

# ---- Quiet noisy libs early ----
warnings.filterwarnings("ignore", message="torchaudio._backend.*")
warnings.filterwarnings("ignore", message="Can't initialize NVML")
warnings.filterwarnings("ignore", message="Module 'speechbrain.pretrained' was deprecated.*")

# Avoid thread oversubscription (tuned later in processes too)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import torchaudio
import soundfile as sf
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, get_start_method

# pyannote
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
from pyannote.audio.pipelines.utils.hook import ProgressHook

# ASR
from faster_whisper import WhisperModel


# ===================== Utility =====================

def set_torch_threads(cpu_threads: int):
    """Constrain intra-op/interop threads to prevent thrashing on CPU."""
    try:
        torch.set_num_threads(max(1, cpu_threads))
        torch.set_num_interop_threads(max(1, min(2, cpu_threads // 2)))
    except Exception:
        pass


def ensure_mono16k(path: Path) -> Tuple[np.ndarray, int]:
    """Load audio, downmix to mono, resample to 16kHz, return float32 waveform + sr."""
    waveform, sr = torchaudio.load(str(path))
    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000
    return waveform.squeeze(0).numpy().astype(np.float32), sr


def write_wav(path: Path, waveform: np.ndarray, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform, sr, subtype="PCM_16")


@dataclass
class SegmentResult:
    speaker: str
    start: float
    end: float
    text: str
    words: Optional[List[Dict]] = None


def segment_to_frames(segment: Segment, sr: int) -> Tuple[int, int]:
    start = max(0, int(round(segment.start * sr)))
    end = max(start, int(round(segment.end * sr)))
    return start, end


def srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - math.floor(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_outputs(results: List[SegmentResult], out_base: Path):
    # JSON
    data = [asdict(r) for r in results]
    out_base.with_suffix(".json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # CSV
    df = pd.DataFrame(
        [{"speaker": r.speaker, "start": f"{r.start:.3f}", "end": f"{r.end:.3f}", "text": r.text} for r in results],
        columns=["speaker", "start", "end", "text"]
    )
    df.to_csv(out_base.with_suffix(".csv"), index=False, encoding="utf-8")
    # SRT
    srt_lines = []
    for i, r in enumerate(results, start=1):
        srt_lines.append(str(i))
        srt_lines.append(f"{srt_timestamp(r.start)} --> {srt_timestamp(r.end)}")
        srt_lines.append(f"{r.speaker}: {r.text}".strip())
        srt_lines.append("")
    out_base.with_suffix(".srt").write_text("\n".join(srt_lines), encoding="utf-8")


# ===================== Diarization & ASR =====================

def load_diarization_pipeline(hf_token: str) -> Pipeline:
    # CPU by default; pyannote 3.1 is pure torch
    return Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)


def load_asr_model(model_name: str, compute_type: str) -> WhisperModel:
    # device="cpu" — faster-whisper uses CTranslate2 CPU kernels
    return WhisperModel(model_name, device="cpu", compute_type=compute_type)


def transcribe_chunk(
    model: WhisperModel,
    audio: np.ndarray,
    sr: int,
    language: Optional[str],
    beam_size: int,
    vad_filter: bool,
) -> Tuple[str, List[Dict]]:
    segments, info = model.transcribe(
        audio,
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter,
        word_timestamps=True,
    )
    text_parts, words = [], []
    for seg in segments:
        if seg.text:
            text_parts.append(seg.text.strip())
        if seg.words:
            for w in seg.words:
                words.append({
                    "word": w.word,
                    "start": float(w.start) if w.start is not None else None,
                    "end": float(w.end) if w.end is not None else None,
                    "prob": float(w.probability) if w.probability is not None else None
                })
    return (" ".join(text_parts).strip(), words)


# ===================== SQLite Index =====================

DB_PRAGMAS = [
    "PRAGMA journal_mode=WAL;",
    "PRAGMA synchronous=NORMAL;",
    "PRAGMA temp_store=MEMORY;",
    "PRAGMA mmap_size=268435456;",   # 256 MiB
    "PRAGMA busy_timeout=5000;"
]

def connect_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.execute("PRAGMA foreign_keys=ON;")
    for p in DB_PRAGMAS:
        try:
            conn.execute(p)
        except sqlite3.OperationalError:
            pass
    return conn


def init_db(conn: sqlite3.Connection, enable_fts: bool):
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY,
        rel_path TEXT UNIQUE,
        abs_path TEXT,
        date_str TEXT,
        year INTEGER, month INTEGER, day INTEGER,
        samplerate INTEGER,
        duration REAL,
        rttm_path TEXT, json_path TEXT, csv_path TEXT, srt_path TEXT,
        processed_at TEXT
    );
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS speakers (
        id INTEGER PRIMARY KEY,
        file_id INTEGER,
        label TEXT,
        UNIQUE(file_id, label),
        FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
    );
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS segments (
        id INTEGER PRIMARY KEY,
        file_id INTEGER,
        speaker_id INTEGER,
        start REAL,
        end REAL,
        text TEXT,
        FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE,
        FOREIGN KEY(speaker_id) REFERENCES speakers(id) ON DELETE CASCADE
    );
    """)
    if enable_fts:
        c.execute("CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(text, content='segments', content_rowid='id');")
        c.executescript("""
        CREATE TRIGGER IF NOT EXISTS segments_ai AFTER INSERT ON segments BEGIN
          INSERT INTO segments_fts(rowid, text) VALUES (new.id, new.text);
        END;
        CREATE TRIGGER IF NOT EXISTS segments_ad AFTER DELETE ON segments BEGIN
          INSERT INTO segments_fts(segments_fts, rowid, text) VALUES('delete', old.id, old.text);
        END;
        CREATE TRIGGER IF NOT EXISTS segments_au AFTER UPDATE ON segments BEGIN
          INSERT INTO segments_fts(segments_fts, rowid, text) VALUES('delete', old.id, old.text);
          INSERT INTO segments_fts(rowid, text) VALUES (new.id, new.text);
        END;
        """)
    conn.commit()


DATE_RE = re.compile(r"^(?P<y>\d{4})/(?P<m>\d{2})/(?P<d>\d{2})(?:/|$)")

def parse_date_from_rel(rel_path: Path) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    # Expect first 3 components: YYYY/MM/DD
    parts = rel_path.as_posix().split("/")
    if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit() and parts[2].isdigit():
        y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
        if 1 <= m <= 12 and 1 <= d <= 31:
            return y, m, d
    # Fallback regex
    m = DATE_RE.match(rel_path.as_posix())
    if m:
        return int(m["y"]), int(m["m"]), int(m["d"])
    return None, None, None


def index_into_db(
    db_path: Path,
    enable_fts: bool,
    rel_path: Path,
    abs_path: Path,
    samplerate: int,
    duration_sec: float,
    rttm_path: Path,
    json_path: Path,
    csv_path: Path,
    srt_path: Path,
    results: List[SegmentResult],
):
    conn = connect_db(db_path)
    try:
        init_db(conn, enable_fts)
        y, m, d = parse_date_from_rel(rel_path)
        date_str = f"{y:04d}-{m:02d}-{d:02d}" if (y and m and d) else None

        # Retry loop in case of WAL lock under concurrency
        for attempt in range(6):
            try:
                cur = conn.cursor()
                cur.execute("BEGIN IMMEDIATE;")
                # Upsert file row
                cur.execute("""
                INSERT INTO files (rel_path, abs_path, date_str, year, month, day, samplerate, duration, rttm_path, json_path, csv_path, srt_path, processed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(rel_path) DO UPDATE SET
                    abs_path=excluded.abs_path,
                    date_str=excluded.date_str,
                    year=excluded.year,
                    month=excluded.month,
                    day=excluded.day,
                    samplerate=excluded.samplerate,
                    duration=excluded.duration,
                    rttm_path=excluded.rttm_path,
                    json_path=excluded.json_path,
                    csv_path=excluded.csv_path,
                    srt_path=excluded.srt_path,
                    processed_at=datetime('now');
                """, (
                    rel_path.as_posix(),
                    str(abs_path),
                    date_str, y, m, d,
                    samplerate,
                    float(duration_sec),
                    str(rttm_path),
                    str(json_path),
                    str(csv_path),
                    str(srt_path),
                ))
                cur.execute("SELECT id FROM files WHERE rel_path = ?", (rel_path.as_posix(),))
                file_id = cur.fetchone()[0]

                # Speakers: ensure rows & map labels -> ids
                labels = sorted(set(r.speaker for r in results))
                for lbl in labels:
                    cur.execute("INSERT OR IGNORE INTO speakers (file_id, label) VALUES (?, ?);", (file_id, lbl))
                spk_id_map: Dict[str, int] = {}
                cur.execute("SELECT id, label FROM speakers WHERE file_id = ?;", (file_id,))
                for sid, lbl in cur.fetchall():
                    spk_id_map[lbl] = sid

                # Remove existing segments (if reprocessing), then insert fresh
                cur.execute("DELETE FROM segments WHERE file_id = ?;", (file_id,))
                cur.executemany("""
                    INSERT INTO segments (file_id, speaker_id, start, end, text)
                    VALUES (?, ?, ?, ?, ?);
                """, [
                    (file_id, spk_id_map[r.speaker], float(r.start), float(r.end), r.text or "")
                    for r in results
                ])

                conn.commit()
                break
            except sqlite3.OperationalError as e:
                conn.rollback()
                if "locked" in str(e).lower() or "busy" in str(e).lower():
                    time.sleep(0.25 * (attempt + 1))
                    continue
                raise
    finally:
        conn.close()


# ===================== Per-file processing =====================

def process_one_file(
    wav_path: Path,
    indir: Path,
    outdir: Path,
    hf_token: str,
    asr_name: str,
    asr_lang: Optional[str],
    asr_beam: int,
    asr_compute: str,
    min_segment_len: float,
    export_segments: bool,
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    skip_existing: bool,
    verbose: bool,
    db_path: Optional[Path],
    enable_fts: bool,
):
    rel_path = wav_path.relative_to(indir)
    out_dir = outdir / rel_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    base = wav_path.stem
    artifact_dir = out_dir / base
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out_base = artifact_dir / f"{base}_speaker_transcript"
    rttm_path = artifact_dir / f"{base}.rttm"
    spk_dir = artifact_dir / "speakers"
    seg_dir = artifact_dir / "segments"

    if skip_existing and out_base.with_suffix(".json").exists():
        if verbose:
            print(f"⏩ Skip {wav_path} (already processed)")
        # Still ensure DB has a row (best-effort) by indexing existing JSON if DB provided
        if db_path and out_base.with_suffix(".json").exists():
            try:
                results = [SegmentResult(**r) for r in json.loads(out_base.with_suffix(".json").read_text(encoding="utf-8"))]
                dur = 0.0
                try:
                    wf, sr = ensure_mono16k(wav_path)
                    dur = len(wf) / 16000.0
                except Exception:
                    sr, dur = 16000, 0.0
                index_into_db(
                    db_path, enable_fts, rel_path, wav_path, 16000, dur,
                    rttm_path, out_base.with_suffix(".json"), out_base.with_suffix(".csv"), out_base.with_suffix(".srt"),
                    results
                )
            except Exception:
                pass
        return str(out_base)

    if verbose:
        print(f"🎧 Loading {wav_path}")

    # Per-process thread tuning (leave one core)
    set_torch_threads(max(1, os.cpu_count() - 1))

    # Load models per process
    diar_pipe = load_diarization_pipeline(hf_token)
    asr = load_asr_model(asr_name, asr_compute)

    waveform, sr = ensure_mono16k(wav_path)
    audio_mem = {"waveform": torch.from_numpy(waveform).unsqueeze(0), "sample_rate": sr}
    duration_sec = len(waveform) / sr

    # Diarization
    if verbose:
        print("🗣️  Diarizing...")
    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    diarization: Annotation
    if verbose:
        with ProgressHook() as hook:
            diarization = diar_pipe(audio_mem, hook=hook, **kwargs)
    else:
        diarization = diar_pipe(audio_mem, **kwargs)

    # Save RTTM
    with rttm_path.open("w") as rttm:
        diarization.write_rttm(rttm)

    # Per-speaker WAVs
    if verbose:
        print("🔊 Exporting per-speaker WAVs...")
    speakers = sorted(set(diarization.labels()))
    per_spk_audio: Dict[str, List[np.ndarray]] = {spk: [] for spk in speakers}

    for segment, _, label in diarization.itertracks(yield_label=True):
        start_f, end_f = segment_to_frames(segment, sr)
        dur = (end_f - start_f) / sr
        if dur < min_segment_len:
            continue
        per_spk_audio[label].append(waveform[start_f:end_f].copy())

    for spk, chunks in per_spk_audio.items():
        if not chunks:
            continue
        arr = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        write_wav(spk_dir / f"{spk}.wav", arr, sr)

    # ASR on diarized segments
    if verbose:
        print("🧩 Transcribing segments...")
    results: List[SegmentResult] = []
    if export_segments:
        seg_dir.mkdir(parents=True, exist_ok=True)

    iterables = list(diarization.itertracks(yield_label=True))
    for (segment, _, label) in iterables:
        start_f, end_f = segment_to_frames(segment, sr)
        dur = (end_f - start_f) / sr
        if dur < min_segment_len:
            continue

        chunk = waveform[start_f:end_f]
        text, words = transcribe_chunk(
            model=asr,
            audio=chunk,
            sr=sr,
            language=asr_lang,
            beam_size=asr_beam,
            vad_filter=True,
        )

        if export_segments:
            seg_name = f"{label}_{segment.start:.3f}-{segment.end:.3f}.wav"
            write_wav(seg_dir / seg_name, chunk, sr)

        results.append(SegmentResult(
            speaker=label,
            start=float(segment.start),
            end=float(segment.end),
            text=text,
            words=words
        ))

    results.sort(key=lambda r: (r.start, r.end))
    out_base = artifact_dir / f"{base}_speaker_transcript"
    write_outputs(results, out_base)

    # Index into SQLite
    if db_path:
        index_into_db(
            db_path=db_path,
            enable_fts=enable_fts,
            rel_path=rel_path,
            abs_path=wav_path,
            samplerate=sr,
            duration_sec=duration_sec,
            rttm_path=rttm_path,
            json_path=out_base.with_suffix(".json"),
            csv_path=out_base.with_suffix(".csv"),
            srt_path=out_base.with_suffix(".srt"),
            results=results,
        )

    if verbose:
        print(f"✅ Done: {wav_path}")
    return str(out_base)


# ===================== Directory walking & Pool =====================

def find_audio(indir: Path, exts: List[str]) -> List[Path]:
    exts = {e.lower().lstrip(".") for e in exts if e}
    out = []
    for p in indir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix and p.suffix.lower().lstrip(".") in exts:
            out.append(p)
    return sorted(out)

def find_wavs(indir: Path) -> List[Path]:
    return sorted(p for p in indir.rglob("*.wav") if p.is_file())


def _worker(args_tuple):
    try:
        return process_one_file(*args_tuple)
    except Exception as e:
        return f"ERROR::{args_tuple[0]}::{repr(e)}"


def main():
    ap = argparse.ArgumentParser(description="Batch diarization + ASR with SQLite index for YYYY/MM/DD trees (CPU).")
    ap.add_argument("--indir", required=True, type=str, help="Input root directory containing YYYY/MM/DD")
    ap.add_argument("--outdir", required=True, type=str, help="Output root directory")
    ap.add_argument("--hf_token", type=str, default=os.getenv("HUGGINGFACE_TOKEN"), help="Hugging Face token")
    ap.add_argument("--asr_model", type=str, default="small", help="faster-whisper model (tiny|base|small|medium|large-v3)")
    ap.add_argument("--asr_compute", type=str, default="int8_float16", help="compute_type for faster-whisper on CPU")
    ap.add_argument("--asr_lang", type=str, default=None, help="Language hint (e.g., en)")
    ap.add_argument("--asr_beam", type=int, default=1, help="Beam size (CPU: 1 is fastest)")
    ap.add_argument("--min_segment_len", type=float, default=0.8, help="Drop diarized segments shorter than this (sec)")
    ap.add_argument("--num_speakers", type=int, default=None, help="If known, fix speaker count")
    ap.add_argument("--min_speakers", type=int, default=None, help="Lower bound on speakers")
    ap.add_argument("--max_speakers", type=int, default=None, help="Upper bound on speakers")
    ap.add_argument("--export_segments", action="store_true", help="Save each diarized segment as a WAV")
    ap.add_argument("--skip_existing", action="store_true", help="Skip if *_speaker_transcript.json exists")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1), help="Worker processes across files")
    ap.add_argument("--db", type=str, default=None, help="Path to SQLite DB (created if missing)")
    ap.add_argument("--fts", action="store_true", help="Enable FTS5 (full-text search) on segments.text")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--exts", type=str, default="wav", help="Comma-separated audio extensions to scan (case-insensitive). e.g. 'wav,WAV,flac,mp3'")
    ap.add_argument("--dry_run", action="store_true", help="List discovered files then exit (no processing)")

    args = ap.parse_args()

    indir = Path(args.indir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if not indir.exists():
        print(f"❌ Input directory not found: {indir}", file=sys.stderr)
        sys.exit(1)
    if not args.hf_token:
        print("❌ Missing Hugging Face token (set HUGGINGFACE_TOKEN or pass --hf_token).", file=sys.stderr)
        sys.exit(1)
    if args.db and not Path(args.db).parent.exists():
        print(f"❌ DB parent directory does not exist: {Path(args.db).parent}", file=sys.stderr)
        sys.exit(1)

    # Constrain threads in parent
    set_torch_threads(max(1, os.cpu_count() - 1))

# after parsing args
    exts = [e.strip() for e in args.exts.split(",") if e.strip()]
    files = find_audio(indir, exts)
    if not files:
        print(f"⚠️ No audio files found under {indir} with extensions: {exts}")
        print("Tip: try --exts 'wav,WAV' or run --dry_run to check discovery.")
        return

    print(f"📂 Found {len(files)} audio files under {indir} (exts={exts})")

    if args.dry_run:
        for p in files[:50]:
            print(" •", p)
        if len(files) > 50:
            print(f"… and {len(files)-50} more")
        return

# build task_args from `files` instead of `wavs`
    task_args = []
    db_path = Path(args.db) if args.db else None
    for wav in files:
        task_args.append((
        wav, indir, outdir,
        args.hf_token,
        args.asr_model, args.asr_lang, args.asr_beam, args.asr_compute,
        args.min_segment_len, args.export_segments,
        args.num_speakers, args.min_speakers, args.max_speakers,
        args.skip_existing, args.verbose,
        db_path, args.fts
        ))

    # Ensure 'fork' start where possible for lower overhead
    try:
        if get_start_method(allow_none=True) is None:
            import multiprocessing as mp
            mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass

    results = []
    if args.workers == 1:
        for t in tqdm(task_args):
            results.append(_worker(t))
    else:
        with Pool(processes=args.workers) as pool:
            for res in tqdm(pool.imap_unordered(_worker, task_args), total=len(task_args)):
                results.append(res)

    failed = [r for r in results if isinstance(r, str) and r.startswith("ERROR::")]
    if failed:
        print(f"❌ {len(failed)} file(s) failed:")
        for f in failed:
            print("   ", f)
    else:
        print("✅ All files processed successfully.")


if __name__ == "__main__":
    main()

