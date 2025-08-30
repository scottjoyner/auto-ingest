#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch diarization + transcription for YYYY/MM/DD/<Contents> directories.

Outputs (mirrored under --outdir):
  - <file>.rttm
  - <file>_speaker_transcript.{json,csv,srt}
  - speakers/<SPEAKER_X>.wav (concatenated)
  - segments/*.wav (optional)

CPU-tuned defaults for i7-7th gen class machines:
  - faster-whisper: small, int8_float16, beam=1, VAD filter on
  - min_segment_len = 0.8s
  - multiprocessing across files (one worker per CPU minus one)
"""

import os
import sys
import io
import json
import math
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

# --------- De-noise logs early ---------
warnings.filterwarnings("ignore", message="torchaudio._backend.*")
warnings.filterwarnings("ignore", message="Can't initialize NVML")
warnings.filterwarnings("ignore", message="Module 'speechbrain.pretrained' was deprecated.*")

# Threading env to avoid over-subscription (adjusted later once we know CPU count)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import torchaudio
import soundfile as sf
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool, cpu_count, get_start_method

# pyannote
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
from pyannote.audio.pipelines.utils.hook import ProgressHook

# ASR
from faster_whisper import WhisperModel


# ---------- Core helpers ----------

def set_torch_threads(cpu_threads: int):
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


# ---------- Diarization & ASR loaders ----------

def load_diarization_pipeline(hf_token: str) -> Pipeline:
    # pyannote pipelines default to CPU; CUDA/MPS not present here.
    pipe = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )
    return pipe


def load_asr_model(model_name: str, compute_type: str) -> WhisperModel:
    # faster-whisper will use CPU when device="cpu"
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


# ---------- Processing of a single file ----------

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
        return str(out_base)

    if verbose:
        print(f"🎧 Loading {wav_path}")

    # Per-process thread tuning (keep majority for diarization math)
    # i7-7xxx has 8 threads -> leave 1 free
    set_torch_threads(max(1, os.cpu_count() - 1))

    # Load models *per process* (safe with fork/spawn)
    diar_pipe = load_diarization_pipeline(hf_token)
    asr = load_asr_model(asr_name, asr_compute)

    # Load audio once; use memory dict for pyannote to skip re-read/resample inside pipeline
    waveform, sr = ensure_mono16k(wav_path)
    audio_mem = {"waveform": torch.from_numpy(waveform).unsqueeze(0), "sample_rate": sr}

    # Diarization with progress (progress hook is per file; keep quiet in workers unless verbose)
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

    # Export per-speaker concatenated audio
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

    # Transcribe diarized segments
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
            vad_filter=True,  # default on for CPU
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
    write_outputs(results, out_base)

    if verbose:
        print(f"✅ Done: {wav_path}")
    return str(out_base)


# ---------- Directory walking & multiprocessing ----------

def find_wavs(indir: Path) -> List[Path]:
    return sorted(p for p in indir.rglob("*.wav") if p.is_file())


def _worker(args_tuple):
    # Unpack and call
    try:
        return process_one_file(*args_tuple)
    except Exception as e:
        return f"ERROR::{args_tuple[0]}::{repr(e)}"


def main():
    ap = argparse.ArgumentParser(description="Batch diarization + ASR for YYYY/MM/DD trees (CPU-optimized).")
    ap.add_argument("--indir", required=True, type=str, help="Input root directory containing YYYY/MM/DD")
    ap.add_argument("--outdir", required=True, type=str, help="Output root directory for results")
    ap.add_argument("--hf_token", type=str, default=os.getenv("HUGGINGFACE_TOKEN"), help="Hugging Face token")
    ap.add_argument("--asr_model", type=str, default="small", help="faster-whisper model (e.g., tiny, base, small, medium, large-v3)")
    ap.add_argument("--asr_compute", type=str, default="int8_float16", help="compute_type for faster-whisper on CPU")
    ap.add_argument("--asr_lang", type=str, default=None, help="Language hint (e.g., en)")
    ap.add_argument("--asr_beam", type=int, default=1, help="Beam size (CPU: 1 is fastest)")
    ap.add_argument("--min_segment_len", type=float, default=0.8, help="Drop diarized segments shorter than this (sec)")
    ap.add_argument("--num_speakers", type=int, default=None, help="If known, fix speaker count")
    ap.add_argument("--min_speakers", type=int, default=None, help="Lower bound on speakers")
    ap.add_argument("--max_speakers", type=int, default=None, help="Upper bound on speakers")
    ap.add_argument("--export_segments", action="store_true", help="Save each diarized segment as a WAV")
    ap.add_argument("--skip_existing", action="store_true", help="Skip if *_speaker_transcript.json exists")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1), help="Parallel worker processes across files")
    ap.add_argument("--verbose", action="store_true")
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

    # CPU thread tuning (process-level; workers will set their own too)
    # i7-7xxx has ~8 threads -> leave one for I/O/main
    set_torch_threads(max(1, os.cpu_count() - 1))

    wavs = find_wavs(indir)
    if not wavs:
        print("⚠️ No .wav files found.")
        return

    print(f"📂 Found {len(wavs)} WAV files under {indir}")
    print(f"🧵 Using {args.workers} worker process(es)")

    # Build argument tuples for workers
    task_args = []
    for wav in wavs:
        task_args.append((
            wav, indir, outdir,
            args.hf_token,
            args.asr_model, args.asr_lang, args.asr_beam, args.asr_compute,
            args.min_segment_len, args.export_segments,
            args.num_speakers, args.min_speakers, args.max_speakers,
            args.skip_existing, args.verbose
        ))

    # On some distros, start method may be 'fork' by default; this is fine here.
    try:
        if get_start_method(allow_none=True) is None:
            import multiprocessing as mp
            mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass

    # Run
    results = []
    if args.workers == 1:
        for t in tqdm(task_args):
            results.append(_worker(t))
    else:
        with Pool(processes=args.workers) as pool:
            for res in tqdm(pool.imap_unordered(_worker, task_args), total=len(task_args)):
                results.append(res)

    # Report failures
    failed = [r for r in results if isinstance(r, str) and r.startswith("ERROR::")]
    if failed:
        print(f"❌ {len(failed)} file(s) failed:")
        for f in failed:
            print("   ", f)
    else:
        print("✅ All files processed successfully.")


if __name__ == "__main__":
    main()

