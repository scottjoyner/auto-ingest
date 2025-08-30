#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple, Iterable

import torch
import whisper
from tqdm import tqdm

# ---------- Utilities ----------

def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

def pick_device(user_device: Optional[str]) -> str:
    if user_device in ("cpu", "cuda"):
        return user_device
    return "cuda" if torch.cuda.is_available() else "cpu"

def run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)

def ffprobe_duration(path: Path) -> float:
    """Get duration (seconds, float) using ffprobe."""
    try:
        cp = run([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=nokey=1:noprint_wrappers=1",
            str(path)
        ])
        return float(cp.stdout.strip())
    except Exception as e:
        raise RuntimeError(f"ffprobe failed for {path}: {e}")

def _is_under(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False

_CHUNK_NAME_RE = re.compile(r"_chunk\d{4}$", re.IGNORECASE)

def _looks_like_our_chunk(p: Path) -> bool:
    # Matches e.g., *_chunk0001.wav/.mp3
    return bool(_CHUNK_NAME_RE.search(p.stem))

def list_audio_files(root: Path, exclude_dirs: Optional[List[Path]] = None) -> List[Path]:
    """
    Find audio files below root, excluding:
      - anything under exclude_dirs
      - any filename that matches our chunk pattern *_chunk####.*
    """
    exts = {".wav", ".mp3"}
    exclude_dirs = [d for d in (exclude_dirs or []) if d is not None and d.exists()]
    out = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        if any(_is_under(p, ex) for ex in exclude_dirs):
            continue
        if _looks_like_our_chunk(p):
            continue
        out.append(p)
    return sorted(out)

def mirror_dir(base: Path, file_path: Path, root: Path) -> Path:
    return base / file_path.relative_to(root).parent

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

# ---------- I/O formats ----------

def save_raw_result(whisper_result: dict, file_path: Path) -> None:
    # JSON content written to .txt per your pattern
    file_path.write_text(json.dumps(whisper_result, ensure_ascii=False, indent=2), encoding="utf-8")

def save_transcription_segments_to_csv(whisper_result: dict, file_path: Path) -> None:
    segments = whisper_result.get("segments") or []
    with file_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["SegmentIndex", "StartTime", "EndTime", "Text"])
        for i, seg in enumerate(segments):
            w.writerow([i, seg.get("start"), seg.get("end"), (seg.get("text") or "").strip()])

# ---------- Chunking ----------

@dataclass
class ChunkPlan:
    starts: List[float]   # start times in seconds
    length: float         # target chunk length (seconds)
    stride: float         # stride between starts (seconds)
    overlap: float        # overlap length (seconds)

def plan_chunks(duration: float, chunk_len: float = 63.0, stride: float = 60.0) -> ChunkPlan:
    if duration <= 60.0:
        return ChunkPlan(starts=[0.0], length=min(chunk_len, duration), stride=stride, overlap=chunk_len - stride)
    starts = []
    t = 0.0
    while t < duration:
        starts.append(t)
        t += stride
    return ChunkPlan(starts=starts, length=chunk_len, stride=stride, overlap=chunk_len - stride)

def chunk_output_name(stem: str, idx: int, ext: str) -> str:
    return f"{stem}_chunk{idx:04d}{ext}"

def cut_chunk(
    src: Path, dst: Path, start: float, length: float,
    fast_copy: bool = True, pcm_wav: bool = False
) -> None:
    """
    Cut a chunk via ffmpeg.
      - fast_copy=True uses stream copy (-c copy) (fast; MP3 cuts at frame boundaries)
      - pcm_wav=True forces decode->WAV PCM (accurate but larger files)
    """
    ensure_dir(dst.parent)
    if pcm_wav:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-t", f"{length:.3f}",
            "-i", str(src),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(dst.with_suffix(".wav")),
        ]
    elif fast_copy:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-t", f"{length:.3f}",
            "-i", str(src),
            "-c", "copy",
            str(dst),
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-t", f"{length:.3f}",
            "-i", str(src),
            "-vn",
            "-c:a", "aac",
            "-b:a", "160k",
            str(dst.with_suffix(".m4a")),
        ]
    try:
        run(cmd)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg chunking failed for {src} → {dst}: {e.stderr}")

# ---------- Whisper ----------

def load_model(model_name: str, device: str):
    try:
        return whisper.load_model(model_name, device=device)
    except Exception as e:
        print(f"❌ Failed to load model '{model_name}' on '{device}': {e}", file=sys.stderr)
        sys.exit(2)

def transcribe_file(
    model,
    audio_path: Path,
    device: str,
    language: Optional[str],
    temperature: float,
    beam_size: int,
) -> dict:
    fp16 = device == "cuda"
    return model.transcribe(
        str(audio_path),
        language=language,
        temperature=temperature,
        beam_size=beam_size,
        word_timestamps=True,
        condition_on_previous_text=True,
        fp16=fp16,
        verbose=False,
    )

# ---------- Merge helpers ----------

def merge_segments(results: List[Tuple[float, dict]], overlap: float) -> dict:
    merged_segments = []
    texts = []
    lang = None

    for idx, (chunk_start, res) in enumerate(results):
        if lang is None:
            lang = res.get("language")
        segs = res.get("segments") or []
        for s in segs:
            start = float(s.get("start", 0.0))
            end = float(s.get("end", 0.0))
            if idx > 0 and end <= overlap:
                continue
            seg_adj = {
                "start": start + chunk_start,
                "end": end + chunk_start,
                "text": (s.get("text") or "").strip(),
            }
            for k in ("avg_logprob", "compression_ratio", "no_speech_prob", "temperature"):
                if k in s and s[k] is not None:
                    seg_adj[k] = s[k]
            merged_segments.append(seg_adj)
            texts.append(seg_adj["text"])

    merged_text = " ".join(texts).strip()
    return {"language": lang, "text": merged_text, "segments": merged_segments}

# ---------- Finished detection ----------

def _merged_dump_path_for_model(merged_dir: Path, stem: str, model_name: str) -> Path:
    return merged_dir / f"{stem}_{model_name}_transcription.txt"

def _merged_csv_path(merged_dir: Path, stem: str) -> Path:
    return merged_dir / f"{stem}_transcription.csv"

def _is_up_to_date(src: Path, *outs: Path) -> bool:
    try:
        src_mtime = src.stat().st_mtime
        for o in outs:
            if not o.exists() or o.stat().st_mtime < src_mtime:
                return False
        return True
    except OSError:
        return False

def file_has_finished_transcription_for_model(
    audio_path: Path,
    audio_root: Path,
    trans_root: Path,
    model_name: str,
) -> bool:
    """Finished if both merged outputs exist (this model’s dump + generic CSV) and are at least as new as the source."""
    stem = audio_path.stem
    merged_dir = mirror_dir(trans_root, audio_path, audio_root)
    merged_dump = _merged_dump_path_for_model(merged_dir, stem, model_name)
    merged_csv = _merged_csv_path(merged_dir, stem)
    return _is_up_to_date(audio_path, merged_dump, merged_csv)

def file_has_finished_transcription_for_any_model(
    audio_path: Path,
    audio_root: Path,
    trans_root: Path,
    model_names: Iterable[str],
) -> Optional[str]:
    """Return the first model name for which the file is finished (up-to-date), else None."""
    for m in model_names:
        if file_has_finished_transcription_for_model(audio_path, audio_root, trans_root, m):
            return m
    return None

# ---------- Main ----------

def main():
    p = argparse.ArgumentParser(
        description="Transcribe audio; avoid chunking when possible; optionally chunk for long files."
    )
    p.add_argument("--audio-root", type=Path, default=Path("/mnt/8TB_2025/fileserver/audio"))
    p.add_argument("--transcriptions-root", type=Path, default=None,
                   help="Defaults to <audio-root>/transcriptions")
    p.add_argument("--chunks-root", type=Path, default=None,
                   help="Defaults to <audio-root>/chunks")

    p.add_argument("--model", choices=["medium", "large", "large-v3"], default="large-v3")
    p.add_argument("--device", choices=["cpu", "cuda"], default=None)
    p.add_argument("--language", default=None)
    p.add_argument("--beam-size", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.0)

    # Skipping and overwrite controls
    p.add_argument("--overwrite", action="store_true",
                   help="Recreate outputs even if they exist for the CURRENT model.")
    p.add_argument("--force-current-model", action="store_true",
                   help="Even if another model already produced finished outputs, still run CURRENT model.")
    p.add_argument("--skip-if-any-model", type=lambda s: s.lower() != "false", default=True,
                   help="If true (default), skip files finished by ANY known model.")
    p.add_argument("--known-models", default="medium,large,large-v3",
                   help="Comma-separated list of model names that count as 'already done'.")

    # Chunking policy
    p.add_argument("--direct-threshold", type=float, default=1800.0,
                   help="If duration ≤ this (seconds), transcribe the whole file directly (no chunks). Default: 1800 (30m).")
    p.add_argument("--no-chunking", action="store_true",
                   help="Never chunk; always transcribe files directly end-to-end.")

    # Chunking params (only used if chunking is chosen)
    p.add_argument("--chunk-len", type=float, default=63.0, help="Chunk length seconds.")
    p.add_argument("--stride", type=float, default=60.0, help="Chunk stride seconds. Overlap = chunk_len - stride.")
    p.add_argument("--fast-copy", action="store_true",
                   help="Use ffmpeg -c copy for chunking (fast; best for WAV; MP3 cuts at frame boundaries).")
    p.add_argument("--pcm-wav", action="store_true",
                   help="Decode chunks to PCM WAV (accurate timestamps; larger).")

    p.add_argument("--merge", action="store_true", help="Also write merged outputs for the full file.")
    args = p.parse_args()

    audio_root = args.audio_root
    trans_root = args.transcriptions_root or (audio_root / "transcriptions")
    chunks_root = args.chunks_root or (audio_root / "chunks")
    device = pick_device(args.device)
    overlap = args.chunk_len - args.stride
    if overlap <= 0:
        print("❌ chunk_len must be greater than stride (need positive overlap).", file=sys.stderr)
        sys.exit(1)

    if not audio_root.exists():
        print(f"❌ Audio root not found: {audio_root}", file=sys.stderr)
        sys.exit(1)
    ensure_dir(trans_root)
    ensure_dir(chunks_root)

    files = list_audio_files(audio_root, exclude_dirs=[chunks_root])
    if not files:
        print("ℹ️ No .wav/.mp3 files found.")
        return

    # Build the worklist BEFORE loading a heavy model
    known_models = [m.strip() for m in args.known_models.split(",") if m.strip()]
    worklist: List[Path] = []
    skipped_any_model = 0
    skipped_same_model = 0

    for audio_path in files:
        # Skip if current model already finished (unless overwrite)
        if (not args.overwrite) and file_has_finished_transcription_for_model(
            audio_path=audio_path,
            audio_root=audio_root,
            trans_root=trans_root,
            model_name=args.model,
        ):
            skipped_same_model += 1
            continue

        # Skip if any known model finished (unless forcing current)
        finished_by = None
        if args.skip_if_any_model and (not args.force_current_model):
            finished_by = file_has_finished_transcription_for_any_model(
                audio_path=audio_path,
                audio_root=audio_root,
                trans_root=trans_root,
                model_names=known_models,
            )
        if finished_by:
            skipped_any_model += 1
            continue

        worklist.append(audio_path)

    print(f"🔎 Found {len(files)} audio files under {audio_root}")
    print(f"🧠 Model: {args.model}  |  🖥️ Device: {device}")
    print(f"📁 Chunks: {chunks_root}")
    print(f"📁 Transcriptions: {trans_root}")
    print(f"✅ Ready to process: {len(worklist)}  |  ⏭️ skipped (same model): {skipped_same_model}  |  ⏭️ skipped (any model): {skipped_any_model}")

    if not worklist:
        print("Nothing to do. Exiting.")
        return

    # Load model only if we have work
    model = load_model(args.model, device)
    total_errors = 0

    for audio_path in worklist:
        # Decide direct vs chunked
        try:
            duration = ffprobe_duration(audio_path)
        except Exception as e:
            print(f"❌ Duration probe failed for {audio_path}: {e}", file=sys.stderr)
            total_errors += 1
            continue

        out_dir = mirror_dir(trans_root, audio_path, audio_root)
        ensure_dir(out_dir)

        stem = audio_path.stem
        merged_dump = _merged_dump_path_for_model(out_dir, stem, args.model)
        merged_csv  = _merged_csv_path(out_dir, stem)

        # Direct path if chosen
        should_chunk = (not args.no_chunking) and (duration > args.direct_threshold)

        if not should_chunk:
            # Direct full-file transcription
            try:
                res = transcribe_file(
                    model=model,
                    audio_path=audio_path,
                    device=device,
                    language=args.language,
                    temperature=args.temperature,
                    beam_size=args.beam_size,
                )
                if args.overwrite or not _is_up_to_date(audio_path, merged_dump, merged_csv):
                    save_raw_result(res, merged_dump)
                    save_transcription_segments_to_csv(res, merged_csv)
                print(f"🎯 Direct transcribed: {audio_path.name} ({duration/60:.1f} min)")
            except Exception as e:
                print(f"❌ Direct transcription failed for {audio_path}: {e}", file=sys.stderr)
                total_errors += 1
            continue

        # ---------- Chunked path (fallback for long files) ----------
        rel_dir = mirror_dir(chunks_root, audio_path, audio_root)
        ensure_dir(rel_dir)
        ext = audio_path.suffix
        plan = plan_chunks(duration, chunk_len=args.chunk_len, stride=args.stride)

        chunk_results: List[Tuple[float, dict]] = []

        for idx, start in enumerate(tqdm(plan.starts, desc=f"Chunking {audio_path.name}", unit="chunk")):
            chunk_name = chunk_output_name(stem, idx, ext)
            chunk_path = rel_dir / chunk_name

            if args.overwrite or not chunk_path.exists():
                try:
                    cut_chunk(
                        src=audio_path,
                        dst=chunk_path,
                        start=start,
                        length=plan.length,
                        fast_copy=args.fast_copy,
                        pcm_wav=args.pcm_wav,
                    )
                except Exception as e:
                    print(f"❌ Chunk creation failed for {audio_path} [{idx}]: {e}", file=sys.stderr)
                    total_errors += 1
                    continue

            dump_path = out_dir / f"{stem}_chunk{idx:04d}_{args.model}_transcription.txt"
            csv_path  = out_dir / f"{stem}_chunk{idx:04d}_transcription.csv"

            if not args.overwrite and dump_path.exists() and csv_path.exists():
                try:
                    res = json.loads(dump_path.read_text(encoding="utf-8"))
                    chunk_results.append((start, res))
                except Exception:
                    pass
                continue

            try:
                res = transcribe_file(
                    model=model,
                    audio_path=chunk_path,
                    device=device,
                    language=args.language,
                    temperature=args.temperature,
                    beam_size=args.beam_size,
                )
                save_raw_result(res, dump_path)
                save_transcription_segments_to_csv(res, csv_path)
                chunk_results.append((start, res))
            except Exception as e:
                print(f"❌ Transcription failed for chunk {chunk_path}: {e}", file=sys.stderr)
                total_errors += 1
                continue

        # Optional merged outputs for the full file
        if args.merge and chunk_results:
            merged = merge_segments(chunk_results, overlap=args.chunk_len - args.stride)
            if args.overwrite or not (merged_dump.exists() and merged_csv.exists()):
                save_raw_result(merged, merged_dump)
                save_transcription_segments_to_csv(merged, merged_csv)
            print(f"🧩 Merged chunks → {merged_dump.name}")

    if total_errors:
        print(f"✅ Done with {total_errors} error(s).")
    else:
        print("✅ Done.")


if __name__ == "__main__":
    main()
