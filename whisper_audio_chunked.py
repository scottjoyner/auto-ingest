#!/usr/bin/env python3
"""
Robust, chunk-aware Whisper transcriber.

Key features:
- Works reliably with WAV/MP3/M4A/FLAC/OGG/AAC inputs.
- Never stream-copies WAV (avoids corrupt headers); re-encodes to 16 kHz mono PCM.
- For compressed inputs, supports fast stream-copy or AAC (.m4a) re-encode.
- Always transcribes the ACTUAL chunk path written (no extension mismatches).
- ffprobe sanity-check + automatic fallback to PCM WAV for problematic chunks.
- Skips work if outputs are already finished (per-model or any model).
- Optional per-chunk CSV + merged CSV/JSON across chunks.
- Clean, configurable logging.

Example:
    python transcribe_chunks.py \
        --audio-root /mnt/8TB_2025/fileserver/audio \
        --model large-v3 \
        --fast-copy \
        --chunk-len 90 --stride 85 \
        --merge \
        --log-level INFO \
        --delete-chunks-after
"""

import argparse
import csv
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
import whisper
from tqdm import tqdm

# -------------------- Logging --------------------

LOG = logging.getLogger("transcribe_chunks")

def setup_logging(level: str) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# -------------------- Utilities --------------------

def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

def pick_device(user_device: Optional[str]) -> str:
    if user_device in ("cpu", "cuda"):
        return user_device
    return "cuda" if torch.cuda.is_available() else "cpu"

def run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        text=True,
    )

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
    # Matches e.g., *_chunk0001.wav/.m4a/.mp3
    return bool(_CHUNK_NAME_RE.search(p.stem))

def list_audio_files(root: Path, exclude_dirs: Optional[List[Path]] = None) -> List[Path]:
    """
    Find audio files below root, excluding:
      - anything under exclude_dirs
      - any filename that matches our chunk pattern *_chunk####.*
    """
    exts = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}
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

def safe_unlink(p: Path) -> None:
    try:
        if p.exists():
            p.unlink()
    except Exception as e:
        LOG.warning("Failed to delete %s: %s", p, e)


# -------------------- I/O formats --------------------

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


# -------------------- Chunking --------------------

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

def chunk_stem(stem: str, idx: int) -> str:
    return f"{stem}_chunk{idx:04d}"


# -------------------- Chunk strategies --------------------

_SUPPORTED_COPY_EXTS = {".mp3", ".m4a", ".aac", ".ogg", ".flac"}  # compressed/packetized

def decide_chunk_strategy(src: Path, fast_copy: bool, pcm_wav: bool) -> Tuple[str, str]:
    """
    Returns (strategy, out_ext):
      strategy in {"pcm_wav","copy","aac"}
      out_ext includes dot, e.g., ".wav"
    """
    src_ext = src.suffix.lower()
    if pcm_wav or src_ext == ".wav":
        return ("pcm_wav", ".wav")
    if fast_copy and src_ext in _SUPPORTED_COPY_EXTS:
        return ("copy", src_ext)
    return ("aac", ".m4a")

def cut_chunk_to(
    src: Path, base_noext: Path, start: float, length: float, strategy: str, out_ext: str
) -> Path:
    """
    Write the chunk to base_noext with the selected out_ext and return the actual Path written.
    """
    ensure_dir(base_noext.parent)
    out = base_noext.with_suffix(out_ext)
    if strategy == "pcm_wav":
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}", "-t", f"{length:.3f}",
            "-i", str(src),
            "-vn",
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(out),
        ]
    elif strategy == "copy":
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}", "-t", f"{length:.3f}",
            "-i", str(src),
            "-c", "copy",
            str(out),
        ]
    else:  # "aac"
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}", "-t", f"{length:.3f}",
            "-i", str(src),
            "-vn",
            "-c:a", "aac", "-b:a", "160k",
            str(out),
        ]
    try:
        cp = run(cmd)
        if LOG.isEnabledFor(logging.DEBUG):
            LOG.debug("ffmpeg out: %s", cp.stdout.strip())
            LOG.debug("ffmpeg err: %s", cp.stderr.strip())
        return out
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg chunking failed for {src} → {out}: {e.stderr}")


# -------------------- Whisper --------------------

def load_model(model_name: str, device: str):
    try:
        return whisper.load_model(model_name, device=device)
    except Exception as e:
        LOG.error("Failed to load model '%s' on '%s': %s", model_name, device, e)
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


# -------------------- Merge helpers --------------------

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
            # Drop overlap from non-first chunks
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


# -------------------- Finished detection --------------------

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


# -------------------- Main --------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Transcribe audio; avoid chunking when possible; robust chunking for long files with safe formats."
    )
    p.add_argument("--audio-root", type=Path, default=Path("/mnt/8TB_2025/fileserver/audio"),
                   help="Root directory containing source audio files (wav/mp3/m4a/flac/ogg/aac).")
    p.add_argument("--transcriptions-root", type=Path, default=None,
                   help="Where to write transcription outputs. Default: <audio-root>/transcriptions")
    p.add_argument("--chunks-root", type=Path, default=None,
                   help="Where to write chunked audio. Default: <audio-root>/chunks")

    p.add_argument("--model", choices=["medium", "large", "large-v3"], default="large-v3",
                   help="Whisper model to use.")
    p.add_argument("--device", choices=["cpu", "cuda"], default=None,
                   help="Force a device. Default: auto-detect (cuda if available).")
    p.add_argument("--language", default=None,
                   help="Hint language code for Whisper (optional).")
    p.add_argument("--beam-size", type=int, default=5,
                   help="Beam size for decoding.")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Decoding temperature.")

    # Skipping and overwrite controls
    p.add_argument("--overwrite", action="store_true",
                   help="Recreate outputs even if they exist for the CURRENT model.")
    p.add_argument("--force-current-model", action="store_true",
                   help="Even if another model already produced finished outputs, still run CURRENT model.")
    p.add_argument("--skip-if-any-model", type=lambda s: s.lower() != "false", default=True,
                   help="If true (default), skip files finished by ANY known model. Pass 'false' to disable.")
    p.add_argument("--known-models", default="medium,large,large-v3",
                   help="Comma-separated list of model names that count as 'already done' when --skip-if-any-model is on.")

    # Chunking policy
    p.add_argument("--direct-threshold", type=float, default=1800.0,
                   help="If duration ≤ this (seconds), transcribe the whole file directly (no chunks). Default: 1800 (30m).")
    p.add_argument("--no-chunking", action="store_true",
                   help="Never chunk; always transcribe files directly end-to-end.")

    # Chunking params (only used if chunking is chosen)
    p.add_argument("--chunk-len", type=float, default=63.0, help="Chunk length seconds.")
    p.add_argument("--stride", type=float, default=60.0, help="Chunk stride seconds. Overlap = chunk_len - stride.")

    # Output format choices for chunks
    p.add_argument("--fast-copy", action="store_true",
                   help="For compressed inputs (mp3/m4a/ogg/aac/flac), attempt -c copy (fast). "
                        "Ignored for WAV (WAV always re-encodes to valid PCM).")
    p.add_argument("--pcm-wav", action="store_true",
                   help="Force chunk output to 16 kHz mono PCM WAV even for compressed inputs "
                        "(most deterministic for Whisper).")

    # Post-processing & housekeeping
    p.add_argument("--merge", action="store_true",
                   help="Also write merged outputs for the full file (JSON dump and CSV).")
    p.add_argument("--delete-chunks-after", action="store_true",
                   help="Delete individual chunk audio files after they are successfully transcribed.")

    # Logging
    p.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")

    args = p.parse_args(argv)
    return args

def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    setup_logging(args.log_level)

    audio_root = args.audio_root
    trans_root = args.transcriptions_root or (audio_root / "transcriptions")
    chunks_root = args.chunks_root or (audio_root / "chunks")
    device = pick_device(args.device)
    overlap = args.chunk_len - args.stride
    if overlap <= 0:
        LOG.error("chunk_len must be greater than stride (need positive overlap).")
        sys.exit(1)

    if not audio_root.exists():
        LOG.error("Audio root not found: %s", audio_root)
        sys.exit(1)
    ensure_dir(trans_root)
    ensure_dir(chunks_root)

    files = list_audio_files(audio_root, exclude_dirs=[chunks_root])
    if not files:
        LOG.info("No supported audio files found under %s.", audio_root)
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

    LOG.info("🔎 Found %d audio files under %s", len(files), audio_root)
    LOG.info("🧠 Model: %s  |  🖥️ Device: %s", args.model, device)
    LOG.info("📁 Chunks dir: %s", chunks_root)
    LOG.info("📁 Transcriptions dir: %s", trans_root)
    LOG.info("✅ Ready to process: %d  |  ⏭️ skipped (same model): %d  |  ⏭️ skipped (any model): %d",
             len(worklist), skipped_same_model, skipped_any_model)

    if not worklist:
        LOG.info("Nothing to do. Exiting.")
        return

    # Load model only if we have work
    model = load_model(args.model, device)
    total_errors = 0

    for audio_path in worklist:
        # Decide direct vs chunked
        try:
            duration = ffprobe_duration(audio_path)
        except Exception as e:
            LOG.error("Duration probe failed for %s: %s", audio_path, e)
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
                LOG.info("🎯 Direct transcribed: %s (%.1f min)", audio_path.name, duration/60.0)
            except Exception as e:
                LOG.error("Direct transcription failed for %s: %s", audio_path, e)
                total_errors += 1
            continue

        # ---------- Chunked path (for long files) ----------
        rel_dir = mirror_dir(chunks_root, audio_path, audio_root)
        ensure_dir(rel_dir)
        plan = plan_chunks(duration, chunk_len=args.chunk_len, stride=args.stride)

        chunk_results: List[Tuple[float, dict]] = []

        for idx, start in enumerate(tqdm(plan.starts, desc=f"Chunking {audio_path.name}", unit="chunk")):
            stem_chunk = chunk_stem(stem, idx)
            base_noext = rel_dir / stem_chunk

            strategy, out_ext = decide_chunk_strategy(audio_path, args.fast_copy, args.pcm_wav)
            chunk_path = base_noext.with_suffix(out_ext)

            # (Re)create chunk if needed
            if args.overwrite or not chunk_path.exists():
                try:
                    chunk_path = cut_chunk_to(
                        src=audio_path,
                        base_noext=base_noext,
                        start=start,
                        length=plan.length,
                        strategy=strategy,
                        out_ext=out_ext,
                    )
                except Exception as e:
                    LOG.error("Chunk creation failed for %s [%d]: %s", audio_path, idx, e)
                    total_errors += 1
                    continue

                # quick sanity check; fallback to PCM WAV if needed
                try:
                    _ = ffprobe_duration(chunk_path)
                except Exception:
                    LOG.warning("Chunk %s unreadable; falling back to PCM WAV.", chunk_path.name)
                    try:
                        chunk_path = cut_chunk_to(
                            src=audio_path,
                            base_noext=base_noext,
                            start=start,
                            length=plan.length,
                            strategy="pcm_wav",
                            out_ext=".wav",
                        )
                        _ = ffprobe_duration(chunk_path)
                    except Exception as e2:
                        LOG.error("Chunk sanity check/fallback failed [%s]: %s", chunk_path.name, e2)
                        total_errors += 1
                        continue

            dump_path = out_dir / f"{stem_chunk}_{args.model}_transcription.txt"
            csv_path  = out_dir / f"{stem_chunk}_transcription.csv"

            # Reuse if already present (unless overwrite)
            if not args.overwrite and dump_path.exists() and csv_path.exists():
                try:
                    res = json.loads(dump_path.read_text(encoding="utf-8"))
                    chunk_results.append((start, res))
                except Exception:
                    pass
                finally:
                    if args.delete_chunks_after:
                        safe_unlink(chunk_path)
                continue

            try:
                res = transcribe_file(
                    model=model,
                    audio_path=chunk_path,  # USE THE ACTUAL PATH WE WROTE
                    device=device,
                    language=args.language,
                    temperature=args.temperature,
                    beam_size=args.beam_size,
                )
                save_raw_result(res, dump_path)
                save_transcription_segments_to_csv(res, csv_path)
                chunk_results.append((start, res))
            except Exception as e:
                LOG.error("Transcription failed for chunk %s: %s", chunk_path, e)
                total_errors += 1
            finally:
                if args.delete_chunks_after:
                    safe_unlink(chunk_path)

        # Optional merged outputs for the full file
        if args.merge and chunk_results:
            merged = merge_segments(chunk_results, overlap=args.chunk_len - args.stride)
            if args.overwrite or not (merged_dump.exists() and merged_csv.exists()):
                save_raw_result(merged, merged_dump)
                save_transcription_segments_to_csv(merged, merged_csv)
            LOG.info("🧩 Merged chunks → %s", merged_dump.name)

    if total_errors:
        LOG.info("✅ Done with %d error(s).", total_errors)
    else:
        LOG.info("✅ Done.")


if __name__ == "__main__":
    main()


# ./.venv/bin/python3 whisper_audio_chunked.py \
#   --model medium \
#   --device cuda \
#   --merge \
#   --audio-root /mnt/8TB_2025/fileserver/audio \
#   --transcriptions-root /mnt/8TB_2025/fileserver/audio/transcriptions \
#   --chunks-root /tmp/chunks \
#   --pcm-wav \
#   --chunk-len 90 --stride 85 \
#   --delete-chunks-after \
#   --log-level INFO
