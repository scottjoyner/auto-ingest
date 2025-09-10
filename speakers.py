#!/usr/bin/env python3
import os, re, sys, subprocess, shutil
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# Optional: faster-whisper for local Python transcription fallback
try:
    from faster_whisper import WhisperModel  # type: ignore
    HAVE_FASTER_WHISPER = True
except Exception:
    HAVE_FASTER_WHISPER = False

import torch  # noqa: F401
from pyannote.audio import Pipeline

# ---------- Config ----------
DASHCAM_BASE           = Path("/mnt/8TB_2025/fileserver/dashcam")
DASHCAM_AUDIO_BASE     = Path("/mnt/8TB_2025/fileserver/dashcam/audio")
DASHCAM_TRANS_BASE     = Path("/mnt/8TB_2025/fileserver/dashcam/transcriptions")

BODYCAM_BASE           = Path("/mnt/8TB_2025/fileserver/bodycam")

AUDIO_BASE             = Path("/mnt/8TB_2025/fileserver/audio")
AUDIO_TRANS_BASE       = Path("/mnt/8TB_2025/fileserver/audio/transcriptions")

HF_TOKEN               = os.getenv("HF_TOKEN")  # do NOT hardcode tokens
MIN_SPEAKERS           = int(os.getenv("MIN_SPEAKERS", "1"))
MAX_SPEAKERS           = int(os.getenv("MAX_SPEAKERS", "5"))

# Transcription controls
DO_TRANSCRIBE          = os.getenv("DO_TRANSCRIBE", "1") not in {"0", "false", "False", ""}
WHISPER_MODEL_SIZE     = os.getenv("WHISPER_MODEL", "medium")  # for CLI or faster-whisper
WHISPER_LANGUAGE       = os.getenv("WHISPER_LANG", "en")

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a"}

# ---------- Helpers ----------
def run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def has_cmd(cmd: str) -> bool:
    return shutil.which(cmd) is not None

def walk_videos(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS)

def walk_audio(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS)

# --------- Key parsing & naming (dashcam & bodycam) ----------
def dashcam_key_and_suffix(stem: str) -> Optional[Tuple[str, Optional[str]]]:
    """
    DASHCAM file stems look like: YYYY_MMDD_HHMMSS_{F,R,FR,...}
    We keep legacy behavior: key = YYYY_MMDD_HHMMSS, and for *_R we drop suffix.
    Return (key, cam_suffix) where cam_suffix is None for *_R (legacy) else e.g. '_F', '_FR'.
    """
    m = re.match(r"^(\d{4}_\d{4}_\d{6})(?:_([A-Za-z]+))?$", stem)
    if not m:
        return None
    key = m.group(1)
    tail = m.group(2)
    # Legacy: only accept *_R for audio extraction; diarization uses base key
    if tail and tail.upper() != "R":
        return (key, f"_{tail.upper()}")
    return (key, None)

def is_dashcam_right_cam(path: Path) -> bool:
    return path.stem.upper().endswith("_R")

def bodycam_key_and_suffix(stem: str) -> Optional[Tuple[str, str]]:
    """
    BODYCAM stems like: 20250405212855_000029 or 20250405212855
    Convert to key 'YYYY_MMDD_HHMMSS' and suffix '_BC' or '_BC-000029' if index present.
    """
    m = re.match(r"^(\d{14})(?:_(\d+))?$", stem)  # 14-digit timestamp
    if not m:
        return None
    ts14 = m.group(1)
    idx = m.group(2)
    y, mo, d, hh, mm, ss = ts14[0:4], ts14[4:6], ts14[6:8], ts14[8:10], ts14[10:12], ts14[12:14]
    key = f"{y}_{mo}{d}_{hh}{mm}{ss}"
    suffix = f"_BC-{idx}" if idx else "_BC"
    return (key, suffix)

def audio_stem(key: str, suffix: Optional[str]) -> str:
    """Build audio base stem: dashcam R => key; others append suffix."""
    return f"{key}{suffix or ''}"

def out_mp3_path(key: str, suffix: Optional[str]) -> Path:
    ensure_dir(AUDIO_BASE)
    return AUDIO_BASE / f"{audio_stem(key, suffix)}.mp3"

def rttm_path_for(key: str, suffix: Optional[str]) -> Path:
    ensure_dir(AUDIO_BASE)
    return AUDIO_BASE / f"{audio_stem(key, suffix)}_speakers.rttm"

def transcript_paths_for(key: str, suffix: Optional[str]) -> Tuple[Path, Path]:
    ensure_dir(AUDIO_TRANS_BASE)
    base = AUDIO_TRANS_BASE / audio_stem(key, suffix)
    return base.with_suffix(".txt"), base.with_suffix(".csv")

def transcript_already_exists(key: str, suffix: Optional[str]) -> bool:
    txt, csv = transcript_paths_for(key, suffix)
    if txt.exists() and csv.exists():
        return True
    # Also check legacy dashcam transcription folder
    legacy_txt = (DASHCAM_TRANS_BASE / audio_stem(key, suffix)).with_suffix(".txt")
    legacy_csv = (DASHCAM_TRANS_BASE / audio_stem(key, suffix)).with_suffix(".csv")
    return legacy_txt.exists() and legacy_csv.exists()

# ---------- Extraction ----------
def extract_audio_ffmpeg(src_video: Path, dst_mp3: Path) -> None:
    if dst_mp3.exists():
        return
    ensure_dir(dst_mp3.parent)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", str(src_video),
        "-vn", "-ac", "1", "-ar", "16000", "-b:a", "128k",
        str(dst_mp3),
    ]
    run(cmd)

# ---------- Diarization ----------
def build_diarization_pipeline() -> Optional[Pipeline]:
    if not HF_TOKEN:
        print("WARN: HF_TOKEN not set; skipping diarization.", file=sys.stderr)
        return None
    return Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)

def diarize_once(pipeline: Optional[Pipeline], audio_path: Path, rttm_path: Path) -> None:
    if rttm_path.exists():
        return
    if pipeline is None:
        return
    diar = pipeline(str(audio_path), min_speakers=MIN_SPEAKERS, max_speakers=MAX_SPEAKERS)
    with rttm_path.open("w") as f:
        diar.write_rttm(f)

# ---------- Transcription ----------
def transcribe_cli_whisper(audio_path: Path, txt_out: Path, csv_out: Path) -> bool:
    """
    Use openai-whisper CLI if available. Returns True if success or already exists.
    """
    if txt_out.exists() and csv_out.exists():
        return True
    if not has_cmd("whisper"):
        return False
    ensure_dir(txt_out.parent)
    # whisper writes multiple formats; we ask for txt & csv to output_dir
    cmd = [
        "whisper", str(audio_path),
        "--model", WHISPER_MODEL_SIZE,
        "--language", WHISPER_LANGUAGE,
        "--task", "transcribe",
        "--output_dir", str(txt_out.parent),
        "--output_format", "txt,csv",
        "--verbose", "False",
    ]
    run(cmd)
    # whisper names files based on input basename; we need to move/rename to our exact stem
    produced_txt = txt_out.parent / (audio_path.stem + ".txt")
    produced_csv = txt_out.parent / (audio_path.stem + ".csv")
    # If whisper output used the input's stem, we're already aligned; otherwise rename if needed
    if produced_txt.exists() and produced_txt != txt_out:
        produced_txt.rename(txt_out)
    if produced_csv.exists() and produced_csv != csv_out:
        produced_csv.rename(csv_out)
    return txt_out.exists() and csv_out.exists()

def transcribe_faster_whisper(audio_path: Path, txt_out: Path, csv_out: Path) -> bool:
    if txt_out.exists() and csv_out.exists():
        return True
    if not HAVE_FASTER_WHISPER:
        return False
    ensure_dir(txt_out.parent)
    model = WhisperModel(WHISPER_MODEL_SIZE, device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16" if torch.cuda.is_available() else "int8")
    segments, _ = model.transcribe(str(audio_path), language=WHISPER_LANGUAGE)
    # Write .txt
    with txt_out.open("w", encoding="utf-8") as ft:
        for seg in segments:
            ft.write(seg.text.strip() + "\n")
    # Re-run to iterate again for CSV (segments is a generator)
    segments, _ = model.transcribe(str(audio_path), language=WHISPER_LANGUAGE)
    with csv_out.open("w", encoding="utf-8") as fc:
        fc.write("start,end,text\n")
        for seg in segments:
            fc.write(f"{seg.start:.2f},{seg.end:.2f},\"{seg.text.replace('\"','')}\"\n")
    return True

def transcribe_audio(audio_path: Path, key: str, suffix: Optional[str]) -> None:
    if not DO_TRANSCRIBE:
        return
    if transcript_already_exists(key, suffix):
        return
    txt_out, csv_out = transcript_paths_for(key, suffix)
    # Prefer CLI whisper if present; fallback to faster-whisper if available
    if transcribe_cli_whisper(audio_path, txt_out, csv_out):
        return
    if transcribe_faster_whisper(audio_path, txt_out, csv_out):
        return
    print(f"WARN: No transcription backend available for {audio_path.name}. Install `whisper` or `faster-whisper`.", file=sys.stderr)

# ---------- Main passes ----------
def process_dashcam_videos(pipeline: Optional[Pipeline]) -> None:
    right_videos = [vp for vp in walk_videos(DASHCAM_BASE) if is_dashcam_right_cam(vp)]
    print(f"🎥 DASHCAM: {len(right_videos)} rear-camera video(s) (*_R.*)")
    for vp in right_videos:
        try:
            parsed = dashcam_key_and_suffix(vp.stem)
            if not parsed:
                print(f"SKIP (DC parse): {vp.name}", file=sys.stderr); continue
            key, cam_suffix = parsed  # cam_suffix is None for *_R (legacy)
            mp3 = out_mp3_path(key, cam_suffix)  # cam_suffix None => legacy stem
            extract_audio_ffmpeg(vp, mp3)
            rttm = rttm_path_for(key, cam_suffix)
            diarize_once(pipeline, mp3, rttm)
            transcribe_audio(mp3, key, cam_suffix)
            print(f"OK DC: {vp.name} → {mp3.name}")
        except subprocess.CalledProcessError as e:
            print(f"ERR (ffmpeg DC): {vp} :: {e}", file=sys.stderr)
        except Exception as e:
            print(f"ERR (DC): {vp} :: {e}", file=sys.stderr)

def process_bodycam_videos(pipeline: Optional[Pipeline]) -> None:
    vids = list(walk_videos(BODYCAM_BASE))
    print(f"🎥 BODYCAM: {len(vids)} video(s)")
    for vp in vids:
        try:
            parsed = bodycam_key_and_suffix(vp.stem)
            if not parsed:
                print(f"SKIP (BC parse): {vp.name}", file=sys.stderr); continue
            key, bc_suffix = parsed
            mp3 = out_mp3_path(key, bc_suffix)
            extract_audio_ffmpeg(vp, mp3)
            rttm = rttm_path_for(key, bc_suffix)
            diarize_once(pipeline, mp3, rttm)
            transcribe_audio(mp3, key, bc_suffix)
            print(f"OK BC: {vp.name} → {mp3.name}")
        except subprocess.CalledProcessError as e:
            print(f"ERR (ffmpeg BC): {vp} :: {e}", file=sys.stderr)
        except Exception as e:
            print(f"ERR (BC): {vp} :: {e}", file=sys.stderr)

def process_audio_dirs_for_diarization_and_transcription(pipeline: Optional[Pipeline]) -> None:
    # Pass over centralized and legacy audio bases
    for root in [AUDIO_BASE, DASHCAM_AUDIO_BASE]:
        auds = list(walk_audio(root))
        print(f"🔊 AUDIO SWEEP: {len(auds)} file(s) in {root}")
        for ap in auds:
            try:
                # Try dashcam naming first, then bodycam
                parsed = dashcam_key_and_suffix(ap.stem)
                suffix = None
                if parsed:
                    key, suffix = parsed
                else:
                    parsed_bc = bodycam_key_and_suffix(ap.stem)
                    if not parsed_bc:
                        # If neither matches, skip (keeps strictness)
                        continue
                    key, suffix = parsed_bc

                rttm = rttm_path_for(key, suffix)
                diarize_once(pipeline, ap, rttm)
                transcribe_audio(ap, key, suffix)
            except Exception as e:
                print(f"ERR (audio sweep): {ap} :: {e}", file=sys.stderr)

def main():
    ensure_dir(AUDIO_BASE)
    ensure_dir(AUDIO_TRANS_BASE)

    # Build diarization pipeline once (if token present)
    pipeline = build_diarization_pipeline()

    # 1) Dashcam: extract from *_R only → diarize+transcribe
    process_dashcam_videos(pipeline)

    # 2) Bodycam: extract from ALL videos → diarize+transcribe
    process_bodycam_videos(pipeline)

    # 3) Sweep existing audio in both locations → fill any missing RTTM/transcripts
    process_audio_dirs_for_diarization_and_transcription(pipeline)

if __name__ == "__main__":
    main()
