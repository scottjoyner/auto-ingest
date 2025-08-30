#!/usr/bin/env python3
import os, re, sys, subprocess, json, time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from pyannote.audio import Pipeline

# ---------- Config ----------
DASHCAM_BASE = Path("/mnt/8TB_2025/fileserver/dashcam")
AUDIO_BASE   = Path("/mnt/8TB_2025/fileserver/audio")  # central audio home (and where *_speakers.rttm lives)
HF_TOKEN     = os.getenv("HF_TOKEN", "<HF_TOKEN>")
MIN_SPEAKERS = 1
MAX_SPEAKERS = 5

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a"}

# ---------- Helpers ----------
def run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def rel_to_base(p: Path, base_candidates: List[Path]) -> Path:
    for b in base_candidates:
        try:
            return p.resolve().relative_to(b.resolve())
        except Exception:
            continue
    # fallback: just use filename
    return Path(p.name)

def is_date_tree(p: Path) -> bool:
    # accept /YYYY/MM/DD somewhere in path
    try:
        parts = [pp for pp in p.parts]
        for i in range(len(parts) - 3):
            candidate = "/".join(parts[i:i+3])
            try:
                datetime.strptime(candidate, "%Y/%m/%d")
                return True
            except Exception:
                pass
        return False
    except Exception:
        return False

def walk_dashcam_videos(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS)

def walk_audio(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS)

def extract_audio_if_needed(video_path: Path, out_root: Path) -> Path:
    """
    Extract mono 16k MP3 to AUDIO_BASE mirroring the source path.
    """
    rel = rel_to_base(video_path.parent, [DASHCAM_BASE, AUDIO_BASE])
    dst_dir = out_root / rel
    ensure_dir(dst_dir)
    stem = re.sub(r"_R$", "", video_path.stem)  # keep your legacy key format
    out_mp3 = dst_dir / f"{stem}.mp3"
    if not out_mp3.exists():
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", "16000",
            "-b:a", "128k",
            str(out_mp3)
        ]
        run(cmd)
    return out_mp3

def diarize_to_rttm(audio_path: Path, rttm_path: Path, speakers: int = 0) -> None:
    ensure_dir(rttm_path.parent)
    if rttm_path.exists():
        return
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
    diarization = pipeline(str(audio_path), min_speakers=MIN_SPEAKERS, max_speakers=MAX_SPEAKERS)
    with rttm_path.open("w") as f:
        diarization.write_rttm(f)
    del pipeline

def rttm_target_for(audio_or_video: Path) -> Path:
    # RTTM always goes alongside the central audio under AUDIO_BASE, mirroring structure
    if audio_or_video.suffix.lower() in VIDEO_EXTS:
        rel = rel_to_base(audio_or_video.parent, [DASHCAM_BASE])
        stem = re.sub(r"_R$", "", audio_or_video.stem)
        return (AUDIO_BASE / rel / f"{stem}_speakers.rttm")
    else:
        # already an audio file: write next to its mirrored location under AUDIO_BASE
        rel = rel_to_base(audio_or_video.parent, [AUDIO_BASE, DASHCAM_BASE])
        return (AUDIO_BASE / rel / f"{audio_or_video.stem}_speakers.rttm")

def main():
    # Pass 1: dashcam videos → extract audio → diarize
    videos = list(walk_dashcam_videos(DASHCAM_BASE))
    print(f"🎥 Found {len(videos)} dashcam video(s)")
    for vp in videos:
        try:
            mp3_path = extract_audio_if_needed(vp, AUDIO_BASE)
            rttm_path = rttm_target_for(vp)
            diarize_to_rttm(mp3_path, rttm_path)
            print(f"OK: {vp.name} → {mp3_path.name} → {rttm_path.name}")
        except Exception as e:
            print(f"ERR (video): {vp} :: {e}", file=sys.stderr)

    # Pass 2: standalone audio under AUDIO_BASE → diarize directly
    auds = list(walk_audio(AUDIO_BASE))
    print(f"🔊 Found {len(auds)} standalone audio file(s)")
    for ap in auds:
        try:
            rttm_path = rttm_target_for(ap)
            diarize_to_rttm(ap, rttm_path)
            print(f"OK: {ap.name} → {rttm_path.name}")
        except Exception as e:
            print(f"ERR (audio): {ap} :: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
