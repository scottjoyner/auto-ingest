#!/usr/bin/env python3
"""
Dashcam Speaker Diarization → RTTM (one per CORE)

- Scans dashcam videos under /dashcam (dated tree) and extracts mono 16k MP3s
  into flat /dashcam/audio (KEEPING _F/_R/_FR in the filename).
- Builds one RTTM per CORE at /dashcam/transcriptions/<CORE>.rttm
  (CORE = YYYY_MMDD_HHMMSS; F/R/FR variants collapse to same RTTM).
- Skips MP3s named exactly <CORE>.mp3 to avoid duplicate/inneficient work.
- Also leverages base audio under /fileserver/audio (dated tree) when helpful.

Env:
  HF_TOKEN          : Hugging Face token for pyannote/speaker-diarization-3.1
  MIN_SPEAKERS      : default 1
  MAX_SPEAKERS      : default 5
"""

import os, re, sys, subprocess, argparse, logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Optional; if unavailable, we error at runtime when needed
try:
    from pyannote.audio import Pipeline
except Exception as e:
    Pipeline = None

# ===== Paths =====
DASHCAM_ROOT       = Path("/mnt/8TB_2025/fileserver/dashcam")     # dated videos live under here
DASHCAM_AUDIO_DIR  = DASHCAM_ROOT / "audio"                        # FLAT dir for extracted MP3s from dashcam
TRANSCRIPTIONS_DIR = DASHCAM_ROOT / "transcriptions"               # FLAT dir for RTTMs (one per CORE)
FILES_AUDIO_ROOT   = Path("/mnt/8TB_2025/fileserver/audio")        # dated audio tree (general/base audio)

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a"}

HF_TOKEN     = os.getenv("HF_TOKEN", "<HF_TOKEN>")
MIN_SPEAKERS = int(os.getenv("MIN_SPEAKERS", "1"))
MAX_SPEAKERS = int(os.getenv("MAX_SPEAKERS", "5"))

# ===== Logging =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("speakers")

# ===== Name parsing =====
CORE_RE = re.compile(r"^(?P<core>\d{4}_\d{4}_\d{6})$")
VIEW_RE = re.compile(r"^(?P<core>\d{4}_\d{4}_\d{6})(?P<view>_(?:FR|F|R))$")

def parse_core(stem: str) -> Optional[Tuple[str, Optional[str]]]:
    """Return (core, view) where view ∈ {None, _F, _R, _FR} if name matches."""
    m = VIEW_RE.match(stem)
    if m:
        return m.group("core"), m.group("view")
    m = CORE_RE.match(stem)
    if m:
        return m.group("core"), None
    return None

def is_core_mp3(p: Path) -> bool:
    """True if file is an MP3 named exactly CORE.mp3 (no _F/_R/_FR)."""
    return p.suffix.lower() == ".mp3" and parse_core(p.stem) == (p.stem, None)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def run(cmd: List[str]) -> None:
    """Run a command, raising with captured output on failure."""
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSTDERR:\n{proc.stderr}")

# ===== Discovery =====
def walk_videos(root: Path) -> Iterable[Path]:
    if not root.exists(): return []
    return (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS)

def walk_audio(root: Path) -> Iterable[Path]:
    if not root.exists(): return []
    for p in root.rglob("*"):
        if not p.is_file(): continue
        if p.suffix.lower() not in AUDIO_EXTS: continue
        got = parse_core(p.stem)
        if not got: continue
        # Exclude bare CORE.mp3 to avoid duplication
        if is_core_mp3(p):
            continue
        yield p

# ===== Extraction =====
def extract_audio_if_needed(video_path: Path, out_dir: Path) -> Path:
    """
    Extract mono 16k MP3 into flat /dashcam/audio, preserving the original stem
    (including _F/_R/_FR). This prevents collisions with core-only MP3s.
    """
    ensure_dir(out_dir)
    out_mp3 = out_dir / f"{video_path.stem}.mp3"
    if not out_mp3.exists():
        log.info(f"[extract] {video_path.name} → {out_mp3.name}")
        run([
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", "16000",
            "-b:a", "128k",
            str(out_mp3)
        ])
    return out_mp3

# ===== Pipeline (singleton) =====
_PIPELINE = None
def get_pipeline() -> "Pipeline":
    global _PIPELINE
    if _PIPELINE is None:
        if Pipeline is None:
            raise RuntimeError("pyannote.audio not installed; please `pip install pyannote.audio`.")
        if not HF_TOKEN or HF_TOKEN == "<HF_TOKEN>":
            raise RuntimeError("HF_TOKEN is not set. Export a valid Hugging Face token for pyannote.")
        log.info("Loading pyannote pipeline: pyannote/speaker-diarization-3.1 …")
        _PIPELINE = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN
        )
    return _PIPELINE

# ===== RTTM target =====
def rttm_path_for_core(core: str) -> Path:
    """RTTM lives in flat transcriptions dir named <CORE>.rttm."""
    ensure_dir(TRANSCRIPTIONS_DIR)
    return TRANSCRIPTIONS_DIR / f"{core}.rttm"

# ===== Strategy: choose one source per CORE =====
def build_core_sources() -> Dict[str, Path]:
    """
    Build a map of core -> audio path with precedence:
      1) dashcam/audio (flat)  [preferred]
      2) fileserver/audio (dated)
    Excludes bare CORE.mp3.
    """
    mapping: Dict[str, Path] = {}
    # 1) Preferred: dashcam/audio
    for ap in walk_audio(DASHCAM_AUDIO_DIR):
        core, view = parse_core(ap.stem)  # type: ignore
        if core not in mapping:
            mapping[core] = ap
    # 2) Fallback: fileserver/audio
    for ap in walk_audio(FILES_AUDIO_ROOT):
        core, view = parse_core(ap.stem)  # type: ignore
        if core not in mapping:
            mapping[core] = ap
    return mapping

# ===== Diarization =====
def diarize_to_rttm(audio_path: Path, out_rttm: Path, overwrite: bool = True, dry_run: bool = False) -> bool:
    """
    Run diarization and write RTTM. Returns True if wrote (new or replaced).
    """
    if out_rttm.exists() and not overwrite:
        return False
    if dry_run:
        log.info(f"[dry-run] would write RTTM: {out_rttm.name} (from {audio_path.name})")
        return True
    diar = get_pipeline()(str(audio_path), min_speakers=MIN_SPEAKERS, max_speakers=MAX_SPEAKERS)
    with out_rttm.open("w", encoding="utf-8") as f:
        diar.write_rttm(f)
    return True

# ===== CLI =====
def main():
    ap = argparse.ArgumentParser(description="Dashcam → Speaker RTTM (one per CORE)")
    ap.add_argument("--skip-extract", action="store_true",
                    help="Do not extract audio from dashcam videos (use existing MP3s only).")
    ap.add_argument("--overwrite", action="store_true", default=True,
                    help="Overwrite existing RTTMs (default: True).")
    ap.add_argument("--no-overwrite", dest="overwrite", action="store_false",
                    help="Do not overwrite existing RTTMs.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Plan only; do not run the diarization model.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N cores (after discovery).")
    ap.add_argument("--only-cores", type=str, nargs="*", default=None,
                    help="Restrict processing to these COREs (space-separated).")
    args = ap.parse_args()

    # 1) Optional extraction from dashcam videos → flat dashcam/audio
    if not args.skip_extract:
        vids = list(walk_videos(DASHCAM_ROOT))
        log.info(f"Found {len(vids)} dashcam video(s) to consider for extraction.")
        for vp in vids:
            got = parse_core(vp.stem)
            if not got:
                continue
            try:
                extract_audio_if_needed(vp, DASHCAM_AUDIO_DIR)
            except Exception as e:
                log.warning(f"[extract ERR] {vp}: {e}")

    # 2) Build core → audio source map (dashcam/audio preferred, then fileserver/audio)
    sources = build_core_sources()
    cores = sorted(sources.keys())

    if args.only-cores:
        only = set(args.only_cores)
        cores = [c for c in cores if c in only]
        missing = [c for c in args.only_cores if c not in sources]
        if missing:
            log.warning(f"--only-cores requested but not found: {missing}")

    if args.limit:
        cores = cores[:args.limit]

    log.info(f"Will process {len(cores)} core(s). overwrite={args.overwrite} dry_run={args.dry_run}")

    wrote, skipped = 0, 0
    for i, core in enumerate(cores, 1):
        src = sources[core]
        out = rttm_path_for_core(core)
        try:
            did = diarize_to_rttm(src, out, overwrite=args.overwrite, dry_run=args.dry_run)
            if did: wrote += 1
            else:   skipped += 1
            log.info(f"[{i}/{len(cores)}] {core} ← {src.name} → {out.name} ({'wrote' if did else 'skipped'})")
        except Exception as e:
            log.exception(f"[{i}/{len(cores)}] FAILED {core}: {e}")

    log.info(f"Done. wrote={wrote} skipped={skipped} total={len(cores)}")

if __name__ == "__main__":
    main()
