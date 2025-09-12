#!/usr/bin/env python3
import os, re, sys, subprocess, shutil, time, logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict

# Optional: faster-whisper for local Python transcription fallback
try:
    from faster_whisper import WhisperModel  # type: ignore
    HAVE_FASTER_WHISPER = True
except Exception:
    HAVE_FASTER_WHISPER = False

import torch  # noqa: F401
from pyannote.audio import Pipeline

# ========= Config =========
DASHCAM_BASE           = Path("/mnt/8TB_2025/fileserver/dashcam")
DASHCAM_AUDIO_BASE     = Path("/mnt/8TB_2025/fileserver/dashcam/audio")           # legacy source (read-only)
DASHCAM_TRANS_BASE     = Path("/mnt/8TB_2025/fileserver/dashcam/transcriptions")  # legacy source (read-only)

BODYCAM_BASE           = Path("/mnt/8TB_2025/fileserver/bodycam")                 # source (read-only)

AUDIO_BASE             = Path("/mnt/8TB_2025/fileserver/audio")                   # canonical sink (read+write)

HF_TOKEN               = os.getenv("HF_TOKEN", "")
MIN_SPEAKERS           = int(os.getenv("MIN_SPEAKERS", "1"))
MAX_SPEAKERS           = int(os.getenv("MAX_SPEAKERS", "5"))

DO_TRANSCRIBE          = os.getenv("DO_TRANSCRIBE", "1") not in {"0", "false", "False", ""}
WHISPER_MODEL_SIZE     = os.getenv("WHISPER_MODEL", "medium")  # for CLI or faster-whisper
WHISPER_LANGUAGE       = os.getenv("WHISPER_LANG", "en")

LOG_LEVEL              = os.getenv("LOG_LEVEL", "INFO").upper()  # DEBUG/INFO/WARN/ERROR
EMA_ALPHA              = float(os.getenv("EMA_ALPHA", "0.2"))    # rolling average smoothing

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".MP4", ".MOV", ".MKV", ".AVI"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".WAV", ".MP3", ".M4A"}

# ========= Logging Setup =========
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("speakers")

# ========= Stats & Timing =========
class StageStats:
    """Tracks counts, total time, and EMA per named stage."""
    def __init__(self, name: str, alpha: float = 0.2):
        self.name = name
        self.alpha = alpha
        self.count = 0
        self.total = 0.0
        self.ema = None  # type: Optional[float]

    def update(self, dt: float):
        self.count += 1
        self.total += dt
        if self.ema is None:
            self.ema = dt
        else:
            self.ema = self.alpha * dt + (1 - self.alpha) * self.ema

    @property
    def avg(self) -> float:
        return (self.total / self.count) if self.count else 0.0

    def summary(self) -> str:
        ema = f"{self.ema:.2f}s" if self.ema is not None else "n/a"
        avg = f"{self.avg:.2f}s"
        return f"{self.name}: n={self.count}, avg={avg}, ema={ema}, total={self.total:.2f}s"

class TimedStage:
    """Context manager for timing a single stage and updating stats."""
    def __init__(self, stats: StageStats, detail: str = ""):
        self.stats = stats
        self.detail = detail
        self.start = None
        self.dt = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        end = time.perf_counter()
        self.dt = end - self.start
        # update only for successful completion
        if exc is None:
            self.stats.update(self.dt)
            log.info(f"[stage:{self.stats.name}] done in {self.dt:.2f}s | {self.stats.summary()} | {self.detail}")
        else:
            log.error(f"[stage:{self.stats.name}] FAILED after {self.dt:.2f}s | {self.detail}")
        return False  # propagate exceptions

GLOBAL_START = time.perf_counter()
st_extract = StageStats("extract", EMA_ALPHA)
st_diar    = StageStats("diarize", EMA_ALPHA)
st_trans   = StageStats("transcribe", EMA_ALPHA)

# ========= Helpers =========
def run(cmd: List[str]) -> None:
    log.debug(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def has_cmd(cmd: str) -> bool:
    return shutil.which(cmd) is not None

def walk_videos(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return (p for p in root.rglob("*") if p.is_file() and p.suffix in VIDEO_EXTS)

def walk_audio(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return (p for p in root.rglob("*") if p.is_file() and p.suffix in AUDIO_EXTS)

# ========= Key parsing =========
def parse_dashcam_key_and_suffix(stem: str) -> Optional[Tuple[str, Optional[str]]]:
    m = re.match(r"^(\d{4}_\d{4}_\d{6})(?:_([A-Za-z]+))?$", stem)
    if not m:
        return None
    key = m.group(1)
    tail = m.group(2)
    if tail:
        t = tail.upper()
        if t == "R":  # legacy behavior
            return (key, None)
        return (key, f"_{t}")
    return (key, None)

def parse_bodycam_key_and_suffix(stem: str) -> Optional[Tuple[str, Optional[str]]]:
    m = re.match(r"^(\d{14})(?:_(\d+))?$", stem)
    if not m:
        return None
    key = m.group(1)
    idx = m.group(2)
    suffix = f"_BC-{idx}" if idx else "_BC"
    return (key, suffix)

def parse_any_key_and_suffix(stem: str) -> Optional[Tuple[str, Optional[str]]]:
    return parse_dashcam_key_and_suffix(stem) or parse_bodycam_key_and_suffix(stem)

def key_to_date_parts(key: str) -> Tuple[str, str, str]:
    if "_" in key:
        y = key[0:4]; mo = key[5:7]; d = key[7:9]
    else:
        y = key[0:4]; mo = key[4:6]; d = key[6:8]
    return y, mo, d

def audio_stem(key: str, suffix: Optional[str]) -> str:
    return f"{key}{suffix or ''}"

def target_day_dir_for_key(key: str) -> Path:
    y, mo, d = key_to_date_parts(key)
    day_dir = AUDIO_BASE / y / mo / d
    ensure_dir(day_dir)
    return day_dir

# ========= Output paths (canonical in AUDIO_BASE/YYYY/MM/DD) =========
def out_mp3_path(key: str, suffix: Optional[str]) -> Path:
    return target_day_dir_for_key(key) / f"{audio_stem(key, suffix)}.mp3"

def rttm_path_for(key: str, suffix: Optional[str]) -> Path:
    return target_day_dir_for_key(key) / f"{audio_stem(key, suffix)}_speakers.rttm"

def transcript_paths_for(key: str, suffix: Optional[str]) -> Tuple[Path, Path]:
    base = target_day_dir_for_key(key) / audio_stem(key, suffix)
    txt = base.with_name(f"{base.name}_{WHISPER_MODEL_SIZE}_transcription.txt")
    csv = base.with_name(f"{base.name}_transcription.csv")
    return txt, csv

def transcript_already_exists(key: str, suffix: Optional[str]) -> bool:
    txt, csv = transcript_paths_for(key, suffix)
    if txt.exists() and csv.exists():
        return True
    # Consider any model TXT sufficient if CSV exists
    alt_txt = list(txt.parent.glob(f"{audio_stem(key, suffix)}_*_transcription.txt"))
    return csv.exists() and any(p.exists() for p in alt_txt)

# ========= Stages =========
def extract_audio_ffmpeg(src_video: Path, dst_mp3: Path) -> None:
    detail = f"src={src_video.resolve()} dst={dst_mp3.resolve()}"
    if dst_mp3.exists():
        log.info(f"[extract] SKIP exists | {detail}")
        return
    ensure_dir(dst_mp3.parent)
    with TimedStage(st_extract, detail=detail):
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-i", str(src_video),
            "-vn", "-ac", "1", "-ar", "16000", "-b:a", "128k",
            str(dst_mp3),
        ]
        run(cmd)

def build_diarization_pipeline() -> Optional[Pipeline]:
    if not HF_TOKEN:
        log.warning("HF_TOKEN not set; diarization will be skipped.")
        return None
    log.info("Loading pyannote diarization pipeline…")
    t0 = time.perf_counter()
    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
    log.info(f"pyannote loaded in {time.perf_counter() - t0:.2f}s")
    return pipe

def diarize_once(pipeline: Optional[Pipeline], audio_path: Path, rttm_path: Path) -> None:
    detail = f"audio={audio_path.resolve()} out={rttm_path.resolve()}"
    if rttm_path.exists():
        log.info(f"[diarize] SKIP exists | {detail}")
        return
    if pipeline is None:
        log.info(f"[diarize] SKIP (no pipeline) | {detail}")
        return
    with TimedStage(st_diar, detail=detail):
        diar = pipeline(str(audio_path), min_speakers=MIN_SPEAKERS, max_speakers=MAX_SPEAKERS)
        with rttm_path.open("w") as f:
            diar.write_rttm(f)

def transcribe_cli_whisper(audio_path: Path, key: str, suffix: Optional[str]) -> bool:
    txt_out, csv_out = transcript_paths_for(key, suffix)
    detail = f"audio={audio_path.resolve()} txt={txt_out.resolve()} csv={csv_out.resolve()}"
    if txt_out.exists() and csv_out.exists():
        log.info(f"[transcribe] SKIP exists | {detail}")
        return True
    if not has_cmd("whisper"):
        log.debug("[transcribe] whisper CLI not found")
        return False

    ensure_dir(txt_out.parent)
    with TimedStage(st_trans, detail=detail):
        cmd = [
            "whisper", str(audio_path),
            "--model", WHISPER_MODEL_SIZE,
            "--language", WHISPER_LANGUAGE,
            "--task", "transcribe",
            "--output_dir", str(txt_out.parent),
            "--output_format", "all",
            "--verbose", "False",
        ]
        run(cmd)

        produced_txt = txt_out.parent / (audio_path.stem + ".txt")
        produced_csv = txt_out.parent / (audio_path.stem + ".csv")

        if produced_txt.exists():
            produced_txt.rename(txt_out)
        if produced_csv.exists():
            produced_csv.rename(csv_out)

    ok = txt_out.exists() and csv_out.exists()
    log.info(f"[transcribe] RESULT via whisper CLI: {'OK' if ok else 'MISSING'} | {detail}")
    return ok

def transcribe_faster_whisper(audio_path: Path, key: str, suffix: Optional[str]) -> bool:
    txt_out, csv_out = transcript_paths_for(key, suffix)
    detail = f"audio={audio_path.resolve()} txt={txt_out.resolve()} csv={csv_out.resolve()}"
    if txt_out.exists() and csv_out.exists():
        log.info(f"[transcribe] SKIP exists | {detail}")
        return True
    if not HAVE_FASTER_WHISPER:
        log.debug("[transcribe] faster-whisper not available")
        return False

    ensure_dir(txt_out.parent)
    with TimedStage(st_trans, detail=detail):
        model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16" if torch.cuda.is_available() else "int8",
        )

        segments, _ = model.transcribe(str(audio_path), language=WHISPER_LANGUAGE)
        with txt_out.open("w", encoding="utf-8") as ft:
            for seg in segments:
                ft.write(seg.text.strip() + "\n")

        # Re-run to iterate again for CSV (generator)
        segments, _ = model.transcribe(str(audio_path), language=WHISPER_LANGUAGE)
        with csv_out.open("w", encoding="utf-8") as fc:
            fc.write("start,end,text\n")
            for seg in segments:
                text = seg.text.replace('"', '')
                fc.write(f"{seg.start:.2f},{seg.end:.2f},\"{text}\"\n")

    log.info(f"[transcribe] RESULT via faster-whisper: OK | {detail}")
    return True

def transcribe_audio(audio_path: Path, key: str, suffix: Optional[str]) -> None:
    if not DO_TRANSCRIBE:
        log.info(f"[transcribe] SKIP (DO_TRANSCRIBE=0) | audio={audio_path.resolve()} key={key} suffix={suffix}")
        return
    if transcript_already_exists(key, suffix):
        txt_out, csv_out = transcript_paths_for(key, suffix)
        log.info(f"[transcribe] SKIP exists | txt={txt_out.resolve()} csv={csv_out.resolve()} key={key} suffix={suffix}")
        return
    if transcribe_cli_whisper(audio_path, key, suffix):
        return
    if transcribe_faster_whisper(audio_path, key, suffix):
        return
    log.warning(f"[transcribe] NO BACKEND available | audio={audio_path.resolve()} key={key} suffix={suffix}")

# ========= Main passes with BEGIN/END banners =========
def _begin_banner(kind: str, src: Path, key: str, suffix: Optional[str], mp3: Path, rttm: Path, txt: Path, csv: Path):
    log.info(
        f"===== BEGIN {kind} =====\n"
        f" key={key} suffix={suffix}\n"
        f" src={src.resolve()}\n"
        f" mp3={mp3.resolve()}\n"
        f" rttm={rttm.resolve()}\n"
        f" txt={txt.resolve()}\n"
        f" csv={csv.resolve()}"
    )

def _end_banner(kind: str, key: str):
    elapsed = time.perf_counter() - GLOBAL_START
    log.info(
        f"===== END {kind} ===== key={key} | "
        f"elapsed={elapsed:.2f}s | "
        f"{st_extract.summary()} | {st_diar.summary()} | {st_trans.summary()}"
    )

def process_dashcam_videos(pipeline: Optional[Pipeline]) -> None:
    right_videos = [vp for vp in walk_videos(DASHCAM_BASE) if vp.stem.upper().endswith("_R")]
    log.info(f"DASHCAM rear-camera videos detected: {len(right_videos)}")
    for vp in right_videos:
        try:
            parsed = parse_dashcam_key_and_suffix(vp.stem)
            if not parsed:
                log.warning(f"SKIP (DC parse) name={vp.name} path={vp.resolve()}")
                continue
            key, cam_suffix = parsed
            mp3 = out_mp3_path(key, cam_suffix)
            rttm = rttm_path_for(key, cam_suffix)
            txt, csv = transcript_paths_for(key, cam_suffix)

            _begin_banner("DASHCAM", vp, key, cam_suffix, mp3, rttm, txt, csv)
            extract_audio_ffmpeg(vp, mp3)
            diarize_once(pipeline, mp3, rttm)
            transcribe_audio(mp3, key, cam_suffix)
            _end_banner("DASHCAM", key)
        except subprocess.CalledProcessError as e:
            log.error(f"ERR (ffmpeg DC) key={key if 'key' in locals() else '?'} src={vp.resolve()} :: {e}")
        except Exception as e:
            log.exception(f"ERR (DC) key={key if 'key' in locals() else '?'} src={vp.resolve()} :: {e}")

def process_bodycam_videos(pipeline: Optional[Pipeline]) -> None:
    vids = list(walk_videos(BODYCAM_BASE))
    log.info(f"BODYCAM videos detected: {len(vids)}")
    for vp in vids:
        try:
            parsed = parse_bodycam_key_and_suffix(vp.stem)
            if not parsed:
                log.warning(f"SKIP (BC parse) name={vp.name} path={vp.resolve()}")
                continue
            key, bc_suffix = parsed
            mp3 = out_mp3_path(key, bc_suffix)
            rttm = rttm_path_for(key, bc_suffix)
            txt, csv = transcript_paths_for(key, bc_suffix)

            _begin_banner("BODYCAM", vp, key, bc_suffix, mp3, rttm, txt, csv)
            extract_audio_ffmpeg(vp, mp3)
            diarize_once(pipeline, mp3, rttm)
            transcribe_audio(mp3, key, bc_suffix)
            _end_banner("BODYCAM", key)
        except subprocess.CalledProcessError as e:
            log.error(f"ERR (ffmpeg BC) key={key if 'key' in locals() else '?'} src={vp.resolve()} :: {e}")
        except Exception as e:
            log.exception(f"ERR (BC) key={key if 'key' in locals() else '?'} src={vp.resolve()} :: {e}")

def process_audio_dir_fill_gaps(pipeline: Optional[Pipeline]) -> None:
    auds = list(walk_audio(AUDIO_BASE))
    log.info(f"AUDIO sweep in {AUDIO_BASE.resolve()}: {len(auds)} file(s)")
    for ap in auds:
        try:
            parsed = parse_any_key_and_suffix(ap.stem)
            if not parsed:
                # Skip non-canonical stems like *.music.json, etc.
                log.debug(f"SKIP (parse-any) name={ap.name} path={ap.resolve()}")
                continue
            key, suffix = parsed
            rttm = rttm_path_for(key, suffix)
            txt, csv = transcript_paths_for(key, suffix)

            _begin_banner("AUDIO-SWEEP", ap, key, suffix, ap, rttm, txt, csv)
            diarize_once(pipeline, ap, rttm)
            transcribe_audio(ap, key, suffix)
            _end_banner("AUDIO-SWEEP", key)
        except Exception as e:
            log.exception(f"ERR (audio sweep) key={key if 'key' in locals() else '?'} src={ap.resolve()} :: {e}")

# ========= Entry =========
def main():
    ensure_dir(AUDIO_BASE)
    pipeline = build_diarization_pipeline()

    process_dashcam_videos(pipeline)
    process_bodycam_videos(pipeline)
    process_audio_dir_fill_gaps(pipeline)

    total_elapsed = time.perf_counter() - GLOBAL_START
    log.info(
        f"ALL DONE | elapsed={total_elapsed:.2f}s | "
        f"{st_extract.summary()} | {st_diar.summary()} | {st_trans.summary()}"
    )

if __name__ == "__main__":
    main()
