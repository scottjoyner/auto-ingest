#!/usr/bin/env python3
import argparse, subprocess, sys, os, re, shlex, json, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import shutil

PATTERN = re.compile(r".*_(F|R|FR)\.MP4$", re.IGNORECASE)  # match *_F.MP4, *_R.MP4, *_FR.MP4

def which_ffmpeg():
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        sys.exit("ERROR: ffmpeg not found in PATH. Please install FFmpeg.")
    return ffmpeg

def ffmpeg_supports(codec: str) -> bool:
    try:
        out = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], stderr=subprocess.STDOUT, text=True)
        return codec in out
    except Exception:
        return False

def detect_codecs():
    # Prefer HEVC; fallback to H.264
    if ffmpeg_supports("libx265"):
        return ("libx265", "hevc")
    elif ffmpeg_supports("hevc_videotoolbox"):  # mac hw accel (if present)
        return ("hevc_videotoolbox", "hevc")
    elif ffmpeg_supports("libx264"):
        return ("libx264", "h264")
    else:
        sys.exit("ERROR: No suitable H.265 (libx265/hevc_videotoolbox) or H.264 (libx264) encoder found in FFmpeg.")

def relative_subpath(p: Path, root: Path) -> Path:
    return p.relative_to(root)

def is_target_outdated(src: Path, dst: Path) -> bool:
    if not dst.exists():
        return True
    if src.stat().st_mtime > dst.stat().st_mtime:
        return True
    if dst.stat().st_size < 64 * 1024:
        return True
    return False

def build_ffmpeg_cmd(
    ffmpeg, src, dst, vcodec, family, crf, preset, max_width, fps, audio_k, tune, extra_x265, tag_apple
):
    vf = []
    if max_width:
        vf.append(f"scale='min({max_width},iw)':'-2':force_original_aspect_ratio=decrease")
    if fps:
        vf.append(f"fps={fps}")
    vf_str = ",".join(vf) if vf else "null"

    common = [
        ffmpeg, "-hide_banner", "-y",
        "-i", str(src),
        "-map_metadata", "0",
        "-map", "0",
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        "-vf", vf_str,
    ]

    v = ["-c:v", vcodec]
    if vcodec == "libx265":
        v += ["-preset", preset, "-crf", str(crf)]
        if extra_x265:
            v += ["-x265-params", extra_x265]
        if tag_apple:
            v += ["-tag:v", "hvc1"]
    elif vcodec == "libx264":
        v += ["-preset", preset, "-crf", str(max(16, min(crf, 30)))]
        if tune:
            v += ["-tune", tune]
    elif vcodec == "hevc_videotoolbox":
        q = max(18, min(crf, 36))
        v += ["-b:v", "0", "-q:v", str(q)]
        v += ["-tag:v", "hvc1"]

    a = ["-c:a", "aac", "-b:a", f"{audio_k}k"]

    sync = ["-vsync", "vfr"] if fps else []
    return common + v + a + sync + [str(dst)]

def process_one(task):
    (
        src, dst, args_effective, ffmpeg, vcodec, family, crf, preset, max_width,
        fps, audio_k, tune, extra_x265, tag_apple, dry_run, overwrite
    ) = task

    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not overwrite and not is_target_outdated(src, dst):
            return (src, dst, "skip", 0, None)

        cmd = build_ffmpeg_cmd(
            ffmpeg, src, dst, vcodec, family, crf, preset, max_width, fps, audio_k, tune, extra_x265, tag_apple
        )
        if dry_run:
            return (src, dst, "dry-run", 0, " ".join(shlex.quote(c) for c in cmd))

        t0 = time.time()
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        dt = time.time() - t0
        if proc.returncode != 0:
            return (src, dst, "error", dt, proc.stdout)

        try:
            os.utime(dst, (src.stat().st_atime, src.stat().st_mtime))
        except Exception:
            pass

        return (src, dst, "ok", dt, None)
    except Exception as e:
        return (src, dst, "error", 0, str(e))

def collect_inputs(root: Path):
    # Walk and yield only files matching *_F/_R/_FR.MP4 (case-insensitive)
    seen = set()
    for ext in ("*.MP4", "*.mp4"):
        for p in root.rglob(ext):
            if PATTERN.match(p.name):
                rp = p.resolve()
                if rp not in seen:
                    seen.add(rp)
                    yield rp

def within_time_window(p: Path, since_ts: float | None, until_ts: float | None) -> bool:
    t = p.stat().st_mtime
    if since_ts is not None and t < since_ts:
        return False
    if until_ts is not None and t > until_ts:
        return False
    return True

def parse_date(s: str) -> float:
    # Accept YYYY-MM-DD
    try:
        dt = datetime.strptime(s, "%Y-%m-%d")
        return dt.timestamp()
    except ValueError:
        sys.exit(f"Invalid date '{s}'. Use YYYY-MM-DD.")

def main():
    ap = argparse.ArgumentParser(description="Compress dashcam MP4s into a mirrored portable tree using FFmpeg.")
    ap.add_argument("--input-root", required=True, help="Root directory containing YYYY/MM/DD/*.MP4 files")
    ap.add_argument("--output-root", required=True, help="Output directory (mirrors YYYY/MM/DD/ structure)")
    ap.add_argument("--max-width", type=int, default=1280, help="Max output width (keep AR). 0 = keep original")
    ap.add_argument("--fps", type=int, default=30, help="Normalize frame rate. 0 = keep original")
    ap.add_argument("--crf", type=int, default=26, help="CRF for libx265 (lower=better). Fallback libx264 uses similar scale.")
    ap.add_argument("--preset", default="medium", help="Encoder preset (x265/x264); e.g. slow, medium, fast")
    ap.add_argument("--audio-k", type=int, default=96, help="AAC audio bitrate kbps")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Parallel workers")
    ap.add_argument("--overwrite", action="store_true", help="Re-encode even if target exists/newer")
    ap.add_argument("--dry-run", action="store_true", help="Print what would run, do not encode")
    ap.add_argument("--tune", default=None, help="x264 tune (e.g. film, psnr) if using libx264")
    ap.add_argument("--x265-params", default="aq-mode=3", help="Extra -x265-params (comma-separated)")
    ap.add_argument("--no-apple-tag", action="store_true", help="Do not force -tag:v hvc1 on HEVC outputs")

    # NEW: ordering & filters
    ap.add_argument("--order", choices=["newest", "oldest", "name"], default="newest",
                    help="Processing order (default: newest)")
    ap.add_argument("--limit", type=int, default=0, help="Only process the first N files after sorting/filtering")
    ap.add_argument("--since", type=str, default=None, help="Only files modified on/after this date (YYYY-MM-DD)")
    ap.add_argument("--until", type=str, default=None, help="Only files modified on/before this date (YYYY-MM-DD)")

    args = ap.parse_args()

    ffmpeg = which_ffmpeg()
    vcodec, family = detect_codecs()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    if not input_root.exists():
        sys.exit(f"ERROR: Input root does not exist: {input_root}")

    files = list(collect_inputs(input_root))
    if not files:
        print("No matching *_F/_R/_FR .MP4 files found.")
        return

    # Apply time filters
    since_ts = parse_date(args.since) if args.since else None
    until_ts = parse_date(args.until) + 24*3600 - 1 if args.until else None  # end of day
    if since_ts is not None or until_ts is not None:
        files = [p for p in files if within_time_window(p, since_ts, until_ts)]

    # Order: newest first by mtime (default)
    if args.order == "newest":
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    elif args.order == "oldest":
        files.sort(key=lambda p: p.stat().st_mtime)
    else:  # name
        files.sort(key=lambda p: str(p).lower())

    # Limit
    if args.limit and args.limit > 0:
        files = files[:args.limit]

    if not files:
        print("No files to process after filters/order.")
        return

    print(f"Found {len(files)} files under {input_root}")
    first_dt = datetime.fromtimestamp(files[0].stat().st_mtime)
    last_dt = datetime.fromtimestamp(files[-1].stat().st_mtime)
    print(f"Order: {args.order}  | newest={first_dt}  oldest={last_dt}")
    print(f"Encoder: {vcodec}  (family: {family})")
    if vcodec == "libx265":
        print(f"HEVC (CRF {args.crf}, preset {args.preset}, x265-params='{args.x265_params}')")
    elif vcodec == "libx264":
        print(f"H.264 (CRF {max(16, min(args.crf, 30))}, preset {args.preset}, tune={args.tune})")

    tasks = []
    for src in files:
        rel = relative_subpath(src, input_root)  # e.g., YYYY/MM/DD/NAME.mp4
        dst = output_root / rel
        dst = dst.with_suffix(".mp4")
        tasks.append((
            src, dst, args, ffmpeg, vcodec, family, args.crf, args.preset,
            args.max_width if args.max_width > 0 else None,
            args.fps if args.fps > 0 else None,
            args.audio_k, args.tune, args.x265_params,
            (False if args.no_apple_tag else True) and (vcodec in ("libx265", "hevc_videotoolbox")),
            args.dry_run, args.overwrite
        ))

    ok, skipped, errors = 0, 0, 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_one, t) for t in tasks]
        for fut in as_completed(futs):
            src, dst, status, dt, info = fut.result()
            if status == "ok":
                ok += 1
                print(f"[OK] {src}  ->  {dst}  ({dt:.1f}s)")
            elif status == "skip":
                skipped += 1
                print(f"[SKIP] {src} (up-to-date)")
            elif status == "dry-run":
                print(f"[DRY] {src} -> {dst}")
                print(f"      ffmpeg: {info}")
            else:
                errors += 1
                print(f"[ERR] {src} -> {dst}")
                if info:
                    tail = "\n".join(info.splitlines()[-20:])
                    print(tail)

    elapsed = time.time() - started
    print(f"\n==== Summary ====")
    print(f"Encoder: {vcodec}")
    print(f"Processed: {ok} ok, {skipped} skipped, {errors} errors in {elapsed/60:.1f} min")

if __name__ == "__main__":
    main()


# # Dry run the last ~10,000 newest files to preview commands
# python3 compress_dashcam.py \
#   --input-root /media/scott/NAS/fileserver/dashcam \
#   --output-root /mnt/8TB_2025/portable/dashcam_portable \
#   --order newest --limit 10000 --dry-run

# # Now actually run, newest first, in parallel
# python3 compress_dashcam.py \
#   --input-root /media/scott/NAS/fileserver/dashcam \
#   --output-root /media/scott/NAS/3863-3833/dashcam \
#   --order newest --limit 1000 --workers 1
