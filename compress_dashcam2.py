#!/usr/bin/env python3
import argparse, subprocess, sys, os, re, shlex, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import shutil

PATTERN = re.compile(r".*_(F|R|FR)\.MP4$", re.IGNORECASE)  # *_F.MP4, *_R.MP4, *_FR.MP4

def which_ffmpeg():
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        sys.exit("ERROR: ffmpeg not found in PATH. Please install FFmpeg.")
    return ffmpeg

def which_ffprobe():
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        sys.exit("ERROR: ffprobe not found in PATH. Please install FFmpeg (includes ffprobe).")
    return ffprobe

def ffmpeg_supports(codec: str) -> bool:
    try:
        out = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], stderr=subprocess.STDOUT, text=True)
        return codec in out
    except Exception:
        return False

def detect_codecs():
    if ffmpeg_supports("libx265"):
        return ("libx265", "hevc")
    elif ffmpeg_supports("hevc_videotoolbox"):
        return ("hevc_videotoolbox", "hevc")
    elif ffmpeg_supports("libx264"):
        return ("libx264", "h264")
    else:
        sys.exit("ERROR: No suitable H.265/H.264 encoder found in FFmpeg (need libx265/hevc_videotoolbox/libx264).")

def relative_subpath(p: Path, root: Path) -> Path:
    return p.relative_to(root)

def probe_duration(ffprobe: str, path: Path) -> float | None:
    """Return duration in seconds via ffprobe, or None if unreadable."""
    try:
        cmd = [
            ffprobe, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "format=duration", "-of", "default=nk=1:nw=1", str(path)
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        if not out:
            return None
        return float(out)
    except Exception:
        return None

def is_valid_output(ffprobe: str, src: Path, dst: Path, min_bytes: int, thresh: float) -> bool:
    """A destination is valid if it is readable and duration >= thresh * src_duration and size >= min_bytes."""
    if not dst.exists():
        return False
    if dst.stat().st_size < min_bytes:
        return False
    src_dur = probe_duration(ffprobe, src)
    dst_dur = probe_duration(ffprobe, dst)
    if src_dur is None or dst_dur is None:
        return False
    # Some sources might be VFR; accept small timing mismatch, but guard substantial truncation
    return (dst_dur >= max(1.0, src_dur * thresh))

def is_target_outdated(src: Path, dst: Path) -> bool:
    if not dst.exists():
        return True
    if dst.stat().st_size < 64 * 1024:
        return True
    if src.stat().st_mtime > dst.stat().st_mtime:
        return True
    return False

def build_ffmpeg_cmd(
    ffmpeg, src, dst_tmp, vcodec, family, crf, preset, max_width, fps, audio_k, tune, extra_x265, tag_apple
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
        q = max(18, min(crf, 36))  # videotooolbox quality param
        v += ["-b:v", "0", "-q:v", str(q)]
        v += ["-tag:v", "hvc1"]

    a = ["-c:a", "aac", "-b:a", f"{audio_k}k"]
    sync = ["-vsync", "vfr"] if fps else []
    return common + v + a + sync + [str(dst_tmp)]

def process_one(task):
    (
        src, dst, ffmpeg, ffprobe, vcodec, family, crf, preset, max_width,
        fps, audio_k, tune, extra_x265, tag_apple, dry_run, overwrite, skip_existing,
        verify, verify_thresh, min_bytes, quarantine_bad
    ) = task

    try:
        dst.parent.mkdir(parents=True, exist_ok=True)

        # Skip logic (with verification)
        if dst.exists() and not overwrite:
            if skip_existing:
                # If verify enabled, ensure destination is healthy; otherwise skip blindly.
                if verify:
                    if is_valid_output(ffprobe, src, dst, min_bytes, verify_thresh):
                        return (src, dst, "skip-exists", 0, None)
                    else:
                        # Quarantine a bad/partial file so we can re-encode cleanly
                        if quarantine_bad:
                            bad = dst.with_suffix(dst.suffix + ".bad")
                            try:
                                if bad.exists():
                                    bad.unlink()
                                dst.rename(bad)
                            except Exception:
                                pass
                        # fall through to re-encode
                else:
                    return (src, dst, "skip-exists", 0, None)
            else:
                # not skipping existing; fall through to smart skip
                pass

        # Smart skip if up-to-date and (if verify) valid
        if not overwrite and dst.exists() and not is_target_outdated(src, dst):
            if not verify or is_valid_output(ffprobe, src, dst, min_bytes, verify_thresh):
                return (src, dst, "skip", 0, None)
            # else: invalid -> re-encode

        # Build temp path for atomic write
        dst_tmp = dst.with_suffix(dst.suffix + ".partial")

        cmd = build_ffmpeg_cmd(
            ffmpeg, src, dst_tmp, vcodec, family, crf, preset, max_width, fps, audio_k, tune, extra_x265, tag_apple
        )
        if dry_run:
            return (src, dst, "dry-run", 0, " ".join(shlex.quote(c) for c in cmd))

        # Remove any stale partial
        try:
            if dst_tmp.exists():
                dst_tmp.unlink()
        except Exception:
            pass

        t0 = time.time()
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        dt = time.time() - t0
        if proc.returncode != 0:
            # Clean up partial
            try:
                if dst_tmp.exists():
                    dst_tmp.unlink()
            except Exception:
                pass
            return (src, dst, "error", dt, proc.stdout)

        # Verify the freshly encoded tmp before moving
        if verify:
            if not is_valid_output(ffprobe, src, dst_tmp, min_bytes, verify_thresh):
                # keep tmp as .bad and report error
                bad = dst_tmp.with_suffix(".bad")
                try:
                    if bad.exists():
                        bad.unlink()
                    dst_tmp.rename(bad)
                except Exception:
                    pass
                return (src, dst, "error", dt, "Verification failed: output appears truncated/corrupt")

        # Atomic move into place
        try:
            if dst.exists():
                dst.unlink()
            os.replace(dst_tmp, dst)
        except Exception as e:
            # fallback: attempt rename then cleanup
            try:
                if dst_tmp.exists():
                    dst_tmp.rename(dst)
            except Exception:
                pass

        # Preserve timestamps from source
        try:
            os.utime(dst, (src.stat().st_atime, src.stat().st_mtime))
        except Exception:
            pass

        return (src, dst, "ok", dt, None)
    except Exception as e:
        return (src, dst, "error", 0, str(e))

def collect_inputs(root: Path):
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
    ap.add_argument("--crf", type=int, default=26, help="CRF for libx265; libx264 uses similar scale.")
    ap.add_argument("--preset", default="medium", help="Encoder preset (x265/x264)")
    ap.add_argument("--audio-k", type=int, default=96, help="AAC audio bitrate kbps")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Parallel workers")
    ap.add_argument("--overwrite", action="store_true", help="Re-encode even if target exists/newer")
    ap.add_argument("--dry-run", action="store_true", help="Print actions only")
    ap.add_argument("--tune", default=None, help="x264 tune (e.g. film)")
    ap.add_argument("--x265-params", default="aq-mode=3", help="Extra -x265-params")
    ap.add_argument("--no-apple-tag", action="store_true", help="Do not force -tag:v hvc1 on HEVC outputs")

    # Ordering & filters
    ap.add_argument("--order", choices=["newest", "oldest", "name"], default="newest",
                    help="Processing order (default: newest)")
    ap.add_argument("--limit", type=int, default=0, help="Only process the first N files after sorting/filtering")
    ap.add_argument("--since", type=str, default=None, help="Only files modified on/after YYYY-MM-DD")
    ap.add_argument("--until", type=str, default=None, help="Only files modified on/before YYYY-MM-DD")

    # Skip/verify options
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                    help="Do NOT skip when destination exists (still obeys --overwrite)")
    ap.set_defaults(skip_existing=True)

    ap.add_argument("--no-verify", dest="verify", action="store_false",
                    help="Disable ffprobe verification of outputs (not recommended)")
    ap.set_defaults(verify=True)

    ap.add_argument("--verify-threshold", type=float, default=0.95,
                    help="Minimum (dst_duration / src_duration) to accept as valid (default 0.95)")
    ap.add_argument("--min-bytes", type=int, default=128*1024,
                    help="Minimum file size in bytes to consider a valid output (default 131072)")
    ap.add_argument("--no-quarantine-bad", dest="quarantine_bad", action="store_false",
                    help="Do not rename invalid existing outputs to .bad before re-encoding")
    ap.set_defaults(quarantine_bad=True)

    args = ap.parse_args()

    ffmpeg = which_ffmpeg()
    ffprobe = which_ffprobe()
    vcodec, family = detect_codecs()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    if not input_root.exists():
        sys.exit(f"ERROR: Input root does not exist: {input_root}")

    files = list(collect_inputs(input_root))
    if not files:
        print("No matching *_F/_R/_FR .MP4 files found.")
        return

    since_ts = parse_date(args.since) if args.since else None
    until_ts = parse_date(args.until) + 24*3600 - 1 if args.until else None
    if since_ts is not None or until_ts is not None:
        files = [p for p in files if within_time_window(p, since_ts, until_ts)]

    if args.order == "newest":
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    elif args.order == "oldest":
        files.sort(key=lambda p: p.stat().st_mtime)
    else:
        files.sort(key=lambda p: str(p).lower())

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
        rel = relative_subpath(src, input_root)  # YYYY/MM/DD/NAME.mp4
        dst = (output_root / rel).with_suffix(".mp4")
        tasks.append((
            src, dst, ffmpeg, ffprobe, vcodec, family, args.crf, args.preset,
            args.max_width if args.max_width > 0 else None,
            args.fps if args.fps > 0 else None,
            args.audio_k, args.tune, args.x265_params,
            (False if args.no_apple_tag else True) and (vcodec in ("libx265", "hevc_videotoolbox")),
            args.dry_run, args.overwrite, args.skip_existing,
            args.verify, args.verify_threshold, args.min_bytes, args.quarantine_bad
        ))

    ok, skipped, skipped_exists, errors = 0, 0, 0, 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_one, t) for t in tasks]
        for fut in as_completed(futs):
            src, dst, status, dt, info = fut.result()
            if status == "ok":
                ok += 1
                print(f"[OK]   {src}  ->  {dst}  ({dt:.1f}s)")
            elif status == "skip-exists":
                skipped_exists += 1
                print(f"[SKIP] {src} (destination exists & verified)")
            elif status == "skip":
                skipped += 1
                print(f"[SKIP] {src} (up-to-date & verified)")
            elif status == "dry-run":
                print(f"[DRY]  {src} -> {dst}")
                print(f"       ffmpeg: {info}")
            else:
                errors += 1
                print(f"[ERR]  {src} -> {dst}")
                if info:
                    tail = "\n".join(info.splitlines()[-20:])
                    print(tail)

    elapsed = time.time() - started
    print(f"\n==== Summary ====")
    print(f"Encoder: {vcodec}")
    print(f"Processed: {ok} ok, {skipped_exists} skipped(existing), {skipped} skipped(up-to-date), {errors} errors in {elapsed/60:.1f} min")

if __name__ == "__main__":
    main()

# python3 compress_dashcam2.py \
#   --input-root /media/scott/NAS/fileserver/dashcam \
#   --output-root /media/scott/NAS/3863-3833/dashcam \
#   --order newest --limit 50000 --workers 1 --verify-threshold 0.95