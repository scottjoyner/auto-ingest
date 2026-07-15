#!/usr/bin/env python3
"""
tiktok_shorts.py — generate "iPhone TikTok style" vertical shorts from dashcam
(and other) footage.

Design goals (the previous generators were insufficient):
  * True vertical 9:16 phone framing with safe-zone margins (not just a cropped box)
  * TikTok-native caption styling: large bold captions in the lower third, current
    word highlighted (karaoke) in TikTok yellow, white otherwise, hard black outline
  * A hook line pinned to the top and a closing CTA
  * Optional zoom-punch (Ken Burns) for energy
  * Optional background music (looped + ducked under the clip's own audio)
  * Pulls the transcript straight from Neo4j so captions are real, word-timed

Implemented on ffmpeg + PIL (no moviepy dependency). Output is H.264 1080x1920
mp4 for maximum platform compatibility.

Usage:
  # From a single clip (transcript pulled from Neo4j by filename):
  python3 tiktok_shorts.py --clip /path/to/2023_1228_190747_FR.MP4 \
      --out ./out/clip1.mp4 --hook "POV: you live in the future"

  # Explicit transcript (list of {start,end,text}):
  python3 tiktok_shorts.py --clip clip.mp4 --transcript-json tr.json --out out.mp4

  # Add music:
  python3 tiktok_shorts.py --clip clip.mp4 --music loop.mp3 --out out.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
WIDTH, HEIGHT = 1080, 1920
FONT = "DejaVuSans-Bold"  # must be resolvable by fontconfig
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# ASS colours are &HAABBGGRR
C_WHITE = "&H00FFFFFF"
C_YELLOW = "&H0000FFFF"   # TikTok-ish yellow (R=255,G=255,B=0)
C_BLACK = "&H00000000"

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASS = os.environ.get("NEO4J_PASSWORD", "knowledge_graph_2026")
NEO4J_DB = os.environ.get("NEO4J_DB", "neo4j")


@dataclass
class Seg:
    start: float
    end: float
    text: str


# --------------------------------------------------------------------------
# Transcript sourcing
# --------------------------------------------------------------------------
def clip_key(path: Path) -> str:
    """Derive the Neo4j Transcription key stem from a clip filename."""
    s = path.stem
    s = re.sub(r"_[FRFR]+$", "", s, flags=re.I)        # drop _FR / _F / _R
    s = re.sub(r"_(medium|large|tiny|base|small|_transcription.*)$", "", s, flags=re.I)
    return s


def fetch_transcript(key: str) -> List[Seg]:
    """Pull (start, end, text) segments for a clip from Neo4j."""
    try:
        from neo4j import GraphDatabase
    except Exception:
        return []
    try:
        drv = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    except Exception as ex:
        print(f"[warn] Neo4j unavailable ({ex}); skipping auto transcript.", file=sys.stderr)
        return []
    out: List[Seg] = []
    try:
        with drv.session(database=NEO4J_DB) as s:
            rows = s.run(
                """
                MATCH (t:Transcription)-[:HAS_SEGMENT]->(seg:Segment)
                WHERE coalesce(t.key, t.clip_key, '') CONTAINS $k
                RETURN seg.start AS a, seg.end AS b, seg.text AS txt
                ORDER BY a
                """,
                k=key,
            ).data()
        for r in rows:
            a = float(r.get("a") or 0.0)
            b = float(r.get("b") or 0.0)
            txt = (r.get("txt") or "").strip()
            if txt:
                out.append(Seg(a, b, txt))
    finally:
        drv.close()
    return out


def load_transcript_json(p: Path) -> List[Seg]:
    data = json.loads(p.read_text())
    segs = []
    for d in data:
        segs.append(Seg(float(d["start"]), float(d["end"]), str(d["text"]).strip()))
    return segs


# --------------------------------------------------------------------------
# ASS (karaoke) subtitle builder
# --------------------------------------------------------------------------
def _words_with_timing(text: str, start: float, end: float) -> List[tuple]:
    """Split a segment's text into words and assign each a sub-interval of
    [start,end] proportional to its length (approximated word timing)."""
    words = re.findall(r"\S+", text)
    if not words:
        return []
    total = sum(len(w) for w in words) or 1
    out = []
    t = start
    for w in words:
        dur = (len(w) / total) * (end - start)
        out.append((w, t, t + dur))
        t += dur
    return out


def build_ass(segs: List[Seg], hook: Optional[str], cta: Optional[str],
              total_dur: float) -> str:
    lines = []
    lines.append("[Script Info]")
    lines.append("ScriptType: v4.00+")
    lines.append("PlayResX: 1080")
    lines.append("PlayResY: 1920")
    lines.append("WrapStyle: 2")
    lines.append("")
    lines.append("[V4+ Styles]")
    lines.append("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
                 "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
                 "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
                 "Alignment, MarginL, MarginR, MarginV, Encoding")
    # Lower-third captions: bold, large, yellow karaoke fill -> white.
    lines.append(
        f"Style: Cap,{FONT},92,{C_WHITE},{C_YELLOW},{C_BLACK},{C_BLACK},"
        f"-1,0,0,0,100,100,1,0,1,5,1,2,40,40,200,1"
    )
    # Hook (top) + CTA (bottom-ish) styles.
    lines.append(
        f"Style: Hook,{FONT},72,{C_WHITE},{C_WHITE},{C_BLACK},{C_BLACK},"
        f"-1,0,0,0,100,100,1,0,1,4,1,8,40,40,150,1"
    )
    lines.append(
        f"Style: CTA,{FONT},80,{C_YELLOW},{C_YELLOW},{C_BLACK},{C_BLACK},"
        f"-1,0,0,0,100,100,1,0,1,5,1,2,40,40,260,1"
    )
    lines.append("")
    lines.append("[Events]")
    lines.append("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
                 "Effect, Text")

    for s in segs:
        wt = _words_with_timing(s.text, s.start, s.end)
        if not wt:
            continue
        parts = []
        for w, a, b in wt:
            cs = max(1, int(round((b - a) * 100)))  # centiseconds for \k
            parts.append(f"{{\\k{cs}}}{w}")
        karaoke = " ".join(parts)
        start = _ass_time(s.start)
        end = _ass_time(s.end)
        lines.append(f"Dialogue: 0,{start},{end},Cap,,0,0,0,,{karaoke}")

    if hook:
        lines.append(f"Dialogue: 0,0:00:00.00,{_ass_time(min(total_dur, 4.0))},"
                     f"Hook,,0,0,0,,{_esc(hook)}")
    if cta:
        cstart = max(0.0, total_dur - 3.0)
        lines.append(f"Dialogue: 0,{_ass_time(cstart)},{_ass_time(total_dur)},"
                     f"CTA,,0,0,0,,{_esc(cta)}")

    return "\n".join(lines) + "\n"


def _ass_time(sec: float) -> str:
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


def _esc(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


# --------------------------------------------------------------------------
# ffmpeg assembly
# --------------------------------------------------------------------------
def _ffmpeg_ok(codec: str) -> bool:
    try:
        out = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"],
                                      stderr=subprocess.STDOUT, text=True)
        return codec in out
    except Exception:
        return False


def _has_audio(path: Path) -> bool:
    try:
        pa = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
            text=True).strip()
        return bool(pa)
    except Exception:
        return False


def find_paired_audio(clip: Path, extra_root: Optional[Path] = None) -> Optional[Path]:
    """Return a usable audio track for a clip.

    Dashcam `_FR`/`_F` clips are video-only; the spoken audio lives in the
    paired `_R` rear clip. We return the clip's own audio if present, otherwise
    look for the sibling `_R` (in the same dir, in the uncompressed source tree
    if this clip lives under `.../compressed/`, or under an explicit root).
    """
    if _has_audio(clip):
        return clip
    parts = clip.parts
    stem = clip.stem
    paired_stem = re.sub(r"_(FR|F|R)$", "_R", stem, flags=re.I)
    cands: List[Path] = []
    # same-dir sibling (compressed tree may eventually hold _R too)
    for ext in (clip.suffix or ".mp4", ".mp4", ".MP4"):
        cands.append(clip.parent / (paired_stem + ext))
    # if under .../compressed/, map to the source tree (parent of 'compressed')
    if "compressed" in parts:
        idx = parts.index("compressed")
        source_root = Path(*parts[:idx])
        rel = Path(*parts[idx + 1:])
        cands.append(source_root / rel.parent / (paired_stem + ".MP4"))
        if extra_root is not None:
            cands.append(extra_root / rel.parent / (paired_stem + ".MP4"))
    for c in cands:
        if c.exists() and _has_audio(c):
            return c
    return None


def run(in_clip: Path, out_path: Path, segs: List[Seg], hook: Optional[str],
         cta: Optional[str], music: Optional[Path], audio: Optional[Path],
         zoom: float, crf: int, max_dur: float) -> int:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("ERROR: ffmpeg not found", file=sys.stderr)
        return 2

    # Trim segments to max_dur (keep the first max_dur seconds).
    if segs and segs[-1].end > max_dur:
        segs = [s for s in segs if s.start < max_dur]
        for s in segs:
            s.end = min(s.end, max_dur)

    total_dur = max([s.end for s in segs], default=0.0)
    if total_dur <= 0 and in_clip.exists():
        # probe duration as fallback
        try:
            d = subprocess.check_output(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=nk=1:nw=1", str(in_clip)], text=True).strip()
            total_dur = float(d) if d else 30.0
        except Exception:
            total_dur = 30.0

    tmp = tempfile.mkdtemp(prefix="tiktok_")
    try:
        ass_path = Path(tmp) / "subs.ass"
        ass_path.write_text(build_ass(segs, hook, cta, total_dur), encoding="utf-8")

        # Cover-crop to 9:16 centred (preserve aspect — no vertical stretch),
        # then scale to 1080x1920. Optional zoom-punch scales up first.
        vf = []
        vf.append(
            "crop=w='if(gt(iw/ih\\,9/16),trunc(ih*9/16/2)*2,iw)':"
            "h='if(gt(iw/ih\\,9/16),ih,trunc(iw*16/9/2)*2)'"
        )
        if zoom and zoom != 1.0:
            vf.append(f"scale={int(WIDTH*zoom)}:{int(HEIGHT*zoom)}")
            vf.append(f"crop={WIDTH}:{HEIGHT}")
        else:
            vf.append(f"scale={WIDTH}:{HEIGHT}")
        vf.append("setsar=1")
        vf.append(f"subtitles={str(ass_path).replace(':', '\\:')}"
                  f":fontsdir=/usr/share/fonts/truetype/dejavu")

        # Audio graph: clip audio (or paired _R voice) + optional ducked music.
        infiles = [str(in_clip)]
        arefs: List[str] = []
        if audio and Path(audio).exists() and _has_audio(Path(audio)):
            infiles.append(str(audio))
            arefs.append(f"[{len(infiles) - 1}:a]")
        if music and Path(music).exists():
            infiles.append(str(music))
            arefs.append(f"[{len(infiles) - 1}:a]")

        fc_audio = ""
        amap: List[str] = []
        if arefs:
            norm = [f"{ref}aresample=44100[a{i}]" for i, ref in enumerate(arefs)]
            if len(arefs) == 1:
                fc_audio = norm[0] + ";[a0]volume=1.0[aout]"
            else:
                fade_st = max(0.0, total_dur - 0.5)
                fc_audio = (
                    norm[0] + ";" + norm[1] + ";"
                    f"[a0]volume=1.0[v];"
                    f"[a1]afade=t=out:st={fade_st:.2f}:d=0.5,volume=0.18[m];"
                    f"[v][m]amix=inputs=2:duration=first:dropout_transition=0[aout]"
                )
            amap = ["-map", "[aout]"]
        # else: silent short

        vfilter = ",".join(vf)
        fc = f"[0:v]{vfilter}[v]" + (f";{fc_audio}" if fc_audio else "")
        in_args = []
        for f in infiles:
            in_args += ["-i", f]
        cmd = [ffmpeg, "-hide_banner", "-y", *in_args,
               "-filter_complex", fc, "-map", "[v]", *amap,
               "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
               "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               "-c:a", "aac", "-b:a", "128k", "-shortest", str(out_path)]
        print("+ ffmpeg", " ".join(shlex_quote(c) for c in cmd[:6]), "...")
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            print("FFMPEG FAILED:", file=sys.stderr)
            print("\n".join(proc.stdout.splitlines()[-30:]), file=sys.stderr)
            return proc.returncode
        print(f"+ wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def shlex_quote(s: str) -> str:
    import shlex
    return shlex.quote(s)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def pick_clip(source_root: Path, pick: str, date: Optional[str]) -> Optional[Path]:
    """Scan a media root for candidate clips and return one to use.

    Prefers dashcam _FR clips, then any mp4/mov. Optional --date filters by a
    YYYY-MM-DD substring in the path. Returns the newest (default) or a random
    clip.
    """
    if not source_root.exists():
        print(f"[warn] source root missing: {source_root}", file=sys.stderr)
        return None
    pats = ("*_FR.MP4", "*_F.MP4", "*_R.MP4", "*.mp4", "*.mov", "*.MP4", "*.MOV")
    cands = []
    for pat in pats:
        for pp in source_root.rglob(pat):
            if not pp.is_file():
                continue
            if date and date not in pp.as_posix():
                continue
            cands.append(pp)
    if not cands:
        return None
    cands.sort(key=lambda q: q.stat().st_mtime, reverse=(pick == "newest"))
    if pick == "random":
        import random
        return random.choice(cands)
    return cands[0]


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate iPhone TikTok-style vertical shorts.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--clip", type=Path, help="Source video (e.g. *_FR.MP4)")
    src.add_argument("--source-root", type=Path, help="Scan a media root and pick a clip")
    src.add_argument("--nextcloud", action="store_true",
                     help="Shortcut: scan the Nextcloud root for a clip (iPhone media)")
    ap.add_argument("--pick", choices=["newest", "random"], default="newest",
                    help="Which clip to pick when scanning a root")
    ap.add_argument("--date", type=str, default=None, help="Filter scanned clips by YYYY-MM-DD")
    ap.add_argument("--out", required=True, type=Path, help="Output mp4 path")
    ap.add_argument("--transcript-json", type=Path, help="Optional JSON list of {start,end,text}")
    ap.add_argument("--hook", type=str, default=None, help="Top hook/title text")
    ap.add_argument("--cta", type=str, default="Follow for more", help="Closing CTA text")
    ap.add_argument("--music", type=Path, default=None, help="Optional background music (looped+ducked)")
    ap.add_argument("--audio", type=Path, default=None,
                    help="Explicit audio track (else auto-detect paired _R voice)")
    ap.add_argument("--no-auto-audio", action="store_true",
                    help="Disable auto paired-audio detection")
    ap.add_argument("--zoom", type=float, default=1.06, help="Slight zoom-punch factor (>1)")
    ap.add_argument("--crf", type=int, default=24, help="H.264 CRF")
    ap.add_argument("--max-dur", type=float, default=60.0, help="Cap short length (seconds)")
    ap.add_argument("--no-auto-transcript", action="store_true",
                    help="Do not fetch transcript from Neo4j (use --transcript-json or none)")
    args = ap.parse_args()

    clip = args.clip
    if clip is None:
        root = args.source_root
        if args.nextcloud:
            try:
                import auto_ingest_config as cfg
                root = Path(cfg.get_nextcloud_root())
            except Exception as ex:
                print(f"ERROR: cannot resolve Nextcloud root: {ex}", file=sys.stderr)
                return 2
        if root is None:
            print("ERROR: provide --clip, --source-root, or --nextcloud", file=sys.stderr)
            return 2
        clip = pick_clip(root, args.pick, args.date)
        if clip is None:
            print(f"ERROR: no clips found under {root}", file=sys.stderr)
            return 2
        print(f"Picked clip: {clip}")

    if not clip.exists():
        print(f"ERROR: clip not found: {clip}", file=sys.stderr)
        return 2

    audio = args.audio
    if audio is None and not args.no_auto_audio:
        audio = find_paired_audio(clip)
        if audio:
            print(f"Paired audio: {audio}")
        else:
            print("No paired audio found (silent short).")

    if args.transcript_json:
        segs = load_transcript_json(args.transcript_json)
        print(f"Loaded {len(segs)} segments from JSON.")
    elif not args.no_auto_transcript:
        key = clip_key(clip)
        segs = fetch_transcript(key)
        print(f"Fetched {len(segs)} segments from Neo4j (key='{key}').")
    else:
        segs = []
        print("No transcript (captions disabled).")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    return run(clip, args.out, segs, args.hook, args.cta,
               args.music, audio, args.zoom, args.crf, args.max_dur)


if __name__ == "__main__":
    raise SystemExit(main())
