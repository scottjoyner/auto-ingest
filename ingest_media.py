#!/usr/bin/env python3
"""
ingest_media.py — ingest iPhone movies + pictures (Nextcloud store / local mirror)
into the knowledge graph and turn them into content.

Scans a media root (the Nextcloud filestore, or a local mirror like photos/) for
movies (.mp4/.mov/...) and pictures (.heic/.jpg/...), then for each file:
  * extracts metadata (date, dimensions, duration, gps, audio presence)
  * renders a thumbnail
  * upserts a MediaFile node in Neo4j (idempotent on content hash)
  * FULL processing:
      - movies  : transcribe audio (transformers Whisper) -> captions -> vertical
                  TikTok-style short (reuses tiktok_shorts.py)
      - pictures: CLIP image embedding (stored on the node) + a Ken Burns
                  slideshow short per capture date

Source can be a local --root, or a live Nextcloud via --nextcloud-url/user/pass
(WebDAV pull to a staging dir). Resumable via --state. Heavy AI steps degrade
gracefully when their models/libs are unavailable.

Usage:
  # local mirror
  python3 ingest_media.py --root /media/scott/SSD_4TB/photos --kind all \
      --state media_ingest.json --limit 20

  # live Nextcloud (WebDAV pull)
  python3 ingest_media.py --nextcloud-url https://cloud.example.com/remote.php/dav/files/ME/Photos \
      --nextcloud-user ME --nextcloud-pass TOKEN --kind all

  # also link MediaFile nodes to SummaryPlace/Trip/PhoneLog by GPS, and build
  # per-date + per-Trip/Place Ken Burns slideshows (near-dup images pruned)
  python3 ingest_media.py --root /media/scott/SSD_4TB/photos --slideshow \
      --link-limit 5000   # --no-link to skip; --no-embed to skip CLIP embedding
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# Config / paths
# --------------------------------------------------------------------------
try:
    import auto_ingest_config as _cfg
    NEXTCLOUD_ROOT = _cfg.get_nextcloud_root()
    _NC = _cfg.get_neo4j_config()
except Exception:
    NEXTCLOUD_ROOT = os.environ.get("NEXTCLOUD_ROOT", "/media/scott/SSD_4TB/nextcloud")
    _NC = {}

NEO4J_URI = os.environ.get("NEO4J_URI") or _NC.get("uri") or "bolt://localhost:7687"
NEO4J_USER = os.environ.get("NEO4J_USER") or _NC.get("user") or "neo4j"
NEO4J_PASS = os.environ.get("NEO4J_PASSWORD") or _NC.get("password") or os.environ.get("NEO4J_PASSWORD_DEFAULT") or "knowledge_graph_2026"
NEO4J_DB = os.environ.get("NEO4J_DB") or "neo4j"

THUMB_ROOT = os.environ.get("MEDIA_THUMB_ROOT", "/media/scott/SSD_4TB/media_thumbs")
SHORT_ROOT = os.environ.get("TIKTOK_OUT_ROOT", "/media/scott/SSD_4TB/tiktok_shorts")
STAGE_ROOT = os.environ.get("NEXTCLOUD_STAGE_ROOT", "/media/scott/SSD_4TB/nextcloud_stage")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "openai/whisper-base")

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".gif", ".bmp"}
DATE_RE = re.compile(r"(?P<y>\d{4})[-_/.]?(?P<m>\d{2})[-_/.]?(?P<d>\d{2})")


def _valid_ymd(y: int, m: int, d: int) -> bool:
    return 1 <= m <= 12 and 1 <= d <= 31 and 1900 <= y <= 2100


def _norm_date(y: int, m: int, d: int) -> Optional[str]:
    if _valid_ymd(y, m, d):
        return f"{y:04d}-{m:02d}-{d:02d}"
    return None


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# Hashing / classification
# --------------------------------------------------------------------------
def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def classify(ext: str) -> Optional[str]:
    ext = ext.lower()
    if ext in VIDEO_EXTS:
        return "movie"
    if ext in IMAGE_EXTS:
        return "picture"
    return None


# --------------------------------------------------------------------------
# Metadata extraction
# --------------------------------------------------------------------------
def video_meta(path: Path) -> Dict:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", str(path)],
            text=True, stderr=subprocess.STDOUT)
        d = json.loads(out)
    except Exception as e:
        _log(f"  ffprobe failed: {e}")
        return {}
    meta: Dict = {}
    dur = None
    for s in d.get("streams", []):
        if s.get("codec_type") == "video" and "width" not in meta:
            meta["width"] = s.get("width")
            meta["height"] = s.get("height")
            meta["codec"] = s.get("codec_name")
        if s.get("codec_type") == "audio":
            meta["has_audio"] = True
    if dur is None and "duration" in d.get("format", {}):
        try:
            dur = float(d["format"]["duration"])
        except Exception:
            dur = None
    if dur is None:
        for s in d.get("streams", []):
            if s.get("codec_type") == "video" and "duration" in s:
                try:
                    dur = float(s["duration"])
                except Exception:
                    pass
    if dur is not None:
        meta["duration"] = round(dur, 2)
    meta.setdefault("has_audio", False)
    # creation_time
    ct = d.get("format", {}).get("tags", {}).get("creation_time")
    if ct:
        m = re.match(r"(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})", ct)
        if m:
            nd = _norm_date(int(m.group("y")), int(m.group("m")), int(m.group("d")))
            if nd:
                meta["date"] = nd
        # best-effort full timestamp from the creation_time tag
        try:
            dt = datetime.strptime(ct.replace("T", " ").replace("Z", ""),
                                   "%Y-%m-%d %H:%M:%S")
            meta["created_at"] = dt.strftime("%Y-%m-%d %H:%M:%S")
            meta["epoch_millis"] = int(dt.timestamp() * 1000)
        except Exception:
            pass
    return meta


def _register_heif() -> None:
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except Exception:
        pass


def image_meta(path: Path) -> Dict:
    _register_heif()
    try:
        from PIL import Image, ExifTags
    except Exception as e:
        _log(f"  PIL unavailable: {e}")
        return {}
    try:
        with Image.open(path) as im:
            meta = {"width": im.width, "height": im.height}
            ex = im.getexif()
            # DateTimeOriginal = 36867 (date only); time from 36868/306
            dt = ex.get(36867) or ex.get(306)
            if dt:
                m = re.match(r"(?P<y>\d{4})[:_](?P<m>\d{2})[:_](?P<d>\d{2})", str(dt))
                if m:
                    nd = _norm_date(int(m.group("y")), int(m.group("m")),
                                    int(m.group("d")))
                    if nd:
                        meta["date"] = nd
            # best-effort full timestamp: 36868 DateTimeDigitized, else 306
            dtt = ex.get(36868) or ex.get(306)
            if dtt:
                m2 = re.match(
                    r"(?P<y>\d{4})[:_](?P<mo>\d{2})[:_](?P<d>\d{2})\s+"
                    r"(?P<h>\d{2})[:_](?P<mi>\d{2})[:_](?P<s>\d{2})", str(dtt))
                if m2:
                    try:
                        full = datetime(int(m2.group("y")), int(m2.group("mo")),
                                        int(m2.group("d")), int(m2.group("h")),
                                        int(m2.group("mi")), int(m2.group("s")))
                        meta["created_at"] = full.strftime("%Y-%m-%d %H:%M:%S")
                        meta["epoch_millis"] = int(full.timestamp() * 1000)
                    except Exception:
                        pass
            # GPS via piexif (handles HEIC nested IFD that Pillow returns as int)
            try:
                import piexif
                raw = getattr(im, "info", {}).get("exif")
                if raw:
                    gps = (piexif.load(raw) or {}).get("GPS") or {}
                    if 2 in gps and 4 in gps:
                        def _rat(t):
                            return t[0] / t[1]
                        lat = _rat(gps[2][0]) + _rat(gps[2][1]) / 60 + _rat(gps[2][2]) / 3600
                        lon = _rat(gps[4][0]) + _rat(gps[4][1]) / 60 + _rat(gps[4][2]) / 3600
                        if gps.get(1) in (b"S", b"W", "S", "W"):
                            lat = -lat
                        if gps.get(3) in (b"S", b"W", "S", "W"):
                            lon = -lon
                        meta["gps_lat"] = round(lat, 6)
                        meta["gps_lon"] = round(lon, 6)
            except Exception as e:
                _log(f"  gps(piexif) failed: {e}")
            return meta
    except Exception as e:
        _log(f"  image_meta failed: {e}")
        return {}


def mtime_date(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d")


def date_for(path: Path, meta: Dict) -> str:
    # 1) EXIF / ffprobe date (already validity-checked when set)
    if meta.get("date"):
        return meta["date"]
    # 2) iPhone filename format: YY-MM-DD HH-MM-SS (2-digit year)
    m = re.match(r"(\d{2})-(\d{2})-(\d{2})\s", path.stem)
    if m:
        nd = _norm_date(2000 + int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if nd:
            return nd
    # 3) any YYYY-MM-DD in the path (validated)
    mm = DATE_RE.search(path.as_posix())
    if mm:
        nd = _norm_date(int(mm.group("y")), int(mm.group("m")), int(mm.group("d")))
        if nd:
            return nd
    return mtime_date(path)


# --------------------------------------------------------------------------
# Thumbnails
# --------------------------------------------------------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def thumb_path_for(sha: str) -> Path:
    return Path(THUMB_ROOT) / (sha[:2]) / f"{sha}.jpg"


def make_thumbnail(path: Path, kind: str, sha: str) -> Optional[str]:
    out = thumb_path_for(sha)
    if out.exists():
        return str(out)
    ensure_dir(out.parent)
    if kind == "movie":
        try:
            dur = video_meta(path).get("duration") or 0
            ss = max(0.1, dur / 2)
            subprocess.run(
                ["ffmpeg", "-y", "-ss", f"{ss:.2f}", "-i", str(path),
                 "-vf", "scale='if(gt(iw,ih),320,-1)':'if(gt(iw,ih),-1,320)'",
                 "-frames:v", "1", "-q:v", "4", str(out)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return str(out)
        except Exception as e:
            _log(f"  thumb(video) failed: {e}")
            return None
    else:
        _register_heif()
        try:
            from PIL import Image
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((320, 320))
                im.save(str(out), "JPEG", quality=80)
            return str(out)
        except Exception as e:
            _log(f"  thumb(image) failed: {e}")
            return None


# --------------------------------------------------------------------------
# Heavy AI: Whisper transcription
# --------------------------------------------------------------------------
_WHISPER = None


def _whisper_pipeline():
    global _WHISPER
    if _WHISPER is not None:
        return _WHISPER
    try:
        from transformers import pipeline
    except Exception as e:
        _log(f"  transformers unavailable, skipping transcription: {e}")
        _WHISPER = False
        return None
    _log(f"  loading Whisper model {WHISPER_MODEL} ...")
    _WHISPER = pipeline("automatic-speech-recognition", model=WHISPER_MODEL,
                        device="cpu", return_timestamps=True)
    return _WHISPER


def transcribe_movie(path: Path) -> Optional[List[Dict]]:
    wav = Path(tempfile.gettempdir()) / f"{path.stem}.wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000",
             "-f", "wav", str(wav)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception as e:
        _log(f"  audio extract failed: {e}")
        return None
    pipe = _whisper_pipeline()
    if not pipe:
        return None
    try:
        res = pipe(str(wav))
    except Exception as e:
        _log(f"  whisper failed: {e}")
        return None
    segs: List[Dict] = []
    for ch in res.get("chunks", []):
        ts = ch.get("timestamp")
        if not ts or ts[0] is None or ts[1] is None:
            continue
        s, e = ts
        txt = (ch.get("text") or "").strip()
        if txt:
            segs.append({"start": round(float(s), 2),
                         "end": round(float(e), 2), "text": txt})
    return segs or None


# --------------------------------------------------------------------------
# Heavy AI: CLIP image embedding
# --------------------------------------------------------------------------
_CLIP = None


def _clip_model():
    global _CLIP
    if _CLIP is not None:
        return _CLIP
    try:
        import open_clip
        import torch
    except Exception as e:
        _log(f"  open_clip unavailable, skipping image embedding: {e}")
        _CLIP = False
        return None
    _log("  loading CLIP ViT-B/32 ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B/32", pretrained="laion2b_s34b_b79k")
    tok = open_clip.get_tokenizer("ViT-B/32")
    model.eval()
    _CLIP = (model, preprocess, tok)
    return _CLIP


# Prefer the canonical personal-recall embed path if available.
try:
    from auto_ingest.personal.embed import embed_image as _shared_embed_image
    embed_image = _shared_embed_image  # noqa: F811  (prefer shared impl)
except Exception:
    pass


def embed_image(path: Path) -> Optional[List[float]]:
    bundle = _clip_model()
    if not bundle:
        return None
    import torch
    from PIL import Image
    _register_heif()
    try:
        model, preprocess, _ = bundle
        with Image.open(path) as im:
            img = preprocess(im.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            vec = model.encode_image(img)
        vec = vec.cpu().numpy().flatten().tolist()
        return vec
    except Exception as e:
        _log(f"  clip embed failed: {e}")
        return None


# --------------------------------------------------------------------------
# Content generation
# --------------------------------------------------------------------------
def generate_movie_short(path: Path, segs: List[Dict], date: str, hook: str) -> Optional[str]:
    out_dir = Path(SHORT_ROOT) / date
    ensure_dir(out_dir)
    out = out_dir / f"{path.stem}_tok.mp4"
    if out.exists():
        return str(out)
    tr_json = Path(tempfile.gettempdir()) / f"{path.stem}_tr.json"
    tr_json.write_text(json.dumps(segs))
    cmd = [sys.executable, "tiktok_shorts.py", "--clip", str(path),
           "--transcript-json", str(tr_json), "--out", str(out), "--hook", hook]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            _log(f"  tiktok failed: {r.stderr.splitlines()[-3:]}")
            return None
        return str(out)
    except Exception as e:
        _log(f"  tiktok error: {e}")
        return None


def _ffmpeg_readable(path: Path) -> Path:
    """Return a path ffmpeg can decode (convert HEIC/HEIF -> temp JPG)."""
    if path.suffix.lower() in (".heic", ".heif"):
        _register_heif()
        try:
            from PIL import Image
            tmp = Path(tempfile.gettempdir()) / f"{path.stem}.jpg"
            with Image.open(path) as im:
                im.convert("RGB").save(str(tmp), "JPEG", quality=88)
            return tmp
        except Exception:
            return path
    return path


def build_slideshow(images: List[Path], out: Path, music: Optional[Path],
                    dur: float = 3.5, fade: float = 0.6, zoom: float = 1.12) -> bool:
    W, H, FPS = 1080, 1920, 30
    images = [_ffmpeg_readable(p) for p in images]
    ensure_dir(out.parent)
    n = len(images)
    if n == 0:
        return False
    infiles = []
    for im in images:
        infiles += ["-loop", "1", "-t", str(dur), "-i", str(im)]
    vf = []
    for i in range(n):
        vf.append(
            f"[{i}:v]scale={W}:{H}:force_original_aspect_ratio=increase,"
            f"crop={W}:{H},"
            f"zoompan=z='min(1.0+{zoom-1.0}/({int(dur*FPS)}*on),{zoom:.3f})':"
            f"d={int(dur*FPS)}:s={W}x{H}:fps={FPS},"
            f"trim=duration={dur},setpts=PTS-STARTPTS,format=yuv420p[v{i}]")
    if n == 1:
        vf.append("[v0]copy[outv]")
    else:
        chain = "[v0][v1]"
        last = "x01"
        vf.append(f"{chain}xfade=transition=fade:duration={fade}:"
                  f"offset={dur-fade:.2f}[{last}]")
        for i in range(2, n):
            nxt = f"x0{i:02d}"
            vf.append(f"[{last}][v{i}]xfade=transition=fade:duration={fade}:"
                      f"offset={i*(dur-fade):.2f}[{nxt}]")
            last = nxt
        vf.append(f"[{last}]copy[outv]")
    fc = ";".join(vf)
    cmd = ["ffmpeg", "-y", *infiles, "-filter_complex", fc, "-map", "[outv]",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    if music and Path(music).exists():
        cmd += ["-stream_loop", "-1", "-i", str(music), "-shortest",
                "-map", "1:a", "-c:a", "aac", "-b:a", "128k"]
    cmd += [str(out)]
    try:
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if r.returncode != 0:
            _log(f"  slideshow failed: {r.stderr.splitlines()[-3:]}")
            return False
        return True
    except Exception as e:
        _log(f"  slideshow error: {e}")
        return False


# --------------------------------------------------------------------------
# Nextcloud WebDAV pull (no FUSE mount required)
# --------------------------------------------------------------------------
def webdav_walk(base_url: str, user: str, password: str) -> List[Tuple[str, int]]:
    """Yield (absolute_url, size) for every file under base_url via PROPFIND.

    Nextcloud returns hrefs as server-absolute paths *without* the web context
    (e.g. /remote.php/dav/... even when the base is /nextcloud/remote.php/...),
    so we re-anchor on the context root rather than urljoin (which would drop it).
    """
    import requests
    from urllib.parse import urlparse
    from xml.etree import ElementTree as ET
    NS = {"d": "DAV:"}
    parsed = urlparse(base_url)
    idx = parsed.path.find("/remote.php")
    context = parsed.path[:idx] if idx >= 0 else ""
    server = f"{parsed.scheme}://{parsed.netloc}{context}"
    out: List[Tuple[str, int]] = []
    stack = [base_url.rstrip("/")]
    while stack:
        url = stack.pop()
        try:
            r = requests.request("PROPFIND", url, auth=(user, password),
                                 headers={"Depth": "1"}, timeout=30)
        except Exception as e:
            _log(f"  PROPFIND {url} failed: {e}")
            continue
        if r.status_code not in (207, 200):
            continue
        try:
            root = ET.fromstring(r.content)
        except Exception:
            continue
        for resp in root.findall("d:response", NS):
            href = resp.findtext("d:href", namespaces=NS)
            if not href:
                continue
            is_coll = resp.find(".//d:propstat/d:prop/d:resourcetype/d:collection", NS)
            length = resp.findtext(".//d:propstat/d:prop/d:getcontentlength", namespaces=NS)
            abs_url = server + href
            if is_coll is not None:
                if abs_url.rstrip("/") != base_url.rstrip("/"):
                    stack.append(abs_url)
                continue
            try:
                size = int(length or 0)
            except Exception:
                size = 0
            out.append((abs_url, size))
    return out


def webdav_pull(base_url: str, user: str, password: str, stage: Path,
                exts: set, limit: int = 0) -> List[Path]:
    import requests
    stage = Path(stage)
    ensure_dir(stage)
    entries = webdav_walk(base_url, user, password)
    got: List[Path] = []
    for url, size in entries:
        p = url.rsplit("/", 1)[-1]
        if classify(Path(p).suffix) is None:
            continue
        rel = url.replace(base_url.rstrip("/") + "/", "").lstrip("/")
        dst = stage / rel
        if dst.exists() and dst.stat().st_size == size:
            got.append(dst)
            continue
        ensure_dir(dst.parent)
        try:
            with requests.get(url, auth=(user, password), stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(dst, "wb") as f:
                    for blk in r.iter_content(1 << 20):
                        f.write(blk)
            got.append(dst)
            _log(f"  pulled {rel} ({size//1024}KB)")
        except Exception as e:
            _log(f"  pull {rel} failed: {e}")
        if limit and len(got) >= limit:
            break
    return got


# --------------------------------------------------------------------------
# Neo4j
# --------------------------------------------------------------------------
def neo4j_driver():
    from neo4j import GraphDatabase
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))


def upsert_media(drv, rec: Dict) -> None:
    q = """
    MERGE (m:MediaFile {sha256:$sha})
    SET m.path=$path, m.source=$source, m.kind=$kind, m.mime=$mime,
        m.date=$date, m.year=$year, m.month=$month, m.day=$day,
        m.width=$width, m.height=$height, m.duration=$duration,
        m.has_audio=$has_audio, m.gps_lat=$gps_lat, m.gps_lon=$gps_lon,
        m.size=$size, m.thumb=$thumb, m.ingested_at=$now,
        m.storage_root=$storage_root, m.relative_path=$relative_path,
        m.host_path=$host_path, m.tailscale_host_hint=$tailscale_host_hint,
        m.retention_class=$retention_class,
        m.captured_at=$captured_at, m.epoch_millis=$epoch_millis
    """
    params = {
        "sha": rec["sha"], "path": rec["path"], "source": rec["source"],
        "kind": rec["kind"], "mime": rec["mime"], "date": rec["date"],
        "year": int(rec["date"][:4]) if rec["date"] else None,
        "month": int(rec["date"][5:7]) if rec["date"] else None,
        "day": int(rec["date"][8:10]) if rec["date"] else None,
        "width": rec.get("width"), "height": rec.get("height"),
        "duration": rec.get("duration"), "has_audio": rec.get("has_audio", False),
        "gps_lat": rec.get("gps_lat"), "gps_lon": rec.get("gps_lon"),
        "size": rec["size"], "thumb": rec.get("thumb"),
        "now": datetime.now(tz=timezone.utc).isoformat(),
        "storage_root": rec.get("storage_root"),
        "relative_path": rec.get("relative_path"),
        "host_path": rec.get("host_path"),
        "tailscale_host_hint": rec.get("tailscale_host_hint"),
        "retention_class": rec.get("retention_class"),
        "captured_at": rec.get("created_at"),
        "epoch_millis": rec.get("epoch_millis"),
    }
    with drv.session(database=NEO4J_DB) as s:
        s.run(q, **params)


def set_processed(drv, sha: str, **props) -> None:
    if not props:
        return
    set_clause = ", ".join(f"m.{k}=${k}" for k in props)
    q = f"MATCH (m:MediaFile {{sha256:$sha}}) SET {set_clause}, m.processed_at=$now"
    props["sha"] = sha
    props["now"] = datetime.now(tz=timezone.utc).isoformat()
    with drv.session(database=NEO4J_DB) as s:
        s.run(q, **props)


# --------------------------------------------------------------------------
# Scanning
# --------------------------------------------------------------------------
def scan_local(root: Path) -> List[Path]:
    out: List[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if classify(p.suffix) is None:
            continue
        out.append(p)
    return out


# --------------------------------------------------------------------------
# Main ingest loop
# --------------------------------------------------------------------------
def process_file(drv, path: Path, source: str, kind: str, state: dict,
                 do_transcribe: bool, do_embed: bool, do_shorts: bool,
                 music: Optional[Path], force: bool) -> dict:
    sha = sha256_of(path)
    if not force and sha in state.get("done", {}):
        return {"sha": sha, "skip": True}
    meta = video_meta(path) if kind == "movie" else image_meta(path)
    date = date_for(path, meta)
    rec = {
        "sha": sha, "path": str(path), "source": source, "kind": kind,
        "mime": path.suffix.lstrip(".").lower(), "date": date,
        "size": path.stat().st_size, **meta,
    }
    try:
        from auto_ingest_config import build_artifact_ref
        ref = build_artifact_ref(str(path), storage_root="local-ssd",
                                 retention_class="keep")
        rec.update({k: ref[k] for k in ("storage_root", "relative_path",
                                        "host_path", "tailscale_host_hint",
                                        "retention_class")})
    except Exception:
        pass
    rec["thumb"] = make_thumbnail(path, kind, sha)
    upsert_media(drv, rec)
    _log(f"  upserted {kind} {path.name} ({date})")

    processed = {}
    if kind == "movie" and (do_transcribe or do_shorts):
        if do_transcribe and meta.get("has_audio"):
            segs = transcribe_movie(path)
            if segs:
                tr_path = Path(THUMB_ROOT).parent / "transcripts" / f"{sha}.json"
                ensure_dir(tr_path.parent)
                tr_path.write_text(json.dumps(segs))
                processed["transcript"] = str(tr_path)
                _log(f"    transcript: {len(segs)} segs")
        if do_shorts:
            segs = None
            if processed.get("transcript"):
                segs = json.loads(Path(processed["transcript"]).read_text())
            hook = f"iPhone memory — {date}"
            short = generate_movie_short(path, segs or [], date, hook)
            if short:
                processed["short_path"] = short
                _log(f"    short: {short}")
    elif kind == "picture" and do_embed:
        emb = embed_image(path)
        if emb:
            processed["embedding"] = emb
            _log(f"    embedding: {len(emb)}d")

    if processed:
        set_processed(drv, sha, **{k: v for k, v in processed.items()
                                   if k != "embedding"} | (
                                   {"embedding_dim": len(processed["embedding"])}
                                   if "embedding" in processed else {}))
        if "embedding" in processed:
            # store the vector on the node
            with drv.session(database=NEO4J_DB) as s:
                s.run("MATCH (m:MediaFile {sha256:$sha}) SET m.embedding=$emb",
                      sha=sha, emb=processed["embedding"])
    state.setdefault("done", {})[sha] = {
        "path": str(path), "kind": kind, "date": date,
        "ok": True, "at": datetime.now(tz=timezone.utc).isoformat(),
        **{f"has_{k}": True for k in processed},
    }
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest iPhone movies/pictures into the graph + content.")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--root", type=Path, help="Local media root to scan")
    src.add_argument("--nextcloud-url", help="Nextcloud WebDAV base URL (live pull)")
    ap.add_argument("--nextcloud-user", help="Nextcloud WebDAV user")
    ap.add_argument("--nextcloud-pass", help="Nextcloud WebDAV password/app-token")
    ap.add_argument("--nextcloud-root", default="Photos",
                    help="Subfolder under the Nextcloud user's files (when --nextcloud-url is the bare server root)")
    ap.add_argument("--source", default="nextcloud", help="source label for MediaFile nodes")
    ap.add_argument("--kind", choices=["movie", "picture", "all"], default="all")
    ap.add_argument("--state", default="./media_ingest.json", help="Resume state file")
    ap.add_argument("--limit", type=int, default=0, help="Max files this run (0=all)")
    ap.add_argument("--no-transcribe", action="store_true", help="Skip Whisper transcription")
    ap.add_argument("--no-embed", action="store_true", help="Skip CLIP image embedding")
    ap.add_argument("--no-shorts", action="store_true", help="Skip short generation")
    ap.add_argument("--no-link", action="store_true",
                    help="Skip GPS-based MediaFile linking (SummaryPlace/Trip/PhoneLog)")
    ap.add_argument("--link-limit", type=int, default=0,
                    help="Max MediaFile nodes to link this run (0=all)")
    ap.add_argument("--music", type=Path, default=None, help="Background music for slideshows")
    ap.add_argument("--slideshow", action="store_true",
                    help="Also build per-date Ken Burns slideshow shorts from pictures")
    ap.add_argument("--force", action="store_true", help="Re-process already-done files")
    ap.add_argument("--dry-run", action="store_true", help="Scan + metadata only, no Neo4j write")
    args = ap.parse_args()

    # Resolve file list
    if not args.nextcloud_url:
        try:
            import auto_ingest_config as _c
            cu, cuser, cpass = _c.get_nextcloud_webdav()
            if cu:
                args.nextcloud_url, args.nextcloud_user, args.nextcloud_pass = cu, cuser, cpass
        except Exception:
            pass
    # Normalise to a full DAV endpoint when given the bare server root.
    if args.nextcloud_url and "/remote.php/dav" not in args.nextcloud_url:
        args.nextcloud_url = (f"{args.nextcloud_url.rstrip('/')}/remote.php/dav/files/"
                               f"{args.nextcloud_user}/{args.nextcloud_root}")
    if args.nextcloud_url:
        if not (args.nextcloud_user and args.nextcloud_pass):
            print("ERROR: --nextcloud-user and --nextcloud-pass required with --nextcloud-url",
                  file=sys.stderr)
            return 2
        stage = Path(STAGE_ROOT)
        _log(f"Pulling from Nextcloud {args.nextcloud_url} -> {stage}")
        files = webdav_pull(args.nextcloud_url, args.nextcloud_user,
                            args.nextcloud_pass, stage, VIDEO_EXTS | IMAGE_EXTS,
                            limit=args.limit)
        source = args.source
    else:
        if not args.root:
            print("ERROR: provide --root, --nextcloud-url, or configure nextcloud: in config.yaml",
                  file=sys.stderr)
            return 2
        root = Path(args.root)
        if not root.exists():
            print(f"ERROR: root missing: {root}", file=sys.stderr)
            return 2
        files = scan_local(root)
        source = args.source

    if args.kind != "all":
        files = [f for f in files if classify(f.suffix) == args.kind]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    _log(f"Found {len(files)} media file(s) [{args.kind}]")

    state: dict = {}
    if Path(args.state).exists() and not args.force:
        try:
            state = json.loads(Path(args.state).read_text())
        except Exception:
            state = {}
    state.setdefault("done", {})

    drv = None
    if not args.dry_run:
        try:
            drv = neo4j_driver()
        except Exception as e:
            print(f"ERROR: Neo4j unavailable: {e}", file=sys.stderr)
            return 2

    count = 0
    for path in files:
        kind = classify(path.suffix)
        if args.dry_run:
            meta = video_meta(path) if kind == "movie" else image_meta(path)
            _log(f"[dry] {kind} {path.name} date={date_for(path, meta)} "
                 f"{meta.get('width')}x{meta.get('height')} dur={meta.get('duration')} "
                 f"gps={meta.get('gps_lat')},{meta.get('gps_lon')}")
            count += 1
        else:
            try:
                process_file(drv, path, source, kind, state,
                             do_transcribe=not args.no_transcribe,
                             do_embed=not args.no_embed,
                             do_shorts=not args.no_shorts,
                             music=args.music, force=args.force)
            except Exception as e:
                _log(f"  ERROR on {path.name}: {e}")
            count += 1
        if args.limit and count >= args.limit:
            break

    if not args.dry_run:
        Path(args.state).write_text(json.dumps(state, indent=2))

    # Optional per-date slideshow from ingested pictures
    if args.slideshow and not args.dry_run and args.kind in ("picture", "all"):
        build_slideshows(drv, source, args.music)

    # Optional GPS-based linking of MediaFile -> SummaryPlace/Trip/PhoneLog.
    if not args.no_link and not args.dry_run:
        try:
            from auto_ingest.personal.embed import ensure_media_indexes_with_retry
            ensure_media_indexes_with_retry()
        except Exception as e:
            _log(f"  ensure_media_indexes failed (continuing): {e}")
        try:
            from auto_ingest.personal.link_media import link_all
            from auto_ingest.shorts.db_retry import with_driver
            with_driver(lambda drv: link_all(drv, limit=args.link_limit))
            _log("  linking complete")
        except Exception as e:
            _log(f"  linking skipped: {e}")

    _log(f"Done. Processed {count} file(s). State: {args.state}")
    return 0


def dedup_by_embedding(paths: List[Path], threshold: float = 0.95) -> List[Path]:
    """Drop near-duplicate consecutive images (cosine sim >= threshold).

    Uses the shared CLIP embed; if CLIP is unavailable, returns paths unchanged.
    """
    try:
        from auto_ingest.personal.embed import embed_image as _embed
    except Exception:
        return paths
    try:
        import numpy as np
        vecs: List[Optional[List[float]]] = []
        for p in paths:
            try:
                vecs.append(_embed(p))
            except Exception:
                vecs.append(None)
        out: List[Path] = []
        for i, p in enumerate(paths):
            v = vecs[i]
            if v is None or not out:
                out.append(p)
                continue
            last_v = vecs[paths.index(out[-1])]
            if last_v is None:
                out.append(p)
                continue
            sim = float(np.dot(v, last_v) / (np.linalg.norm(v) * np.linalg.norm(last_v)))
            if sim < threshold:
                out.append(p)
        return out
    except Exception:
        return paths


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(s))[:80]


def build_slideshows(drv, source: str, music: Optional[Path]) -> None:
    _log("Building picture slideshows (per-date + per-Trip/Place) ...")
    groups: Dict[str, List[str]] = {}

    def _add(grp: str, paths: List[str]) -> None:
        if len(paths) >= 2:
            groups.setdefault(grp, paths)

    with drv.session(database=NEO4J_DB) as s:
        rows = s.run(
            "MATCH (m:MediaFile {source:$src, kind:'picture'}) "
            "WHERE m.date IS NOT NULL RETURN m.date AS date, m.path AS path "
            "ORDER BY m.date, m.path", src=source).data()
    by_date: Dict[str, List[str]] = {}
    for r in rows:
        by_date.setdefault(r["date"], []).append(r["path"])
    for date, paths in sorted(by_date.items()):
        _add(f"date:{date}", paths)

    # Group by linked SummaryPlace / Trip when available.
    with drv.session(database=NEO4J_DB) as s:
        for r in s.run(
            "MATCH (m:MediaFile {source:$src, kind:'picture'})-[:AT_PLACE]->"
            "(p:SummaryPlace) RETURN p.name AS grp, m.path AS path "
            "ORDER BY grp, m.path", src=source).data():
            groups.setdefault(f"place:{_safe(r['grp'])}", []).append(r["path"])
        for r in s.run(
            "MATCH (m:MediaFile {source:$src, kind:'picture'})-[:DURING]->"
            "(t:Trip) RETURN coalesce(t.uniqueKey, t.tripId) AS grp, m.path AS path "
            "ORDER BY grp, m.path", src=source).data():
            groups.setdefault(f"trip:{_safe(r['grp'])}", []).append(r["path"])

    for grp, paths in sorted(groups.items()):
        deduped = dedup_by_embedding([Path(p) for p in paths])
        if len(deduped) < 2:
            continue
        out = Path(SHORT_ROOT) / grp / f"slideshow_{_safe(grp)}.mp4"
        if out.exists():
            continue
        ok = build_slideshow(deduped, out, music)
        if ok:
            with drv.session(database=NEO4J_DB) as s2:
                s2.run(
                    "MATCH (m:MediaFile {source:$src, kind:'picture'}) "
                    "WHERE m.path IN $ps SET m.slideshow=$p",
                    src=source, ps=[str(p) for p in deduped], p=str(out))
            _log(f"  slideshow {grp}: {len(deduped)} pics -> {out}")


if __name__ == "__main__":
    raise SystemExit(main())
