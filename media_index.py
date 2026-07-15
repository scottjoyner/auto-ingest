#!/usr/bin/env python3
"""
media_index.py — build a unified media catalog for the content pipeline.

Scans the configured roots (dashcam video, the hot audio store, and the
Nextcloud iPhone store) and writes a JSON catalog of every media file,
classified by type with an extracted date. The TikTok generator and the
background worker use this catalog to pick sources and queue work.

Nextcloud is only mounted on some machines; if its root is absent it is
skipped gracefully (so the catalog still covers dashcam + audio).

Usage:
  python3 media_index.py                 # (re)build ./media_index.json
  python3 media_index.py --cache media_index.json --refresh
  python3 media_index.py --roots /a /b   # override roots
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".gif", ".bmp"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}

DATE_RE = re.compile(r"(?P<y>\d{4})[/_-]?(?P<m>\d{2})[/_-]?(?P<d>\d{2})")


def _classify(ext: str) -> Optional[str]:
    ext = ext.lower()
    if ext in VIDEO_EXTS:
        return "video"
    if ext in IMAGE_EXTS:
        return "image"
    if ext in AUDIO_EXTS:
        return "audio"
    return None


def _date_for(path: Path) -> Optional[str]:
    m = DATE_RE.search(path.as_posix())
    if m:
        return f"{m.group('y')}-{m.group('m')}-{m.group('d')}"
    try:
        t = path.stat().st_mtime
        return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        return None


def default_roots() -> List[str]:
    roots: List[str] = []
    try:
        import auto_ingest_config as cfg
        roots.append(cfg.get_fileserver_path("dashcam"))
        roots.append(cfg.get_hot_root())
        roots.append(cfg.get_nextcloud_root())
    except Exception:
        pass
    # de-dup, keep order
    seen, out = set(), []
    for r in roots:
        if r and r not in seen:
            seen.add(r)
            out.append(r)
    return out


def scan_root(root: str, progress: bool = True) -> Dict[str, List[dict]]:
    cat: Dict[str, List[dict]] = {"video": [], "image": [], "audio": []}
    rp = Path(root)
    if not rp.exists():
        if progress:
            print(f"  (skip missing root: {root})")
        return cat
    count = 0
    for p in rp.rglob("*"):
        if not p.is_file():
            continue
        kind = _classify(p.suffix)
        if not kind:
            continue
        count += 1
        if progress and count % 5000 == 0:
            print(f"  scanned {count} files...")
        cat[kind].append({
            "path": str(p),
            "stem": p.stem,
            "date": _date_for(p),
            "size": p.stat().st_size,
        })
    if progress:
        print(f"  {root}: {len(cat['video'])} video, {len(cat['image'])} image, "
              f"{len(cat['audio'])} audio")
    return cat


def build(roots: List[str], progress: bool = True) -> dict:
    merged = {"video": [], "image": [], "audio": []}
    for r in roots:
        c = scan_root(r, progress=progress)
        for k in merged:
            merged[k].extend(c[k])
    merged["video"].sort(key=lambda x: x["date"] or "")
    merged["image"].sort(key=lambda x: x["date"] or "")
    merged["audio"].sort(key=lambda x: x["date"] or "")
    return {
        "generated": datetime.now(tz=timezone.utc).isoformat(),
        "roots": roots,
        **merged,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a unified media catalog.")
    ap.add_argument("--cache", default="./media_index.json", help="Output JSON path")
    ap.add_argument("--roots", nargs="*", default=None, help="Override scan roots")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    roots = args.roots or default_roots()
    if not roots:
        print("ERROR: no roots to scan", file=sys.stderr)
        return 2
    print(f"Scanning {len(roots)} root(s)...")
    data = build(roots, progress=not args.no_progress)
    Path(args.cache).write_text(json.dumps(data, indent=2))
    print(f"Wrote {args.cache}: {len(data['video'])} video, "
          f"{len(data['image'])} image, {len(data['audio'])} audio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
