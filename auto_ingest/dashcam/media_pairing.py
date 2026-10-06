"""Pairing a YOLO detection csv with the clip it describes.

Pure filesystem logic, deliberately dependency-free.

This lives apart from ``yolo_embeddings`` for two reasons. That module imports
moviepy, cv2, ultralytics and sentence_transformers at module scope, so the one
piece of logic that decides whether a recording is processed at all could not be
tested without a GPU image. And the pairing rule is a naming contract shared with
``auto_ingest.custody.staging``, which files a clip and its csv into the same
``YYYY/MM/DD`` directory; keeping it here makes that contract checkable on both
sides.

The rule itself: a ``{key}_YOLOv8n.csv`` names its clip by stem, in the same
directory, with any media extension in any case. It used to probe for exactly
``{key}.MP4``, which silently skipped every card that wrote ``.mp4`` - the clip
was present and playable, and the only trace was one log line per recording.
Staging preserves the source card's extension rather than rewriting it, so that
case is routine, not hypothetical.
"""
from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple

#: Containers a dashcam clip may arrive in, compared case-insensitively. Matches
#: the video subset of ``transcripts.py``'s PAT_MEDIA.
MEDIA_EXTENSIONS: Tuple[str, ...] = (".mp4", ".mov", ".mkv")

#: The detection filename this detector produces. Anchored at the end, as before.
DETECTION_SUFFIX = "_YOLOv8n.csv"

_DETECTION_RE = re.compile(r"_YOLOv8n\.csv$", re.IGNORECASE)


def detection_stem(filename: str) -> Optional[str]:
    """The clip stem a detection filename names, or None if it is not one."""
    m = _DETECTION_RE.search(filename)
    if not m:
        return None
    return filename[: m.start()].rsplit("_YOLOv8n", 1)[0]


def find_sibling_media(directory: str, key: str) -> Optional[str]:
    """The playable clip ``key`` names in ``directory``, or None.

    An empty clip counts as missing: a zero-byte file is a truncated recording,
    not a usable one, and passing it downstream only moves the failure somewhere
    less obvious.
    """
    wanted = key.casefold()
    try:
        entries = os.listdir(directory)
    except OSError:
        return None
    for name in entries:
        stem, dot, ext = name.rpartition(".")
        if not dot or stem.casefold() != wanted:
            continue
        if f".{ext}".lower() not in MEDIA_EXTENSIONS:
            continue
        path = os.path.join(directory, name)
        try:
            if os.path.getsize(path) > 0:
                return path
        except OSError:
            continue
    return None


def find_file_keys(directory: str) -> List[str]:
    """Every clip stem in ``directory`` that has a detection csv and a clip.

    Sorted, and never inferred: a csv without a playable clip is skipped rather
    than reported as processed.
    """
    try:
        entries = sorted(os.listdir(directory))
    except OSError:
        return []
    keys = set()
    for name in entries:
        key = detection_stem(name)
        if key is None:
            continue
        if find_sibling_media(directory, key) is not None:
            keys.add(key)
    return sorted(keys)


def missing_media(directory: str) -> List[Tuple[str, str]]:
    """Detection files whose clip is absent or empty: ``(stem, csv_name)``.

    Reported rather than swallowed, because "no detections" and "the clip was
    never looked at" are indistinguishable downstream otherwise.
    """
    try:
        entries = sorted(os.listdir(directory))
    except OSError:
        return []
    out = []
    for name in entries:
        key = detection_stem(name)
        if key is None:
            continue
        if find_sibling_media(directory, key) is None:
            out.append((key, name))
    return out


# ---------------------------------------------------------------------------
# Finding the trees worth processing
# ---------------------------------------------------------------------------

#: Directory names that hold preserved artifacts rather than work to do.
#:
#: `custody stage` routes detection files whose clip is absent into
#: `orphaned-detections/`, and that path is a YYYY/MM/DD shape - so a walker that
#: recognises date directories descends into it and then finds no clip for any of
#: them. On the real card that is 2,934 directories and one "missing media"
#: warning each, on every pass, forever. The files are meant to be kept, not
#: processed.
QUARANTINE_DIRNAMES: Tuple[str, ...] = ("orphaned-detections",)


def is_quarantined(path: str, names: Tuple[str, ...] = QUARANTINE_DIRNAMES) -> bool:
    """Whether any component of ``path`` is a quarantine directory."""
    if not names:
        return False
    parts = {p for p in os.path.normpath(str(path)).split("/") if p}
    return any(name in parts for name in names)


def walk_date_dirs(
    base: str,
    *,
    exclude: Tuple[str, ...] = QUARANTINE_DIRNAMES,
) -> List[str]:
    """Date-shaped directories under ``base``, minus any quarantine namespace.

    Quarantined trees are pruned at the walk rather than filtered afterwards, so
    their contents are never even stat()ed - which matters when the excluded tree
    is 2,934 directories on a CIFS share.
    """
    out: List[str] = []
    for root, dirs, _files in os.walk(base):
        # Prune in place; os.walk honours the mutation of `dirs`.
        dirs[:] = [d for d in dirs if d not in exclude]
        parts = [p for p in os.path.normpath(root).split("/") if p]
        if len(parts) < 3:
            continue
        y, m, d = parts[-3], parts[-2], parts[-1]
        if not (len(y) == 4 and len(m) == 2 and len(d) == 2):
            continue
        if not (y.isdigit() and m.isdigit() and d.isdigit()):
            continue
        if is_quarantined(root, exclude):
            continue
        out.append(root)
    return sorted(out)
