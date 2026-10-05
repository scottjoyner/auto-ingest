"""How the pipeline finds media, and how it names a recording.

Pure: compiled regexes, datetime parsing, hashlib, os. No torch, no transformers,
no Neo4j driver, no model.

This lives apart from ``transcripts`` because it is a *contract* rather than an
implementation. ``auto_ingest.custody.staging`` has to agree with it exactly -
stage renames each object into a layout, and these are the rules that decide
whether the renamed object is found. Importing the contract through the ingest
module meant a custody test asserting "this staged filename is discoverable"
could not run without torch installed, and could not run without downloading an
embedding model at import time.

That is not hypothetical, and it is not a style complaint: an earlier version of
the staging tests asserted against hand-copied versions of these regexes. The
copies agreed with each other while the real patterns rejected every name staging
produced, and the bug shipped. A test that guards a shared contract must import
the shared contract, and the shared contract must be importable from somewhere
that does not need a GPU.

``transcripts`` re-exports every name here, so existing importers are unaffected.
"""
from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime, timezone
from typing import Optional

try:
    from zoneinfo import ZoneInfo  # py>=3.9
except Exception:  # pragma: no cover - platforms without tzdata
    ZoneInfo = None

#: Timezone a filename stamp is interpreted in. The same variable run_ingest_all.sh
#: exports, so the two agree by construction rather than by coincidence.
LOCAL_TZ = os.getenv("LOCAL_TZ", "America/New_York")

# ---------------------------------------------------------------------------
# Discovery patterns
# ---------------------------------------------------------------------------
# Discovery matches on the *tail* of a filename, never on the directory it sits
# in. That single fact is what makes staging a naming problem: a file can be
# copied anywhere and still be found, as long as its name still ends correctly.
PAT_TRANS_JSON_TXT = re.compile(r"_([^_]+)_transcription\.txt$", re.IGNORECASE)
PAT_TRANS_CSV = re.compile(r"_transcription\.csv$", re.IGNORECASE)
PAT_ENTITIES = re.compile(r"_transcription_(entites|entities)\.csv$", re.IGNORECASE)
PAT_RTTM = re.compile(r"_speakers\.rttm$", re.IGNORECASE)
PAT_MEDIA = re.compile(
    r"\.(wav|mp3|m4a|flac|mp4|mov|mkv|MP4|MOV|MKV)$", re.IGNORECASE)
PAT_META_CSV = re.compile(r"_metadata\.csv$", re.IGNORECASE)


def stable_id(*parts: str) -> str:
    """A stable id for something with no usable name."""
    h = hashlib.md5()
    for p in parts:
        h.update((p or "").encode("utf-8", errors="ignore"))
        h.update(b"|")
    return h.hexdigest()


def _to_localized(dt: datetime) -> datetime:
    if ZoneInfo:
        return dt.replace(tzinfo=ZoneInfo(LOCAL_TZ))
    return dt.replace(tzinfo=timezone.utc)


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = _to_localized(dt)
    return dt.astimezone(timezone.utc)


def parse_key_datetime_utc_from_string(s: str) -> Optional[datetime]:
    """The moment a name or path refers to, in UTC, or None.

    Ordered from most to least specific, because the order is the whole
    behaviour: a bare 14-digit run wins over a directory date plus a bare
    six-digit time, and either wins over the caller's fallback. Adding a pattern
    at the top would silently re-key existing recordings.
    """
    s = s.strip()
    m = re.search(r"(?P<dt14>\d{14})", s)
    if m:
        try:
            return _to_utc(_to_localized(
                datetime.strptime(m.group("dt14"), "%Y%m%d%H%M%S")))
        except Exception:
            pass
    for pat, fmt in [
        (r"(\d{4})_(\d{4})_(\d{6})", "%Y_%m%d_%H%M%S"),
        (r"(\d{8})_(\d{6})", "%Y%m%d_%H%M%S"),
        (r"(\d{8})(\d{6})", "%Y%m%d%H%M%S"),
        (r"(\d{4})-(\d{2})-(\d{2})[_\-](\d{2})-(\d{2})-(\d{2})", "%Y-%m-%d_%H-%M-%S"),
        (r"(\d{4})_(\d{2})_(\d{2})[_\-](\d{2})_(\d{2})_(\d{2})", "%Y_%m_%d_%H_%M_%S"),
    ]:
        m2 = re.search(pat, s)
        if m2:
            try:
                return _to_utc(_to_localized(datetime.strptime(m2.group(0), fmt)))
            except Exception:
                pass
    m = re.search(r"/(?P<Y>\d{4})/(?P<M>\d{2})/(?P<D>\d{2})/", s)
    if m:
        Y, M, D = m.group("Y"), m.group("M"), m.group("D")
        m2 = re.search(r"(?<!\d)(\d{6})(?!\d)", os.path.basename(s))
        if m2:
            try:
                return _to_utc(_to_localized(datetime.strptime(
                    f"{Y}{M}{D}{m2.group(1)}", "%Y%m%d%H%M%S")))
            except Exception:
                pass
    return None


def canonicalize_key(name_without_suffix: str, full_path: str) -> str:
    """The join key for a discovered object.

    UTC, and always. Two cameras covering one moment agree on this value or they
    do not join, so the conversion has to happen in exactly one place - here -
    rather than in each producer. Callers pass the basename with the matched tail
    already removed, because the tail can itself contain digits.
    """
    dt = (parse_key_datetime_utc_from_string(name_without_suffix)
          or parse_key_datetime_utc_from_string(full_path))
    if dt:
        return dt.astimezone(timezone.utc).strftime("%Y_%m%d_%H%M%S")
    base = re.sub(r"[^\w\-]+", "_", name_without_suffix).strip("_")
    return base or stable_id(full_path)
