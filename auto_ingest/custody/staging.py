"""Deciding what to stage from a card, and where it has to land to be found.

The problem this solves
-----------------------

The ingest pipeline discovers media by **filename suffix**, not by directory:
`auto_ingest/ingest/transcripts.py:94-99` matches `_transcription.txt`,
`_transcription.csv`, `_speakers.rttm`, `_metadata.csv` and
`\\.(wav|mp3|m4a|flac|mp4|mov|mkv)$`, and the dashcam path
(`yolo_embeddings.py:1695-1697`) looks for `{key}_YOLOv8n.csv` beside
`{key}.MP4`. `canonicalize_key` (`:245-249`) then rebuilds the join key from any
14-digit `YYYY_MMDD_HHMMSS` it finds in the name **or the full path**.

So a card copied verbatim keeps its *camera* layout and the pipeline sees almost
none of it. On the surveyed card, `DCIM/2026_0829_123850_F.MP4` happens to be
keyable because the filename carries the stamp, but nothing guarantees that, and
`VIDEO/MOVI0000.avi` never will be.

This module is the one place that decides:

* **what is in scope** - media and detection sidecars are; another program's
  JSON, a Python package and heatmap renders are not. The surveyed card was 69%
  out-of-scope by file count;
* **where each object lands** - the `YYYY/MM/DD/<key>[_F|_R|_FR].<ext>` layout
  discovery already expects, with sidecars sharing their media's stem;
* **what cannot be staged** - objects with no derivable key are reported, never
  given a fabricated one and never silently dropped.

Deliberately not here
---------------------

The copy itself. `auto_ingest.custody.executor` already copies per object,
atomically, refusing to overwrite, and recording a ledger. This module produces
the *plan*; it writes nothing.

Vocabulary, in this file's own terms:

``role``
    What kind of thing an object is: video, audio, a sidecar of a particular
    kind, or not content at all.

``key``
    The pipeline's join identity - ``YYYY_MMDD_HHMMSS``, optionally with a camera
    suffix. Two objects with the same key are two views of one recording.

``recording``
    The key with any camera suffix removed. This is what sidecars attach to.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

#: Roles, most specific first. Order matters: ``_YOLOv8n.csv`` must be tested
#: before the generic csv arm, exactly as the discovery chain does.
VIDEO_ROLE = "video"
AUDIO_ROLE = "audio"
DETECTION_CSV_ROLE = "detection_csv"

#: The marker this detector puts on its csv. Held once because `pairing` has to
#: remove it to compare against a clip's stem, and two literals that must agree
#: is exactly how that comparison starts silently reporting zero.
DETECTION_MARKER = "_YOLOv8n"
TRANSCRIPT_TXT_ROLE = "transcript_txt"
TRANSCRIPT_CSV_ROLE = "transcript_csv"
ENTITIES_CSV_ROLE = "entities_csv"
RTTM_ROLE = "rttm"
METADATA_CSV_ROLE = "metadata_csv"
OTHER_CSV_ROLE = "other_csv"
OTHER_ROLE = "other"

#: Roles that represent content the pipeline ingests.
CONTENT_ROLES = frozenset({VIDEO_ROLE, AUDIO_ROLE})

#: Roles that are sidecars - derived artefacts, ingested but never the payload.
SIDECAR_ROLES = frozenset({
    DETECTION_CSV_ROLE, TRANSCRIPT_TXT_ROLE, TRANSCRIPT_CSV_ROLE,
    ENTITIES_CSV_ROLE, RTTM_ROLE, METADATA_CSV_ROLE,
})

VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".mkv", ".m4v", ".avi", ".3gp",
                              ".mts", ".m2ts", ".webm"})
AUDIO_EXTENSIONS = frozenset({".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg",
                              ".opus", ".wma", ".amr", ".aiff"})

#: The dashcam key shape the pipeline joins on: ``YYYY_MMDD_HHMMSS`` plus an
#: OPTIONAL camera suffix.
#:
#: The suffix matters and is easy to miss: it lives in the *filename*, not in the
#: timestamp, so a pattern that stops at the six digits returns the same key for
#: ``..._123850_F.MP4`` and ``..._123850_R.MP4`` - and the two cameras of one
#: moment stage onto one destination path. `(?![0-9])` stops a longer digit run
#: from matching its first fourteen digits.
KEY_PATTERN = re.compile(
    r"(\d{4})_(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})(?![0-9])"
    # Longest alternatives FIRST: with (F|R|FR) the alternation matches F and
    # leaves the R behind, so "..._FR.mp4" stages as "..._F.mp4" - a wrong
    # filename for a clip that exists.
    r"(?:_(FR|RF|BC(?:-\d+)?|F|R))?",
    re.IGNORECASE,
)

#: Camera suffixes that identify two views of ONE recording. Stripped to form the
#: recording identity so a front/rear pair shares a key. ``transcripts.py:809``
#: carries the same list for the ingest path.
CAMERA_SUFFIX_PATTERN = re.compile(r"_(F|R|FR|RF|BC(-\d+)?)$", re.IGNORECASE)

#: The extension the dashcam detection path hardcodes when locating a clip beside
#: its CSV (``yolo_embeddings.py:1697``). Staging must not fight it.
DASHCAM_MEDIA_EXTENSION = ".MP4"


@dataclass(frozen=True)
class StagedObject:
    """One source object and the path it should occupy in the pipeline layout."""

    #: POSIX-relative path under the source root - the custody key.
    source_key: str
    #: POSIX-relative path under the destination root, or ``None`` when the object
    #: has no derivable key and therefore no place in the layout.
    destination_key: Optional[str]
    role: str
    #: ``YYYY_MMDD_HHMMSS`` when derivable.
    key: Optional[str]
    #: Key with any camera suffix removed - what sidecars attach to.
    recording: Optional[str]
    #: Camera suffix found on the source, e.g. ``_F``. ``None`` when absent.
    camera: Optional[str]
    #: Why this object cannot be staged. ``None`` when it can.
    reason: Optional[str] = None

    @property
    def stageable(self) -> bool:
        return self.destination_key is not None

    def to_dict(self) -> Dict[str, object]:
        return {
            "camera": self.camera,
            "destination_key": self.destination_key,
            "key": self.key,
            "reason": self.reason,
            "recording": self.recording,
            "role": self.role,
            "source_key": self.source_key,
            "stageable": self.stageable,
        }


def classify(name: str) -> str:
    """What kind of thing is this, by name alone.

    Sidecars are recognised **before** their generic extension, because
    ``_YOLOv8n.csv`` is a dashcam detection and must not be filed as a plain csv.
    Matching is case-insensitive throughout: the source is vfat and the
    destination exFAT, and the real card carries ``.MP4`` and ``.mp4`` alike.
    """
    lowered = name.lower()
    if lowered.endswith("_yolov8n.csv"):
        return DETECTION_CSV_ROLE
    if lowered.endswith("_speakers.rttm"):
        return RTTM_ROLE
    if lowered.endswith("_transcription.txt") or lowered.endswith("_transcription.jsonl"):
        return TRANSCRIPT_TXT_ROLE
    if "_transcription_entities.csv" in lowered or "_transcription_entites.csv" in lowered:
        return ENTITIES_CSV_ROLE
    if lowered.endswith("_transcription.csv"):
        return TRANSCRIPT_CSV_ROLE
    if lowered.endswith("_metadata.csv"):
        return METADATA_CSV_ROLE
    if lowered.endswith(".csv"):
        return OTHER_CSV_ROLE
    stem, dot, ext = lowered.rpartition(".")
    if dot:
        if f".{ext}" in VIDEO_EXTENSIONS:
            return VIDEO_ROLE
        if f".{ext}" in AUDIO_EXTENSIONS:
            return AUDIO_ROLE
    return OTHER_ROLE


def in_scope(role: str, *, include_sidecars: bool = True) -> bool:
    """Whether a role is content or a sidecar the pipeline ingests.

    Everything else - another program's JSON, a Python package, heatmap renders -
    is out. It is reported, never deleted, never copied.
    """
    return role in CONTENT_ROLES or (include_sidecars and role in SIDECAR_ROLES)


def derive_key(source_key: str) -> Optional[str]:
    """The pipeline join key for a source path, or ``None`` if there is none.

    The stamp may live in the filename or anywhere in the path, because
    ``canonicalize_key`` searches the full path too (``transcripts.py:236-242``)
    - which is what makes a ``YYYY/MM/DD/`` directory component sufficient.
    """
    match = KEY_PATTERN.search(source_key)
    # The whole match, source case preserved: the dashcam detector rebuilds
    # ``{key}.MP4`` from the CSV's own spelling, so a normalised key would point
    # it at a file that is not there.
    return match.group(0) if match else None


def split_camera(key: str) -> Tuple[str, Optional[str]]:
    """Split ``YYYY_MMDD_HHMMSS_F`` into (``YYYY_MMDD_HHMMSS``, ``_F``)."""
    match = CAMERA_SUFFIX_PATTERN.search(key)
    if not match:
        return key, None
    return key[: match.start()], match.group(0)


#: Roles that are PER-RECORDING, so the camera suffix is dropped and a front/rear
#: pair deliberately shares one staged name. These are the artefacts of a
#: conversation, not of a clip: one transcript, one speaker map, one entity table.
#: `canonicalize_key` rebuilds `YYYY_MMDD_HHMMSS` from the name regardless
#: (`transcripts.py:245-249`), so this collapse is the pipeline's own intent.
PER_RECORDING_ROLES = frozenset({
    TRANSCRIPT_TXT_ROLE, TRANSCRIPT_CSV_ROLE, ENTITIES_CSV_ROLE,
    RTTM_ROLE, METADATA_CSV_ROLE,
})


def date_directory(key: str) -> str:
    """``YYYY_MM_DD_HHMMSS`` -> ``YYYY/MM/DD``.

    Built from the regex groups, never from string slicing. The key carries
    underscores between its fields, so ``key[4:6]`` is ``_0``, not the month -
    and an off-by-one here yields ``2026/_0/82``, a plausible-looking path that
    nothing will ever discover again. The groups cannot drift.
    """
    match = KEY_PATTERN.search(key)
    if match is None:
        raise ValueError(f"not a pipeline key: {key!r}")
    year, month, day = match.group(1), match.group(2), match.group(3)
    return f"{year}/{month}/{day}"


def staged_filename(stem: str, suffix: str, role: str, *,
                    key: Optional[str] = None, base: Optional[str] = None) -> str:
    """The basename this object takes in the pipeline layout.

    Two different rules, because the two subsystems pair differently:

    * **Per-clip** objects are named from the key. The dashcam detector recovers a
      stem with ``rsplit("_YOLOv8n", 1)`` and then opens ``{stem}.MP4`` by exact
      name (``yolo_embeddings.py:1695-1697``), so the camera suffix must survive:
      stripping it would point both cameras' CSVs at one file, which is why
      `StagingPlan.collisions` exists to surface this class rather than resolve
      it.
    * **Per-recording** sidecars are named from the source, minus the camera
      suffix, so ``_F`` and ``_R`` of one moment share a transcript - and the rest
      of the name is kept exactly as it was.

    That last clause is the whole point, and getting it wrong is invisible from
    the destination alone. The pipeline finds sidecars by their tails, not their
    stems: ``_([^_]+)_transcription.txt``, ``_speakers.rttm``, ``_metadata.csv``,
    ``_transcription_entities.csv`` (``transcripts.py:94-99``). Renaming
    ``2025_0202_171732_medium_transcription.txt`` to ``2025_0202_171732.txt``
    lands the file perfectly and makes it undiscoverable, because the model tag
      *is* the handle. Worse, it is a collision waiting to happen: every model
      that transcribed one moment would land on the same path.

    So the suffix is rewritten - camera removed - and nothing else is touched.
    When the key came from the path rather than the basename there is no prefix
    to rewrite, and the name is kept verbatim.
    """
    if role == DETECTION_CSV_ROLE:
        return f"{stem}{DETECTION_MARKER}.csv"
    if role in PER_RECORDING_ROLES and key and base:
        # `base` is the source basename without its extension. Rewriting only the
        # key prefix leaves every marker after it in place.
        if base.startswith(key):
            return f"{stem}{base[len(key):]}{suffix}"
        return f"{base}{suffix}"
    return f"{stem}{suffix}"


def destination_for(source_key: str, *, role: Optional[str] = None,
                    include_sidecars: bool = True) -> StagedObject:
    """Plan one source object's place in the pipeline layout.

    Pure: reads nothing, writes nothing, and never invents a key. An object with
    no derivable stamp comes back ``stageable=False`` with a reason, so the caller
    decides rather than the plan quietly dropping it.
    """
    import posixpath

    name = posixpath.basename(source_key)
    stem_only, _, _ = name.rpartition(".")
    base = stem_only if "." in name else name
    role = role or classify(name)
    # Scope is judged BEFORE the key. A Python file in overland/ has no
    # timestamp and never will, so "no key" describes a problem that does not
    # exist while hiding the one that does: it is not pipeline content at all.
    # The order also keeps --media-only honest, since a CSV it declines is
    # genuinely declined rather than merely unkeyed.
    if not in_scope(role, include_sidecars=include_sidecars):
        return StagedObject(
            source_key=source_key,
            destination_key=None,
            role=role,
            key=None,
            recording=None,
            camera=None,
            reason=f"out_of_scope:{role}",
        )
    key = derive_key(source_key)
    if key is None:
        return StagedObject(
            source_key=source_key,
            destination_key=None,
            role=role,
            key=None,
            recording=None,
            camera=None,
            reason="no_YYYY_MMDD_HHMMSS_in_name_or_path",
        )

    stem, camera = split_camera(key)
    _, dot, ext = name.rpartition(".")
    suffix = f".{ext}" if dot else ""
    # Per-clip roles keep the camera suffix; per-recording roles drop it. Both
    # then have their marker-bearing tail preserved (see staged_filename).
    name_stem = stem if role in PER_RECORDING_ROLES else key
    destination = (f"{date_directory(key)}/"
                   f"{staged_filename(name_stem, suffix, role, key=key, base=base)}")
    return StagedObject(
        source_key=source_key,
        destination_key=destination,
        role=role,
        key=key,
        recording=stem,
        camera=camera,
        reason=None,
    )


@dataclass(frozen=True)
class StagingPlan:
    """Every source object, split by whether it has a place in the layout."""

    staged: Tuple[StagedObject, ...] = ()
    unstaged: Tuple[StagedObject, ...] = ()

    @property
    def destination_keys(self) -> Tuple[str, ...]:
        return tuple(o.destination_key for o in self.staged if o.destination_key)

    @property
    def source_keys(self) -> Tuple[str, ...]:
        return tuple(o.source_key for o in self.staged if o.destination_key)

    def by_reason(self) -> Dict[str, int]:
        """Counts per unstaged reason. Bounded by the number of reasons."""
        counts: Dict[str, int] = {}
        for obj in self.unstaged:
            reason = obj.reason or "unknown"
            counts[reason] = counts.get(reason, 0) + 1
        return counts

    def by_role(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for obj in self.staged:
            counts[obj.role] = counts.get(obj.role, 0) + 1
        return counts

    def pairing(self) -> Dict[str, int]:
        """How staged clips and detection files relate, per destination directory.

        This is the measurement that turns "we staged 3,558 files" into an
        operator decision. On the real card it reported 624 clips and 2,934
        detection files across *disjoint* date ranges - 2026/08-29..09/01 for the
        media, 2024/07/13..08/03 for the detections - with no 2024 video present
        at all. So every one of those 2,934 csv files described a recording that
        is not on the card.

        That is a legitimate state and not an error: YOLO csvs are produced by
        running the detector, so a card can hold the outputs of a session whose
        media was already archived, plus fresh media whose detections do not exist
        yet. But it is invisible unless something counts it, and "2,934 staged
        files, none of them paired" would otherwise read as a naming failure.

        Counts, not paths: a bounded summary is what belongs in a status report.
        """
        clips: Dict[str, set] = {}
        detections: Dict[str, set] = {}
        for obj in self.staged:
            directory, _, name = obj.destination_key.rpartition("/")
            base, dot, _ = name.rpartition(".")
            if not dot:
                continue
            if obj.role == DETECTION_CSV_ROLE:
                detections.setdefault(directory, set()).add(
                    base[: -len(DETECTION_MARKER)]
                    if base.endswith(DETECTION_MARKER) else base)
            elif obj.role == VIDEO_ROLE:
                clips.setdefault(directory, set()).add(base)
        paired = sum(len(clips.get(d, set()) & stems) for d, stems in detections.items())
        clips_total = sum(len(s) for s in clips.values())
        det_total = sum(len(s) for s in detections.values())
        return {
            "clips": clips_total,
            "detections": det_total,
            "paired": paired,
            # A csv whose clip is not here: a leftover from an archived session.
            "detections_without_clip": det_total - paired,
            # A clip with no csv: not yet detected, which is the normal state of
            # footage arriving from a card.
            "clips_without_detection": clips_total - paired,
        }

    def collisions(self) -> Tuple[str, ...]:
        """Destination paths claimed by more than one source object.

        Two different recordings writing the same staged path is a layout bug, not
        a filesystem quirk, and it must never be resolved by pick-a-winner: the
        caller has to decide which recording the name belongs to.
        """
        seen: Dict[str, List[str]] = {}
        for obj in self.staged:
            if obj.destination_key:
                seen.setdefault(obj.destination_key, []).append(obj.source_key)
        return tuple(sorted(
            destination for destination, sources in seen.items() if len(sources) > 1
        ))

    def to_dict(self) -> Dict[str, object]:
        return {
            "staged_files": len(self.staged),
            "unstaged_files": len(self.unstaged),
            "by_role": self.by_role(),
            "unstaged_by_reason": self.by_reason(),
            "destination_collisions": list(self.collisions()),
        }


def plan_staging(source_keys: Iterable[str], *,
                 include_sidecars: bool = True) -> StagingPlan:
    """Plan the whole card in one pass.

    ``source_keys`` are POSIX-relative paths under the source root, in any order.
    Sorted internally so two runs over an unchanged card produce an identical plan
    - the same determinism the ledgers rely on.
    """
    staged: List[StagedObject] = []
    unstaged: List[StagedObject] = []
    for source_key in sorted(source_keys):
        obj = destination_for(source_key, include_sidecars=include_sidecars)
        (staged if obj.stageable else unstaged).append(obj)
    return StagingPlan(staged=tuple(staged), unstaged=tuple(unstaged))


__all__ = [
    "AUDIO_EXTENSIONS",
    "AUDIO_ROLE",
    "CAMERA_SUFFIX_PATTERN",
    "CONTENT_ROLES",
    "DASHCAM_MEDIA_EXTENSION",
    "DETECTION_CSV_ROLE",
    "KEY_PATTERN",
    "METADATA_CSV_ROLE",
    "OTHER_CSV_ROLE",
    "OTHER_ROLE",
    "PER_RECORDING_ROLES",
    "RTTM_ROLE",
    "SIDECAR_ROLES",
    "TRANSCRIPT_CSV_ROLE",
    "TRANSCRIPT_TXT_ROLE",
    "VIDEO_EXTENSIONS",
    "VIDEO_ROLE",
    "StagedObject",
    "StagingPlan",
    "classify",
    "date_directory",
    "derive_key",
    "destination_for",
    "in_scope",
    "plan_staging",
    "split_camera",
    "staged_filename",
]
