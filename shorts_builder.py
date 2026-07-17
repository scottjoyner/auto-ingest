#!/usr/bin/env python3
from __future__ import annotations

from auto_ingest_config import get_fileserver_path
# -*- coding: utf-8 -*-
"""
Build vertical 9:16 'shorts' with dynamic, speaker-aware captions from *_FR.MP4.

Adds:
- Optional diarization (RTTM) -> per-word speaker labels
- Optional local->global speaker mapping (JSON)
- Optional Neo4j manifest upsert for Video/Short/Segments/Speakers
"""


import argparse
import csv
import hashlib
import json
import logging
import os
import re
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import CompositeVideoClip, ImageClip, VideoFileClip

# Lazy import whisper only if needed
_whisper = None
try:
    from neo4j import GraphDatabase
except Exception:
    GraphDatabase = None  # Only needed if --neo4j-uri is used

# --------------------------------------------------------------------------- #
# Shared caption / compose helpers live in the modern package (single source of
# truth). The legacy CLI below delegates to them so behavior is unchanged.
# --------------------------------------------------------------------------- #
from auto_ingest.shorts.captions import (
    DEFAULT_PROFILES,
    load_profiles_from_json,
    SpeakerMapEntry,
    load_speaker_map,
    _hash_color,
    _normalize_colors_dict,
    _normalize_profile_from_json,
    _apply_speaker_override,
    _ensure_parent_dir,
    _temp_path_with_same_ext,
    _atomic_replace,
    _write_videofile_safely,
    _load_font,
    _wrap_text,
    _speaker_display_name,
    _speaker_color,
    _render_tag,
    _cue_kind_style,
    _draw_scrim,
    _apply_kenburns,
    _draw_speed_hud,
    _draw_end_card,
    _draw_sentence,
    _draw_sentence_highlight,
    _draw_wordgrid,
    _render_cached,
    _cap_key,
    _build_caption_image,
    _cue_word_timings,
)
from auto_ingest.shorts.compose import compose_scripted_short


DATE_DIR_RE = re.compile(r"^\d{4}/\d{2}/\d{2}$")





# ---------------------------
# Diarization (RTTM) parsing
# ---------------------------

@dataclass
class DiarSegment:
    start: float
    duration: float
    end: float
    speaker: str


def parse_rttm(rttm_path: Path) -> List[DiarSegment]:
    """
    Parse a simple RTTM file where lines contain:
      SPEAKER <file> 1 <onset> <dur> <ortho> <stype> <name> <conf> <speaker>
    We'll read onset, dur, and speaker label.
    """
    segs: List[DiarSegment] = []
    for line in rttm_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] != "SPEAKER":
            continue
        try:
            onset = float(parts[3])
            dur = float(parts[4])
            spk = parts[7] if len(parts) > 7 else parts[-1]
            segs.append(DiarSegment(onset, dur, onset + dur, spk))
        except Exception:
            continue
    segs.sort(key=lambda s: s.start)
    return segs


def align_words_to_speakers(whisper_result: dict, diar: List[DiarSegment]) -> None:
    """
    Mutates whisper_result["segments"][i]["words"][j] to include "speaker" where possible.
    Strategy: assign by word midpoint ∈ diarization segment.
    """
    if not diar:
        return
    for seg in whisper_result.get("segments", []):
        for w in seg.get("words", []):
            st = w.get("start")
            en = w.get("end")
            if st is None or en is None:
                continue
            mid = 0.5 * (st + en)
            label = None
            for ds in diar:
                if ds.start <= mid <= ds.end:
                    label = ds.speaker
                    break
            if label:
                w["speaker"] = label
        labels = [w.get("speaker") for w in seg.get("words", []) if w.get("speaker")]
        if labels:
            seg["speaker"] = max(set(labels), key=labels.count)




# ---------------------------
# Transcription I/O
# ---------------------------

def _default_transcript_paths(fr_file: Path, model: str) -> Tuple[Path, Path]:
    base = fr_file.with_suffix("")
    json_path = base.with_name(f"{base.name}_{model}_whisper.json")
    csv_path = base.with_name(f"{base.name}_transcription.csv")
    return json_path, csv_path


def save_whisper_json(result: dict, path: Path) -> None:
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def save_segments_csv(result: dict, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["SegmentIndex", "Start", "End", "Speaker", "Text"])
        for i, seg in enumerate(result.get("segments", [])):
            w.writerow([i, seg.get("start"), seg.get("end"), seg.get("speaker", ""), (seg.get("text") or "").strip()])


def load_or_transcribe(
    audio_path: Path,
    json_path: Path,
    *,
    model_size: str,
    device: str,
    language: Optional[str],
    overwrite: bool,
) -> dict:
    if json_path.exists() and not overwrite:
        logging.info(f"Using cached transcript: {json_path.name}")
        return json.loads(json_path.read_text(encoding="utf-8"))
    global _whisper
    if _whisper is None:
        import whisper as _w
        _whisper = _w
    logging.info(f"Transcribing {audio_path.name} with Whisper({model_size}, device={device})")
    model = _whisper.load_model(model_size, device=None if device == "auto" else device)
    kwargs = {"word_timestamps": True}
    if language:
        kwargs["language"] = language
    result = model.transcribe(str(audio_path), **kwargs)
    save_whisper_json(result, json_path)
    return result


# ---------------------------
# Safe video writing utilities
# ---------------------------


# ---------------------------
# Video: crop to 9:16
# ---------------------------

def crop_to_vertical(src_video: Path, dst_video: Path, target_w: int, target_h: int) -> Tuple[float, float]:
    """
    Center-crop the source to target aspect (9:16 by default), resize to (target_w, target_h),
    write safely to dst_video, and return (fps, duration).
    """
    with VideoFileClip(str(src_video)) as clip:
        w, h = clip.size
        current_aspect = w / float(h)
        target_aspect = target_w / float(target_h)

        if current_aspect > target_aspect:
            new_w = int(h * target_aspect)
            x1 = (w - new_w) // 2
            cropped = clip.crop(x1=x1, y1=0, x2=x1 + new_w, y2=h)
        else:
            new_h = int(w / target_aspect)
            y1 = (h - new_h) // 2
            cropped = clip.crop(x1=0, y1=y1, x2=w, y2=y1 + new_h)

        final = cropped.resize((target_w, target_h))
        fps = float(clip.fps or 30.0)
        duration = float(final.duration or clip.duration or 0.0)

        _write_videofile_safely(
            final,
            dst_video,
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            bitrate="6M",
        )

        try:
            final.close()
        except Exception:
            pass

        return fps, duration


def compose_shorts(
    vertical_video: Path,
    transcript: dict,
    out_path: Path,
    *,
    profile_name: str,
    profile: Dict,
    y_ratio: float,
    speaker_map: Dict[str, SpeakerMapEntry],
) -> None:
    _ensure_parent_dir(out_path)
    with VideoFileClip(str(vertical_video)) as vclip:
        H, W = vclip.h, vclip.w
        base_y = int(H * y_ratio)

        clips = [vclip]

        if profile.get("wordgrid", False):
            for seg in transcript.get("segments", []):
                words = seg.get("words", [])
                if not words:
                    continue
                for idx, w in enumerate(words):
                    st, en = w.get("start"), w.get("end")
                    if st is None or en is None:
                        continue
                    img_key = _cap_key(
                        "wordgrid",
                        words=words,
                        active_idx=idx,
                        width=W,
                        profile=profile,
                        speaker_map={k: v.__dict__ for k, v in speaker_map.items()},
                    )
                    arr = _render_cached(img_key)
                    ic = ImageClip(arr).set_duration(en - st).set_start(st).set_position(("center", base_y))
                    clips.append(ic)
        else:
            for seg in transcript.get("segments", []):
                s_start, s_end = seg.get("start"), seg.get("end")
                sentence = (seg.get("text") or "").strip()
                if not sentence or s_start is None or s_end is None:
                    continue
                seg_speaker = seg.get("speaker")

                base_key = _cap_key(
                    "sentence",
                    sentence=sentence,
                    width=W,
                    profile=profile,
                    speaker_label=seg_speaker,
                    speaker_map={k: v.__dict__ for k, v in speaker_map.items()},
                )
                base_arr = _render_cached(base_key)
                base_clip = ImageClip(base_arr).set_duration(s_end - s_start).set_start(s_start).set_position(("center", base_y))
                clips.append(base_clip)

                if profile.get("karaoke", True):
                    for w in seg.get("words", []):
                        token = (w.get("word") or "").strip()
                        st, en = w.get("start"), w.get("end")
                        if not token or st is None or en is None:
                            continue
                        w_speaker = w.get("speaker", seg_speaker)
                        kara_key = _cap_key(
                            "karaoke",
                            sentence=sentence,
                            word=token,
                            width=W,
                            profile=profile,
                            speaker_label=w_speaker,
                            speaker_map={k: v.__dict__ for k, v in speaker_map.items()},
                        )
                        kara_arr = _render_cached(kara_key)
                        kara_clip = ImageClip(kara_arr).set_duration(en - st).set_start(st).set_position(("center", base_y))
                        clips.append(kara_clip)

        logging.info(f"Writing {out_path.name}")
        comp = CompositeVideoClip(clips)
        _write_videofile_safely(
            comp,
            out_path,
            fps=float(vclip.fps or 30.0),
            codec="libx264",
            audio_codec="aac",
            bitrate="6M",
        )
        try:
            comp.close()
        except Exception:
            pass
    logging.info(f"Wrote {out_path}")




# ---------------------------
# Discovery & orchestration
# ---------------------------

def is_valid_date_structure(base: Path, d: Path) -> bool:
    try:
        rel = d.relative_to(base).as_posix()
    except ValueError:
        return False
    if not DATE_DIR_RE.match(rel):
        return False
    y, m, day = rel.split("/")
    import datetime as _dt
    try:
        _dt.datetime(int(y), int(m), int(day))
        return True
    except ValueError:
        return False


def iter_date_dirs(base: Path) -> Iterable[Path]:
    for y in sorted(p for p in base.iterdir() if p.is_dir() and p.name.isdigit() and len(p.name) == 4):
        for m in sorted(p for p in y.iterdir() if p.is_dir() and p.name.isdigit() and len(p.name) == 2):
            for d in sorted(p for p in m.iterdir() if p.is_dir() and p.name.isdigit() and len(p.name) == 2):
                if is_valid_date_structure(base, d):
                    yield d


def iter_fr_mp4(root: Path) -> Iterable[Path]:
    for p in root.glob("*_FR.MP4"):
        if p.is_file():
            yield p


def find_sidecar(path: Path, suffix: str) -> Optional[Path]:
    cand = path.with_name(path.stem.replace("_FR", "") + suffix)
    return cand if cand.exists() else None


# ---------------------------
# Neo4j upsert
# ---------------------------

def neo4j_upsert_manifest(
    uri: str,
    user: str,
    password: str,
    *,
    video_key: str,
    fr_path: str,
    short_profile: str,
    short_path: str,
    width: int,
    height: int,
    fps: float,
    duration: float,
    whisper_model: str,
    transcript: dict,
    speaker_map: Dict[str, SpeakerMapEntry],
) -> None:
    if GraphDatabase is None:
        logging.error("neo4j Python driver not installed; cannot upsert manifest.")
        return

    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as sess:
        sess.run(
            """
        MERGE (v:Video {key:$video_key})
          ON CREATE SET v.path=$fr_path, v.created_at=timestamp()
          ON MATCH  SET v.path=$fr_path
        MERGE (s:Short {key:$video_key, profile:$profile})
          ON CREATE SET s.path=$short_path, s.width=$width, s.height=$height,
                        s.fps=$fps, s.duration=$duration, s.model=$model, s.created_at=timestamp()
          ON MATCH  SET s.path=$short_path, s.width=$width, s.height=$height,
                        s.fps=$fps, s.duration=$duration, s.model=$model
        MERGE (s)-[:FROM_VIDEO]->(v)
        """,
            video_key=video_key,
            fr_path=fr_path,
            profile=short_profile,
            short_path=short_path,
            width=width,
            height=height,
            fps=fps,
            duration=duration,
            model=whisper_model,
        )

        for i, seg in enumerate(transcript.get("segments", [])):
            s_start = seg.get("start")
            s_end = seg.get("end")
            s_text = (seg.get("text") or "").strip()
            s_speaker = seg.get("speaker")
            entry = speaker_map.get(s_speaker) if s_speaker else None
            global_id = entry.global_id if entry and entry.global_id else None

            sess.run(
                """
            MERGE (seg:Segment {key:$video_key, index:$idx, profile:$profile})
              ON CREATE SET seg.start=$start, seg.end=$end, seg.text=$text, seg.speaker_label=$speaker, seg.created_at=timestamp()
              ON MATCH  SET seg.start=$start, seg.end=$end, seg.text=$text, seg.speaker_label=$speaker
            MERGE (short:Short {key:$video_key, profile:$profile})
            MERGE (short)-[:HAS_SEGMENT]->(seg)
            """,
                video_key=video_key,
                idx=i,
                profile=short_profile,
                start=s_start,
                end=s_end,
                text=s_text,
                speaker=s_speaker,
            )

            if s_speaker:
                sess.run(
                    """
                MERGE (sp:Speaker {label:$label})
                MERGE (seg:Segment {key:$video_key, index:$idx, profile:$profile})
                MERGE (seg)-[:SPOKEN_BY]->(sp)
                """,
                    label=s_speaker,
                    video_key=video_key,
                    idx=i,
                    profile=short_profile,
                )

                if global_id:
                    sess.run(
                        """
                    MERGE (g:GlobalSpeaker {global_id:$gid})
                      ON CREATE SET g.name=$gname, g.created_at=timestamp()
                      ON MATCH  SET g.name=coalesce($gname, g.name)
                    MERGE (sp:Speaker {label:$label})
                    MERGE (sp)-[:IS]->(g)
                    """,
                        gid=global_id,
                        gname=entry.name,
                        label=s_speaker,
                    )

    driver.close()


# ---------------------------
# File-level processing
# ---------------------------

def process_one(
    fr_path: Path,
    *,
    profiles: Dict[str, Dict],
    model_size: str,
    device: str,
    language: Optional[str],
    target_w: int,
    target_h: int,
    overwrite: bool,
    dry_run: bool,
    diarization_path: Optional[Path],
    speaker_map: Dict[str, SpeakerMapEntry],
    neo4j_cfg: Optional[dict],
) -> Tuple[int, int]:
    planned = 0
    done = 0

    base = fr_path.with_suffix("")
    key = base.name  # e.g., 2025_0101_173933
    r_audio = fr_path.with_name(f"{key}_R.MP4")
    audio_path = r_audio if r_audio.exists() else fr_path

    json_path, csv_path = _default_transcript_paths(fr_path, model_size)
    vertical_path = fr_path.with_name(f"{key}_FR_vertical.mp4")

    outputs = [fr_path.with_name(f"{key}_short-{name}.mp4") for name in profiles.keys()]
    if all(p.exists() for p in outputs) and vertical_path.exists() and json_path.exists() and csv_path.exists() and not overwrite:
        logging.info(f"Skip (all outputs exist): {fr_path.name}")
        return (0, 0)

    planned = len(outputs)

    if dry_run:
        logging.info(f"[DRY-RUN] Would process {fr_path.name} -> {', '.join(p.name for p in outputs)}")
        return (0, planned)

    # Transcribe
    transcript = load_or_transcribe(
        audio_path=audio_path,
        json_path=json_path,
        model_size=model_size,
        device=device,
        language=language,
        overwrite=overwrite,
    )

    # Diarization (optional)
    diar_file = diarization_path or find_sidecar(fr_path, "_diarization.rttm")
    diar_segments: List[DiarSegment] = parse_rttm(diar_file) if (diar_file and diar_file.exists()) else []
    if diar_segments:
        align_words_to_speakers(transcript, diar_segments)

    save_segments_csv(transcript, csv_path)

    # Vertical crop/resize
    if overwrite or not vertical_path.exists():
        fps, duration = crop_to_vertical(fr_path, vertical_path, target_w, target_h)
    else:
        with VideoFileClip(str(vertical_path)) as vc:
            fps, duration = float(vc.fps or 30.0), float(vc.duration or 0.0)

    # Build each profile
    for pname, prof in profiles.items():
        out = fr_path.with_name(f"{key}_short-{pname}.mp4")
        if out.exists() and not overwrite:
            logging.debug(f"Skip (exists): {out.name}")
            continue
        try:
            compose_shorts(
                vertical_video=vertical_path,
                transcript=transcript,
                out_path=out,
                profile_name=pname,
                profile=prof,
                y_ratio=prof.get("y_pos_ratio", 0.82),
                speaker_map=speaker_map,
            )
            done += 1

            if neo4j_cfg:
                neo4j_upsert_manifest(
                    neo4j_cfg["uri"],
                    neo4j_cfg["user"],
                    neo4j_cfg["password"],
                    video_key=key,
                    fr_path=str(fr_path),
                    short_profile=pname,
                    short_path=str(out),
                    width=target_w,
                    height=target_h,
                    fps=fps,
                    duration=duration,
                    whisper_model=model_size,
                    transcript=transcript,
                    speaker_map=speaker_map,
                )
        except Exception as e:
            logging.exception(f"Failed building {out.name}: {e}")

    return (done, planned)


def process_root(
    root: Path,
    *,
    profiles: Dict[str, Dict],
    model_size: str,
    device: str,
    language: Optional[str],
    target_w: int,
    target_h: int,
    overwrite: bool,
    dry_run: bool,
    diarization_path: Optional[Path],
    speaker_map: Dict[str, SpeakerMapEntry],
    neo4j_cfg: Optional[dict],
) -> Tuple[int, int]:
    planned = 0
    done = 0
    for date_dir in iter_date_dirs(root):
        for fr in iter_fr_mp4(date_dir):
            d, p = process_one(
                fr,
                profiles=profiles,
                model_size=model_size,
                device=device,
                language=language,
                target_w=target_w,
                target_h=target_h,
                overwrite=overwrite,
                dry_run=dry_run,
                diarization_path=diarization_path,
                speaker_map=speaker_map,
                neo4j_cfg=neo4j_cfg,
            )
            planned += p
            done += d
    return (done, planned)


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build speaker-aware phone shorts from *_FR.MP4 with optional Neo4j manifest.")
    # Inputs
    p.add_argument("--base", dest="bases", action="append", default=[], help="Dashcam root with YYYY/MM/DD (repeatable).")
    p.add_argument("--file", type=Path, default=None, help="Single *_FR.MP4 to process.")
    # Whisper
    p.add_argument("--whisper-model", default="base", help="tiny/base/small/medium/large")
    p.add_argument("--device", default="auto", help="auto/cpu/cuda")
    p.add_argument("--language", default=None, help="Force language (e.g., en)")
    # Geometry
    p.add_argument("--width", type=int, default=1080)
    p.add_argument("--height", type=int, default=1920)
    # Behavior
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--profiles", nargs="+", default=["clean", "karaoke"], help="Subset of clean/karaoke/wordgrid")
    p.add_argument("--profiles-file", type=Path, default=None, help="Path to a JSON file containing profile dicts to merge (keys become profile names).")
    p.add_argument("--log-level", default="INFO")
    # Diarization / speakers
    p.add_argument("--diarization", type=Path, default=None, help="RTTM file (if omitted, auto-detect <key>_diarization.rttm).")
    p.add_argument("--speaker-map", type=Path, default=None, help="JSON mapping local speaker -> {global_id,name}.")
    # Neo4j
    p.add_argument("--neo4j-uri", type=str, default=None)
    p.add_argument("--neo4j-user", type=str, default=None)
    p.add_argument("--neo4j-pass", type=str, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    if args.device == "auto":
        args.device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "-1") else "cpu"

    # Load/merge profiles once
    profiles: Dict[str, Dict] = {k: DEFAULT_PROFILES[k] for k in args.profiles if k in DEFAULT_PROFILES}
    if args.profiles_file:
        if not Path(args.profiles_file).exists():
            logging.error(f"--profiles-file not found: {args.profiles_file}")
            return 2
        ext_profiles = load_profiles_from_json(Path(args.profiles_file))
        profiles.update(ext_profiles)
        if args.profiles:
            requested = set(args.profiles)
            filtered = {k: v for k, v in profiles.items() if k in requested}
            profiles = filtered or profiles  # don't end up empty if names mismatch

    smap = load_speaker_map(args.speaker_map)

    neo4j_cfg = None
    if args.neo4j_uri and args.neo4j_user and args.neo4j_pass:
        neo4j_cfg = {"uri": args.neo4j_uri, "user": args.neo4j_user, "password": args.neo4j_pass}

    total_done = 0
    total_planned = 0

    if args.file:
        fr = args.file.resolve()
        if not fr.exists() or not fr.name.endswith("_FR.MP4"):
            logging.error("--file must be an existing *_FR.MP4")
            return 2
        d, p = process_one(
            fr,
            profiles=profiles,
            model_size=args.whisper_model,
            device=args.device,
            language=args.language,
            target_w=args.width,
            target_h=args.height,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            diarization_path=args.diarization,
            speaker_map=smap,
            neo4j_cfg=neo4j_cfg,
        )
        total_done += d
        total_planned += p
    else:
        bases = [Path(b).resolve() for b in (args.bases or [get_fileserver_path("dashcam/")])]
        for base in bases:
            if not base.is_dir():
                logging.warning(f"--base not found/dir: {base}")
                continue
            d, p = process_root(
                base,
                profiles=profiles,
                model_size=args.whisper_model,
                device=args.device,
                language=args.language,
                target_w=args.width,
                target_h=args.height,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                diarization_path=args.diarization,
                speaker_map=smap,
                neo4j_cfg=neo4j_cfg,
            )
            total_done += d
            total_planned += p

    logging.info(f"Completed {total_done}/{total_planned} short(s).")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
