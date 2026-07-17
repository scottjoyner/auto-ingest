"""Cinematic scripted-short composition for the modern shorts package.

This is the in-package replacement for the legacy root-level
``shorts_builder.compose_scripted_short``. It produces the same 9:16 MP4 given
the same inputs (profiles, cues, ken burns, end card, scrim, speed HUD). The
legacy ``shorts_builder.py`` still exists for its CLI but now delegates the
shared caption helpers to :mod:`auto_ingest.shorts.captions`.

Behavior is preserved verbatim from the original implementation.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from auto_ingest.shorts.captions import (
    DEFAULT_PROFILES,
    _apply_kenburns,
    _build_caption_image,
    _cue_kind_style,
    _cue_word_timings,
    _draw_end_card,
    _draw_scrim,
    _draw_sentence_highlight,
    _draw_speed_hud,
    _load_font,
    _normalize_profile_from_json,
    _render_tag,
    _write_videofile_safely,
)

log = logging.getLogger("shorts.compose")


def compose_scripted_short(
    shots: List[Dict[str, Any]],
    cues: List[Dict[str, Any]],
    out_path: Path,
    *,
    profile_name: str = "clean",
    profile: Optional[Dict] = None,
    y_ratio: float = 0.78,
    width: int = 1080,
    height: int = 1920,
    target_w: int = 1080,
    target_h: int = 1920,
    bitrate: str = "6M",
    fade: float = 0.12,
    narration_audio: Optional[Path] = None,
    end_card: bool = True,
    hashtag: str = "",
) -> None:
    """Compose a research-scripted short from highway B-roll + a cue track.

    Overlays an externally-authored **script** (the ``cues`` list of
    ``{start, end, text, kind}``) on top of one or more highway clips
    (``shots`` of ``{fr_path, t_sec, dur}``). Falls back gracefully when
    moviepy is unavailable by raising ``ImportError`` so the planner/curator
    remain unit-testable without the render stack.
    """
    try:
        from moviepy.editor import (
            CompositeVideoClip,
            ImageClip,
            VideoFileClip,
            concatenate_videoclips,
        )
    except Exception as e:  # pragma: no cover - environment dependent
        raise ImportError("moviepy is required to render scripted shorts") from e

    if profile is None:
        profile = DEFAULT_PROFILES.get(profile_name, DEFAULT_PROFILES["clean"])
    profile = _normalize_profile_from_json(profile)

    empty_smap: Dict[str, Any] = {}

    # Build the B-roll piece(s) from highway shots.
    pieces = []
    src_clips = []
    shot_windows: List[Tuple[float, float, Optional[float]]] = []
    cum = 0.0
    for i, sh in enumerate(shots):
        frp = Path(sh.get("fr_path") or "")
        if not frp or not frp.exists() or not frp.is_file():
            logging.warning("Skipping shot %d: missing footage %r", i, sh.get("fr_path"))
            cum += float(sh.get("dur", 6.0))
            continue
        dur = float(sh.get("dur", 6.0))
        v = VideoFileClip(str(frp)).without_audio()
        src_clips.append(v)
        clip_dur = float(v.duration or 0.0)
        t0 = float(sh.get("t_sec", 0.0))
        if clip_dur and t0 + dur > clip_dur:
            t0 = max(0.0, clip_dur - dur)
        t1 = min(clip_dur, t0 + dur) if clip_dur else t0 + dur
        sub = v.subclip(max(0.0, t0), t1)
        src_w, src_h = sub.w, sub.h
        tgt_aspect = float(width) / float(height)
        if src_w / float(src_h) > tgt_aspect:
            cw = int(src_h * tgt_aspect)
            ch = src_h
        else:
            cw = src_w
            ch = int(src_w / tgt_aspect)
        x1 = max(0, (src_w - cw) // 2)
        y1 = max(0, (src_h - ch) // 2)
        vcrop = sub.crop(x1=x1, y1=y1, width=cw, height=ch)
        vcrop = vcrop.resize((width, height))
        if profile.get("kenburns", True):
            vcrop = _apply_kenburns(vcrop)
        if i > 0 and fade:
            vcrop = vcrop.crossfadein(fade)
        pieces.append(vcrop)
        mph = sh.get("mph")
        shot_windows.append((cum, cum + dur, float(mph) if mph is not None else None))
        cum += dur

    if not pieces:
        for sc in src_clips:
            try:
                sc.close()
            except Exception:
                pass
        raise RuntimeError("No highway shots to render (shot list empty).")
    broll = concatenate_videoclips(pieces, method="compose") if len(pieces) > 1 else pieces[0]
    total = float(broll.duration or 0.0)

    # Speed HUD: a small top-left readout per shot that carries an mph value.
    hud_clips = []
    if profile.get("show_speed_hud", True):
        for (t0, t1, mph) in shot_windows:
            if mph is None:
                continue
            hud = ImageClip(np.array(_draw_speed_hud(mph, width, height))).set_start(t0)
            hud = hud.set_duration(max(0.0, min(t1, total) - t0))
            hud = hud.crossfadein(0.3).crossfadeout(0.3) if hud.duration and hud.duration > 0.6 else hud
            hud_clips.append(hud)

    # Overlay cues on top of the B-roll.
    base_y = int(height * y_ratio)
    end_card_top = height - 520
    caption_clips = []
    anim_t = float(profile.get("anim_sec", 0.35))

    def _safe_y(band_h: int, cstart: float, cend: float) -> Tuple[int, int]:
        cap_y = base_y
        ec_dur = min(2.8, total * 0.25) if (end_card and total > 3.0) else 0.0
        ec_start = max(0.0, total - ec_dur)
        if cend > ec_start and ec_dur > 0:
            max_bottom = end_card_top - 24
            cap_y = min(base_y, max(0, max_bottom - band_h // 2))
        scrim_y = max(0, min(int(cap_y - band_h // 2), height - band_h))
        return int(cap_y), scrim_y

    for c in cues:
        start = float(c.get("start", 0.0))
        end = float(c.get("end", start + 3.0))
        if end <= start:
            end = start + 3.0
        if start >= total:
            continue
        end = min(end, total)
        text = (c.get("text") or "").strip()
        if not text:
            continue
        kind = c.get("kind", "line")

        do_karaoke = bool(profile.get("karaoke", True)) and len(text.split()) > 1
        if do_karaoke:
            wtimings = _cue_word_timings(text, start, end, c.get("words"))
            for (w, ws, we) in wtimings:
                kstart = max(start, ws)
                kend = min(end, we)
                if kend <= kstart:
                    continue
                kprof = _normalize_profile_from_json(profile)
                kprof = dict(kprof)
                kstyle = _cue_kind_style(kind)
                kprof["font_size"] = int(kprof.get("font_size", 64) * kstyle["font_scale"])
                kimg = _draw_sentence_highlight(text, w, width, kprof, None, empty_smap).convert("RGBA")
                if kstyle.get("tag"):
                    tag_font = _load_font(kprof["font_path"], int(kprof.get("font_size", 64) * 0.5))
                    tag_img = _render_tag(kstyle["tag"], tag_font, kstyle["accent"])
                    framed = Image.new("RGBA", (width, kimg.height + tag_img.height + 8), (0, 0, 0, 0))
                    framed.paste(tag_img, ((width - tag_img.width) // 2, 0), tag_img)
                    framed.paste(kimg, (0, tag_img.height + 8), kimg)
                    kimg = framed
                band_h = int(kimg.height + 120)
                cap_y, scrim_y = _safe_y(band_h, kstart, kend)
                scrim = ImageClip(np.array(_draw_scrim(width, band_h))).set_start(kstart).set_duration(kend - kstart)
                scrim = scrim.set_position(("center", scrim_y))
                kc = ImageClip(np.array(kimg)).set_start(kstart).set_duration(kend - kstart)
                if anim_t > 0 and kend - kstart > anim_t:
                    kc = kc.fadein(anim_t)
                kc = kc.set_position(("center", cap_y))
                caption_clips.append(scrim)
                caption_clips.append(kc)
            continue

        img = _build_caption_image(text, width, profile, kind)
        band_h = int(img.height + 120)
        cap_y, scrim_y = _safe_y(band_h, start, end)
        scrim = ImageClip(np.array(_draw_scrim(width, band_h))).set_start(start).set_duration(end - start)
        scrim = scrim.set_position(("center", scrim_y))

        ic = ImageClip(np.array(img)).set_start(start).set_duration(end - start)

        if anim_t > 0 and end - start > anim_t:
            ic = ic.fadein(anim_t)
            rise = int(_normalize_profile_from_json(profile).get("line_height_px", 90) * 0.6)

            def _pos(t, base_y=cap_y, rise=rise):  # noqa: B008
                if t <= 0:
                    return ("center", base_y + rise)
                frac = min(1.0, t / anim_t)
                return ("center", int(base_y + rise * (1.0 - frac)))

            ic = ic.set_position(_pos)
        else:
            ic = ic.set_position(("center", cap_y))

        caption_clips.append(scrim)
        caption_clips.append(ic)

    comp = CompositeVideoClip([broll] + hud_clips + caption_clips)
    if end_card and total > 3.0:
        ec_dur = min(2.8, total * 0.25)
        ec_start = max(0.0, total - ec_dur)
        tag = hashtag or ""
        if not tag:
            for c in cues:
                if c.get("kind") == "hook" and c.get("text"):
                    slug = re.sub(r"[^a-z0-9]+", " ", str(c["text"]).lower()).strip()
                    tag = slug.replace(" ", "")[:24]
                    break
        ec_img = _draw_end_card(width, height, tag)
        ec = ImageClip(np.array(ec_img)).set_start(ec_start).set_duration(total - ec_start)
        ec = ec.fadein(min(0.5, ec_dur * 0.4))
        comp = CompositeVideoClip([broll] + hud_clips + caption_clips + [ec])

    narration_reader = None
    if narration_audio and Path(narration_audio).exists():
        try:
            from moviepy.editor import AudioFileClip
            a = AudioFileClip(str(narration_audio))
            narration_reader = a
            if float(a.duration or 0.0) > total:
                a = a.subclip(0.0, total)
            elif total > float(a.duration or 0.0):
                from moviepy.audio.fx.all import audio_loop
                n = max(1, int(total // max(float(a.duration or 1.0), 0.1)))
                a = a.fx(audio_loop, nloops=n).subclip(0.0, total)
            comp = comp.set_audio(a)
        except Exception as e:  # pragma: no cover - environment dependent
            logging.warning("Narration audio unavailable, rendering silent: %s", e)

    has_audio = bool(getattr(comp, "audio", None))
    try:
        _write_videofile_safely(
            comp, out_path, fps=float(broll.fps or 30.0),
            codec="libx264",
            audio_codec="aac" if has_audio else None,
            bitrate=bitrate,
        )
    except Exception as e:  # pragma: no cover - depends on ffmpeg/moviepy
        logging.warning("Video+audio write failed (%s); retrying silent: %s", type(e).__name__, e)
        try:
            silent = comp.without_audio() if has_audio else comp
            _write_videofile_safely(
                silent, out_path, fps=float(broll.fps or 30.0),
                codec="libx264", audio_codec=None, bitrate=bitrate,
            )
        except Exception as e2:
            logging.error("Silent fallback write also failed: %s", e2)
            raise
    finally:
        try:
            comp.close()
        except Exception:
            pass
        if narration_reader is not None:
            try:
                narration_reader.close()
            except Exception:
                pass
        for sc in src_clips:
            try:
                sc.close()
            except Exception:
                pass
    logging.info(f"Wrote scripted short: {out_path}")
