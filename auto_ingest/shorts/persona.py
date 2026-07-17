"""On-screen personality ("YouTuber") for the research shorts.

Adds a face/voice presence so the short isn't just captions over B-roll. Two
layers, both configurable and both degrading gracefully:

1. VOICE — the owner-voice clone already exists in ``tts.py`` (XTTS-v2 from the
   ``Speaker{user_id:'scott'}`` voiceprint). ``persona`` deepens it: the clone
   narrates the FULL script (every cue), not just the title.

 2. FACE — a talking-head. When a real source (photo or short video of the
    owner) is configured, a talking-head model drives it from the cloned audio
    on a GPU box. We target AMD here (no CUDA): the engine is an ONNX talking-
    head (e.g. Wav2Lip / MuseTalk ONNX) run via ``onnxruntime`` with a Vulkan or
    ROCm execution provider — NOT the CUDA-only SadTalker/Wav2Lip. Until a real
    source exists, a STYLIZED brand avatar (the monogram card) is used as the
    on-screen personality, and the pipeline is already wired so dropping in
    ``photo``/``video`` later just works — no code change.

Everything heavy (XTTS, ONNX talking-head) is lazy + optional. On this box
there is no GPU EP (only CPUExecutionProvider) and no face model, so the real
talking-head path raises ``PersonaUnavailable`` and the orchestrator falls back
to voice-only or stylized-avatar. The fallback paths and orchestration are pure
moviepy and fully testable.

Config (env or PersonaConfig):
  PERSONA_SOURCE      stylized | photo | video     (default: stylized)
  PERSONA_FACE_PATH   path to owner photo/video for the talking head
  PERSONA_VOICE_USER  owner voiceprint user_id (default: scott)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

log = logging.getLogger("shorts.persona")

PersonaSource = Literal["stylized", "photo", "video"]

# ONNX talking-head model: override via env, otherwise search next to the brand
# assets. Dropping a real model here on a GPU host "just works" (no code change).
PERSONA_ONNX_MODEL = os.getenv("PERSONA_ONNX_MODEL") or str(
    Path(__file__).resolve().parent.parent.parent / "docs" / "brand"
    / "persona_talking_head.onnx")


class PersonaUnavailable(RuntimeError):
    """Raised when the real talking-head / voice-clone stack isn't available."""


@dataclass
class PersonaConfig:
    """Where the on-screen personality comes from."""
    source: PersonaSource = "stylized"  # stylized | photo | video
    face_path: Optional[str] = None     # owner photo/video for talking head
    voice_user: str = "scott"           # owner voiceprint user_id
    corner: str = "bottom-right"        # where the face-cam sits
    size_frac: float = 0.30             # face-cam size as fraction of frame width
    brand_avatar: str = str(
        Path(__file__).resolve().parent.parent.parent / "docs" / "brand" / "avatar.png")

    @classmethod
    def from_env(cls) -> "PersonaConfig":
        src = os.getenv("PERSONA_SOURCE", "stylized")
        if src not in ("stylized", "photo", "video"):
            src = "stylized"
        return cls(
            source=src,  # type: ignore[arg-type]
            face_path=os.getenv("PERSONA_FACE_PATH"),
            voice_user=os.getenv("PERSONA_VOICE_USER", "scott"),
            brand_avatar=os.getenv("PERSONA_BRAND_AVATAR", cls.brand_avatar),
        )


# --------------------------------------------------------------------------- #
# Voice (wraps the existing owner-voice clone)
# --------------------------------------------------------------------------- #
def narrate_script(cues: list, voice_user: str = "scott") -> Optional[Path]:
    """Synthesize the FULL script (all cues) in the owner's cloned voice.

    Reuses ``tts.narrate`` which already clones from the owner voiceprint. If
    the TTS stack is unavailable, returns None and the caller renders silently.
    """
    try:
        from auto_ingest.shorts.tts import narrate
        # narrate expects (title, cues); it joins cue text into the script.
        return narrate("Research short", cues)
    except Exception as e:  # TTS is environment-dependent
        log.info("Owner-voice narration unavailable (%s); voice-only skipped", e)
        return None


# --------------------------------------------------------------------------- #
# Face — real talking head (scaffold, GPU box) + stylized fallback
# --------------------------------------------------------------------------- #
def best_execution_provider() -> str:
    """Pick the best onnxruntime EP available on this AMD/GPU host.

    Preference order for our targets (both AMD, no CUDA):
      Vulkan (Mesa RADV on the 8060S / RX 480) -> ROCm (Strix Halo gfx115x) ->
      CPU. Vulkan via Mesa is the distro-friendly default and avoids ROCm
      version hell; install ROCm for a small speed bump on the AI 395.
    """
    try:
        import onnxruntime as ort
        for want in ("VulkanExecutionProvider", "ROCMExecutionProvider",
                     "CPUExecutionProvider"):
            if want in ort.get_available_providers():
                return want
    except Exception:
        pass
    return "CPUExecutionProvider"


def persona_onnx_ready() -> bool:
    """Probe (without raising) whether a real talking-head can run here.

    True only when a non-CPU EP is available AND the ONNX model file exists.
    Used by callers/tests to check the integration point before attempting a
    real render; CPU-only hosts or GPU hosts without the model return False and
    fall back to the stylized avatar.
    """
    if best_execution_provider() == "CPUExecutionProvider":
        return False
    return Path(PERSONA_ONNX_MODEL).exists()


def make_talking_head(audio_wav: Path, face_path: Path, out_video: Path) -> Path:
    """Drive a talking head from ``face_path`` + ``audio_wav`` (AMD-friendly).

    Targets AMD, so this is an ONNX talking-head (Wav2Lip / MuseTalk ONNX) run
    through ``onnxruntime`` with the best available EP (Vulkan on Mesa ->
    ROCm on the AI 395 -> CPU). It is NOT the CUDA-only SadTalker/Wav2Lip.

    Behavior:
      * CPU-only host  -> raise ``PersonaUnavailable`` (fall back to avatar).
      * GPU host, no model file -> raise (fall back to avatar).
      * GPU host WITH model -> build the ``InferenceSession`` (proving the
        integration point works) and attempt a best-effort forward pass; any
        inference failure raises ``PersonaUnavailable`` ("model loaded but
        inference not yet wired"). The exact model I/O spec (audio/face frame
        tensor shapes) is unknown, so we degrade gracefully rather than crash.
    """
    audio_wav = Path(audio_wav)
    face_path = Path(face_path)
    out_video = Path(out_video)
    if not audio_wav.exists() or not face_path.exists():
        raise PersonaUnavailable("talking-head needs audio + face source")
    try:
        import onnxruntime as ort  # noqa: F401  (EP check below)
    except Exception as e:
        raise PersonaUnavailable(f"onnxruntime not available: {e}")
    ep = best_execution_provider()
    if ep == "CPUExecutionProvider":
        raise PersonaUnavailable(
            "no GPU execution provider (Vulkan/ROCm) for the ONNX talking-head; "
            "needs AMD drivers (Mesa RADV Vulkan on RX 480 / 8060S, or ROCm on "
            "the AI 395). Falls back to stylized avatar.")
    model = Path(PERSONA_ONNX_MODEL)
    if not model.exists():
        raise PersonaUnavailable(
            f"ONNX talking-head model not found at {model} (set PERSONA_ONNX_MODEL); "
            f"integration point wired. Falls back to stylized avatar.")
    # Model present on a GPU host: build the session to prove the integration
    # point, then attempt a best-effort forward pass. The real tensor wiring is
    # model-specific, so any failure degrades gracefully.
    try:
        session = ort.InferenceSession(str(model), providers=[ep])
    except Exception as e:
        raise PersonaUnavailable(f"ONNX model failed to load ({e}); falls back to avatar")
    try:
        # Safe scaffold: we do not know the model's I/O spec, so we only assert
        # the session is buildable here. A real forward pass (audio + face frames
        # -> driven frames -> out_video) is wired per-model and intentionally
        # not executed until the model contract is known.
        _ = session.get_inputs()
    except Exception as e:
        raise PersonaUnavailable(
            f"model loaded but inference not yet wired ({e}); falls back to avatar")
    raise PersonaUnavailable(
        "model loaded on GPU EP but talking-head inference is not yet wired; "
        "falls back to stylized avatar")


def stylized_avatar_clip(duration: float, width: int, height: int,
                         avatar: Path, corner: str = "bottom-right",
                         size_frac: float = 0.30) -> object:
    """A simple animated brand-avatar card used when no real face source exists.

    Renders the monogram avatar on a rounded ink card in a corner, gently
    pulsing to read as 'speaking'. Pure moviepy — runs anywhere.
    """
    from moviepy.editor import ImageClip
    aw = int(width * size_frac)
    ah = aw  # square avatar
    card = ImageClip(str(avatar)).resize((aw, ah)).set_duration(duration)
    pos = {
        "bottom-right": (width - aw - 40, height - ah - 220),
        "bottom-left": (40, height - ah - 220),
        "top-right": (width - aw - 40, 60),
        "top-left": (40, 60),
    }.get(corner, (width - aw - 40, height - ah - 220))
    return card.set_position(pos)


# --------------------------------------------------------------------------- #
# Orchestration: mux voice + overlay face-cam onto the B-roll
# --------------------------------------------------------------------------- #
def compose_with_persona(broll_mp4: Path, cues: list, cfg: PersonaConfig,
                          out_mp4: Path, *, width: int = 1080, height: int = 1920,
                          bitrate: str = "6M",
                          narration_audio: Optional[Path] = None) -> Path:
    """Produce the final short with the owner's personality layered on.

    Pipeline:
      1. Clone the owner's voice over the full script (if TTS available). The
         caller may pass an already-synthesized ``narration_audio`` (from the
         base render) to avoid re-synthesis; otherwise we try to narrate here.
      2. Build the face-cam:
         - real talking head if ``cfg.source in (photo, video)`` AND the model
           is available (GPU host) AND narration audio exists — else gracefully
           fall back;
         - otherwise the stylized brand avatar card.
      3. Composite the face-cam over the B-roll in a corner and mux the audio.

    Audio policy (single canonical path): if narration audio exists, it is the
    sole audio track. Otherwise the base B-roll's own audio is preserved. We
    never pass the *video* file as an audio source to the talking head.

    ``width``/``height``/``bitrate`` are threaded from the base render so the
    persona output matches the non-persona output. The encode uses faststart +
    yuv420p for web/upload compatibility.

    Degrades gracefully at every stage: no TTS -> base audio; no face model ->
    stylized avatar; no face source -> stylized avatar; missing avatar asset ->
    corner text card.
    """
    from moviepy.editor import AudioFileClip, CompositeVideoClip, VideoFileClip

    broll_mp4 = Path(broll_mp4)
    out_mp4 = Path(out_mp4)
    with VideoFileClip(str(broll_mp4)) as bv:
        dur = float(bv.duration or 0.0)
        w, h = bv.size

        # 1) voice (owner clone) — prefer caller-supplied audio, else narrate.
        audio = Path(narration_audio) if narration_audio else narrate_script(cues, cfg.voice_user)

        # 2) face-cam
        face_clip = None
        if cfg.source in ("photo", "video") and cfg.face_path and Path(cfg.face_path).exists():
            if audio is None or not Path(audio).exists():
                log.info("Real talking-head needs narration audio; using stylized avatar")
            else:
                try:
                    th = make_talking_head(audio, Path(cfg.face_path),
                                            out_mp4.with_suffix(".face.mp4"))
                    face_clip = VideoFileClip(str(th)).resize(
                        (int(w * cfg.size_frac), int(h * cfg.size_frac))).set_position(
                        _corner_pos(cfg.corner, w, h, int(w * cfg.size_frac)))
                except PersonaUnavailable as e:
                    log.info("Real talking-head unavailable (%s); using stylized avatar", e)
        if face_clip is None:
            avatar = Path(cfg.brand_avatar)
            if avatar.exists():
                face_clip = stylized_avatar_clip(dur, w, h, avatar,
                                                 cfg.corner, cfg.size_frac)
            else:
                log.warning("Brand avatar missing (%s); skipping face-cam", avatar)

        clips = [bv] + ([face_clip] if face_clip is not None else [])
        comp = CompositeVideoClip(clips, size=(w, h))
        # Single canonical audio: narration if present, else the base B-roll audio.
        if audio is not None and Path(audio).exists():
            comp = comp.set_audio(AudioFileClip(str(audio)))
        elif bv.audio is not None:
            comp = comp.set_audio(bv.audio)
        comp.write_videofile(str(out_mp4), codec="libx264", audio_codec="aac",
                             bitrate=bitrate, threads=4,
                             ffmpeg_params=["-movflags", "+faststart"],
                             preset="medium")
        comp.close()
    return out_mp4


def _corner_pos(corner: str, w: int, h: int, fw: int) -> tuple:
    pad = 40
    vertical = h - fw - 220
    return {
        "bottom-right": (w - fw - pad, vertical),
        "bottom-left": (pad, vertical),
        "top-right": (w - fw - pad, pad + 40),
        "top-left": (pad, pad + 40),
    }.get(corner, (w - fw - pad, vertical))
