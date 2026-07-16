"""Owner-voice TTS for research shorts.

The graph already holds the owner's voice: ``Speaker {user_id:'scott'}`` is the
owner voiceprint (``is_owner_voiceprint: true``) and it :RECORDED:`Audio`
captures from the mobile app. We use those captures as the reference clip for a
**voice-cloning** TTS model (Coqui XTTS-v2) so shorts are narrated in Scott's
own voice instead of a generic synthetic one.

Everything here is lazy + optional:

* ``extract_voice_reference`` pulls a ~12s clean mono 22.05kHz WAV of the
  owner's voice from the recorded ``Audio`` nodes (resolving their paths
  through the fileserver remap). It is cached under ``<audio_root>/.cache/tts``.
* ``synthesize`` lazily imports ``TTS`` (Coqui XTTS-v2). If the package is not
  installed it raises ``TTSUnavailable`` and the caller renders the short
  silently (the proven default path is untouched).

No fine-tuning is required: XTTS-v2 clones a target voice from a short
reference clip, which is exactly what we have in the DB. Fine-tuning would only
matter if the reference set were tiny/unreliable; with the owner's mobile
captures (often several minutes) the clone is already high-fidelity.
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import List, Optional

log = logging.getLogger("shorts.tts")

OWNER_USER_ID = os.getenv("TTS_OWNER_USER_ID", "scott")
REFERENCE_SECONDS = float(os.getenv("TTS_REF_SECONDS", "12"))
XTTS_SR = 22050  # XTTS-v2 expects 22.05kHz mono


class TTSUnavailable(RuntimeError):
    """Raised when the Coqui TTS stack is not installed."""


def _cache_dir() -> Path:
    try:
        from auto_ingest_config import get_audio_root
        root = get_audio_root()
    except Exception:
        root = os.getcwd()
    d = Path(root) / ".cache" / "tts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _neo4j_creds():
    from auto_ingest_config import get_neo4j_password
    return (
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        os.getenv("NEO4J_USER", "neo4j"),
        get_neo4j_password(),
        os.getenv("NEO4J_DB", "neo4j"),
    )


def _resolve_path(p: str) -> Optional[Path]:
    from auto_ingest_config import get_fileserver_path
    cand = get_fileserver_path(p)
    if cand and Path(cand).exists():
        return Path(cand)
    # Mobile captures are stored as `/captures/<id>.<ext>` in the graph but
    # live under <fileserver_root>/audio/sophia-captures/<id>.<ext>.
    if p.startswith("/captures/"):
        name = Path(p).name
        sc = get_fileserver_path(os.path.join("audio", "sophia-captures", name))
        if sc and Path(sc).exists():
            return Path(sc)
    raw = Path(p)
    if raw.exists():
        return raw
    return None


def owner_audio_captures() -> List[Path]:
    """Return resolved on-disk paths of the owner's recorded voice captures."""
    from neo4j import GraphDatabase
    uri, user, pw, db = _neo4j_creds()
    out: List[Path] = []
    try:
        with GraphDatabase.driver(uri, auth=(user, pw)).session(database=db) as sess:
            rows = sess.run(
                """
                MATCH (sp:Speaker {user_id:$uid})-[:RECORDED]->(a:Audio)
                WHERE a.path IS NOT NULL
                RETURN a.path AS p, coalesce(a.duration_ms, 0) AS dur
                ORDER BY dur DESC
                LIMIT 25
                """,
                uid=OWNER_USER_ID,
            ).data()
        for r in rows:
            p = _resolve_path(r["p"])
            if p:
                out.append(p)
    except Exception as e:  # graph optional for TTS
        log.warning("Could not read owner audio captures: %s", e)
    return out


def owner_local_audio() -> List[Path]:
    """Fallback: the owner's own recordings in the audio archive.

    The graph ``Audio`` captures are sometimes stale/missing on disk, but the
    owner's personal recorder files under the fileserver audio root are
    unambiguously his voice. We target the clearly-personal clips (phone-call
    ``R*.MP3``, handheld ``MOVI*.mp3``, dashcam-mic ``YYYY*.WAV``) and skip the
    large bulk folders, to keep the scan fast and the reference speech-dense.
    """
    from auto_ingest_config import get_audio_root
    root = Path(get_audio_root())
    out: List[Path] = []
    if not root.exists():
        return out

    def _name(p: Path) -> str:
        return p.name

    # Targeted globs (non-recursive at the archive top level) for the owner's
    # personal voice; avoids walking the entire (huge) transcription tree.
    patterns = ("R*.MP3", "R*.mp3", "MOVI*.mp3", "MOVI*.MP3",
                "2023*.WAV", "2024*.WAV", "2025*.WAV", "2026*.WAV",
                "20[0-9][0-9]*.WAV")
    seen = set()
    for pat in patterns:
        for p in root.glob(pat):
            if "sophia-captures" in str(p):
                continue
            if p.stat().st_size < 20_000:
                continue
            if p not in seen:
                seen.add(p)
                out.append(p)
    # Longer clips first (more speech context), but cap the candidate pool so
    # the ffmpeg concat stays quick.
    out.sort(key=_audio_seconds, reverse=True)
    return out[:12]


def extract_voice_reference(force: bool = False) -> Optional[Path]:
    """Build (and cache) a ~12s clean 22.05kHz mono WAV of the owner's voice."""
    cache = _cache_dir() / "owner_voice_reference.wav"
    if cache.exists() and not force:
        return cache
    caps = owner_audio_captures() + owner_local_audio()
    if not caps:
        log.warning("No owner voice captures resolved on disk; TTS reference unavailable")
        return None
    # Prefer the longest clips so we always gather enough clean owner voice
    # even when the graph `Audio` captures are missing/empty on disk.
    caps.sort(key=_audio_seconds, reverse=True)
    targets: List[Path] = []
    acc = 0.0
    for c in caps:
        dur = _audio_seconds(c)
        targets.append(c)
        acc += dur
        if acc >= REFERENCE_SECONDS:
            break
    if not targets:
        return None
    concat_list = cache.with_suffix(".list.txt")
    with concat_list.open("w") as fh:
        for t in targets:
            fh.write(f"file '{t.resolve()}'\n")
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_list), "-vn", "-ac", "1", "-ar", str(XTTS_SR),
        "-t", str(int(REFERENCE_SECONDS) + 1), str(cache),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    except Exception as e:
        log.warning("Voice reference build failed: %s", e)
        return None
    finally:
        try:
            concat_list.unlink()
        except Exception:
            pass
    if not cache.exists():
        return None
    log.info("Built owner voice reference: %s", cache)
    return cache


def _audio_seconds(p: Path) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(p)],
            capture_output=True, text=True, timeout=30,
        )
        return float(out.stdout.strip() or 0.0)
    except Exception:
        return 5.0


def _tts_venv_python() -> Optional[Path]:
    """Locate a Python interpreter that has Coqui TTS (py3.11 venv).

    Coqui TTS only supports Python < 3.12, so on this host it lives in a
    dedicated ``.venv-tts`` (python3.11) rather than the render ``.venv``.
    """
    env = os.environ.get("TTS_VENV")
    if env:
        p = Path(env)
        if p.exists():
            return p
    # Walk up from this file to find a .venv-tts/bin/python.
    here = Path(__file__).resolve()
    for parent in [here.parent, here.parent.parent, here.parent.parent.parent]:
        cand = parent / ".venv-tts" / "bin" / "python"
        if cand.exists():
            return cand
    return None


def _synthesize_via_venv(text: str, ref_wav: Path, out_wav: Path, py: Path) -> Path:
    """Run XTTS synthesis in the dedicated TTS venv (subprocess)."""
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    script = (
        "from TTS.api import TTS;"
        f"m=TTS('tts_models/multilingual/multi-dataset/xtts_v2');"
        f"m.tts_to_file(text={text!r},speaker_wav={str(ref_wav)!r},"
        f"language='en',file_path={str(out_wav)!r})"
    )
    try:
        subprocess.run([str(py), "-c", script], check=True, capture_output=True, timeout=600)
    except subprocess.CalledProcessError as e:
        raise TTSUnavailable(
            f"TTS synthesis failed in venv ({py}): {e.stderr.decode()[:300]}"
        ) from e
    return out_wav


def synthesize(text: str, ref_wav: Path, out_wav: Path) -> Path:
    """Synthesize ``text`` in the owner's voice -> ``out_wav`` via XTTS-v2.

    Uses an in-process ``TTS`` import when available (e.g. a py3.11 render
    env); otherwise shells out to the dedicated ``.venv-tts`` interpreter.
    Raises :class:`TTSUnavailable` if TTS cannot be found anywhere, so
    callers fall back to a silent render.
    """
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    try:
        from TTS.api import TTS  # type: ignore
    except Exception:
        vpy = _tts_venv_python()
        if vpy and vpy.exists():
            _synthesize_via_venv(text, ref_wav, out_wav, vpy)
            log.info("Synthesized %d chars (venv) -> %s", len(text), out_wav)
            return out_wav
        raise TTSUnavailable(
            "Coqui TTS not installed; create .venv-tts (python3.11) and "
            "`pip install TTS==0.22.0`. Falling back to silent render."
        )
    model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    model.tts_to_file(
        text=text,
        speaker_wav=str(ref_wav),
        language="en",
        file_path=str(out_wav),
    )
    log.info("Synthesized %d chars of owner-voice TTS -> %s", len(text), out_wav)
    return out_wav


def narrate(item_title: str, cues: List[dict]) -> Optional[Path]:
    """Build a narration WAV from a short's scripted cues.

    Returns the WAV path, or ``None`` if TTS is unavailable / no reference
    voice can be built. Callers treat ``None`` as 'render silently'.
    """
    ref = extract_voice_reference()
    if not ref:
        return None
    script = " ".join((c.get("text") or "").strip() for c in cues if c.get("text"))
    if not script:
        return None
    out = _cache_dir() / f"narration_{abs(hash(item_title))}.wav"
    if out.exists():
        return out
    try:
        return synthesize(script, ref, out)
    except TTSUnavailable as e:
        log.info("TTS skipped: %s", e)
        return None
