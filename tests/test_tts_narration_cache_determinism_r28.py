"""Narration cache filenames must be deterministic across processes (r28).

``narrate()`` caches a synthesized WAV under a name derived from the short's
title.  Python's built-in ``hash()`` on ``str`` is salted per process via
``PYTHONHASHSEED``, so a title-derived name built from it changes on every
restart: the cached narration is never found again and the same script is
synthesized (voice-clone inference) over and over.

These tests are hermetic: no TTS model, no Neo4j, no audio files, no network.
They stub the reference/clip discovery and synthesis, and only observe the
cache filename that ``narrate()`` chooses.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TITLE = "Deterministic narration title r28"

# Stub every external dependency; narrate() only needs a path back.
_SNIPPET = """
import os, sys
from pathlib import Path
sys.path.insert(0, os.environ["R28_REPO_ROOT"])
import auto_ingest.shorts.tts as tts
cache = Path(os.environ["R28_CACHE"]); cache.mkdir(parents=True, exist_ok=True)
tts._cache_dir = lambda: cache
tts.extract_voice_reference = lambda: cache / "ref.wav"
tts.synthesize = lambda text, ref, out: Path(out)
result = tts.narrate(os.environ["R28_TITLE"], [{"text": "hello world"}])
print(Path(result).name)
"""


def _cache_name(tmp_path: Path, seed: str, title: str = TITLE) -> str:
    """Ask a fresh interpreter with a fixed hash seed for the cache basename."""
    cache = tmp_path / f"cache-seed-{seed}"
    env = {
        **os.environ,
        "PYTHONHASHSEED": seed,
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": os.environ.get("TMPDIR", str(tmp_path)),
        "R28_REPO_ROOT": str(REPO_ROOT),
        "R28_CACHE": str(cache),
        "R28_TITLE": title,
    }
    proc = subprocess.run(
        [sys.executable, "-c", _SNIPPET],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"probe interpreter failed: {proc.stderr}"
    return proc.stdout.strip()


def test_narration_cache_name_is_stable_across_hash_seeds(tmp_path: Path) -> None:
    """The same title must map to the same cache file in every process."""
    seed_one = _cache_name(tmp_path, "1")
    seed_two = _cache_name(tmp_path, "2")
    assert seed_one, "narrate() returned no cache filename"
    assert seed_one == seed_two, (
        "narration cache name is salted by PYTHONHASHSEED "
        f"(seed=1 -> {seed_one!r}, seed=2 -> {seed_two!r})"
    )


def test_narration_cache_names_distinguish_titles(tmp_path: Path) -> None:
    """A stable name must still separate distinct scripts."""
    first = _cache_name(tmp_path, "7", TITLE)
    second = _cache_name(tmp_path, "7", TITLE + " (part 2)")
    assert first != second, "distinct titles collapsed to one cache filename"
