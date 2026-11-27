"""An invalid LOCAL_TZ must fail closed, not silently dejoin recordings.

`auto_ingest.ingest.discovery` is the pure naming contract: `canonicalize_key`
promises a UTC ``YYYY_MMDD_HHMMSS`` key ("UTC, and always") so two cameras
covering one moment agree. The local stamp is interpreted in ``LOCAL_TZ`` and
converted to UTC by `_to_localized()`.

Before the fix, `LOCAL_TZ` was read from the environment and passed straight to
``ZoneInfo(LOCAL_TZ)``. An invalid value raised inside every datetime pattern,
but each pattern is wrapped in a broad ``except Exception: pass``, so all of
them silently failed and `canonicalize_key` fell back to the raw local stamp.
That is a fail-open on configuration: a typo turned a UTC join key into a
camera-local string and dejoined the two cameras, with rc=0 and no warning.

These tests are hermetic: they launch a fresh interpreter with only a tiny
environment and call the pure contract. No ML stack, Neo4j, network, or live
state is involved. The subprocess boundary is deliberate -- LOCAL_TZ is process
configuration, so the assertion must exercise it at process start.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_PROGRAM = (
    "import auto_ingest.ingest.discovery as d\n"
    "print(d.canonicalize_key('2025_0202_171732', "
    "'x/2025_0202_171732_medium_transcription.txt'))\n"
)


def _import_and_canonicalize(local_tz: str) -> subprocess.CompletedProcess:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": str(ROOT),
        "PYTHONDONTWRITEBYTECODE": "1",
        "LOCAL_TZ": local_tz,
    }
    return subprocess.run(
        [sys.executable, "-c", _PROGRAM],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=30,
    )


@pytest.mark.parametrize("bad_tz", ["Mars/Olympus", "Not/AZone", ""])
def test_invalid_local_tz_fails_closed(bad_tz: str) -> None:
    """A misconfigured timezone must be an error, not a silent key downgrade."""
    result = _import_and_canonicalize(bad_tz)
    assert result.returncode != 0, (
        f"LOCAL_TZ={bad_tz!r} silently produced {result.stdout.strip()!r}"
    )
    assert "LOCAL_TZ" in result.stderr, result.stderr[-2000:]
    if bad_tz:
        assert bad_tz in result.stderr, result.stderr[-2000:]


def test_valid_local_tz_still_yields_utc_key() -> None:
    """Positive control: the ordinary contract is unchanged."""
    result = _import_and_canonicalize("America/New_York")
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip() == "2025_0202_221732"
