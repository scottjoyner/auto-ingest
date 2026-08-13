"""Canonical transcript-ingest process entrypoint.

This replaces the historical shell orchestration in run_ingest_all.sh. It
resolves shared roots and Neo4j credentials through auto_ingest_config, exposes
batch/meta tuning via environment variables, and runs the package module in the
foreground so the fenced orchestrator owns supervision/logging/retry policy.
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Sequence


def _env_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def build_command() -> tuple[list[str], dict[str, str]]:
    from auto_ingest_config import (
        get_audio_root,
        get_dashcam_root,
        get_fileserver_root,
        get_neo4j_config,
    )

    env = dict(os.environ)
    fs = os.environ.get("FILESERVER_ROOT") or str(get_fileserver_root())
    dashcam = os.environ.get("DASHCAM_ROOT") or str(get_dashcam_root())
    audio = os.environ.get("AUDIO_ROOT") or str(get_audio_root())
    cfg = get_neo4j_config()

    env["FILESERVER_ROOT"] = fs
    env["DASHCAM_ROOT"] = dashcam
    env["AUDIO_ROOT"] = audio
    env["NEO4J_URI"] = os.environ.get("NEO4J_URI") or cfg["uri"]
    env["NEO4J_USER"] = os.environ.get("NEO4J_USER") or cfg["user"]
    env["NEO4J_PASSWORD"] = os.environ.get("NEO4J_PASSWORD") or cfg["password"]
    env["NEO4J_DB"] = os.environ.get("NEO4J_DB", "neo4j")
    env["LOCAL_TZ"] = os.environ.get("LOCAL_TZ", "America/New_York")
    env["SCAN_ROOTS"] = os.environ.get(
        "SCAN_ROOTS",
        ",".join(
            [
                dashcam,
                f"{dashcam}/audio",
                f"{dashcam}/transcriptions",
                f"{dashcam}/metadata",
                f"{dashcam}/yolo",
                audio,
                f"{audio}/transcriptions",
                f"{fs}/bodycam",
                f"{fs}/headcam",
            ]
        ),
    )
    env["EMBED_MODEL_NAME"] = os.environ.get(
        "EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )
    env["EMBED_DIM"] = os.environ.get("EMBED_DIM", "384")
    env["EMBED_BATCH"] = os.environ.get("EMBED_BATCH", "32")
    env["MODEL_PREF"] = os.environ.get(
        "MODEL_PREF",
        "large-v3,large-v2,large,turbo,medium.en,medium,small.en,small,base.en,base,"
        "tiny.en,tiny,faster-whisper:large-v3,faster-whisper:large-v2,"
        "faster-whisper:large,faster-whisper:medium,faster-whisper:small,"
        "faster-whisper:base,faster-whisper:tiny",
    )

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "auto_ingest.ingest.transcripts",
        "--tx-seg-batch",
        str(_env_int("TX_SEG_BATCH", 120)),
        "--tx-utt-batch",
        str(_env_int("TX_UTT_BATCH", 120)),
        "--tx-edge-batch",
        str(_env_int("TX_EDGE_BATCH", 300)),
        "--tx-ent-batch",
        str(_env_int("TX_ENT_BATCH", 300)),
        "--tx-loc-batch",
        str(_env_int("TX_LOC_BATCH", 500)),
        "--tx-timeout-sec",
        str(_env_int("TX_TIMEOUT_SEC", 120)),
        "--fetch-size",
        str(_env_int("FETCH_SIZE", 100)),
        "--ingest-dashcam-meta",
        "--lon-auto-west",
        "--allow-latlon-swap",
        "--geo-bbox",
        os.environ.get("GEO_BBOX", "33.0,38.5,-83.0,-70.0"),
        "--meta-fps",
        os.environ.get("META_FPS", "30"),
        "--meta-downsample-sec",
        os.environ.get("META_DOWNSAMPLE_SEC", "1"),
        "--meta-max-speed-mph",
        os.environ.get("META_MAX_SPEED_MPH", "120"),
        "--meta-min-keep-ratio",
        os.environ.get("META_MIN_KEEP_RATIO", "0.6"),
        "--meta-skip-when-bad",
        "--log-level",
        os.environ.get("LOG_LEVEL", "INFO"),
    ]
    limit = _env_int("LIMIT", 0)
    if limit:
        cmd += ["--limit", str(limit)]
    if os.environ.get("DRY_RUN", "0") == "1":
        cmd.append("--dry-run")
    if os.environ.get("FORCE", "0") == "1":
        cmd.append("--force")
    return cmd, env


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        raise ValueError("transcript_entrypoint takes configuration from environment")
    cmd, env = build_command()
    ionice = os.environ.get("IONICE", "ionice -c2 -n7").split()
    nice = os.environ.get("NICE", "nice -n 10").split()
    prefix = []
    if ionice and __import__("shutil").which(ionice[0]):
        prefix += ionice
    if nice and __import__("shutil").which(nice[0]):
        prefix += nice
    return subprocess.call(prefix + cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
