"""Smoke tests for worker pipeline wiring.

Dependency-light checks run in the normal unit lane. Tests that import modules
requiring torch/model runtimes are explicitly marked ``ml`` and belong in the
separate ML compatibility lane.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


class TestConfigResolution:
    """auto_ingest_config is the single source of truth."""

    def test_get_neo4j_password_returns_str(self):
        from auto_ingest_config import get_neo4j_password

        pw = get_neo4j_password()
        assert isinstance(pw, str)
        assert len(pw) > 0

    def test_get_neo4j_config_returns_dict(self):
        from auto_ingest_config import get_neo4j_config

        cfg = get_neo4j_config()
        assert isinstance(cfg, dict)
        assert {"uri", "user", "password"} <= set(cfg)

    def test_get_neo4j_env_returns_tuple(self):
        from auto_ingest_config import get_neo4j_env

        uri, user, pw, db = get_neo4j_env()
        assert all(isinstance(v, str) for v in (uri, user, pw, db))

    def test_no_tailscale_ip_in_generic_config(self):
        from auto_ingest_config import get_neo4j_config

        cfg = get_neo4j_config()
        assert not cfg["uri"].startswith("bolt://100."), (
            "generic/unmatched hosts must not silently inherit a fleet Tailscale IP"
        )


class TestSpeakerLinking:
    """bin/auto-ingest link-speakers delegates to the package module."""

    @pytest.mark.ml
    def test_link_global_speakers_module_exists(self):
        mod = importlib.import_module("auto_ingest.diarize.link_global_speakers")
        assert hasattr(mod, "main") or hasattr(mod, "run_linking")

    def test_cli_link_speakers_help(self):
        r = subprocess.run(
            [sys.executable, "bin/auto-ingest", "link-speakers", "--help"],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert r.returncode == 0, r.stderr


class TestContentGeneration:
    def test_worker_content_importable(self):
        import worker_content  # noqa: F401

    def test_tiktok_shorts_importable(self):
        import tiktok_shorts  # noqa: F401


class TestIngestMedia:
    def test_ingest_media_importable(self):
        import ingest_media  # noqa: F401

    def test_embed_image_is_callable(self):
        import ingest_media

        assert callable(ingest_media.embed_image)


class TestPersonalRecall:
    def test_embed_module(self):
        from auto_ingest.personal.embed import embed_image, embed_text

        assert callable(embed_image)
        assert callable(embed_text)

    def test_link_media_module(self):
        from auto_ingest.personal.link_media import link_all

        assert callable(link_all)

    def test_recall_module(self):
        from auto_ingest.personal.recall import recall_media

        assert callable(recall_media)

    def test_recall_in_all(self):
        import auto_ingest.personal as personal

        assert "recall" in personal.__all__


class TestCliSubcommands:
    @pytest.mark.parametrize(
        "cmd",
        [
            ("link-concepts", ["--help"]),
            ("worker", ["--help"]),
            ("recall", ["--help"]),
        ],
    )
    def test_subcommand_help(self, cmd: tuple):
        name, args = cmd
        r = subprocess.run(
            [sys.executable, "bin/auto-ingest", name, *args],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert r.returncode == 0, f"{name} {' '.join(args)} failed: {r.stderr}"


@pytest.mark.ml
class TestTranscriptShim:
    """Transcript module imports require the ML runtime and are tested there."""

    def test_shim_importable(self):
        import ingest_transcriptsv5_3  # noqa: F401

    def test_transcripts_module_main(self):
        from auto_ingest.ingest.transcripts import main

        assert callable(main)


class TestDeployScripts:
    @pytest.mark.parametrize(
        "script",
        [
            "run_ingest_all.sh",
            "run_worker.sh",
            "deploy/worker_ingest.sh",
            "deploy/manage.sh",
            "deploy/create_job.sh",
        ],
    )
    def test_script_exists(self, script: str):
        p = REPO / script
        assert p.exists(), f"Missing: {script}"

    @pytest.mark.parametrize(
        "script",
        [
            "run_ingest_all.sh",
            "run_worker.sh",
            "deploy/worker_ingest.sh",
            "deploy/manage.sh",
            "deploy/create_job.sh",
        ],
    )
    def test_script_executable(self, script: str):
        p = REPO / script
        assert os.access(p, os.X_OK), f"Not executable: {script}"
