"""Path-hygiene guards for active entrypoints migrated off historical host paths."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

MIGRATED_FILES = (
    "process_usb_device.py",
    "transcribe_usb_batch.py",
    "scripts/get_embedding_endpoint.py",
    "scripts/run_narrated_mix.py",
    "finish_location_backfill.sh",
    "install-signal-bridge-cron.sh",
    "auto_ingest/events.py",
)

STALE_LITERALS = (
    "/media/scott/data/git",
    "/media/scott/SSD_4TB/hermes-home",
    "/media/scott/USB DISK",
    "/media/scott/NAS5/fileserver",
    "bolt://100.64.43.123:7687",
    "bolt://localhost:7687",
)


def _text(rel: str) -> str:
    return (REPO / rel).read_text()


def test_migrated_entrypoints_have_no_historical_checkout_or_storage_literals():
    offenders = []
    for rel in MIGRATED_FILES:
        text = _text(rel)
        for literal in STALE_LITERALS:
            if literal in text:
                offenders.append(f"{rel}: {literal}")
    assert not offenders, "; ".join(offenders)


def test_root_level_shell_helpers_resolve_their_actual_checkout():
    for rel in ("finish_location_backfill.sh", "install-signal-bridge-cron.sh"):
        text = _text(rel)
        assert 'dirname "${BASH_SOURCE[0]}"' in text
        assert 'dirname "${BASH_SOURCE[0]}")/..' not in text


def test_usb_processor_uses_config_resolvers_and_repo_relative_ingest():
    text = _text("process_usb_device.py")
    for symbol in (
        "get_audio_root",
        "get_dashcam_root",
        "get_fileserver_root",
        "get_neo4j_config",
    ):
        assert symbol in text
    assert 'Path(__file__).resolve().parent / "ingest_transcriptsv5_3.py"' in text
    assert 'parser.error("--usb-path is required (or set USB_PATH)")' in text


def test_usb_batch_requires_explicit_source_correction_and_local_audio_resolver():
    text = _text("transcribe_usb_batch.py")
    assert "USB_CORRECTION_FILE" in text
    assert "USB_PROGRESS_FILE" in text
    assert "from auto_ingest_config import get_audio_root" in text
    assert "--correction-file is required" in text
    assert "Path(__file__).resolve().parent" in text


def test_embedding_registry_is_checkout_relative():
    text = _text("scripts/get_embedding_endpoint.py")
    assert 'Path(__file__).resolve().parent.parent / "embedding_endpoints.json"' in text


def test_narrated_mix_uses_shared_namespace_and_canonical_graph_resolver():
    text = _text("scripts/run_narrated_mix.py")
    assert "get_shared_path" in text
    assert "get_neo4j_config" in text
    for component in ("dashcam_timelapse", "_mix_run", "narrated"):
        assert f'"{component}"' in text
    assert "GraphDatabase.driver(" in text
    assert 'cfg["uri"]' in text


def test_event_contract_names_package_not_a_checkout_path():
    text = _text("auto_ingest/events.py")
    assert "assistx.contracts.event_envelope" in text
    assert "home_scott_git_auto-assist" not in text
