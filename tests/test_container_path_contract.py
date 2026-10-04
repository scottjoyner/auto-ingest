"""Executable contract for container-visible shared storage paths."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

CONTAINER_VISIBLE_VARS = (
    "SCAN_ROOTS",
    "DASHCAM_ROOT",
    "AUDIO_ROOT",
    "BODYCAM_ROOT",
    "TRANSCRIPT_ROOT",
    "LEGACY_DROP_ROOT",
    "YOLO_ROOT",
    "META_ROOT",
    "DROP_ROOT",
)


def _env_values(filename="deploy/path_profiles.env.example"):
    values = {}
    for raw in (REPO / filename).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def test_container_visible_paths_use_only_the_nas_namespace():
    values = _env_values()
    assert values["NAS_ROOT"] == "/nas"

    for key in CONTAINER_VISIBLE_VARS:
        value = values[key]
        paths = value.split(",") if key == "SCAN_ROOTS" else [value]
        assert paths
        assert all(path.startswith("/nas") for path in paths), (key, value)
        assert all("/nas/fileserver/" not in path for path in paths), (key, value)
        assert all("/nas/S/" not in path for path in paths), (key, value)


def test_env_examples_agree_on_shared_namespace():
    primary = _env_values("deploy/path_profiles.env.example")
    example = _env_values(".env.example")

    for key in ("NAS_ROOT",) + CONTAINER_VISIBLE_VARS:
        assert example[key] == primary[key], key


def test_compose_mounts_shared_root_at_nas():
    compose = (REPO / "docker-compose.yml").read_text()
    assert "${NAS_ROOT}:/nas" in compose


def test_system_design_uses_the_same_container_namespace():
    design = (REPO / "docs/system_design.md").read_text()
    assert "/nas/fileserver/" not in design
    assert "/nas/S/" not in design

    for path in (
        "/nas/audio",
        "/nas/dashcam",
        "/nas/bodycam",
        "/nas/incoming/deathstar",
    ):
        assert path in design
