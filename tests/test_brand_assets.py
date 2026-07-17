"""Tests for scripts/gen_brand_assets.py — asset generation + --check mode.

Runs under system python3 (only needs Pillow):
    python3 -m pytest tests/test_brand_assets.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SPEC = REPO / "scripts" / "gen_brand_assets.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("gen_brand_assets", SPEC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny_manifest(brand_dir: Path) -> None:
    manifest = {
        "brand_name": "Research in the Fast Lane",
        "handles": {
            "youtube": "ResearchInTheFastLane",
            "tiktok": "@researchfastlane",
            "instagram": "@researchfastlane",
        },
        "assets": {
            "avatar": "avatar.png",
            "banner_youtube": "banner_youtube.png",
            "avatar_instagram": "avatar_instagram.png",
            "avatar_tiktok": "avatar_tiktok.png",
        },
        "palette": {"ink": [11, 14, 20], "cyan": [0, 200, 255]},
    }
    (brand_dir / "brand_manifest.json").write_text(json.dumps(manifest),
                                                   encoding="utf-8")
    (brand_dir / "brand_spec.md").write_text(
        "Handles: ResearchInTheFastLane @researchfastlane @researchfastlane",
        encoding="utf-8")


def test_generate_all_produces_ig_tiktok(tmp_path):
    mod = _load_module()
    brand = tmp_path / "brand"
    brand.mkdir()
    mod.make_avatar(brand / "avatar.png")
    mod.make_banner(brand / "banner_youtube.png")
    mod.make_instagram(brand / "avatar_instagram.png")
    mod.make_tiktok(brand / "avatar_tiktok.png")
    assert (brand / "avatar_instagram.png").exists()
    assert (brand / "avatar_tiktok.png").exists()

    from PIL import Image
    with Image.open(brand / "avatar_instagram.png") as im:
        assert im.size == (640, 640)
    with Image.open(brand / "avatar_tiktok.png") as im:
        assert im.size == (720, 720)


def test_check_ok_on_generated(tmp_path):
    mod = _load_module()
    brand = tmp_path / "brand"
    brand.mkdir()
    _tiny_manifest(brand)
    mod.make_avatar(brand / "avatar.png")
    mod.make_banner(brand / "banner_youtube.png")
    mod.make_instagram(brand / "avatar_instagram.png")
    mod.make_tiktok(brand / "avatar_tiktok.png")
    ok, issues = mod.check_assets(brand)
    assert ok, issues


def test_check_detects_wrong_palette(tmp_path):
    mod = _load_module()
    from PIL import Image
    brand = tmp_path / "brand"
    brand.mkdir()
    _tiny_manifest(brand)
    # All-red images have neither ink nor cyan.
    for name, size in [("avatar.png", (1000, 1000)),
                       ("banner_youtube.png", (2560, 1440)),
                       ("avatar_instagram.png", (640, 640)),
                       ("avatar_tiktok.png", (720, 720))]:
        Image.new("RGB", size, (255, 0, 0)).save(brand / name)
    ok, issues = mod.check_assets(brand)
    assert not ok
    assert any("ink" in i or "cyan" in i for i in issues)


def test_check_detects_handle_mismatch(tmp_path):
    mod = _load_module()
    brand = tmp_path / "brand"
    brand.mkdir()
    _tiny_manifest(brand)
    mod.make_avatar(brand / "avatar.png")
    mod.make_banner(brand / "banner_youtube.png")
    mod.make_instagram(brand / "avatar_instagram.png")
    mod.make_tiktok(brand / "avatar_tiktok.png")
    # Spec that omits the handles.
    (brand / "brand_spec.md").write_text("no handles here", encoding="utf-8")
    ok, issues = mod.check_assets(brand)
    assert not ok
    assert any("not declared in brand_spec.md" in i for i in issues)


def test_real_repo_assets_pass_check():
    """The committed docs/brand assets should pass --check as shipped."""
    mod = _load_module()
    ok, issues = mod.check_assets(REPO / "docs" / "brand")
    assert ok, issues
