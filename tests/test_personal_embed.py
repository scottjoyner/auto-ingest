"""Unit tests for personal memory embed helpers (no model / no DB)."""
from __future__ import annotations

from pathlib import Path

import auto_ingest.personal.embed as embed


def test_clip_dim_is_512():
    assert embed.CLIP_DIM == 512


def test_embed_image_returns_none_without_model(monkeypatch):
    monkeypatch.setattr(embed, "_CLIP", False)
    assert embed.embed_image(Path("/nonexistent")) is None


def test_embed_text_returns_none_without_model(monkeypatch):
    monkeypatch.setattr(embed, "_CLIP", False)
    assert embed.embed_text("hello") is None


def test_embed_text_signature(monkeypatch):
    monkeypatch.setattr(embed, "_CLIP", False)
    assert embed.embed_text("x") is None


def test_haversine_known_distance():
    from auto_ingest.personal.link_media import haversine

    # 1 degree of latitude ~ 111 km.
    d_lat = haversine(0.0, 0.0, 1.0, 0.0)
    assert abs(d_lat - 111320.0) < 2000

    # NYC -> Boston ~ 306 km (within 10%).
    nyc = (40.7128, -74.0060)
    bos = (42.3601, -71.0589)
    d = haversine(*nyc, *bos)
    assert 270_000 < d < 340_000
