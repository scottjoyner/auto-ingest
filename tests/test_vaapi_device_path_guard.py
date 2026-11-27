"""Hermetic guard tests for the explicit VAAPI render-node override.

An operator-set ``AUTO_INGEST_VAAPI_DEVICE`` is a *device path*, not just any
path that happens to exist. Accepting a regular file or directory here makes
``enable_vaapi()`` set VAAPI acceleration on an unusable node, so decode can
fail or silently fall back after the caller believed hardware decode was on.
No OpenCV, GPU, or render node is needed for these tests.
"""

from __future__ import annotations

from auto_ingest.cv2_vaapi import vaapi_device


def test_regular_file_override_is_rejected(monkeypatch, tmp_path):
    bogus = tmp_path / "renderD128"
    bogus.write_text("not a render node", encoding="utf-8")
    monkeypatch.setenv("AUTO_INGEST_VAAPI_DEVICE", str(bogus))
    assert vaapi_device() is None


def test_directory_override_is_rejected(monkeypatch, tmp_path):
    bogus = tmp_path / "dri"
    bogus.mkdir()
    monkeypatch.setenv("AUTO_INGEST_VAAPI_DEVICE", str(bogus))
    assert vaapi_device() is None


def test_missing_override_is_still_rejected(monkeypatch, tmp_path):
    missing = tmp_path / "renderD999"
    monkeypatch.setenv("AUTO_INGEST_VAAPI_DEVICE", str(missing))
    assert vaapi_device() is None
