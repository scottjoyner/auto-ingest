"""Retention limits are config-supplied and deletion is irreversible.

birdcam's ``apply_retention`` unlinks clip files permanently. Its limits come
from YAML (``birdcam/config.py::RetentionConfig``), which happily yields ``0``,
negative numbers, booleans and ``.inf``/``.nan``. Before the fix a zero or
negative cap made the size loop delete *every* clip, and a negative/zero age
made the age loop do the same -- a config typo silently wiped the archive.
Non-finite caps crashed mid-pass instead of refusing cleanly.

Every test here is hermetic: a pytest ``tmp_path`` clip tree, no cv2, no Neo4j,
no network mount, no clock mocking. The existing ``test_birdcam.py`` is skipped
in CI because cv2 is absent, so these run independently of it.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from birdcam.retention import apply_retention


@pytest.fixture()
def clip(tmp_path: Path) -> Path:
    directory = tmp_path / "clips" / "2026" / "01" / "01"
    directory.mkdir(parents=True)
    path = directory / "old.mp4"
    path.write_bytes(b"a" * 100)
    return path


def test_zero_gb_refuses_without_deleting(clip: Path) -> None:
    with pytest.raises(ValueError, match="max_gb"):
        apply_retention(str(clip.parents[4]), max_gb=0, max_age_days=999)
    assert clip.exists()


def test_negative_gb_refuses_without_deleting(clip: Path) -> None:
    with pytest.raises(ValueError, match="max_gb"):
        apply_retention(str(clip.parents[4]), max_gb=-1, max_age_days=999)
    assert clip.exists()


def test_infinite_gb_refuses_without_deleting(clip: Path) -> None:
    with pytest.raises(ValueError, match="max_gb"):
        apply_retention(str(clip.parents[4]), max_gb=math.inf, max_age_days=999)
    assert clip.exists()


def test_nan_gb_refuses_without_deleting(clip: Path) -> None:
    with pytest.raises(ValueError, match="max_gb"):
        apply_retention(str(clip.parents[4]), max_gb=math.nan, max_age_days=999)
    assert clip.exists()


def test_bool_gb_is_not_a_number(clip: Path) -> None:
    # bool is an int subclass; `max_gb: true` must not silently mean 1 GiB.
    with pytest.raises(ValueError, match="max_gb"):
        apply_retention(str(clip.parents[4]), max_gb=True, max_age_days=999)
    assert clip.exists()


def test_negative_age_days_refuses_without_deleting(clip: Path) -> None:
    with pytest.raises(ValueError, match="max_age_days"):
        apply_retention(str(clip.parents[4]), max_gb=20, max_age_days=-1)
    assert clip.exists()


def test_zero_age_days_refuses_without_deleting(clip: Path) -> None:
    with pytest.raises(ValueError, match="max_age_days"):
        apply_retention(str(clip.parents[4]), max_gb=20, max_age_days=0)
    assert clip.exists()


def test_non_numeric_age_days_refuses_without_deleting(clip: Path) -> None:
    with pytest.raises(ValueError, match="max_age_days"):
        apply_retention(str(clip.parents[4]), max_gb=20, max_age_days="thirty")
    assert clip.exists()


def test_huge_finite_gb_never_overflows(clip: Path) -> None:
    # A finite but astronomical cap is a safe "never prune by size"; it must not
    # turn into int(+inf) and crash the pass.
    removed = apply_retention(str(clip.parents[4]), max_gb=1e308, max_age_days=999)
    assert removed == []
    assert clip.exists()


def test_valid_config_leaves_small_archive(clip: Path) -> None:
    # Positive control: the shipped defaults must keep their old behaviour.
    removed = apply_retention(str(clip.parents[4]), max_gb=20, max_age_days=30)
    assert removed == []
    assert clip.exists()


def test_small_cap_still_prunes(clip: Path) -> None:
    # Positive control: a genuine, tiny positive cap must still evict.
    removed = apply_retention(str(clip.parents[4]), max_gb=1e-8, max_age_days=999)
    assert removed
    assert not clip.exists()
