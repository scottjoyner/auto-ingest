"""Capacity headroom must be valid, not silently reduced to zero.

`headroom_for` used `max(int(outstanding*fraction), minimum)`, so a negative
fraction produced zero margin when the floor was also zero; `check_capacity`
clamped a negative `headroom_bytes` to zero. A malformed policy/CLI value could
therefore disable the margin that exists to stop a copy dying at 99% while the
capacity gate still reported sufficient.
"""
from __future__ import annotations

import pytest

from auto_ingest.custody.capacity import check_capacity, headroom_for


@pytest.mark.parametrize("fraction", [-0.1, -1, "x", None])
def test_invalid_headroom_fraction_is_refused(fraction):
    with pytest.raises(ValueError, match="headroom_fraction"):
        headroom_for(1000, headroom_fraction=fraction)


@pytest.mark.parametrize("minimum", [-1, -1024, "x", None])
def test_invalid_headroom_minimum_is_refused(minimum):
    with pytest.raises(ValueError, match="headroom_min_bytes"):
        headroom_for(1000, headroom_min_bytes=minimum)


def test_negative_explicit_headroom_is_refused(tmp_path):
    with pytest.raises(ValueError, match="headroom_bytes"):
        check_capacity(str(tmp_path), 10, headroom_bytes=-5)


def test_valid_headroom_still_works(tmp_path):
    assert headroom_for(1000, 0.05, 10) == 50
    assert headroom_for(0, 0.05, 10) == 10
    report = check_capacity(str(tmp_path), 10, headroom_bytes=5)
    assert report.headroom_bytes == 5
    assert report.required_bytes == 15
