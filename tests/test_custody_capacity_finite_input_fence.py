"""Capacity inputs must be finite and non-negative before any verdict.

`check_capacity` clamped a negative `needed_bytes` to zero, so a malformed
requirement could be reported as `sufficient` on any filesystem. The margin
helpers also accepted `nan`/`inf`, which either crashed in `int()` with an
opaque message or bypassed the intended comparison.
"""
from __future__ import annotations

import math

import pytest

from auto_ingest.custody.capacity import check_capacity, headroom_for


@pytest.mark.parametrize("needed", [-1, -1024, float("nan"), float("inf"), None, "x"])
def test_invalid_requirement_is_refused(tmp_path, needed):
    with pytest.raises(ValueError, match="needed_bytes"):
        check_capacity(str(tmp_path), needed)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_headroom_inputs_are_refused(value):
    with pytest.raises(ValueError, match="finite"):
        headroom_for(1000, headroom_fraction=value)
    with pytest.raises(ValueError, match="finite"):
        headroom_for(1000, headroom_min_bytes=value)


def test_valid_capacity_inputs_still_work(tmp_path):
    report = check_capacity(str(tmp_path), 10, headroom_bytes=5)
    assert report.sufficient is True
    assert report.required_bytes == 15
    assert headroom_for(1000, 0.05, 10) == 50
