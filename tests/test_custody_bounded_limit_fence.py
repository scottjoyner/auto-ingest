"""Bounded evidence lists require a positive integer cap.

`_bounded` accepted a zero/negative/non-integer limit, silently truncating every
diagnostic summary to one entry (or behaving oddly) while the policy looked
valid.
"""
from __future__ import annotations

import pytest

from auto_ingest.custody.evidence import _bounded


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "2"])
def test_invalid_limit_is_refused(limit):
    with pytest.raises(ValueError, match="limit"):
        _bounded(["a", "b"], limit)


def test_valid_limit_still_bounds_the_list():
    assert _bounded(["a", "b"], 1) == ("a",)
    assert _bounded(["a", "b"], 20) == ("a", "b")
