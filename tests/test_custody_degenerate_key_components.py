"""Degenerate custody-key components must be reported, not silently normalised.

``filename_problem`` is the detector that stops a key the destination cannot
keep distinct from becoming a release-eligible custody proof. A key with an
empty path component (``a//b.txt``) is not representable as a distinct name:
``Path`` collapses it onto ``a/b.txt`` when the destination path is built, so
two different source objects would be verified against the same file. The
existing checks cover over-long components and trailing dots/spaces, but an
empty component passes all of them.

Hermetic: pure functions, in-memory keys only. No disk, DB, network, mounts,
credentials or services.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from auto_ingest.custody.hashing import (
    UNREPRESENTABLE_NAME,
    detect_name_problems,
    filename_problem,
)

EMPTY_COMPONENT_KEYS = [
    "a//b.txt",      # empty component in the middle
    "/a/b.txt",      # leading empty component
    "a/b.txt/",      # trailing empty component
    "//a//b.txt",    # several
]


@pytest.mark.parametrize("key", EMPTY_COMPONENT_KEYS)
def test_empty_path_component_is_unrepresentable(key):
    """fail -> pass: a collapsed path component cannot round-trip."""
    problem = filename_problem(key)
    assert problem is not None, f"{key!r} silently passed the name detector"
    assert "empty" in problem


def test_detect_name_problems_flags_only_the_degenerate_key():
    """fail -> pass: the producer records the bad name, not the good one."""
    keys = ["a/b.txt", "a//b.txt"]
    flagged = [p.key for p in detect_name_problems(keys)
               if p.kind == UNREPRESENTABLE_NAME]
    assert flagged == ["a//b.txt"]


def test_degenerate_key_aliases_onto_the_canonical_destination_path():
    """Control (passes before and after): why the detector must fail closed."""
    assert Path("/dest") / "a//b.txt" == Path("/dest") / "a/b.txt"


def test_ordinary_names_are_still_not_flagged():
    """Control (passes before and after): the fix must not cry wolf."""
    for good in ["clip_0001.mp4", "DCIM/Movie/2026_0412_100205_F.mp4",
                 "no-extension", "..leading-dots.mp4", "two.dots.in.a.name.mp4"]:
        assert filename_problem(good) is None, good
