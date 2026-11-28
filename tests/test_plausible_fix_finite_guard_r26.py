"""Hermetic regression tests: `plausible_fix()` must reject non-finite GPS data.

`plausible_fix()` is the single admission gate shared by the Frame,
LocationEvent, PhoneLog and metadata.csv branches of
`resolve_location_for_second()` / `choose_location()` in
`auto_ingest/dashcam/yolo_embeddings.py`. A NaN or inf coordinate/speed must
never be admitted as a plausible fix: `location_feature()` turns a NaN lat/lon
into a NaN location vector and the NaN scalars are written to Neo4j, silently
poisoning a second/minute embedding and its `NEAR` links.

The ML-heavy imports are stubbed at module load exactly as in the repo's
existing `auto_ingest/tests/test_ml_pure.py`. Every call below uses synthetic
Python floats: no Neo4j, moviepy, pandas file I/O, network, service or media is
touched.
"""

from __future__ import annotations

import importlib
import sys
import unittest.mock as mock

import pytest

pd = pytest.importorskip("pandas")


@pytest.fixture(scope="module")
def Y():
    stubs = {
        "moviepy": mock.MagicMock(),
        "moviepy.editor": mock.MagicMock(),
        "neo4j": mock.MagicMock(),
        "neo4j.exceptions": mock.MagicMock(),
    }
    with mock.patch.dict(sys.modules, stubs):
        yield importlib.import_module("auto_ingest.dashcam.yolo_embeddings")


NAN = float("nan")
INF = float("inf")


def test_nan_latitude_is_not_a_plausible_fix(Y):
    assert Y.plausible_fix(None, NAN, -120.0, 30.0) is False


def test_nan_longitude_is_not_a_plausible_fix(Y):
    assert Y.plausible_fix(None, 35.0, NAN, 30.0) is False


def test_infinite_latitude_is_not_a_plausible_fix(Y):
    assert Y.plausible_fix(None, INF, -120.0, 30.0) is False


def test_nan_speed_is_not_a_plausible_fix(Y):
    assert Y.plausible_fix(None, 35.0, -120.0, NAN) is False


def test_infinite_speed_is_not_a_plausible_fix(Y):
    assert Y.plausible_fix(None, 35.0, -120.0, INF) is False


def test_nan_coordinate_is_rejected_even_with_a_valid_previous_fix(Y):
    # Pre-fix the jump test computed haversine_m(nan, ...) -> nan, and
    # `nan > max_jump_m` is False, so the gate vacuously admitted the point.
    assert Y.plausible_fix((35.0, -120.0, 30.0), NAN, -120.0, 30.0) is False


def test_non_finite_previous_fix_cannot_validate_a_jump(Y):
    assert Y.plausible_fix((NAN, -120.0, 30.0), 35.0, -120.0, 30.0) is False


def test_finite_valid_fix_is_still_admitted(Y):
    assert Y.plausible_fix(None, 35.0, -120.0, 30.0) is True
    assert Y.plausible_fix((35.0, -120.0, 30.0), 35.0, -120.0, 30.0) is True


def test_existing_rejections_are_preserved(Y):
    assert Y.plausible_fix(None, None, -120.0, 30.0) is False
    assert Y.plausible_fix(None, 35.0, None, 30.0) is False
    assert Y.plausible_fix(None, 35.0, -120.0, 999.0) is False
    assert Y.plausible_fix(None, 35.0, -120.0, -1.0) is False
    # A finite point still fails when it is too far from a finite previous fix.
    assert Y.plausible_fix((35.0, -120.0, 30.0), 36.0, -120.0, 30.0) is False
