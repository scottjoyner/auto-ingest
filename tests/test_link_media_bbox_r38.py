"""Regression tests for the GPS bounding-box prefilter in link_media (r38).

`_bbox` is the cheap, index-backed prefilter that bounds the Neo4j query before
the exact haversine join runs. It must never exclude a point that is genuinely
within `radius_m` of the centre, otherwise a real media/place match is lost
with no error and no retry. These tests are pure: no DB, clock, file, network or
model IO.
"""
from __future__ import annotations

import math

from auto_ingest.personal.link_media import _bbox, haversine

_M_PER_DEG_LAT = 111_320.0
_RADIUS = 50_000.0
_DLAT = _RADIUS / _M_PER_DEG_LAT


def _within_centre_box(lat: float, lon: float, radius: float) -> tuple[float, ...]:
    return _bbox(lat, lon, radius)


def test_northern_point_inside_radius_is_inside_bbox() -> None:
    lat0, lat1, lon0, lon1 = _within_centre_box(60.0, 0.0, _RADIUS)
    # Deterministic witness: a small latitude step plus a longitude between the
    # centre-latitude padding and the poleward-corrected padding.
    witness_lat = 60.0 + _DLAT * 0.005
    witness_lon = 0.899
    assert haversine(60.0, 0.0, witness_lat, witness_lon) <= _RADIUS
    assert lat0 <= witness_lat <= lat1
    assert lon0 <= witness_lon <= lon1


def test_southern_point_inside_radius_is_inside_bbox() -> None:
    lat0, lat1, lon0, lon1 = _within_centre_box(-60.0, 10.0, _RADIUS)
    witness_lat = -60.0 - _DLAT * 0.005
    witness_lon = 10.899
    assert haversine(-60.0, 10.0, witness_lat, witness_lon) <= _RADIUS
    assert lat0 <= witness_lat <= lat1
    assert lon0 <= witness_lon <= lon1


def test_latitude_band_is_symmetric_and_unpadded() -> None:
    """Control: the latitude half-width is the exact spherical north/south step."""
    lat0, lat1, lon0, lon1 = _within_centre_box(60.0, 0.0, _RADIUS)
    assert math.isclose(lat0, 60.0 - _DLAT, rel_tol=1e-12)
    assert math.isclose(lat1, 60.0 + _DLAT, rel_tol=1e-12)
    assert lon0 < 0.0 < lon1


def test_equator_and_small_radius_still_sane() -> None:
    """Control: ordinary short-radius queries are not broken by the fix."""
    lat0, lat1, lon0, lon1 = _within_centre_box(0.0, 0.0, 200.0)
    dlat = 200.0 / _M_PER_DEG_LAT
    assert math.isclose(lat0, -dlat, rel_tol=1e-12)
    assert math.isclose(lat1, dlat, rel_tol=1e-12)
    assert lon0 < 0.0 < lon1
    assert math.isfinite(lon1)


def test_padding_stays_bounded_at_high_latitude() -> None:
    """Control: the safe poleward padding is conservative, not unbounded."""
    _lat0, _lat1, lon0, lon1 = _within_centre_box(60.0, 0.0, _RADIUS)
    assert 0.0 < lon1 < 1.0
    assert math.isclose(lon0, -lon1, rel_tol=1e-12)
