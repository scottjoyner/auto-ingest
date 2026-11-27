"""Round 25 — bounded nearest-place geofence admission (``shorts.planner``).

``auto_ingest.shorts.planner._nearest_place(lat, lon, places, max_miles)`` is the
single pure admission step that turns a dashcam frame's averaged (lat, lon) into
the human-readable geofence label of a ``SummaryPlace``. It is used only by
``_trip_clips`` (trip-story captioning), and its documented contract is
deliberately narrow:

  * return the place nearest to (lat, lon) by great-circle distance;
  * ignore places with a missing latitude or longitude;
  * fail closed (``None``) when even the nearest place is beyond ``max_miles``;
  * otherwise return the exact place object that was supplied.

No test or prior review report exercised this helper: the trip-story suite pins
only assembled plans with a fake driver, never the admission math itself. These
tests are hermetic — in-memory dicts only, no driver, graph, mount, network,
credential, model or file I/O.

The bounds below use the module's own Earth radius (3959 mi): one degree of
latitude is ~69.1 mi, so the bracket values 69.0 / 69.2 straddle a single
degree without duplicating the distance formula under test.
"""
from auto_ingest.shorts import planner


def _place(name, lat, lon, **extra):
    return {"name": name, "lat": lat, "lon": lon, **extra}


def test_single_place_within_default_bounds_is_returned_verbatim():
    """A single in-range place is returned as the same object, not a copy."""
    near = _place("home", 35.0, -78.0)
    assert planner._nearest_place(35.0, -78.0, [near]) is near
    assert planner._nearest_place(35.0, -78.0, [near]) == near


def test_nearest_place_wins_even_when_it_is_last_in_the_list():
    """Selection is by distance, never by list position."""
    far = _place("far", 0.2, 0.0)    # ~13.8 mi
    mid = _place("mid", 0.1, 0.0)    # ~6.9 mi
    near = _place("near", 0.01, 0.0)  # ~0.7 mi
    assert planner._nearest_place(0.0, 0.0, [far, mid, near]) is near


def test_place_beyond_default_max_miles_fails_closed():
    """Default bound is 25 mi: a distant place is not admitted at all."""
    far = _place("far-away", 2.0, 0.0)  # ~138 mi
    assert planner._nearest_place(0.0, 0.0, [far]) is None
    # Control: the very same place is admitted when the bound is raised.
    assert planner._nearest_place(0.0, 0.0, [far], max_miles=200.0) is far


def test_custom_max_miles_bounds_admission_both_ways():
    """An explicit bound replaces the default, in both directions."""
    place = _place("one-degree", 1.0, 0.0)  # ~69.1 mi
    assert planner._nearest_place(0.0, 0.0, [place], max_miles=50.0) is None
    assert planner._nearest_place(0.0, 0.0, [place], max_miles=80.0) is place


def test_distance_boundary_is_inclusive():
    """Exactly-at-the-boundary is admitted; just beyond it is not."""
    place = _place("one-degree", 0.0, 1.0)  # ~69.1 mi at the equator
    assert planner._nearest_place(0.0, 0.0, [place], max_miles=69.2) is place
    assert planner._nearest_place(0.0, 0.0, [place], max_miles=69.0) is None


def test_places_missing_either_coordinate_are_skipped():
    """Incomplete places never win, even when they are closer."""
    no_lon = _place("no-lon", 0.001, None)   # ~0.07 mi if it counted
    no_lat = {"name": "no-lat", "lon": 0.0}
    valid = _place("valid", 0.1, 0.0)        # ~6.9 mi
    assert planner._nearest_place(0.0, 0.0, [no_lon, no_lat, valid]) is valid


def test_no_usable_place_returns_none():
    """Empty input and all-incomplete input both fail closed."""
    assert planner._nearest_place(0.0, 0.0, []) is None
    broken = [{"name": "a", "lat": None, "lon": 0.0},
              {"name": "b", "lat": 0.0, "lon": None}]
    assert planner._nearest_place(0.0, 0.0, broken) is None


def test_exact_distance_tie_keeps_the_first_place():
    """The strict `<` comparison makes ties resolve to the first candidate."""
    first = _place("north", 0.1, 0.0)
    second = _place("south", -0.1, 0.0)
    assert planner._nearest_place(0.0, 0.0, [first, second]) is first
    assert planner._nearest_place(0.0, 0.0, [second, first]) is second


def test_result_is_deterministic_across_calls_and_permutations():
    """Same candidates in any order yield the same selected place, byte-stably."""
    near = _place("near", 0.01, 0.0)
    far = _place("far", 0.5, 0.0)
    candidates = [far, near]
    results = [planner._nearest_place(0.0, 0.0, candidates) for _ in range(4)]
    assert all(r is near for r in results)
    assert planner._nearest_place(0.0, 0.0, list(reversed(candidates))) is near
    assert planner._nearest_place(0.0, 0.0, candidates) == near


def test_zero_max_miles_admits_only_exact_coincidence():
    """A zero bound is not a wildcard: only distance zero passes it."""
    exact = _place("exact", 35.0, -78.0)
    offset = _place("offset", 35.0001, -78.0)
    assert planner._nearest_place(35.0, -78.0, [exact], max_miles=0) is exact
    assert planner._nearest_place(35.0, -78.0, [offset], max_miles=0) is None


def test_negative_max_miles_never_admits():
    """A negative bound fails closed even for an exact coordinate match."""
    exact = _place("exact", 35.0, -78.0)
    assert planner._nearest_place(35.0, -78.0, [exact], max_miles=-1.0) is None


def test_input_places_are_not_mutated():
    """Admission is read-only over its input list and place dicts."""
    near = _place("near", 0.01, 0.0)
    far = _place("far", 0.5, 0.0)
    places = [far, near]
    snapshot = [dict(p) for p in places]
    planner._nearest_place(0.0, 0.0, places)
    assert places == snapshot
    assert [id(p) for p in places] == [id(far), id(near)]
