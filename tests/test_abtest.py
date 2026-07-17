"""Tests for A/B test variant generation + tracking (auto_ingest.shorts.abtest).

Runs under system python3 with `python3 -m pytest tests/test_abtest.py`.
"""
from pathlib import Path

from auto_ingest.shorts import abtest
from auto_ingest.shorts.models import Brief


def _dummy_brief():
    return Brief(topic="llm", title="Cool Topic", hook="h",
                 points=["p"], tags=["concept"])


def test_make_thumbnail_variants_naming(monkeypatch, tmp_path):
    # Monkeypatch thumbnail.make_thumbnail + the internal alt renderer to just
    # touch distinct dummy files (no moviepy / ffmpeg required).
    from auto_ingest.shorts import thumbnail as thumb_mod

    written = {}

    def _fake_v0(mp4, out_path, *, title, topic, width=1080, height=1920):
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"v0")
        written["v0"] = p
        return p

    def _fake_alt(mp4, out_path, *, title, topic, accent, width=1080, height=1920):
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"v1")
        written["v1"] = p
        return p

    monkeypatch.setattr(thumb_mod, "make_thumbnail", _fake_v0)
    monkeypatch.setattr(abtest, "_make_alt_thumbnail", _fake_alt)

    mp4 = tmp_path / "short.mp4"
    mp4.write_bytes(b"fake")
    out = tmp_path / "thumbs"
    paths = abtest.make_thumbnail_variants(
        mp4, out, title="Cool Topic", topic="llm", n=2)
    assert len(paths) == 2
    names = {p.name for p in paths}
    assert "short.v0.jpg" in names
    assert "short.v1.jpg" in names
    # Distinct files (different bytes).
    data = {p.read_bytes() for p in paths}
    assert len(data) == 2


def test_make_thumbnail_variants_partial_failure(monkeypatch, tmp_path):
    from auto_ingest.shorts import thumbnail as thumb_mod

    def _boom(*a, **k):
        raise RuntimeError("no frame")
    monkeypatch.setattr(thumb_mod, "make_thumbnail", _boom)
    monkeypatch.setattr(abtest, "_make_alt_thumbnail", _boom)
    mp4 = tmp_path / "short.mp4"
    mp4.write_bytes(b"fake")
    paths = abtest.make_thumbnail_variants(
        mp4, tmp_path / "thumbs", title="t", topic="t", n=2)
    assert paths == []  # both failed -> return what succeeded (nothing)


def test_title_variants_distinct():
    titles = abtest.title_variants(_dummy_brief(), n=2)
    assert len(titles) >= 2
    assert titles[0] != titles[1]


def test_title_variants_from_short():
    from auto_ingest.shorts.models import PlannedShort
    ps = PlannedShort(id="s1", brief_topic="llm", title="Cool Topic")
    titles = abtest.title_variants(ps, n=2)
    assert len(titles) >= 1


def test_variant_plan_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("AB_VARIANTS_PATH", str(tmp_path / "ab.jsonl"))
    # Reload module-level store path by re-importing isn't needed; setattr.
    monkeypatch.setattr(abtest, "VARIANT_STORE", tmp_path / "ab.jsonl")

    plan = abtest.plan_variants(
        "s1", "youtube",
        [str(tmp_path / "s1.v0.jpg"), str(tmp_path / "s1.v1.jpg")],
        ["Title A", "Title B"],
    )
    assert plan.short_id == "s1"
    assert plan.active_variant == 0
    assert plan.winner is None

    loaded = abtest.load_variants(short_id="s1", platform="youtube")
    assert len(loaded) == 1
    assert loaded[0].thumb_variants[1].endswith("s1.v1.jpg")
    assert loaded[0].title_variants == ["Title A", "Title B"]

    updated = abtest.choose_winner("s1", "youtube", 1)
    assert updated is not None
    assert updated.winner == 1
    assert updated.chosen_at is not None

    reloaded = abtest.load_variants(short_id="s1", platform="youtube")[0]
    assert reloaded.winner == 1
    assert reloaded.active_variant == 0  # active left as-is; winner recorded


def test_assign_variant_returns_active(tmp_path):
    plan = abtest.VariantPlan(
        short_id="s1", platform="youtube",
        thumb_variants=[str(tmp_path / "a.jpg"), str(tmp_path / "b.jpg")],
        title_variants=["A", "B"], active_variant=1,
    )

    class _Item:
        title = "fallback"
        key = "s1"

    thumb, title = abtest.assign_variant(_Item(), plan)
    assert thumb.endswith("b.jpg")
    assert title == "B"


def test_assign_variant_missing_active_uses_zero(tmp_path):
    plan = abtest.VariantPlan(
        short_id="s1", platform="youtube",
        thumb_variants=[str(tmp_path / "a.jpg")],
        title_variants=[], active_variant=0,
    )

    class _Item:
        title = "fallback"
        key = "s1"

    thumb, title = abtest.assign_variant(_Item(), plan)
    assert thumb.endswith("a.jpg")
    assert title == "fallback"


def test_variant_plan_schema_version(tmp_path, monkeypatch):
    monkeypatch.setattr(abtest, "VARIANT_STORE", tmp_path / "ab.jsonl")
    abtest.plan_variants("s1", "youtube", ["a.jpg"], ["A"])
    import json
    line = (tmp_path / "ab.jsonl").read_text(encoding="utf-8").splitlines()[0]
    assert json.loads(line)["_v"] == abtest.ABTEST_SCHEMA_VERSION


def test_variant_plan_missing_version_defaults(tmp_path, monkeypatch, caplog):
    import json
    store = tmp_path / "ab.jsonl"
    monkeypatch.setattr(abtest, "VARIANT_STORE", store)
    legacy = {"short_id": "old", "platform": "youtube",
              "thumb_variants": ["a.jpg"], "title_variants": ["A"],
              "active_variant": 0}
    store.write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    with caplog.at_level("WARNING"):
        plans = abtest.load_variants(short_id="old")
    assert len(plans) == 1
    assert plans[0].short_id == "old"
    assert any("missing _v" in r.message for r in caplog.records)


def test_variant_plan_version_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(abtest, "VARIANT_STORE", tmp_path / "ab.jsonl")
    abtest.plan_variants("s1", "youtube", ["a.jpg", "b.jpg"], ["A", "B"])
    loaded = abtest.load_variants(short_id="s1")[0]
    assert loaded.to_dict()["_v"] == 1
    assert loaded.thumb_variants == ["a.jpg", "b.jpg"]
