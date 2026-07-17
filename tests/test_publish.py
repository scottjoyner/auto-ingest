"""
Publishing pipeline tests — pure, no network, no credentials.

Covers the dry-run / metadata plumbing around the upload adapters:
  - scan_rendered + queue_for_publish build queue entries from disk
  - _title_for / _hashtags / _description_for assemble per-platform metadata
  - plan_report renders a full offline upload plan (no creds, no network)
  - validate_queue reports missing files / thumbnails
"""

from auto_ingest.shorts import publish, scheduling, uploader
from auto_ingest.shorts.models import Brief, Cue, Plan, PlannedShort


def _item(key="abc123", topic="diffusion_models", title="Cool Title",
          platform="youtube_shorts", brief_hook=None, hook_cue=None,
          thumbnail_path=None):
    return uploader.QueueItem(
        key=key, topic=topic, title=title,
        out_path="/tmp/does_not_exist.mp4", platform=platform,
        brief_hook=brief_hook, hook_cue=hook_cue,
        thumbnail_path=thumbnail_path)


# --------------------------------------------------------------------------- #
# queue from disk
# --------------------------------------------------------------------------- #
def test_scan_rendered_finds_mp4s(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, "QUEUE_PATH", tmp_path / "queue.jsonl")
    topic_dir = tmp_path / "llm"
    topic_dir.mkdir()
    (topic_dir / "s1.mp4").write_bytes(b"x")
    (topic_dir / "s2.mp4").write_bytes(b"x")
    (topic_dir / "notes.txt").write_text("ignore")

    items = publish.scan_rendered(tmp_path)
    assert {i.key for i in items} == {"s1", "s2"}
    assert all(i.topic == "llm" for i in items)
    assert all(i.out_path.endswith(".mp4") for i in items)


def test_queue_all_on_disk_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, "QUEUE_PATH", tmp_path / "queue.jsonl")
    topic_dir = tmp_path / "llm"
    topic_dir.mkdir()
    (topic_dir / "s1.mp4").write_bytes(b"x")
    n1 = publish.queue_all_on_disk(tmp_path, ["youtube_shorts", "tiktok"])
    n2 = publish.queue_all_on_disk(tmp_path, ["youtube_shorts", "tiktok"])
    # idempotent: second run adds nothing
    assert n1 == 2
    assert n2 == 0
    lines = [ln for ln in (tmp_path / "queue.jsonl").read_text().splitlines() if ln]
    assert len(lines) == 2


# --------------------------------------------------------------------------- #
# metadata builders
# --------------------------------------------------------------------------- #
def test_title_for_per_platform():
    it = _item(title="Cool Paper Title", topic="large_language_models")
    yt = uploader._title_for("youtube", it)
    tk = uploader._title_for("tiktok", it)
    ig = uploader._title_for("instagram", it)
    assert yt == "Large Language Models: Cool Paper Title"
    assert tk == "cool paper title"
    assert "#largelanguagemodels" in ig


def test_hashtags_per_platform():
    it = _item(topic="diffusion_models")
    yt = uploader._hashtags("youtube", it)
    tk = uploader._hashtags("tiktok", it)
    ig = uploader._hashtags("instagram", it)
    assert "#diffusionmodels" in yt and "#shorts" in yt
    assert "#ai" in tk and "#research" in tk
    assert "learnontiktok" in ig and "savethis" in ig


def test_description_pulls_hook_from_cue_then_brief():
    cue_item = _item(hook_cue="Why scaling hits limits")
    desc = uploader._description_for("youtube", cue_item)
    assert "Why scaling hits limits" in desc
    assert "Part of the research-shorts series" in desc
    assert "#diffusionmodels" in desc

    brief_item = _item(brief_hook="Papers made simple")
    desc2 = uploader._description_for("tiktok", brief_item)
    assert "Papers made simple" in desc2


def test_description_falls_back_to_title_when_no_hook():
    it = _item(title="My Short", brief_hook=None, hook_cue=None)
    desc = uploader._description_for("youtube", it)
    assert "My Short" in desc


# --------------------------------------------------------------------------- #
# offline plan report + validation
# --------------------------------------------------------------------------- #
def test_plan_report_offline_no_creds(tmp_path, monkeypatch):
    it = _item(title="Great Talk", brief_hook="Hook line here",
               thumbnail_path=str(tmp_path / "cover.jpg"))
    rep = uploader.plan_report(it)
    assert rep["platform"] == "youtube_shorts"
    assert rep["title"] == "Diffusion Models: Great Talk"
    assert "Hook line here" in rep["description"]
    assert rep["file_exists"] is False
    assert rep["thumbnail"] is None  # thumbnail path doesn't exist -> None


def test_plan_report_generated_thumbnail_path(tmp_path, monkeypatch):
    # A real mp4 + existing thumbnail on disk -> reported path.
    mp4 = tmp_path / "v.mp4"
    mp4.write_bytes(b"x")
    cover = tmp_path / "v.thumb.jpg"
    cover.write_bytes(b"x")
    it = uploader.QueueItem(
        key="k", topic="t", title="T", out_path=str(mp4),
        platform="instagram", thumbnail_path=str(cover))
    rep = uploader.plan_report(it)
    assert rep["thumbnail"] == str(cover)
    assert rep["file_exists"] is True


def test_validate_queue_reports_missing_file_and_thumb(tmp_path, monkeypatch):
    monkeypatch.setattr(uploader, "QUEUE_PATH", tmp_path / "queue.jsonl")
    # queue a missing file (yt) + a 9:16-only short lacking 16:9 (yt)
    uploader.load_queue.cache_clear() if hasattr(uploader.load_queue, "cache_clear") else None
    missing = uploader.QueueItem(key="m", topic="t", title="T",
                                 out_path="/nope/missing.mp4", platform="youtube_shorts")
    have_file = uploader.QueueItem(key="h", topic="t", title="T",
                                  out_path=str(tmp_path / "real.mp4"),
                                  platform="tiktok")
    (tmp_path / "real.mp4").write_bytes(b"x")
    lines = [
        {"key": missing.key, "topic": missing.topic, "title": missing.title,
         "out_path": missing.out_path, "platform": missing.platform,
         "queued_at": "2026-01-01T00:00:00+00:00"},
        {"key": have_file.key, "topic": have_file.topic, "title": have_file.title,
         "out_path": have_file.out_path, "platform": have_file.platform,
         "queued_at": "2026-01-01T00:00:00+00:00"},
    ]
    (tmp_path / "queue.jsonl").write_text(
        "\n".join(__import__("json").dumps(d) for d in lines) + "\n")

    issues = uploader.validate_queue()
    assert any(i["key"] == "m" and i["issue"] == "missing_file" for i in issues)
    assert all(i["key"] != "h" for i in issues)


# --------------------------------------------------------------------------- #
# enrich from plan (best-effort, offline)
# --------------------------------------------------------------------------- #
def test_enrich_item_from_plan_pulls_hook_and_thumb(tmp_path, monkeypatch):
    plan_dir = tmp_path / "plans"
    plan_dir.mkdir()
    brief = Brief(topic="llm", title="LLMs", hook="Brief hook text",
                  points=["a"], sources=[])
    plan = Plan(topic="llm", brief=brief, shorts=[
        PlannedShort(id="abc123", brief_topic="llm", title="LLM Short",
                     cues=[Cue(0.0, 3.0, "Cue hook text", kind="hook")],
                     thumbnail_path="/x/cover.jpg"),
    ])
    plan.save(plan_dir / "plan.json")

    it = _item(thumbnail_path=None, brief_hook=None, hook_cue=None)
    enriched = uploader.enrich_item_from_plan(it, plan_dir)
    assert enriched.hook_cue == "Cue hook text"
    assert enriched.brief_hook == "Brief hook text"
    assert enriched.thumbnail_path == "/x/cover.jpg"


# --------------------------------------------------------------------------- #
# scheduling (pure planning, no network/creds)
# --------------------------------------------------------------------------- #
def test_build_calendar_warmup_ramp():
    avail = [{"short_id": f"s{i}", "topic": t} for i, t in enumerate(
        ["large_language_models", "diffusion_models", "graph_neural_networks",
         "reinforcement_learning", "computer_vision",
         "retrieval_augmented_generation", "robotics"] * 3)]
    cal = scheduling.build_calendar(days=15, available=avail)
    by_day = {dp.day: len(dp.slots) for dp in cal}
    assert by_day[1] == 1          # days 1-3: 1/day total
    assert by_day[3] == 1
    assert by_day[7] == 3          # days 4-7: 3/day
    assert by_day[14] == 6         # days 8-14: 6/day
    # day 15 is outside the 15-day window (days=1..15 -> day 15 not built)
    # per-platform split on day 14 should be even across the 3 platforms
    d14 = [dp for dp in cal if dp.day == 14][0]
    byp = {}
    for s in d14.slots:
        byp[s.platform] = byp.get(s.platform, 0) + 1
    assert set(byp.values()) == {2}


def test_build_calendar_flags_needs_render_when_short():
    cal = scheduling.build_calendar(
        days=15, available=[{"short_id": "only1", "topic": "large_language_models"}])
    out = scheduling.calendar_to_json(cal)
    assert out["needs_render"] > 0


def test_build_calendar_topic_rotation():
    cal = scheduling.build_calendar(days=7, available=[
        {"short_id": f"s{i}", "topic": "x"} for i in range(40)])
    leads = [dp.slots[0].topic for dp in cal if dp.slots]
    # 7 distinct lead topics in order (rotation_7day)
    assert leads == ["large_language_models", "diffusion_models",
                     "graph_neural_networks", "reinforcement_learning",
                     "computer_vision", "retrieval_augmented_generation", "robotics"]


# --------------------------------------------------------------------------- #
# brand-check (pure, no network/creds)
# --------------------------------------------------------------------------- #
def test_brand_check_passes_on_real_assets():
    from pathlib import Path

    from auto_ingest.shorts.cli import _brand_check
    brand_dir = Path(__file__).resolve().parent.parent / "docs" / "brand"
    ok, issues = _brand_check(brand_dir)
    assert ok, issues
    assert issues == []


def test_brand_check_fails_when_asset_missing(tmp_path, monkeypatch):
    import json as _json

    from auto_ingest.shorts.cli import _brand_check
    # Copy only the manifest (with a bogus asset path) — no image files.
    manifest = {
        "brand_name": "X", "handles": {"youtube": "y"}, "bios": {"youtube": "b"},
        "assets": {"avatar": "nope.png", "banner_youtube": "nope2.png"},
    }
    (tmp_path / "brand_manifest.json").write_text(_json.dumps(manifest))
    ok, issues = _brand_check(tmp_path)
    assert not ok
    assert any("asset missing" in i for i in issues)


# --------------------------------------------------------------------------- #
# Live-publish safety guard (P-G1): publishing is held by default.
# --------------------------------------------------------------------------- #
def test_live_forbidden_without_opt_in(monkeypatch):
    from auto_ingest.shorts.publish_guard import (
        LivePublishForbidden,
        require_live_mode,
        safe_to_run,
    )
    for v in ("AUTO_INGEST_LIVE", "YT_TOKEN_JSON", "TIKTOK_ACCESS_TOKEN",
              "IG_ACCESS_TOKEN"):
        monkeypatch.delenv(v, raising=False)
    assert safe_to_run() is False
    try:
        require_live_mode()
    except LivePublishForbidden:
        pass
    else:
        raise AssertionError("expected LivePublishForbidden without opt-in")


def test_live_opt_in_without_creds_still_forbidden(monkeypatch):
    from auto_ingest.shorts.publish_guard import LivePublishForbidden, require_live_mode
    for v in ("AUTO_INGEST_LIVE", "YT_TOKEN_JSON", "TIKTOK_ACCESS_TOKEN",
              "IG_ACCESS_TOKEN"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("AUTO_INGEST_LIVE", "1")
    try:
        require_live_mode()
    except LivePublishForbidden:
        pass
    else:
        raise AssertionError("opt-in without creds must still be forbidden")


def test_live_opt_in_with_creds_allowed(monkeypatch):
    from auto_ingest.shorts.publish_guard import require_live_mode, safe_to_run
    for v in ("AUTO_INGEST_LIVE", "YT_TOKEN_JSON", "TIKTOK_ACCESS_TOKEN",
              "IG_ACCESS_TOKEN"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("AUTO_INGEST_LIVE", "1")
    monkeypatch.setenv("YT_TOKEN_JSON", "/tmp/yt_token.json")
    assert safe_to_run() is True
    require_live_mode()  # must not raise


def test_process_queue_live_blocked_without_opt_in(monkeypatch):
    from auto_ingest.shorts import uploader
    from auto_ingest.shorts.publish_guard import LivePublishForbidden
    for v in ("AUTO_INGEST_LIVE", "YT_TOKEN_JSON", "TIKTOK_ACCESS_TOKEN",
              "IG_ACCESS_TOKEN"):
        monkeypatch.delenv(v, raising=False)
    tmp = __import__("pathlib").Path("/tmp/opencode/pq_test.jsonl")
    monkeypatch.setenv("SHORTS_PUBLISH_QUEUE", str(tmp))
    if tmp.exists():
        tmp.unlink()
    try:
        uploader.process_queue(dry_run=False)
    except LivePublishForbidden:
        pass
    else:
        raise AssertionError("process_queue must refuse live without opt-in")


# --------------------------------------------------------------------------- #
# A/B variant -> uploader override (P-G10): the active variant reaches plan_report.
# --------------------------------------------------------------------------- #
def test_ab_variant_reaches_plan_report(tmp_path, monkeypatch):
    from auto_ingest.shorts import abtest, uploader

    # Isolate the A/B variant store (module global is read at import time).
    store = tmp_path / "ab_variants.jsonl"
    monkeypatch.setattr(abtest, "VARIANT_STORE", store)
    if store.exists():
        store.unlink()

    item = uploader.QueueItem(
        key="abkey123", topic="diffusion_models", title="Base Title",
        out_path="/tmp/does_not_exist.mp4", platform="youtube_shorts")

    # No variant yet -> standard title.
    rep0 = uploader.plan_report(item)
    assert rep0["ab_variant"] is None
    assert rep0["title"] == "Diffusion Models: Base Title"

    # Plan a variant with a distinct title + thumbnail and apply it.
    thumb = tmp_path / "abkey123.v1.jpg"
    thumb.write_text("fake")
    titles = ["Variant One Title", "Variant Two Title"]
    abtest.plan_variants("abkey123", "youtube_shorts", [str(thumb)], titles)
    # active_variant defaults to 0 -> "Variant One Title"
    rep1 = uploader.plan_report(item)
    assert rep1["ab_variant"] == 0
    assert rep1["title"] == "Variant One Title"
    assert rep1["thumbnail"] == str(thumb)

    # Switch active variant and confirm it propagates.
    plans = abtest.load_variants(short_id="abkey123", platform="youtube_shorts")
    plans[0].active_variant = 1
    abtest._save_all(plans)
    rep2 = uploader.plan_report(item)
    assert rep2["ab_variant"] == 1
    assert rep2["title"] == "Variant Two Title"
