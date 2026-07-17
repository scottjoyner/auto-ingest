"""
Tests for the research-scripted shorts package (auto_ingest.shorts).

Covers the parts that do NOT need moviepy or a live LLM:
  - models round-trip (Brief/Plan/PlannedShort <-> JSON)
  - planner: deterministic planning + iteration (reject + reseed)
  - backdrop: shot picking (gap enforcement, local-only)
  - curator: Neo4j curation query shape (uses a fake driver, no live DB)
"""
from pathlib import Path

import pytest

from auto_ingest.shorts import (
    backdrop,
    content_miner,
    curator,
    hook_bank,
    planner,
    render,
    tts,
    virality,
)
from auto_ingest.shorts.models import Brief, Cue, Plan, PlannedShort, Shot, SourceRef


# --------------------------------------------------------------------------- #
# models round-trip
# --------------------------------------------------------------------------- #
def test_brief_roundtrip():
    b = Brief(topic="llm", title="LLMs", hook="h", points=["a", "b"],
              sources=[SourceRef("paper", "p1", "T", year=2024)],
              tags=["x"])
    b2 = Brief.from_dict(b.to_dict())
    assert b2.topic == "llm" and b2.points == ["a", "b"]
    assert b2.sources[0].year == 2024


def test_plan_save_load(tmp_path):
    brief = Brief(topic="llm", title="LLMs", hook="h", points=["a"])
    plan = Plan(topic="llm", brief=brief, shorts=[
        PlannedShort(id="s1", brief_topic="llm", title="LLMs 1",
                     cues=[Cue(0.0, 3.0, "hi")],
                     shots=[Shot("k", "/x.mp4", 1.0, 6.0, 60.0)]),
    ])
    p = tmp_path / "plan.json"
    plan.save(p)
    loaded = Plan.load(p)
    assert loaded.topic == "llm"
    assert loaded.shorts[0].cues[0].text == "hi"
    assert loaded.shorts[0].shots[0].mph == 60.0
    assert loaded.shorts[0].duration() == 3.0


# --------------------------------------------------------------------------- #
# planner
# --------------------------------------------------------------------------- #
def _brief(points):
    return Brief(topic="t", title="T", hook="hook",
                points=points, sources=[SourceRef("paper", "p1", "P")])


_ANCHORS = [
    {"clip_key": f"c{i}", "fr_path": f"/mnt/clip{i}.MP4",
     "t_sec": float(i * 7), "mph": 60.0}
    for i in range(20)
]


def test_plan_shorts_deterministic():
    plan = planner.plan_shorts(_brief(["a", "b", "c", "d"]), _ANCHORS,
                                  short_count=3, short_dur=24.0, seed=1)
    assert len(plan.shorts) == 3
    # same seed -> identical ids
    plan2 = planner.plan_shorts(_brief(["a", "b", "c", "d"]), _ANCHORS,
                                   short_count=3, short_dur=24.0, seed=1)
    assert plan.shorts[0].id == plan2.shorts[0].id
    assert any(c.kind == "hook" for c in plan.shorts[0].cues)
    for s in plan.shorts:
        assert s.shots, "expected highway shots"
        for sh in s.shots:
            assert sh.fr_path


def test_plan_distributes_points_across_shorts():
    plan = planner.plan_shorts(_brief(["p1", "p2", "p3", "p4", "p5", "p6"]),
                                  _ANCHORS, short_count=3, short_dur=30.0, seed=2)
    all_cue_texts = [c.text for s in plan.shorts for c in s.cues]
    assert any(c.kind == "hook" for c in plan.shorts[0].cues)
    # sources are still present on the final short (payoff cue now caps it)
    assert any(c.kind == "source" for c in plan.shorts[-1].cues)
    assert any(t.startswith("p") for t in all_cue_texts)


def test_plan_partitions_points_disjoint_and_complete():
    # No point is repeated and all points are covered (even divide).
    points = [f"p{i}" for i in range(12)]
    plan = planner.plan_shorts(_brief(points), _ANCHORS,
                               short_count=3, short_dur=30.0, seed=4)
    assert len(plan.shorts) == 3
    seen: list[str] = []
    for s in plan.shorts:
        pt_cues = [c.text for c in s.cues if c.kind == "point"]
        seen.extend(pt_cues)
    # every curated point appears exactly once across shorts
    assert sorted(seen) == sorted(points), seen
    assert len(seen) == len(set(seen)), "a point was repeated across shorts"

    # Uneven divide: coverage is maximal (no point dropped, none duplicated).
    points2 = [f"q{i}" for i in range(10)]
    plan2 = planner.plan_shorts(_brief(points2), _ANCHORS,
                                short_count=3, short_dur=30.0, seed=4)
    seen2 = [c.text for s in plan2.shorts for c in s.cues if c.kind == "point"]
    assert sorted(seen2) == sorted(points2)
    assert len(seen2) == len(set(seen2))


def test_iterate_drops_rejected_and_reseeds():
    plan = planner.plan_shorts(_brief(["a", "b", "c", "d"]), _ANCHORS,
                                  short_count=3, seed=1)
    reject_id = plan.shorts[0].id
    new = planner.iterate_plan(plan, _ANCHORS, seed=5, reject_ids=[reject_id])
    assert new.iteration == plan.iteration + 1
    assert reject_id not in [s.id for s in new.shorts]
    assert len(new.shorts) == 3  # dropped 1, regenerated 1


# --------------------------------------------------------------------------- #
# backdrop
# --------------------------------------------------------------------------- #
def test_pick_shots_enforces_gap_and_local_only():
    anchors = [
        {"clip_key": "cA", "fr_path": "/mnt/A.mp4", "t_sec": 1.0, "mph": 60.0},
        {"clip_key": "cA", "fr_path": "/mnt/A.mp4", "t_sec": 3.0, "mph": 60.0},  # too close (gap 2 < 8)
        {"clip_key": "cB", "fr_path": None, "t_sec": 50.0, "mph": 60.0},     # not local
        {"clip_key": "cC", "fr_path": "/mnt/C.mp4", "t_sec": 80.0, "mph": 60.0},
    ]
    shots = backdrop.pick_shots(anchors, count=5, min_gap_sec=8.0, clip_dur=6.0,
                               rng_seed=1)
    assert len(shots) == 2
    assert all(s["fr_path"] for s in shots)
    assert shots[0]["clip_key"] == "cA" and shots[1]["clip_key"] == "cC"


# --------------------------------------------------------------------------- #
# curator (fake driver, no live Neo4j)
# --------------------------------------------------------------------------- #
class _FakeResult:
    def __init__(self, data): self._d = data
    def __iter__(self): return iter(self._d)
    def data(self): return self._d
    def single(self): return self._d[0] if self._d else None


class _FakeSession:
    def __init__(self, table): self.table = table
    def __enter__(self): return self
    def __exit__(self, *e): return False
    def run(self, *_a, **_k):
        q = (_a[0] if _a else "").lower()
        if "topic {name" in q and "paper" not in q:
            return _FakeResult([{"title": "Large Language Models"}])
        if "mentions" in q and "topic" in q:
            return _FakeResult([
                {"uid": "u1", "text": "I think LLMs will change everything",
                 "tkey": "2025_0101_120000", "started": None,
                 "concept": "large language model", "score": 0.81},
                {"uid": "u2", "text": "agents can call tools",
                 "tkey": "2025_0101_120030", "started": None,
                 "concept": "agent", "score": 0.72},
            ])
        if "has_concept" in q:
            return _FakeResult([{"name": "llm"}, {"name": "agent"}])
        if "belongs_to_topic" in q and "keyword" in q:
            return _FakeResult([
                {"id": "2401.xxxx", "title": "Paper One", "year": "2024-01-01",
                 "url": "http://x/1", "cites": 10, "kws": ["llm"]},
                {"id": "2402.yyyy", "title": "Paper Two", "year": "2024-02-02",
                 "url": "http://x/2", "cites": 5, "kws": ["agent"]},
            ])
        if "has_chunk" in q:
            return _FakeResult([{"id": "ch1", "text": "This paper shows that LLMs can reason."}])
        if "mentions" in q and "topic" in q:
            return _FakeResult([
                {"uid": "u1", "text": "I think LLMs will change everything",
                 "tkey": "2025_0101_120000", "started": None,
                 "concept": "large language model", "score": 0.81},
                {"uid": "u2", "text": "agents can call tools",
                 "tkey": "2025_0101_120030", "started": None,
                 "concept": "agent", "score": 0.72},
            ])
        if "is_lyrics" in q or "review_needed" in q or "music_overlap" in q:
            if "review_needed" in q:
                return _FakeResult([
                    {"clip_key": "segclip_rev1", "idx": 1, "start": 12.0,
                     "text": "uncertain utterance here", "speaker": "Driver"},
                    {"clip_key": "segclip_rev2", "idx": 4, "start": 55.0,
                     "text": "another flagged segment", "speaker": "Passenger"},
                ])
            return _FakeResult([
                {"clip_key": "segclip_mus1", "idx": 2, "start": 20.0,
                 "text": "la la la singing along", "score": 0.91, "music_overlap": True},
                {"clip_key": "segclip_mus2", "idx": 7, "start": 88.0,
                 "text": "lyrics in the background", "score": 0.74, "music_overlap": False},
            ])
        if "belongs_to" in q and "mph" in q:
            return _FakeResult([
                {"clip_key": f"c{i}", "frame": i * 210, "mph": 55.0 + (i % 20),
                 "fps": 30.0, "path": f"/data/c{i}"}
                for i in range(20)
            ])
        if "trip" in q and "uniquekey" in q and "in_trip" not in q:
            # _select_trips by uniqueKey
            return _FakeResult([
                {"key": "Dashcam_TRIP1", "clips": 50, "tracker": "Dashcam",
                 "start": None},
            ])
        if "trip" in q and "clipcount" in q:
            # _select_trips longest
            return _FakeResult([
                {"key": "Dashcam_TRIP1", "clips": 50, "tracker": "Dashcam"},
                {"key": "Dashcam_TRIP2", "clips": 30, "tracker": "Dashcam"},
                {"key": "Dashcam_TRIP3", "clips": 20, "tracker": "Dashcam"},
            ])
        if "in_trip" in q:
            # _trip_clips: DashcamClip-[:IN_TRIP]->Trip (clips only)
            return _FakeResult([
                {"clip_key": "2025_0101_120000_F", "path": "/data/t1a",
                 "fps": 30.0, "cstart": None},
                {"clip_key": "2025_0101_123000_F", "path": "/data/t1b",
                 "fps": 30.0, "cstart": None},
            ])
        if "stop" in q and "location" in q:
            # _trip_clips located Stops (round-robin captions)
            return _FakeResult([
                {"g": "Scott 6 MI Raleigh house", "loc": "124 S Dawson"},
                {"g": None, "loc": "I-40 W"},
            ])
        return _FakeResult([])
    def close(self): pass


class _FakeDriver:
    def session(self, *a, **k): return _FakeSession({})
    def close(self): pass


def test_curate_topic_shape():
    facts = curator.curate_topic(_FakeDriver(), "large_language_models",
                                 top_papers=6, top_concepts=12, chunk_per_paper=1)
    assert facts.topic_title == "Large Language Models"
    assert len(facts.papers) == 2
    assert facts.papers[0]["id"] == "2401.xxxx"
    assert "llm" in facts.concepts and "agent" in facts.concepts
    assert len(facts.chunks) == 2  # 2 papers x chunk_per_paper=1


def test_synthesize_brief_from_facts():
    facts = curator.curate_topic(_FakeDriver(), "large_language_models")

    class _FakeClient:
        def generate(self, model, prompt):
            return ('{"title":"LLMs Explained","hook":"Hook line",'
                    '"points":["p1","p2","p3"],"tags":["llm","ai"]}')

    brief = curator.synthesize_brief(facts, client=_FakeClient())
    assert brief.title == "LLMs Explained"
    assert brief.hook == "Hook line"
    assert len(brief.points) == 3
    assert brief.sources and brief.sources[0].ref_id == "2401.xxxx"


def test_discusses_topic_returns_spoken_discussions():
    clips = curator.discusses_topic(_FakeDriver(), "large_language_models", min_score=0.65, limit=10)
    assert len(clips) == 2
    assert all(c.score >= 0.65 for c in clips)
    assert clips[0].concept == "large language model"
    assert "LLMs" in clips[0].text


def test_brief_from_discussions_no_llm():
        clips = [
            curator.DiscussionClip("u1", "I think LLMs will change everything",
                                   "k1", "large language model", 0.81),
            curator.DiscussionClip("u2", "agents can call tools",
                                   "k2", "agent", 0.72),
            curator.DiscussionClip("u1", "I think LLMs will change everything",  # dup
                                   "k1", "large language model", 0.81),
        ]
        brief = curator.brief_from_discussions("large_language_models", clips,
                                            topic_title="Large Language Models")
        assert brief.hook == "I think LLMs will change everything"
        assert len(brief.points) == 2  # de-duped
        assert brief.sources[0].kind == "utterance"
        assert brief.sources[0].ref_id == "u1"
        assert "large language model" in brief.tags


def test_brief_from_discussions_empty_raises():
    import pytest
    with pytest.raises(ValueError):
        curator.brief_from_discussions("x", [])


def test_plan_montage_builds_ambient_plan(monkeypatch, tmp_path):
    monkeypatch.setattr(backdrop, "_fr_path_for_key",
                        lambda key, root: Path(f"/mnt/{key}.MP4"))

    plan = planner.plan_montage(
        _FakeDriver(), count=3, dur=30.0, mood="calm", limit=400, seed=7,
    )
    assert len(plan.shorts) == 3
    for s in plan.shorts:
        assert s.shots, "montage short must have highway shots"
        for sh in s.shots:
            assert sh.fr_path
        assert s.cues, "montage cues must be non-empty"
        assert all(c.kind == "mood" for c in s.cues)
        assert all(c.text for c in s.cues)

    out = tmp_path / "montage.json"
    plan.save(out)
    loaded = Plan.load(out)
    assert loaded.topic == "montage"
    assert len(loaded.shorts) == 3


def test_plan_discusses_flag_wires_through(monkeypatch, tmp_path):
    from auto_ingest.shorts import cli as shorts_cli

    caught = {}

    def _fake_discusses(driver, topic, *, min_score=0.65, min_text_len=30, limit=40):
        caught["topic"] = topic
        caught["min"] = min_score
        caught["minlen"] = min_text_len
        caught["limit"] = limit
        return [
            curator.DiscussionClip("u1", "spoken point about the topic here",
                                   "k1", "concept_a", 0.7),
        ]

    monkeypatch.setattr(curator, "discusses_topic", _fake_discusses)
    monkeypatch.setattr(backdrop, "select_highway_pool", lambda d, limit=400: [])

    args = shorts_cli.build_parser().parse_args([
        "plan", "large_language_models", "--discusses",
        "--min-score", "0.7", "--discuss-limit", "5",
        "--plans-dir", str(tmp_path),
    ])
    rc = args.func(args)
    assert rc == 0
    assert caught == {"topic": "large_language_models", "min": 0.7, "minlen": 30, "limit": 5}
    written = list(tmp_path.glob("*.json"))
    assert written and "large_language_models" in written[0].name


def test_plan_highlights_builds_event_shorts(monkeypatch, tmp_path):
    import auto_ingest.shorts.planner as pl

    monkeypatch.setattr(pl.backdrop, "_fr_path_for_key",
                        lambda key, root: Path(f"/mnt/{key}.MP4"))

    plan = pl.plan_highlights(
        _FakeDriver(), kinds=("music", "review", "speed"),
        per_kind=2, dur=20.0, limit=400, seed=3,
    )
    assert plan.topic == "highlights"
    assert plan.brief.topic == "highlights"
    kinds = {s.notes.split()[0].split("=")[1] for s in plan.shorts}
    assert "music" in kinds
    assert "review" in kinds
    assert "speed" in kinds
    for s in plan.shorts:
        assert s.shots, "highlight short must have a shot"
        assert s.shots[0].fr_path
        assert s.cues, "highlight cues must be non-empty"
        assert all(c.text for c in s.cues)

    out = tmp_path / "highlights.json"
    plan.save(out)
    loaded = Plan.load(out)
    assert loaded.topic == "highlights"
    assert len(loaded.shorts) >= 3


def test_plan_highlights_skips_missing_kind(monkeypatch):
    import auto_ingest.shorts.planner as pl

    monkeypatch.setattr(pl.backdrop, "_fr_path_for_key",
                        lambda key, root: Path(f"/mnt/{key}.MP4"))

    plan = pl.plan_highlights(
        _FakeDriver(), kinds=("speed",), per_kind=2, dur=20.0, limit=400, seed=1,
    )
    assert plan.shorts, "speed events should be produced"
    for s in plan.shorts:
        assert s.notes.split()[0] == "kind=speed"

def test_plan_trip_story_builds_journey_plan():
    from auto_ingest.shorts import planner

    plan = planner.plan_trip_story(
        _FakeDriver(), trip_key="Dashcam_TRIP1", count=1,
        short_dur=30.0, shots_per_trip=2, clip_dur=6.0, seed=1)
    assert plan.topic == "trip_story"
    assert len(plan.shorts) == 1
    s0 = plan.shorts[0]
    assert len(s0.shots) == 2
    # Journey captions are present (kind=trip) and read as real places.
    kinds = {c.kind for c in s0.cues}
    assert "trip" in kinds
    place_cues = [c.text for c in s0.cues if c.text]
    assert place_cues, "expected place/location captions along the trip"
    assert s0.notes.startswith("trip=Dashcam_TRIP1")


# --------------------------------------------------------------------------- #
# TTS (owner-voice) lazy import + graceful fallback
# --------------------------------------------------------------------------- #

def test_tts_narrate_graceful_without_reference(monkeypatch, tmp_path):

    # When TTS synthesis is unavailable, narrate returns None so callers fall
    # back to a silent render. (Owner audio reference may be cached on disk,
    # so we stub the synthesis step itself.)
    monkeypatch.setattr(
        tts, "synthesize",
        lambda *a, **k: (_ for _ in ()).throw(tts.TTSUnavailable("no tts")),
    )
    out = tts.narrate("demo", [{"text": "hello world", "start": 0, "end": 2}])
    assert out is None


def test_tts_synthesize_raises_when_package_missing(monkeypatch, tmp_path):

    # Force both in-process and venv paths to be unavailable.
    monkeypatch.setattr(tts, "_tts_venv_python", lambda: None)

    def _boom(*a, **k):
        raise ImportError("no TTS")
    monkeypatch.setattr(tts, "TTS", _boom, raising=False)
    import builtins
    real_import = builtins.__import__

    def _fake_import(name, *a, **k):
        if name.startswith("TTS"):
            raise ImportError("TTS not installed")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", _fake_import)
    try:
        with pytest.raises(tts.TTSUnavailable):
            tts.synthesize("hi", tmp_path / "ref.wav", tmp_path / "out.wav")
    finally:
        monkeypatch.setattr(builtins, "__import__", real_import)


# --------------------------------------------------------------------------- #
# render_plan dedup (skip_used_clips)
# --------------------------------------------------------------------------- #

def test_render_plan_skip_history(monkeypatch, tmp_path):

    # skip_history=True reads prior rendered :Short clip_keys and skips them.
    plan = Plan(
        topic="demo",
        brief=Brief(topic="demo", title="Demo", hook="hi", points=["a", "b"], sources=[]),
        shorts=[
            PlannedShort(id="s1", title="S1", brief_topic="demo",
                         shots=[Shot(clip_key="clip_A", fr_path="/x.mp4", t_sec=0, dur=6)],
                         cues=[Cue(text="a", start=0, end=2, kind="narration")]),
            PlannedShort(id="s2", title="S2", brief_topic="demo",
                         shots=[Shot(clip_key="clip_B", fr_path="/y.mp4", t_sec=0, dur=6)],
                         cues=[Cue(text="b", start=0, end=2, kind="narration")]),
        ],
    )
    monkeypatch.setattr(render, "_already_rendered_clips", lambda: {"clip_A"})
    rendered = []
    def _fake(item, out, **kw):
        rendered.append(item.id)
        item.out_path = str(out / f"{item.id}.mp4")
        return out / f"{item.id}.mp4"
    monkeypatch.setattr(render, "render_short", _fake)
    out = render.render_plan(plan, tmp_path, skip_history=True)
    assert rendered == ["s2"], "s1 should be skipped (clip_A already used in history)"
    assert len(out) == 1


def test_render_short_rejects_when_no_footage(monkeypatch, tmp_path):
    # Pure test (no moviepy): a short with no resolvable footage must be
    # rejected, return early, and never invoke compose_scripted_short.
    item = PlannedShort(
        id="nope", title="Nope", brief_topic="demo",
        shots=[Shot(clip_key="k", fr_path="/does/not/exist.mp4", t_sec=0, dur=6)],
        cues=[Cue(text="a", start=0, end=2, kind="narration")],
    )
    called = {}
    def _boom(*a, **k):
        called["compose"] = True
        raise AssertionError("compose must not be called for rejected short")
    monkeypatch.setattr("auto_ingest.shorts.compose.compose_scripted_short", _boom)

    out = render.render_short(item, tmp_path)
    assert item.status == "rejected"
    assert not called, "compose_scripted_short should not be called"
    # No MP4 should exist (it returns early before writing video).
    assert not Path(out).exists(), "rejected short must not write an mp4"


def test_publish_queue_dedup_idempotent(tmp_path, monkeypatch):
    from auto_ingest.shorts import publish

    # Point the queue at a temp file and avoid any live DB calls.
    monkeypatch.setattr(publish, "QUEUE_PATH", tmp_path / "queue.jsonl")

    class _S:
        key = "abc123"
        topic = "x"
        title = "t"
        out_path = "/tmp/x.mp4"

    publish.queue_for_publish(_S(), platform="youtube_shorts")
    publish.queue_for_publish(_S(), platform="youtube_shorts")  # dup
    publish.queue_for_publish(_S(), platform="tiktok")          # different platform
    qlines = [ln for ln in publish.QUEUE_PATH.read_text().splitlines() if ln.strip()]
    # 2 distinct (key,platform) pairs -> 2 lines, not 3.
    assert len(qlines) == 2, qlines
    keys = [__import__("json").loads(ln)["platform"] for ln in qlines]
    assert keys == ["youtube_shorts", "tiktok"]


def test_render_plan_in_plan_dedup(monkeypatch, tmp_path):

    # In-plan dedup: a clip reused within the same plan is rendered once.
    plan = Plan(
        topic="demo",
        brief=Brief(topic="demo", title="Demo", hook="hi", points=["a"], sources=[]),
        shorts=[
            PlannedShort(id="s1", title="S1", brief_topic="demo",
                         shots=[Shot(clip_key="clip_A", fr_path="/x.mp4", t_sec=0, dur=6)],
                         cues=[Cue(text="a", start=0, end=2, kind="narration")]),
            PlannedShort(id="s2", title="S2", brief_topic="demo",
                         shots=[Shot(clip_key="clip_A", fr_path="/x.mp4", t_sec=0, dur=6)],
                         cues=[Cue(text="b", start=0, end=2, kind="narration")]),
        ],
    )
    rendered = []
    def _fake2(item, out, **kw):
        rendered.append(item.id)
        item.out_path = str(out / f"{item.id}.mp4")
        return out / f"{item.id}.mp4"
    monkeypatch.setattr(render, "render_short", _fake2)
    out = render.render_plan(plan, tmp_path, skip_used_clips=True)
    assert rendered == ["s1"], "s2 should be skipped (clip_A reused in-plan)"
    assert len(out) == 1


# --------------------------------------------------------------------------- #
# hook_bank (content strategy — no Neo4j / no moviepy)
# --------------------------------------------------------------------------- #
def test_topic_type_of_infers():
    assert hook_bank.topic_type_of(Brief(topic="t", title="T", hook="h", tags=["paper", "arxiv"])) == "paper"
    assert hook_bank.topic_type_of(Brief(topic="t", title="T", hook="h", tags=["debate"])) == "debate"
    assert hook_bank.topic_type_of(Brief(topic="t", title="T", hook="h", tags=["opinion"])) == "opinion"
    assert hook_bank.topic_type_of(Brief(topic="t", title="T", hook="h", tags=["utterance"])) == "utterance"
    assert hook_bank.topic_type_of(Brief(topic="t", title="T", hook="h", tags=["concept"])) == "concept"


def test_build_hook_cues_single_part():
    b = Brief(topic="t", title="Cool Topic", hook="h", points=["p"],
              tags=["concept"])
    cues = hook_bank.build_hook_cues(b, 12.0, short_index=0, total_parts=1)
    assert cues[0].kind == "hook"
    assert len(cues) == 1  # no callback when single part


def test_build_hook_cues_series_adds_callback_and_payoff():
    b = Brief(topic="t", title="Cool Topic", hook="h", points=["p"],
              tags=["concept"])
    cues = hook_bank.build_hook_cues(b, 12.0, short_index=1, total_parts=3)
    kinds = [c.kind for c in cues]
    assert "hook" in kinds and "callback" in kinds
    assert any("Part 2 of 3" in c.text for c in cues)
    payoff = hook_bank.build_payoff_cue(b, 12.0)
    assert payoff.kind == "payoff"
    assert payoff.start > 8.0  # reserved near the end


def test_hook_rotates_across_variants():
    b = Brief(topic="t", title="Cool Topic", hook="h", points=["p"],
              tags=["concept"])
    v0 = hook_bank.hook_for(b, variant=0, topic_type="concept")
    v1 = hook_bank.hook_for(b, variant=1, topic_type="concept")
    assert v0 != v1  # different cold-open per part


def test_myth_fact_pair():
    cues = hook_bank.myth_fact_pair("bigger models always smarter",
                                     "scaling hits diminishing returns")
    assert [c.kind for c in cues] == ["myth", "fact"]
    assert "bigger models" in cues[0].text and "scaling" in cues[1].text


def test_curiosity_reveal_returns_question_and_reveal():
    cues = hook_bank.curiosity_reveal("why does it work?", "the math is simple", 12.0)
    assert [c.kind for c in cues] == ["question", "reveal"]
    assert cues[0].end <= cues[1].start  # reveal follows the question
    assert cues[1].text == "the math is simple"


def test_series_intro_and_outro():
    b = Brief(topic="t", title="Cool Topic", hook="h", points=["p"], tags=["concept"])
    intro = hook_bank.build_series_intro(b, 3, 0)
    assert intro.kind == "series" and "Part 1 of 3" in intro.text
    mid = hook_bank.build_series_outro(b, 3, 0, duration=12.0)
    assert "Next: Part 2 of 3" in mid.text
    last = hook_bank.build_series_outro(b, 3, 2, duration=12.0)
    assert "Series complete" in last.text
    # outro sits near the end, not on top of the hook
    assert mid.start > 8.0


def test_distribute_cues_series_has_intro_callback_outro():
    plan = planner.plan_shorts(_brief(["p1", "p2", "p3"]), _ANCHORS,
                                short_count=2, short_dur=20.0, seed=3)
    kinds0 = [c.kind for c in plan.shorts[0].cues]
    assert "series" in kinds0  # intro + outro
    assert "callback" in kinds0
    assert "reveal" in kinds0  # curiosity gap
    assert "payoff" in kinds0


def test_distribute_cues_single_part_has_no_series():
    plan = planner.plan_shorts(_brief(["p1", "p2"]), _ANCHORS,
                                short_count=1, short_dur=20.0, seed=3)
    kinds = [c.kind for c in plan.shorts[0].cues]
    assert "series" not in kinds
    assert "callback" not in kinds
    assert "reveal" in kinds


# --------------------------------------------------------------------------- #
# virality scorer (pure, no Neo4j)
# --------------------------------------------------------------------------- #
def _short_with(*kinds, duration=20.0, hook_text="You've been using LLMs wrong"):
    cues = [Cue(0.4, 3.4, hook_text, kind="hook")]
    t = 3.6
    for k in kinds:
        cues.append(Cue(round(t, 2), round(t + 2.6, 2), f"{k} line here", kind=k))
        t += 2.8
    return PlannedShort(id="x", brief_topic="t", title="T", cues=cues)


def test_virality_scores_weak_short_low():
    weak = _short_with("point", "point")  # no gap hook, no callback/payoff/reveal
    vs = virality.score_short(weak)
    assert 0.0 <= vs.total <= 100.0
    assert vs.total < 60.0
    assert any("callback" in s or "payoff" in s for s in vs.suggestions)


def test_virality_scores_strong_short_high():
    strong = _short_with("point", "callback", "reveal", "payoff", "myth", "fact")
    vs = virality.score_short(strong)
    assert vs.total > 70.0
    assert vs.grade() in ("A", "B")


def test_virality_plan_ranking():
    plan = Plan(topic="t", brief=_brief(["p"]), shorts=[
        _short_with("point", "point"),
        _short_with("point", "callback", "reveal", "payoff", "myth", "fact"),
    ])
    res = virality.score_plan(plan)
    assert res["ranked"][0] == plan.shorts[1].id
    assert res["best"] > res["worst"]


def test_virality_short_with_real_myth_fact_scores_share():
    s = _short_with("myth", "fact", "reveal", "payoff", "callback")
    vs = virality.score_short(s)
    assert vs.factors["share"] >= 80.0


# --------------------------------------------------------------------------- #
# content_miner (fake driver)
# --------------------------------------------------------------------------- #
class _FakeMinerDriver:
    def __init__(self, rows): self._rows = rows
    def session(self):
        return _MinerSession(self._rows)


class _MinerSession:
    def __init__(self, rows): self._rows = rows
    def __enter__(self): return self
    def __exit__(self, *e): return False
    def run(self, q, **kw):
        low = q.lower()
        if "has_chunk" in low:
            return _FakeResult(self._rows)
        return _FakeResult([])


def test_content_miner_myth_fact_and_reveal():
    rows = [{"ptitle": "Cool Paper", "pid": "1234", "text":
             "However, the model unexpectedly outperforms baselines without fine-tuning on the held-out set.",
             "excerpt": "However, the model unexpectedly outperforms baselines without fine-tuning on the held-out set."}]
    drv = _FakeMinerDriver(rows)
    mf = content_miner.myth_fact_for_topic(drv, "large_language_models")
    assert mf and "large language models" in mf[0]
    assert "unexpectedly" in mf[1]
    tw = content_miner.reveal_twist_for_topic(drv, "large_language_models")
    assert tw and "unexpectedly" in tw


def test_content_miner_returns_none_on_empty():
    drv = _FakeMinerDriver([])
    assert content_miner.myth_fact_for_topic(drv, "x") is None
    assert content_miner.reveal_twist_for_topic(drv, "x") is None
