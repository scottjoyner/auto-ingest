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

from auto_ingest.shorts import backdrop, curator, planner, render, tts
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
    assert plan.shorts[0].cues[0].text == "hook"
    for s in plan.shorts:
        assert s.shots, "expected highway shots"
        for sh in s.shots:
            assert sh.fr_path


def test_plan_distributes_points_across_shorts():
    plan = planner.plan_shorts(_brief(["p1", "p2", "p3", "p4", "p5", "p6"]),
                                  _ANCHORS, short_count=3, short_dur=30.0, seed=2)
    all_cue_texts = [c.text for s in plan.shorts for c in s.cues]
    assert plan.shorts[0].cues[0].kind == "hook"
    assert plan.shorts[-1].cues[-1].kind == "source"
    assert any(t.startswith("p") for t in all_cue_texts)


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
