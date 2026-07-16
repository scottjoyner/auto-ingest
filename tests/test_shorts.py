"""
Tests for the research-scripted shorts package (auto_ingest.shorts).

Covers the parts that do NOT need moviepy or a live LLM:
  - models round-trip (Brief/Plan/PlannedShort <-> JSON)
  - planner: deterministic planning + iteration (reject + reseed)
  - backdrop: shot picking (gap enforcement, local-only)
  - curator: Neo4j curation query shape (uses a fake driver, no live DB)
"""
from pathlib import Path

from auto_ingest.shorts import models, planner, backdrop, curator
from auto_ingest.shorts.models import Brief, SourceRef, Plan, PlannedShort, Cue, Shot


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
