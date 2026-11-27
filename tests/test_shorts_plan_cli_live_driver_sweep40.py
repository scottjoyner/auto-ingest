"""Sweep40: `shorts plan` must hand an OPEN driver to the planner.

The CLI comment promises that a live driver is passed to ``plan_shorts`` so the
myth/fact + reveal beats come from real graph content (S-G10).  A driver closed
before that call makes every mining query fail, and ``plan_shorts`` swallows the
error and silently falls back to templated text.

Hermetic: a fake driver + stubbed curator/backdrop/planner, no Neo4j, network,
credentials, moviepy or real media.
"""
from __future__ import annotations

from types import SimpleNamespace

from auto_ingest.shorts import cli
from auto_ingest.shorts.models import Brief, Plan


class _FakeDriver:
    """Minimal driver: records close() and refuses sessions once closed."""

    def __init__(self) -> None:
        self.closed = False
        self.sessions = 0

    def close(self) -> None:
        self.closed = True

    def session(self, *args, **kwargs):
        if self.closed:
            raise RuntimeError("Driver is closed")
        self.sessions += 1
        raise AssertionError("no session expected in this hermetic test")


def _args(tmp_path):
    return SimpleNamespace(
        discusses=False,
        topic="large_language_models",
        top_papers=6,
        pool=10,
        shorts=2,
        dur=30.0,
        shots_per_short=3,
        seed=1,
        min_score=0.65,
        min_text_len=30,
        discuss_limit=40,
        plans_dir=tmp_path,
    )


def _stub_graph_reads(monkeypatch):
    brief = Brief(topic="large_language_models", title="LLM", hook="Hook",
                  points=["point"])
    monkeypatch.setattr(cli.curator, "curate_brief",
                        lambda driver, topic, **kw: brief)
    monkeypatch.setattr(cli.backdrop, "select_highway_pool",
                        lambda driver, **kw: [])
    return brief


def test_plan_passes_open_driver_to_planner(monkeypatch, tmp_path):
    """The driver must still be open when plan_shorts is called."""
    drv = _FakeDriver()
    monkeypatch.setattr(cli, "_driver", lambda: (drv, "neo4j"))
    _stub_graph_reads(monkeypatch)
    seen = {}

    def fake_plan_shorts(brief_arg, anchors, **kwargs):
        seen["driver"] = kwargs.get("driver")
        seen["closed_at_call"] = getattr(kwargs.get("driver"), "closed", None)
        return Plan(topic=brief_arg.topic, brief=brief_arg, shorts=[])

    monkeypatch.setattr(cli.planner, "plan_shorts", fake_plan_shorts)

    assert cli._cmd_plan(_args(tmp_path)) == 0
    assert seen["driver"] is drv
    # A closed driver makes content_miner's graph queries fail, so the planner
    # silently drops real reveal/myth-fact content and uses templated text.
    assert seen["closed_at_call"] is False


def test_discusses_branch_does_not_reuse_driver(monkeypatch, tmp_path):
    """Control: the --discusses path returns inside try and never reuses it."""
    drv = _FakeDriver()
    monkeypatch.setattr(cli, "_driver", lambda: (drv, "neo4j"))
    monkeypatch.setattr(cli.curator, "discusses_topic",
                        lambda driver, topic, **kw: [])
    called = []
    monkeypatch.setattr(cli.planner, "plan_discussion",
                        lambda *a, **kw: called.append(kw))

    args = _args(tmp_path)
    args.discusses = True
    assert cli._cmd_plan(args) == 1
    assert called == []
    assert drv.closed is True
