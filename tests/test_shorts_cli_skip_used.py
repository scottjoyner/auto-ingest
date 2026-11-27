"""Contract: ``shorts render --skip-used`` must actually control in-plan dedup.

The flag used to be parsed and then ignored: ``_cmd_render`` passed the hardcoded
``skip_used_clips=True`` to ``render.render_plan``, so neither omitting nor
passing the flag changed anything, and there was no way to render a plan that
reuses a clip across its shorts. These tests are hermetic: no Neo4j, no moviepy,
no filesystem writes outside pytest's tmp_path.
"""
from __future__ import annotations

import pytest

from auto_ingest.shorts import cli


def _fake_plan():
    class _Plan:
        shorts: list = []

        def save(self, path):
            pass

    return _Plan()


@pytest.fixture
def render_calls(monkeypatch):
    """Capture the kwargs ``_cmd_render`` hands to ``render.render_plan``."""
    captured: dict = {}

    class _Driver:
        def close(self):
            pass

    monkeypatch.setattr(cli.Plan, "load", staticmethod(lambda path: _fake_plan()))
    monkeypatch.setattr(cli, "_driver", lambda: (_Driver(), "neo4j"))
    monkeypatch.setattr(cli.backdrop, "select_highway_pool", lambda driver, limit: [])

    def _fake_render(plan, out_dir, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(cli.render, "render_plan", _fake_render)
    monkeypatch.setattr(cli.render, "upsert_manifest", lambda *a, **k: None)
    return captured


def test_parser_exposes_skip_used_and_its_negation():
    p = cli.build_parser()
    # The historical default (dedup within the plan) must be preserved.
    assert p.parse_args(["render", "plan.json"]).skip_used is True
    assert p.parse_args(["render", "plan.json", "--skip-used"]).skip_used is True
    # But an operator must be able to turn it off: before the fix this was an
    # argparse error, because the flag had no negation.
    assert p.parse_args(["render", "plan.json", "--no-skip-used"]).skip_used is False


@pytest.mark.parametrize(
    "extra, expected",
    [([], True), (["--skip-used"], True), (["--no-skip-used"], False)],
)
def test_render_forwards_skip_used_flag(render_calls, tmp_path, extra, expected):
    plan = tmp_path / "plan.json"
    args = cli.build_parser().parse_args(["render", str(plan), *extra])

    assert cli._cmd_render(args) == 0
    assert render_calls["skip_used_clips"] is expected, (
        "the --skip-used flag must reach render_plan; a hardcoded value makes "
        "the flag a silent no-op")
