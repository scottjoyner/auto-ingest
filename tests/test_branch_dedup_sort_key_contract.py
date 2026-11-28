"""Hermetic contract for the documented band-ranking order of the read-only
branch-dedup report.

``Report.render()`` promises a precise ranking in its header prose::

    Ranking: A by branch name; B by fewest behind then name;
    C by fewest ahead, then fewest behind, then name.

That order is implemented solely by ``_sort_key(band)``. The existing tests pin
Band A only implicitly (via ``test_limit_bounds_each_band``) and never exercise
B or C, so a regression that reorders those bands would silently mis-rank the
report while every current test still passed. This test loads
``tools/branch_dedup_report.py`` by exact path and pins the pure sort key for
all three bands: no git subprocess, no filesystem state, no network, and no
heavy imports (the tool itself is stdlib-only).
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_TOOL = _ROOT / "tools" / "branch_dedup_report.py"

spec = importlib.util.spec_from_file_location("branch_dedup_report", _TOOL)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod  # dataclass decorator resolves cls.__module__
spec.loader.exec_module(mod)


def stat(name: str, behind: int, ahead: int) -> mod.BranchStat:
    return mod.BranchStat(name=name, behind=behind, ahead=ahead, subject="")


def names_in_ranked_order(band: str, rows: list[mod.BranchStat]) -> list[str]:
    return [row.name for row in sorted(rows, key=mod._sort_key(band))]


class SortKeyContract(unittest.TestCase):
    def test_band_a_ranks_by_name_only(self) -> None:
        # Behind/ahead are irrelevant for safe candidates: only name decides.
        rows = [
            stat("z-branch", 0, 1),
            stat("a-branch", 99, 42),
            stat("m-branch", 3, 7),
        ]
        self.assertEqual(
            names_in_ranked_order(mod.BAND_SAFE, rows),
            ["a-branch", "m-branch", "z-branch"],
        )
        # The key itself must be the single name, not a tuple including counts.
        self.assertEqual(mod._sort_key(mod.BAND_SAFE)(rows[0]), ("z-branch",))

    def test_band_b_ranks_by_fewest_behind_then_name(self) -> None:
        rows = [
            stat("z-late", 5, 1),
            stat("z-early", 2, 1),
            stat("a-early", 2, 1),
        ]
        self.assertEqual(
            names_in_ranked_order(mod.BAND_REBASE, rows),
            ["a-early", "z-early", "z-late"],
        )
        self.assertEqual(mod._sort_key(mod.BAND_REBASE)(rows[0]), (5, "z-late"))

    def test_band_c_ranks_by_ahead_then_behind_then_name(self) -> None:
        # Ahead is primary: 1 ahead beats 4 behind/0 ahead, even though the
        # latter has fewer behind commits.
        rows = [
            stat("d-many-ahead", 1, 4),
            stat("c-few-ahead-many-behind", 9, 1),
            stat("b-few-ahead-few-behind", 2, 1),
            stat("a-few-ahead-few-behind", 2, 1),
        ]
        self.assertEqual(
            names_in_ranked_order(mod.BAND_DIVERGED, rows),
            [
                "a-few-ahead-few-behind",
                "b-few-ahead-few-behind",
                "c-few-ahead-many-behind",
                "d-many-ahead",
            ],
        )
        self.assertEqual(
            mod._sort_key(mod.BAND_DIVERGED)(rows[0]), (4, 1, "d-many-ahead")
        )

    def test_render_prose_matches_the_pinned_ranking(self) -> None:
        # Fail closed if the documented order drifts away from the keys above.
        report = mod.Report(repo="r", head="h", prefix=mod.DEFAULT_PREFIX, limit=40)
        prose = report.render()
        self.assertIn("A by branch name", prose)
        self.assertIn("B by fewest behind then name", prose)
        self.assertIn("C by fewest ahead, then fewest behind, then name", prose)


if __name__ == "__main__":
    unittest.main()
