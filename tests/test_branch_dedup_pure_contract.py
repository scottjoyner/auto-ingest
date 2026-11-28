"""Hermetic contract for the read-only branch-dedup report's pure helpers.

Complements tests/test_branch_dedup_report.py: no git subprocess, no filesystem
state, no network. Loads tools/branch_dedup_report.py by exact path and pins the
band-classification and markdown-cell escaping semantics.
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


class ClassifyContract(unittest.TestCase):
    def test_band_a_only_zero_behind_single_commit(self) -> None:
        self.assertEqual(mod.classify(0, 1), mod.BAND_SAFE)

    def test_band_b_any_behind_with_single_commit(self) -> None:
        for behind in (1, 2, 17):
            with self.subTest(behind=behind):
                self.assertEqual(mod.classify(behind, 1), mod.BAND_REBASE)

    def test_band_c_covers_all_other_unmerged_shapes(self) -> None:
        for behind, ahead in ((0, 0), (0, 2), (1, 0), (3, 2), (5, 7)):
            with self.subTest(behind=behind, ahead=ahead):
                self.assertEqual(mod.classify(behind, ahead), mod.BAND_DIVERGED)

    def test_bands_and_titles_are_consistent(self) -> None:
        self.assertEqual(tuple(mod.BAND_TITLES), mod.BANDS)
        self.assertEqual(len(mod.BANDS), 3)


class CellContract(unittest.TestCase):
    def test_collapses_whitespace(self) -> None:
        self.assertEqual(mod._cell("  hello \t world \n next  "), "hello world next")

    def test_escapes_markdown_pipe(self) -> None:
        self.assertEqual(mod._cell("a|b|c"), "a\\|b\\|c")

    def test_empty_subject_gets_placeholder(self) -> None:
        self.assertEqual(mod._cell(""), "(no subject)")
        self.assertEqual(mod._cell(" \n\t "), "(no subject)")


if __name__ == "__main__":
    unittest.main()
