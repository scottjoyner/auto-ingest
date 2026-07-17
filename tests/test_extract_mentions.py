"""
Tests for scripts/extract_mentions.py — the safe, batched Entity/Mention
extractor. Only the importable surface + the pure text extractor are tested
(no live graph).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import extract_mentions as em  # noqa: E402


def test_module_importable():
    assert callable(em.main)
    assert callable(em.extract_mentions_from_text)


def test_extract_self_reference():
    ex = em.extract_mentions_from_text("I went to my car and we drove")
    assert ex["self"] is True


def test_extract_dates_and_times():
    ex = em.extract_mentions_from_text("Let's meet on 2026-06-14 at 3:30pm")
    assert "2026-06-14" in ex["dates"]
    assert any("3:30" in t for t in ex["times"])


def test_extract_no_signals():
    ex = em.extract_mentions_from_text("The weather was nice.")
    assert ex["self"] is False
    assert ex["dates"] == []
    assert ex["times"] == []
