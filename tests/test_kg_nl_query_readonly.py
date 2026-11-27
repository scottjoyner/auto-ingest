"""Hermetic contract tests for the NL-query read-only Cypher guard.

Loads ``scripts/kg_nl_query.py`` by path. No LM Studio, no Neo4j, no network:
one test injects a fake driver that records every query handed to
``Session.run`` so a write statement that slips past the guard is observable.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "kg_nl_query.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("kg_nl_query_contract", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


kg = _load_module()


@pytest.mark.parametrize("cypher", [
    "MATCH (n) SET\nn.flag = 1 RETURN n",
    "MATCH (n) SET\tn.flag = 1 RETURN n",
    "MATCH (n) SET\r\nn.flag = 1 RETURN n",
    "MATCH (n) WITH\nWRITE RETURN n",
])
def test_write_spelled_with_non_space_whitespace_is_rejected(cypher):
    # The guard must classify Cypher tokens, not the literal substring
    # "SET " / "WITH WRITE": a newline or tab after the keyword is still a write.
    assert kg.is_readonly(cypher) is False, cypher


@pytest.mark.parametrize("cypher", [
    "MATCH (n) RETURN n LIMIT 5",
    "CALL db.labels() YIELD label RETURN label",
    "MATCH (p:Paper) WHERE p.title CONTAINS 'graph' RETURN p.arxiv_id",
])
def test_genuine_read_queries_stay_allowed(cypher):
    assert kg.is_readonly(cypher) is True, cypher


def test_extracted_write_query_is_rejected():
    raw = "```cypher\nMATCH (n) SET\nn.pwned = true\nRETURN n\n```"
    cypher = kg.extract_cypher(raw)
    assert "SET" in cypher.upper()
    assert kg.is_readonly(cypher) is False


def test_main_never_runs_a_write_query(monkeypatch):
    """End-to-end: a generated write must be rejected before Session.run()."""
    generated = "```cypher\nMATCH (n) SET\nn.pwned = true\nRETURN n\n```"
    monkeypatch.setattr(kg, "chat_with_fallback", lambda q: (generated, "fake-model"))
    monkeypatch.setattr(sys, "argv", ["kg_nl_query.py", "mark everything"])

    executed: list[str] = []

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def run(self, cypher):
            executed.append(cypher)
            return self

        def data(self):
            return [{"n": "row"}]

    class FakeDriver:
        def session(self, database=None):
            return FakeSession()

        def close(self):
            pass

    monkeypatch.setattr(kg, "get_neo4j", lambda: (FakeDriver(), "neo4j"))
    kg.main()
    assert executed == [], "write Cypher reached Session.run()"
