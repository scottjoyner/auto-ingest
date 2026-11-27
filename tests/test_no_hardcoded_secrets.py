"""Guard test: the secrets linter must pass (no un-routed hardcoded passwords).

Standardization (W-53): the historical ``knowledge_graph_2026`` default is only
allowed when routed through the canonical ``NEO4J_PASSWORD_DEFAULT`` env var. This
test runs tools/lint_no_hardcoded_secrets.py and fails if any bare literal creeps
back in.
"""
import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path

LINT_PATH = Path(__file__).resolve().parent.parent / "tools" / "lint_no_hardcoded_secrets.py"


def _load_lint():
    loader = SourceFileLoader("lint_no_hardcoded_secrets", str(LINT_PATH))
    spec = importlib.util.spec_from_loader("lint_no_hardcoded_secrets", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def test_no_hardcoded_neo4j_password():
    lint = _load_lint()
    assert lint.main() == 0, "hardcoded Neo4j password not routed through NEO4J_PASSWORD_DEFAULT"


def _lint_in_temp_root(monkeypatch, tmp_path):
    """Point the linter at a hermetic tree so its exit status can be exercised."""
    lint = _load_lint()
    (tmp_path / "tools").mkdir()
    monkeypatch.setattr(lint, "__file__", str(tmp_path / "tools" / "lint_no_hardcoded_secrets.py"))
    return lint


def test_a_non_utf8_source_cannot_hide_a_bare_literal(monkeypatch, tmp_path, capsys):
    """A legacy PEP 263 file must not silently opt its contents out of the gate.

    Strict decoding made one ``\xe9`` anywhere in a file abort the read and the
    linter skipped it entirely, returning 0 with a bare password on another
    line. That is a fail-open CI gate, not a tolerant one.
    """
    lint = _lint_in_temp_root(monkeypatch, tmp_path)
    (tmp_path / "legacy.py").write_bytes(
        b"# -*- coding: latin-1 -*-\n"
        b"name = 'caf\xe9'\n"
        b"password = 'knowledge_graph_2026'\n"
    )
    assert lint.main() == 1
    assert "legacy.py:3" in capsys.readouterr().out


def test_a_non_utf8_source_still_allows_the_routed_literal(monkeypatch, tmp_path):
    """The fix broadens the scan, not the rule: the documented fallback stays OK."""
    lint = _lint_in_temp_root(monkeypatch, tmp_path)
    (tmp_path / "legacy.py").write_bytes(
        b"# -*- coding: latin-1 -*-\n"
        b"name = 'caf\xe9'\n"
        b"pw = os.getenv('NEO4J_PASSWORD') or os.getenv('NEO4J_PASSWORD_DEFAULT') "
        b"or 'knowledge_graph_2026'\n"
    )
    assert lint.main() == 0
