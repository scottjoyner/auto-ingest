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
