"""The naming contract must stay importable without the ML stack.

`auto_ingest.ingest.discovery` holds the suffix patterns and `canonicalize_key`
that `auto_ingest.custody.staging` has to agree with. It exists as a separate
module precisely so importing it does not require torch, transformers or a Neo4j
driver - and these tests are the reason, because they are what proves a staged
filename is actually discoverable.

If discovery ever imports the ML stack again, every one of those guards silently
stops running on CI: pytest.importorskip would turn a real check into a green
skip, which is precisely how the hand-copied-regex bug got in. So the dependency
boundary is asserted directly rather than trusted.
"""
from __future__ import annotations

import ast
import pathlib
import sys

DISCOVERY = pathlib.Path(
    __file__).resolve().parents[1] / "auto_ingest" / "ingest" / "discovery.py"

FORBIDDEN = {
    "torch", "transformers", "sentence_transformers", "ultralytics", "cv2",
    "moviepy", "gliner", "neo4j", "numpy", "pandas",
}
ALLOWED = {
    "__future__", "hashlib", "os", "re", "datetime", "typing", "zoneinfo",
    "auto_ingest", "dataclasses", "pathlib", "collections", "functools",
    "itertools", "enum", "typing_extensions",
}


def _top_level_imports(path: pathlib.Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0], node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.level:      # relative import within the package
                continue
            if node.module:
                yield node.module.split(".")[0], node.lineno
        elif isinstance(node, ast.Try):
            # A guarded import is exactly what we are looking for; look inside.
            for sub in ast.walk(node):
                if isinstance(sub, ast.Import):
                    for alias in sub.names:
                        yield alias.name.split(".")[0], sub.lineno
                elif isinstance(sub, ast.ImportFrom) and sub.module and not sub.level:
                    yield sub.module.split(".")[0], sub.lineno


def test_discovery_imports_nothing_from_the_ml_stack():
    offenders = [(name, line) for name, line in _top_level_imports(DISCOVERY)
                 if name in FORBIDDEN]
    assert not offenders, (
        f"{DISCOVERY.name} must stay dependency-free; found {offenders}"
    )


def test_discovery_imports_only_what_it_needs():
    """Named, rather than merely "not forbidden", so an unrelated new dependency
    is a deliberate decision instead of a diff nobody reads."""
    unexpected = sorted({name for name, _ in _top_level_imports(DISCOVERY)}
                        - ALLOWED)
    assert not unexpected, f"unexpected imports in discovery.py: {unexpected}"


def test_the_contract_imports_with_the_ml_stack_blocked():
    """The assertion that actually matters, and the reason this module exists.

    Executed in a subprocess with the heavy modules made unimportable, because
    sys.modules and import caches make that unreliable in-process: by the time
    this test runs, something else may already have imported torch.
    """
    import subprocess

    program = """
import builtins, sys
_real = builtins.__import__
BLOCKED = {"torch", "transformers", "sentence_transformers", "ultralytics",
           "cv2", "moviepy", "gliner", "neo4j"}
def guard(name, *a, **k):
    if name.split(".")[0] in BLOCKED:
        raise ModuleNotFoundError(f"No module named {name!r}")
    return _real(name, *a, **k)
builtins.__import__ = guard
from auto_ingest.ingest.discovery import (
    PAT_MEDIA, PAT_TRANS_JSON_TXT, PAT_RTTM, canonicalize_key)
assert PAT_MEDIA.search("clip.MP4")
assert PAT_RTTM.search("2025_0202_171732_speakers.rttm")
assert not PAT_MEDIA.search("MOVI0000.avi")
assert canonicalize_key("2025_0202_171732", "x/2025_0202_171732_medium_transcription.txt")
print("OK")
"""
    root = pathlib.Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", program], cwd=root, capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": "/tmp",
             "PYTHONPATH": str(root), "LOCAL_TZ": "America/New_York"})
    assert result.returncode == 0, result.stderr[-2000:]
    assert "OK" in result.stdout


def test_transcripts_still_exports_the_contract_for_existing_callers():
    """The extraction must not break the importers that predate it."""
    from auto_ingest.ingest import discovery as d
    from auto_ingest.ingest import transcripts as tx

    assert tx.PAT_MEDIA is d.PAT_MEDIA
    assert tx.PAT_TRANS_JSON_TXT is d.PAT_TRANS_JSON_TXT
    assert tx.canonicalize_key is d.canonicalize_key
    assert tx.stable_id is d.stable_id
