"""Ingest subpackage: transcript + media ingestion into Neo4j.

Imports are lazy, and that is a correctness fix rather than a nicety. `transcripts` imports torch, sentence-transformers and the Neo4j driver at module scope,
so importing it here made this subpackage unimportable without the ML stack - including
the discovery patterns inside it, which are plain compiled regexes and the one thing the
custody layout tests must import in order to assert against real discovery behaviour
rather than a hand-copied version of it. Skipping those tests when torch is absent
would put the guard back exactly where it was: they exist because a hand-copied regex
agreed with itself while the real one rejected every name staging produced.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from .transcripts import main as run_transcripts

__all__ = ["run_transcripts", "transcripts"]


def __getattr__(name: str) -> Any:
    if name in ("run_transcripts", "transcripts"):
        # importlib, not `from . import x`. The latter consults __getattr__ for the
        # name before the submodule is bound, so a handler that resolves it by
        # importing recurses until the stack gives out. import_module binds the
        # submodule on the package, so this runs once.
        module = importlib.import_module(f"{__name__}.transcripts")
        return module if name == "transcripts" else getattr(module, "main")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
