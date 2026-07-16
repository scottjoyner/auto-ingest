"""Content subpackage: summary generation (Ollama) for ingested transcripts."""

from __future__ import annotations

from .build_summaries import main as run_build_summaries

__all__ = ["run_build_summaries", "build_summaries"]
