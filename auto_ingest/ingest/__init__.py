"""Ingest subpackage: transcript + media ingestion into Neo4j."""

from __future__ import annotations

from .transcripts import main as run_transcripts

__all__ = ["run_transcripts", "transcripts"]
