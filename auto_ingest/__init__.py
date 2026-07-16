"""auto_ingest - unified packaging for the auto-ingest repo.

This package promotes the previously-flat script dump (LLD §3.4 / W-42) into
importable subpackages:

  auto_ingest.ingest   - transcript / media ingestion into Neo4j
  auto_ingest.diarize  - global speaker linking (no voice auth; Sophia owns auth)
  auto_ingest.dashcam  - dashcam YOLO + embedding ingestion
  auto_ingest.content  - content/summary generation (Ollama)

Shared config (Neo4j, Nextcloud, storage roots) lives in the repo-root module
``auto_ingest_config`` and is imported directly by the subpackages. Cross-repo
events are mirrored locally via ``auto_ingest.events`` (a shim over the shared
contract envelope) until the swarm-contracts package is wired in.
"""

from __future__ import annotations

from .ingest import transcripts as ingest_transcripts
from .diarize import link_global_speakers
from .dashcam import yolo_embeddings
from .content import build_summaries

__all__ = [
    "ingest_transcripts",
    "link_global_speakers",
    "yolo_embeddings",
    "build_summaries",
]
