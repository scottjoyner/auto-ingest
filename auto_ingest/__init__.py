"""auto_ingest - unified packaging for the auto-ingest repo.

This package promotes the previously-flat script dump (LLD §3.4 / W-42) into
importable subpackages:

  auto_ingest.ingest   - transcript / media ingestion into Neo4j
  auto_ingest.diarize  - global speaker linking (no voice auth; Sophia owns auth)
  auto_ingest.dashcam  - dashcam YOLO + embedding ingestion
  auto_ingest.content  - content/summary generation (Ollama)

The heavy subpackages (ingest/diarize/dashcam/content) import ML stacks
(torch, pandas, transformers, ...) at module load, so they are NOT imported
eagerly here. Import them explicitly only when you need them, e.g.::

    from auto_ingest.diarize import link_global_speakers

Shared, dependency-free helpers (config shim, events envelope, outbox) ARE
safe to import anywhere:

  from auto_ingest.outbox import GraphOutbox
  from auto_ingest.events import emit

Shared config (Neo4j, Nextcloud, storage roots) lives in the repo-root module
``auto_ingest_config`` and is imported directly by the subpackages. Cross-repo
events are mirrored locally via ``auto_ingest.events`` (a shim over the shared
contract envelope) until the swarm-contracts package is wired in.
"""

from __future__ import annotations

__all__ = ["ingest", "diarize", "dashcam", "content", "outbox", "events"]
