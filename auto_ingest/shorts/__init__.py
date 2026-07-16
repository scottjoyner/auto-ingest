"""auto_ingest.shorts — research-scripted vertical shorts from the knowledge graph.

Turn a Neo4j research Topic (papers, concepts, chunks) into narrated 9:16 shorts
using generic highway dashcam footage as the visual backdrop. The pipeline is
plan -> render -> iterate: a Plan JSON on disk is the iteration artifact, and a
:ShortPlan / :Short manifest is upserted into Neo4j.

Typical use (from the moviepy/whisper image):

    python -m auto_ingest.shorts.cli plan large_language_models --shorts 3
    python -m auto_ingest.shorts.cli render shorts_plans/large_language_models__XXXX.json
    python -m auto_ingest.shorts.cli iterate shorts_plans/large_language_models__XXXX.json --seed 2

Or via the CLI: ``auto-ingest shorts plan <topic>``.
"""
from __future__ import annotations

__all__ = ["curator", "backdrop", "planner", "render", "models", "cli"]
