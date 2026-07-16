#!/usr/bin/env python3
"""Run the canonical global speaker linker with correct creds and report results.

Prefer `bin/auto-ingest link-speakers` for production runs; this is a thin
debug wrapper that execs the package linker the same way. Creds come from
NEO4J_* env (defaults: localhost:7687 / neo4j / knowledge_graph_2026 / neo4j).
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

env = {**os.environ,
       "NEO4J_URI": os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
       "NEO4J_USER": os.environ.get("NEO4J_USER", "neo4j"),
       "NEO4J_PASSWORD": os.environ.get("NEO4J_PASSWORD", "knowledge_graph_2026"),
       "NEO4J_DB": os.environ.get("NEO4J_DB", "neo4j")}

# Mirror bin/auto-ingest link-speakers: exec the package module as a module so
# the relative config import resolves from the repo root.
result = subprocess.run(
    [sys.executable, "-u", "-m", "auto_ingest.diarize.link_global_speakers",
     "--max-speakers", "50",  # process more speakers for full coverage
     "--global-prefilter",
     "--faiss-prefilter"],
    cwd=HERE, env=env,
)

print(f"\nExit code: {result.returncode}")
