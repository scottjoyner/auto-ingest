#!/usr/bin/env python3
"""Run link_global_speakers_2.py with correct creds and report results.

Prefer `bin/auto-ingest link-speakers` for production runs; this is a thin
debug wrapper. Creds come from NEO4J_* env (defaults: localhost:7687, db neo4j).
"""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Run the linker with all flags
result = subprocess.run(
    [sys.executable, "-u", "link_global_speakers_2.py",
     "--max-speakers", "50",  # process more speakers for full coverage
     "--global-prefilter",
     "--faiss-prefilter"],
    capture_output=True, text=True, cwd=HERE,
    env={**os.environ,
         "NEO4J_URI": os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
         "NEO4J_USER": os.environ.get("NEO4J_USER", "neo4j"),
         "NEO4J_PASSWORD": os.environ.get("NEO4J_PASSWORD", "knowledge_graph_2026"),
         "NEO4J_DB": os.environ.get("NEO4J_DB", "neo4j")}
)

print("=== STDOUT ===")
for line in result.stdout.split("\n"):
    print(line)

if result.stderr:
    print("\n=== STDERR (last 30 lines) ===")
    for line in result.stderr.strip().split("\n")[-30:]:
        print(line)

print(f"\nExit code: {result.returncode}")
