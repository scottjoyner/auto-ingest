#!/usr/bin/env python3
"""Run link_global_speakers_2.py with correct password and report results."""
import subprocess, sys

# Run the linker with all flags
result = subprocess.run(
    [sys.executable, "-u", "link_global_speakers_2.py",
     "--max-speakers", "50",  # process more speakers for full coverage
     "--dry-run",
     "--global-prefilter",
     "--faiss-prefilter"],
    capture_output=True, text=True, cwd="/home/scott/git/auto-ingest",
    env={**__import__("os").environ, "NEO4J_URI": "bolt://100.64.43.123:7687",
         "NEO4J_USER": "neo4j", "NEO4J_PASSWORD": "knowledge_graph_2026",
         "NEO4J_DB": "neo4j"}
)

print("=== STDOUT ===")
for line in result.stdout.split("\n"):
    print(line)

if result.stderr:
    print("\n=== STDERR (last 30 lines) ===")
    for line in result.stderr.strip().split("\n")[-30:]:
        print(line)

print(f"\nExit code: {result.returncode}")
