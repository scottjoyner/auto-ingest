#!/usr/bin/env python3
"""Run summarize_from_segments pipeline with local defaults.

summarize_from_segments reads its own env (LLM_ENDPOINT/LLM_MODEL/OLLAMA_*,
NEO4J_*) with sensible local defaults, so this wrapper only needs to delegate.
Override via environment when running against a remote LLM.
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Resolve the Neo4j password through the shared config loader; keep the
# historical NEO4J_PASSWORD_DEFAULT fallback for parity.
try:
    from auto_ingest_config import get_neo4j_password
    os.environ.setdefault('NEO4J_PASSWORD', get_neo4j_password())
except Exception:
    os.environ.setdefault('NEO4J_PASSWORD',
                          os.environ.get('NEO4J_PASSWORD_DEFAULT', 'knowledge_graph_2026'))

from summarize_from_segments import main
main()