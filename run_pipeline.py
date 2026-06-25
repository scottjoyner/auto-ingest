#!/usr/bin/env python3
"""Run summarize_from_segments pipeline."""
import sys, os

sys.path.insert(0, '/home/scott/git/auto-ingest')
os.environ['NEO4J_PASSWORD'] = 'knowledge_graph_2026'
# LM Studio on Scott's MacBook Air — Gemma 4 e2b
os.environ['LLM_ENDPOINT'] = 'http://scotts-macbook-air.tailcb8954.ts.net:1234/v1'
os.environ['LLM_MODEL']      = 'google/gemma-4-e2b'

from summarize_from_segments import main
main()
