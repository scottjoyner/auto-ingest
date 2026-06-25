#!/usr/bin/env python3
"""Cluster transcriptions via KMeans + topic labels from Gemma -> Cluster nodes."""
import sys, os, json, time, requests, subprocess, csv, io
sys.path.insert(0, '/home/scott/git/auto-ingest')

ENDPOINT = "http://scotts-macbook-air.tailcb8954.ts.net:1234/v1"

def run_cypher(query):
    result = subprocess.run(['docker', 'exec', '-i', 'neo4j', 
        'cypher-shell', '-u', 'neo4j', '-p', 'knowledge_graph_2026'],
        input=query.encode(), capture_output=True, text=True)
    if result.returncode != 0:
        print("CYPHER ERROR:", query[:80], "...", result.stderr[:300])
        return None
    reader = csv.reader(io.StringIO(result.stdout))
    rows = list(reader)
    if not rows or len(rows) < 2:
        return None
    headers = [h.strip() for h in rows[0]]
    data_rows = []
    for row in rows[1:]:
        row = [x.strip() for x in row]
        record = {}
        for i, h in enumerate(headers):
            record[h] = row[i] if i < len(row) else None
        data_rows.append(record)
    return data_rows

print("Phase 1: Counting...")
r = run_cypher('''MATCH (t:Transcription)-[:HAS_SUMMARY]->(s) 
                 WHERE size(s.text)>10 AND s.embedding IS NOT NULL RETURN count(DISTINCT t)''')
total_embedded = r[0]['count(DISTINCT t)'] if r else 0
print("  {} transcriptions with embedded summaries".format(total_embedded))

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

# Try to get embeddings via APOC's apoc.meta.data or explicit property access  
r2 = run_cypher('''MATCH (s:Summary) WHERE size(s.text)>10 AND s.embedding IS NOT NULL 
                  RETURN collect([head(keys(s)), head([y IN [s] | y[\"embedding\"]])]) LIMIT 3''')
print("Test:", r2[:3] if r2 else "None")
