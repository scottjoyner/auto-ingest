#!/usr/bin/env python3
"""Cluster transcriptions via KMeans + topic labels -> Cluster nodes in Neo4j."""
import sys, os, json, time, requests, subprocess, csv, io
sys.path.insert(0, '/home/scott/git/auto-ingest')

ENDPOINT = "http://scotts-macbook-air.tailcb8954.ts.net:1234/v1"
BATCH = 2000
try:
    from auto_ingest_config import get_neo4j_password
    _NEO4J_PW = get_neo4j_password()
except Exception:
    _NEO4J_PW = os.environ.get("NEO4J_PASSWORD") or os.environ.get("NEO4J_PASSWORD_DEFAULT") or "knowledge_graph_2026"

def run_cypher(query):
    """Run cypher-shell via docker exec, return parsed rows."""
    result = subprocess.run(['docker', 'exec', '-i', 'neo4j', 
        'cypher-shell', '--format=plain', '-u', 'neo4j', '-p', _NEO4J_PW],
        input=query, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    lines = result.stdout.strip().split('\n')
    if not lines or len(lines) < 2:
        return None
    headers = [h.strip() for h in lines[0].strip().split('\t')]
    data_rows = []
    for line in lines[1:]:
        row = [x.strip() for x in line.split('\t')]
        record = {}
        for i, h in enumerate(headers):
            record[h] = row[i] if i < len(row) else None
        data_rows.append(record)
    return data_rows

print("Phase 1: Counting...")
r = run_cypher('''MATCH (t:Transcription)-[:HAS_SUMMARY]->(s) 
                 WHERE size(s.text)>10 AND s.embedding IS NOT NULL RETURN count(DISTINCT t)''')
total_embedded = int(r[0]['count(DISTINCT t)']) if r else 0
print("  {} transcriptions with embedded summaries".format(total_embedded))

import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import silhouette_score

# Fetch all summary IDs  
print("Phase 2: Loading summary IDs...")
all_ids = []
for batch in range(0, total_embedded + BATCH, BATCH):
    r = run_cypher('''MATCH (t:Transcription)-[:HAS_SUMMARY]->(s) 
                     WHERE size(s.text)>10 AND s.embedding IS NOT NULL 
                     SKIP {} LIMIT {} RETURN s.id AS summary_id'''.format(batch, BATCH))
    if not r:
        continue
    for rec in r:
        all_ids.append(rec['summary_id'])

print("  Loaded {} summaries".format(len(all_ids)))

# Fetch embeddings — cypher-shell plain format returns tab-separated values  
# The embedding field comes as a string like [-0.039, -0.066, ...] that we parse with ast.literal_eval
print("Phase 2b: Fetching embedding arrays...")
embedding_arrays = []

for j, sid in enumerate(all_ids):
    query = "MATCH (s:Summary {{id:'{}'}}) RETURN s.embedding".format(sid.replace("'", "\\'"))
    result = subprocess.run(['docker', 'exec', '-i', 'neo4j', 
        'cypher-shell', '--format=plain', '-u', 'neo4j', '-p', _NEO4J_PW],
        input=query, capture_output=True, text=True)
    
    if result.returncode != 0:
        continue
    
    lines = result.stdout.strip().split('\n')
    if len(lines) < 2 or not lines[1].strip():
        continue
    
    parts = [p.strip() for p in lines[1].split('\t')]
    
    try:
        import ast
        emb_str = parts[0] if parts else ''
        if emb_str and emb_str != 'null':
            emb_data = list(ast.literal_eval(emb_str))
            embedding_arrays.append(np.array(emb_data, dtype=np.float32))
        
        if j % 500 == 0:
            print("  Progress: {}/{}".format(j+1, len(all_ids)))
    except Exception as e:
        pass

print("\nPhase 3: Loaded {} embeddings of dim {}".format(
    len(embedding_arrays), embedding_arrays[0].shape if embedding_arrays else 'N/A'))

# Run KMeans  
if embedding_arrays:
    X = np.array(embedding_arrays)
    
    print("\nPhase 4: Finding optimal k...")
    sil_scores = {}
    for k in [50, 75, 100]:
        km = MiniBatchKMeans(n_clusters=k, random_state=42).fit(X[:min(5000, len(all_ids))])
        X_tfm = km.transform(X[:min(5000, len(all_ids))])
        sil = silhouette_score(X_tfm, km.labels_[:min(5000, len(all_ids))])
        sil_scores[k] = round(sil, 4)
    
    print("  Silhouette scores:", sil_scores)
    best_k = max(sil_scores, key=sil_scores.get)
    print("  Best k={}".format(best_k))
    
    # Fit final model  
    km = MiniBatchKMeans(n_clusters=best_k, random_state=42).fit(X)
    labels = km.labels_
    
    # Assign clusters to summaries and store back in Neo4j
    print("\nPhase 5: Writing cluster assignments...")
    cluster_map = {}
    for i, sid in enumerate(all_ids):
        if labels[i] not in cluster_map:
            cluster_map[labels[i]] = []
        cluster_map[labels[i]].append(sid)
