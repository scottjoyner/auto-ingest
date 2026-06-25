#!/bin/bash
set -euo pipefail

cd ~/git/auto-ingest || { echo "Failed to cd"; exit 1; }

# Set all Neo4j env vars explicitly
export NEO4J_URI=bolt://100.64.43.123:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=knowledge_graph_2026
export NEO4J_DB=neo4j

echo "=== Running link_global_speakers_2.py ==="
echo "NEO4J_URI=$NEO4J_URI"
echo "NEO4J_USER=$NEO4J_USER"
echo "NEO4J_PASSWORD=knowledge_graph_2026"
echo "NEO4J_DB=$NEO4J_DB"

python3 link_global_speakers_2.py \
    --global-prefilter \
    --global-thresh 0.78 \
    --global-k 8 \
    --skip-already-linked \
    --faiss-prefilter \
    --source-level auto \
    --min-seg 0.7 \
    --max-snips 8 \
    --max-per-file 3 \
    --snip-len 1.6 \
    --thresh 0.72 \
    --holdout \
    --holdout-min 0.62 \
    --holdout-action drop-members \
    --priority-name "Scott|Kipnerter" \
    --rank-and-label \
    --emb-cache ./emb_cache.sqlite \
    "$@"

echo "=== Done ==="
