#!/bin/bash
set -euo pipefail

export NEO4J_URI=bolt://100.64.43.123:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=*** NEO4J_DB=neo4j

echo "=== Env vars ==="
echo "NEO4J_URI=$NEO4J_URI"
echo "NEO4J_USER=$NEO4J_USER"
echo "NEO4J_PASSWORD=*** (len=${#NEO4J_PASSWORD})"
echo "NEO4J_DB=$NEO4J_DB"

cd ~/git/auto-ingest
python3 -u link_global_speakers_2.py \
    --max-speakers 5 \
    --dry-run \
    --global-prefilter \
    --faiss-prefilter \
    2>&1 | head -80
