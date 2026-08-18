#!/bin/bash
set -u
LOG=/tmp/xwing_research_mini12.log
echo "$(date) xwing research Chunk emb_mini12 -> x1-370 research Neo4j (parallel, decoupled from whisper)" >>"$LOG"
cd /home/scott/embed_xwing
source /home/scott/venv-embed/bin/activate
export PATH=/opt/rocm-7.0.0/core-7.14/bin:$PATH
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export NEO4J_URI=bolt://100.64.43.123:7687
export NEO4J_PASSWORD=knowledge_graph_2026
python3 reembed.py Chunk --model sentence-transformers/all-MiniLM-L12-v2 --prop emb_mini12 --batch-size 512 --resume >>"$LOG" 2>&1
echo "$(date) xwing research emb_mini12 DONE" >>"$LOG"
