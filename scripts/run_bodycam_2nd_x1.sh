#!/bin/bash
set -u
LOG=/tmp/x1_bodycam_2nd.log
echo "$(date) bodycam 2nd-model (emb_mini12) launcher start (waits for entity job)" >>"$LOG"

while pgrep -f "run_entity_x1.sh" >/dev/null; do
  echo "$(date) waiting for entity embed job..." >>"$LOG"
  sleep 60
done
echo "$(date) entity done; starting bodycam 2nd-model pass (emb_mini12)" >>"$LOG"

cd /home/scott/embed_x1
VENV=/home/scott/venv-rocm/bin/python3
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export WATCH_EMBED_PY="$VENV"
LABELS="Segment Utterance Transcription Summary Entity Concept Topic Keyword KgNode Note Speaker GlobalSpeaker"

ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=10 -N -L 17687:127.0.0.1:7687 deathstar@192.168.1.128 >>"$LOG" 2>&1 </dev/null &
TUN=$!
for i in $(seq 1 80); do (echo > /dev/tcp/127.0.0.1/17687) 2>/dev/null && break; sleep 1; done
export NEO4J_URI=bolt://127.0.0.1:17687
for L in $LABELS; do
  echo "$(date) === 2nd-model $L (emb_mini12) ===" >>"$LOG"
  "$VENV" reembed.py "$L" --model sentence-transformers/all-MiniLM-L12-v2 --prop emb_mini12 --batch-size 512 --resume >>"$LOG" 2>&1
done
kill $TUN 2>/dev/null
echo "$(date) bodycam 2nd-model DONE" >>"$LOG"
