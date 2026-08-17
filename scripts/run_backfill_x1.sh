#!/bin/bash
set -u
LOG=/tmp/x1_backfill.log
echo "$(date) backfill embed_hash launcher start (waits for entity job)" >>"$LOG"

while pgrep -f "run_entity_x1.sh" >/dev/null; do
  echo "$(date) waiting for entity embed job..." >>"$LOG"
  sleep 60
done
echo "$(date) entity done; backfilling embed_hash on both KGs (no re-embed)" >>"$LOG"

cd /home/scott/embed_x1
VENV=/home/scott/venv-rocm/bin/python3
export CUDA_VISIBLE_DEVICES=0
export WATCH_EMBED_PY="$VENV"
BODYCAM="Segment Utterance Transcription Summary Entity Concept Topic Keyword KgNode Note Speaker GlobalSpeaker"
RESEARCH="Chunk"

# bodycam KG (deathstar) via tunnel
ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=10 -N -L 17687:127.0.0.1:7687 deathstar@192.168.1.128 >>"$LOG" 2>&1 </dev/null &
TUN=$!
for i in $(seq 1 80); do (echo > /dev/tcp/127.0.0.1/17687) 2>/dev/null && break; sleep 1; done
export NEO4J_URI=bolt://127.0.0.1:17687
for L in $BODYCAM; do
  echo "$(date) backfill $L emb_e5_large" >>"$LOG"
  "$VENV" reembed.py "$L" --prop emb_e5_large --backfill-hash --batch-size 500 >>"$LOG" 2>&1
done
kill $TUN 2>/dev/null

# research KG (x1-370 local)
export NEO4J_URI=bolt://100.64.43.123:7687
for L in $RESEARCH; do
  echo "$(date) backfill $L emb_e5_large" >>"$LOG"
  "$VENV" reembed.py "$L" --prop emb_e5_large --backfill-hash --batch-size 500 >>"$LOG" 2>&1
  echo "$(date) backfill $L emb_mini12" >>"$LOG"
  "$VENV" reembed.py "$L" --prop emb_mini12 --backfill-hash --batch-size 500 >>"$LOG" 2>&1
done
echo "$(date) backfill DONE (safe to enable --stale in watcher now)" >>"$LOG"
