#!/bin/bash
set -u
LOG=/tmp/x1_watch_embed.log
echo "$(date) watch_embed launcher start (waits for entity embed job)" >>"$LOG"

# Wait for the entity-node embed launcher to fully exit
while pgrep -f "run_entity_x1.sh" >/dev/null; do
  echo "$(date) waiting for entity embed job..." >>"$LOG"
  sleep 60
done
echo "$(date) entity done; starting incremental embed watcher" >>"$LOG"

cd /home/scott/embed_x1
VENV=/home/scott/venv-rocm/bin/python3
export CUDA_VISIBLE_DEVICES=-1
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:256
export WATCH_EMBED_PY="$VENV"
INTERVAL=900
BODYCAM="Segment Utterance Transcription Summary Entity Concept Topic Keyword KgNode Note Speaker GlobalSpeaker DashcamFrame"
RESEARCH="Chunk"

while true; do
  # --- bodycam KG on deathstar (localhost-only there) via tunnel ---
  ssh -o ConnectTimeout=10 -o ServerAliveInterval=60 -o ServerAliveCountMax=10 -N -L 17687:127.0.0.1:7687 deathstar@100.78.106.121 >>"$LOG" 2>&1 </dev/null &
  TUN=$!
  for i in $(seq 1 80); do (echo > /dev/tcp/127.0.0.1/17687) 2>/dev/null && break; sleep 1; done
  export NEO4J_URI=bolt://127.0.0.1:17687
  echo "$(date) === watch bodycam (deathstar) ===" >>"$LOG"
  "$VENV" watch_embed.py --once --batch-size 256 --labels $BODYCAM >>"$LOG" 2>&1
  kill $TUN 2>/dev/null

  # --- research KG on x1-370 (local) ---
  export NEO4J_URI=bolt://100.64.43.123:7687
  echo "$(date) === watch research (x1-370) ===" >>"$LOG"
  "$VENV" watch_embed.py --once --batch-size 512 --labels $RESEARCH >>"$LOG" 2>&1

  echo "$(date) full sweep complete; sleeping $INTERVAL" >>"$LOG"
  sleep $INTERVAL
done
