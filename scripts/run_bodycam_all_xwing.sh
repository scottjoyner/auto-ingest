#!/bin/bash
set -u
LOG=/tmp/xwing_bodycam_all.log
echo "$(date) xwing bodycam ALL labels catch-up (batch 256) -> deathstar (parallel with x1-370 whisper)" >>"$LOG"
# wait for xwing's research-Chunk embedding to finish before touching deathstar
while pgrep -f "reembed[.]py Chunk" >/dev/null 2>&1; do sleep 30; done
cd /home/scott/embed_xwing
source /home/scott/venv-embed/bin/activate
export PATH=/opt/rocm-7.0.0/core-7.14/bin:$PATH
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
ssh -o StrictHostKeyChecking=no -N -L 17687:127.0.0.1:7687 deathstar@192.168.1.128 >>"$LOG" 2>&1 </dev/null &
TUN=$!
for i in $(seq 1 60); do
  if python3 -c "import socket,sys; socket.create_connection(('127.0.0.1',17687),2); sys.exit(0)" 2>/dev/null; then break; fi
  sleep 2
done
export NEO4J_URI=bolt://127.0.0.1:17687
# taxonomy-ish labels: one sweep (not whisper-produced)
for L in Entity Speaker Keyword Concept Topic KgNode GlobalSpeaker Note; do
  echo "$(date) === $L (emb_e5_large, batch 256) ===" >>"$LOG"
  python3 reembed.py "$L" --model intfloat/multilingual-e5-large --prop emb_e5_large --batch-size 64 --resume >>"$LOG" 2>&1
done
# whisper/summarize-produced text: catchup (loop until no new nodes) so the
# backlog cannot grow while whisper + the summarize job keep creating nodes.
# NOTE: batch 64 (not 256) — xwing's 11.6GB GPU OOMs at 256 on large labels.
for L in Summary Segment Utterance Transcription; do
  echo "$(date) === $L (emb_e5_large, batch 64, catchup) ===" >>"$LOG"
  python3 reembed.py "$L" --model intfloat/multilingual-e5-large --prop emb_e5_large --batch-size 64 --resume --catchup >>"$LOG" 2>&1
done
kill $TUN 2>/dev/null
echo "$(date) xwing bodycam ALL done" >>"$LOG"
