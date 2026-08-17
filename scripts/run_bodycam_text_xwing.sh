#!/bin/bash
set -u
LOG=/tmp/xwing_bodycam_text.log
echo "$(date) xwing bodycam TEXT catch-up (Segment/Utterance/Transcription) -> deathstar (parallel with x1-370 whisper)" >>"$LOG"
cd /home/scott/embed_xwing
export PATH=/opt/rocm-7.0.0/core-7.14/bin:$PATH
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
VENV=/home/scott/venv-embed/bin/python3

ssh -o ServerAliveInterval=60 -N -L 17687:127.0.0.1:7687 deathstar@192.168.1.128 >>"$LOG" 2>&1 </dev/null &
TUN=$!
for i in $(seq 1 80); do (echo > /dev/tcp/127.0.0.1/17687) 2>/dev/null && break; sleep 1; done
export NEO4J_URI=bolt://127.0.0.1:17687

# --catchup loops (re-scanning from 0) until no new nodes appear, so it keeps
# absorbing the Transcription/Utterance/Segment nodes whisper is still creating.
for L in Segment Utterance Transcription; do
  echo "$(date) === $L (emb_e5_large, catchup) ===" >>"$LOG"
  "$VENV" reembed.py "$L" --model intfloat/multilingual-e5-large --prop emb_e5_large --batch-size 128 --resume --catchup >>"$LOG" 2>&1
done
kill $TUN 2>/dev/null
echo "$(date) xwing bodycam TEXT catch-up DONE" >>"$LOG"
