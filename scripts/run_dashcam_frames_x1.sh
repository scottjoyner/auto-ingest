#!/bin/bash
# run_dashcam_frames_x1.sh  (x1-370)
#
# QUEUED AFTER the embedding chain AND the whisper/diarize catchup.
# Stage 1: run dashcam_frame_vision.py on deathstar (extract the FIRST frame of
#   each minute per clip, send to the MacBook Air vision LLM, store DashcamFrame
#   nodes in the bodycam KG). Resumable — re-runs skip done (key, minute) pairs.
# Stage 2: embed the new DashcamFrame descriptions into emb_e5_large on x1-370.
#
# The watcher is paused during stage 2 (GPU) and restarted at the end.
set -u
LOG=/tmp/x1_dashcam_frames.log
echo "$(date) dashcam-frame batch launcher start (gates on chain + whisper/diarize catchup)" >>"$LOG"

while pgrep -f "run_bodycam_x1.sh" >/dev/null || \
      pgrep -f "run_research_2nd_x1.sh" >/dev/null || \
      pgrep -f "run_entity_x1.sh" >/dev/null || \
      pgrep -f "run_bodycam_2nd_x1.sh" >/dev/null || \
      pgrep -f "run_backfill_x1.sh" >/dev/null || \
      pgrep -f "run_whisper_diarize_catchup.sh" >/dev/null; do
  echo "$(date) waiting for prior stages..." >>"$LOG"
  sleep 60
done
echo "$(date) prior stages done; starting dashcam frame batch" >>"$LOG"

# Stage 1: extract + vision + store (deathstar). Detached so it survives; poll for completion.
echo "$(date) === stage1: extract+vision+store (deathstar) ===" >>"$LOG"
ssh -o ConnectTimeout=60 -o ServerAliveInterval=60 deathstar@192.168.1.128 \
  "setsid bash -c 'cd /home/deathstar/git/auto-ingest && .venv/bin/python dashcam_frame_vision.py --sleep 0.3 >>/tmp/deathstar_dashcam_frames.log 2>&1' </dev/null >/dev/null 2>&1 &"
while ssh -o ConnectTimeout=30 deathstar@192.168.1.128 "pgrep -f '[d]ashcam_frame_vision.py' >/dev/null"; do
  echo "$(date) frame pipeline still running..." >>"$LOG"
  sleep 120
done
echo "$(date) stage1 done" >>"$LOG"

# Stage 2: embed DashcamFrame descriptions into emb_e5_large (x1-370 GPU). Pause watcher.
pkill -f "run_watch_x1.sh" 2>/dev/null || true
pkill -f "[w]atch_embed.py" 2>/dev/null || true
pkill -f "[r]eembed[.]py" 2>/dev/null || true
pkill -f "[-]L 17687" 2>/dev/null || true
sleep 10
echo "$(date) === stage2: embed DashcamFrame e5 (x1-370) ===" >>"$LOG"
cd /home/scott/embed_x1
VENV=/home/scott/venv-rocm/bin/python3
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=10 -N -L 17687:127.0.0.1:7687 deathstar@192.168.1.128 >>"$LOG" 2>&1 </dev/null &
TUN=$!
for i in $(seq 1 80); do (echo > /dev/tcp/127.0.0.1/17687) 2>/dev/null && break; sleep 1; done
export NEO4J_URI=bolt://127.0.0.1:17687
"$VENV" reembed.py DashcamFrame --model intfloat/multilingual-e5-large --prop emb_e5_large --batch-size 64 --catchup >>"$LOG" 2>&1
kill $TUN 2>/dev/null
echo "$(date) stage2 done" >>"$LOG"

# Restart watcher (now also embeds DashcamFrame)
echo "$(date) restarting watcher" >>"$LOG"
setsid bash -c 'cd /home/scott/embed_x1 && nohup bash run_watch_x1.sh >>/tmp/x1_watch_embed.log 2>&1 &' </dev/null >/dev/null 2>&1 &
echo "$(date) dashcam-frame batch ALL DONE" >>"$LOG"
