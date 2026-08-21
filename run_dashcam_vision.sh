#!/usr/bin/env bash
# Resilient launcher for the dashcam vision pipeline.
# - Waits for the MacBook Air vision endpoint (100.85.64.117) before doing work.
# - Processes BOTH dashcam archives (8TB_2025 + 8TBHDD).
# - Re-runs continuously so newly-added clips get picked up.
# Intended to be run by a systemd service (Restart=always) for auto-resume on boot.
set -u
BASES="/mnt/8TB_2025/fileserver/dashcam /mnt/8TBHDD/fileserver/dashcam"
VISION_HOST="100.85.64.117:1234"
SCRIPT="/home/deathstar/git/auto-ingest/dashcam_frame_vision.py"
WORKERS="${DASHCAM_WORKERS:-2}"
MAX_MIN="${DASHCAM_MAX_MIN:-5}"
NEO4J_URI="${DASHCAM_NEO4J_URI:-bolt://localhost:7687}"
NEO4J_PWD="${NEO4J_PASSWORD:-knowledge_graph_2026}"

macbook_up() { curl -s --max-time 5 "http://${VISION_HOST}/v1/models" >/dev/null 2>&1; }
neo4j_up()   { curl -s --max-time 5 "http://localhost:7474" >/dev/null 2>&1; }

wait_for() {  # $1=descr $2=fn
  local i=0
  until "$2"; do
    echo "[wrapper $(date -Is)] waiting for $1 (try $((++i)))"
    sleep 15
  done
  echo "[wrapper $(date -Is)] $1 is up"
}

while true; do
  wait_for "neo4j" neo4j_up
  wait_for "macbookair vision endpoint" macbook_up
  for B in $BASES; do
    if [ -d "$B" ]; then
      echo "[wrapper $(date -Is)] === orchestrate $B (workers=$WORKERS) ==="
      python3 "$SCRIPT" --base "$B" --orchestrate --max-minutes "$MAX_MIN" \
        --view both --sleep 0.1 --timeout 120 --clip-timeout 500 \
        --vision-retries 4 --workers "$WORKERS" \
        --neo4j-uri "$NEO4J_URI" --neo4j-password "$NEO4J_PWD" || \
        echo "[wrapper $(date -Is)] orchestrate $B exited with error"
      # self-heal: retry clips that failed to decode (re-mux / re-encode)
      echo "[wrapper $(date -Is)] === salvage failed clips for $B ==="
      python3 /home/deathstar/git/auto-ingest/dashcam_salvage.py --base "$B" \
        --neo4j-uri "$NEO4J_URI" --neo4j-password "$NEO4J_PWD" || \
        echo "[wrapper $(date -Is)] salvage $B exited with error"
    else
      echo "[wrapper $(date -Is)] base missing: $B"
    fi
  done
  echo "[wrapper $(date -Is)] cycle complete; sleeping 300s before re-scan"
  sleep 300
done
