#!/bin/bash
# Full global speaker-linking pass — run in tmux/nohup on a real shell.
#
# Pauses the live ingest containers (frees NVMe I/O + avoids Neo4j write-lock
# contention), runs the complete linker over ALL qualifying speakers, then
# restarts the containers. Resumable: embeddings are cached in emb_cache.sqlite
# and the audio file index in audio_index.json, so re-runs skip recomputation.
set -uo pipefail
cd "$(dirname "$0")"

STATE_FILE="${STATE_FILE:-./linker_state.json}"
export NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
export NEO4J_USER="${NEO4J_USER:-neo4j}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:-${NEO4J_PASSWORD_DEFAULT:-knowledge_graph_2026}}"
export NEO4J_DB="${NEO4J_DB:-neo4j}"

# Pause the live ingest containers to avoid Neo4j write-lock contention / deadlock.
# Portable: only stop containers that actually exist on THIS host (so the script
# is safe to run on other boxes in the distributed fleet that may not run the
# auto-ingest-service/auto-ingest-worker containers — N8/K-N8).
INGEST_CONTAINERS="auto-ingest-service auto-ingest-worker"
present=""
if command -v docker >/dev/null 2>&1; then
  for c in $INGEST_CONTAINERS; do
    if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx "$c"; then
      present="$present $c"
    fi
  done
fi
if [ -n "$present" ]; then
  echo "=== pausing live ingest containers:$present ==="
  # shellcheck disable=SC2086
  docker stop $present 2>/dev/null || true
else
  echo "=== no ingest containers found on this host; skipping pause (portable mode) ==="
fi

cleanup() {
  if [ -n "$present" ]; then
    echo "=== restarting ingest containers:$present ==="
    # shellcheck disable=SC2086
    docker start $present 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "=== FULL global speaker-linking pass ==="
# No --max-speakers => process every qualifying speaker. --faiss widens nets.
# Add --rank-and-label ONLY as a final, separate pass (heavy full-graph query).
# --state-file records processed speaker IDs so a re-launch (after a crash or
# Ctrl-C) resumes where it left off instead of reprocessing the same speakers.
python3 -u bin/auto-ingest link-speakers --faiss --state-file "$STATE_FILE" "$@"

echo "=== anchoring Scott (idempotent) + optional final rank/label ==="
python3 -u bin/auto-ingest whoami --merge
echo "Full pass complete. Run 'python3 bin/auto-ingest status' to see coverage."
