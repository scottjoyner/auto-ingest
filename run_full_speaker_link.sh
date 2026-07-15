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
export NEO4J_PASSWORD="${NEO4J_PASSWORD:-knowledge_graph_2026}"
export NEO4J_DB="${NEO4J_DB:-neo4j}"

echo "=== pausing live ingest containers ==="
docker stop auto-ingest-service auto-ingest-worker 2>/dev/null || true

cleanup() {
  echo "=== restarting ingest containers ==="
  docker start auto-ingest-service auto-ingest-worker 2>/dev/null || true
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
