#!/bin/bash
# Compress dashcam footage (dashcam _F/_R/_FR .MP4) into a mirrored portable tree
# on NAS5 at /fileserver/dashcam/compressed/YYYY/MM/DD/<file>.mp4
#
# Reads source from the fileserver dashcam root (SSD_4TB/fileserver -> NAS5) and
# writes HEVC (libx265) re-encodes. Resumable: existing/verified outputs are
# skipped, so re-launching after a crash/cont/Ctrl-C continues where it left off.
#
# Run in tmux/nohup on a real shell — this is a long job (tens of thousands of files).
set -uo pipefail
cd "$(dirname "$0")"

export NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
export NEO4J_USER="${NEO4J_USER:-neo4j}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:-knowledge_graph_2026}"
export NEO4J_DB="${NEO4J_DB:-neo4j}"

INPUT_ROOT="${INPUT_ROOT:-/media/scott/SSD_4TB/fileserver/dashcam}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/scott/NAS5/fileserver/dashcam/compressed}"
WORKERS="${WORKERS:-$(nproc)}"
CRF="${CRF:-26}"
LIMIT="${LIMIT:-0}"   # 0 = no limit (process everything)

echo "=== dashcam compression ==="
echo "input : $INPUT_ROOT"
echo "output: $OUTPUT_ROOT"
echo "workers: $WORKERS  crf: $CRF  limit: $LIMIT"

python3 -u compress_dashcam2.py \
  --input-root "$INPUT_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --workers "$WORKERS" \
  --crf "$CRF" \
  --max-width 1280 \
  --fps 30 \
  --audio-k 96 \
  --order newest \
  ${LIMIT:+--limit "$LIMIT"} \
  "$@"

echo "Compression pass complete. Output mirrored at: $OUTPUT_ROOT"
