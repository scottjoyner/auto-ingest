#!/bin/bash
# run_pipeline.sh — the EASY one-shot: a clip -> a finished iPhone TikTok short.
#
# Usage:
#   ./run_pipeline.sh                         # picks the newest dashcam _FR clip
#   ./run_pipeline.sh /path/to/clip_FR.MP4    # specific clip
#   ./run_pipeline.sh /path/to/clip.mp4 "POV: you live in the future"   # with hook
#
# Output lands at $OUT_ROOT/<date>/<clip>_tok.mp4 (default
# /media/scott/SSD_4TB/tiktok_shorts/<date>/...). The short pulls its captions
# from Neo4j automatically and applies the TikTok-style framing.
set -uo pipefail
cd "$(dirname "$0")"

export NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
export NEO4J_USER="${NEO4J_USER:-neo4j}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:-knowledge_graph_2026}"
export NEO4J_DB="${NEO4J_DB:-neo4j}"

OUT_ROOT="${OUT_ROOT:-/media/scott/SSD_4TB/tiktok_shorts}"
CLIP="${1:-}"
HOOK="${2:-}"
MUSIC="${MUSIC:-}"

if [ -z "$CLIP" ]; then
  CLIP=$(find /media/scott/SSD_4TB/fileserver/dashcam -maxdepth 4 -iname "*_FR.MP4" \
          -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
  if [ -z "$CLIP" ]; then
    echo "ERROR: no clip found and none provided." >&2
    exit 2
  fi
  echo "Auto-picked newest clip: $CLIP"
fi

if [ ! -f "$CLIP" ]; then
  echo "ERROR: clip not found: $CLIP" >&2
  exit 2
fi

DATE=$(echo "$CLIP" | grep -oE "[0-9]{4}[_/-][0-9]{2}[_/-][0-9]{2}" | head -1 | tr '_' '-')
DATE="${DATE:-today}"
OUT_DIR="$OUT_ROOT/$DATE"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/$(basename "${CLIP%.*}")_tok.mp4"

echo "=== pipeline: $CLIP -> $OUT ==="
python3 bin/auto-ingest tiktok --clip "$CLIP" --out "$OUT" ${HOOK:+--hook "$HOOK"} ${MUSIC:+--music "$MUSIC"}
rc=$?
[ $rc -eq 0 ] && echo "=== done: $OUT ==="
exit $rc
