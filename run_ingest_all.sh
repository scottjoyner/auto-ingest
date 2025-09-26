#!/usr/bin/env bash
set -euo pipefail

# === Edit these paths if needed ===
VENV="${VENV:-./.venv}"
PY="${PY:-$VENV/bin/python3}"
SCRIPT="${SCRIPT:-./ingest_transcriptsv5_3.py}"

# Where logs go
LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "$LOG_DIR"
STAMP="$(date +'%Y%m%d_%H%M%S')"
LOG_FILE="$LOG_DIR/ingest_$STAMP.log"

# Prevent accidental concurrent runs
LOCK_FILE="${LOCK_FILE:-/tmp/ingest_transcripts.lock}"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another ingest is running (lock: $LOCK_FILE). Exiting."
  exit 1
fi

# Be gentle on the box
IONICE="${IONICE:-ionice -c2 -n7}"   # best-effort, lowest prio
NICE="${NICE:-nice -n 10}"           # lower CPU priority

# ========== Environment (override with env vars if you want) ==========
# Where to scan for transcripts & media
export SCAN_ROOTS="${SCAN_ROOTS:-/mnt/8TB_2025/fileserver/dashcam/audio,/mnt/8TB_2025/fileserver/dashcam/transcriptions,/mnt/8TB_2025/fileserver/audio,/mnt/8TB_2025/fileserver/audio/transcriptions,/mnt/8TB_2025/fileserver/bodycam,/mnt/8TB_2025/fileserver/dashcam}"

# Where to aggressively look for *_metadata.csv
export DASHCAM_ROOT="${DASHCAM_ROOT:-/mnt/8TB_2025/fileserver/dashcam}"

# Timezone for key→absolute timestamps
export LOCAL_TZ="${LOCAL_TZ:-America/New_York}"

# Embedding model (384-dim st-only; fast & light)
export EMBED_MODEL_NAME="${EMBED_MODEL_NAME:-sentence-transformers/all-MiniLM-L6-v2}"
export EMBED_DIM="${EMBED_DIM:-384}"
export EMBED_BATCH="${EMBED_BATCH:-32}"  # tokenizer batch size for GPU/CPU

# Neo4j connection (edit if your credentials differ)
export NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
export NEO4J_USER="${NEO4J_USER:-neo4j}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:-livelongandprosper}"
export NEO4J_DB="${NEO4J_DB:-neo4j}"

# Model preference ordering (optional override)
export MODEL_PREF="${MODEL_PREF:-large-v3,large-v2,large,turbo,medium.en,medium,small.en,small,base.en,base,tiny.en,tiny,faster-whisper:large-v3,faster-whisper:large-v2,faster-whisper:large,faster-whisper:medium,faster-whisper:small,faster-whisper:base,faster-whisper:tiny}"

# ========== Dashcam metadata quality defaults (tweak if you move) ==========
# BBox for the CLT area (approx); used for auto-fix and gating
export GEO_BBOX="${GEO_BBOX:-33.0,38.5,-83.0,-70.0}"
export META_FPS="${META_FPS:-30}"
export META_DOWNSAMPLE_SEC="${META_DOWNSAMPLE_SEC:-1}"
export META_MAX_SPEED_MPH="${META_MAX_SPEED_MPH:-120}"
export META_MIN_KEEP_RATIO="${META_MIN_KEEP_RATIO:-0.6}"  # require at least 60% good points

# ========== Neo4j transaction batching (keeps memory in check) ==========
export TX_SEG_BATCH="${TX_SEG_BATCH:-120}"
export TX_UTT_BATCH="${TX_UTT_BATCH:-120}"
export TX_EDGE_BATCH="${TX_EDGE_BATCH:-300}"
export TX_ENT_BATCH="${TX_ENT_BATCH:-300}"
export TX_LOC_BATCH="${TX_LOC_BATCH:-500}"
export TX_TIMEOUT_SEC="${TX_TIMEOUT_SEC:-120}"
export FETCH_SIZE="${FETCH_SIZE:-100}"

# Logging noise control
export LOG_LEVEL="${LOG_LEVEL:-INFO}"

# Optional: limit how many keys you process in one go (0 = all)
LIMIT="${LIMIT:-0}"

# Optional: set DRY_RUN=1 to test without writing
DRY_RUN="${DRY_RUN:-0}"

# Optional: set FORCE=1 to reingest everything regardless of existing nodes
FORCE="${FORCE:-0}"

# Compose the command
CMD=( "$PY" "$SCRIPT"
  --tx-seg-batch "$TX_SEG_BATCH"
  --tx-utt-batch "$TX_UTT_BATCH"
  --tx-edge-batch "$TX_EDGE_BATCH"
  --tx-ent-batch "$TX_ENT_BATCH"
  --tx-loc-batch "$TX_LOC_BATCH"
  --tx-timeout-sec "$TX_TIMEOUT_SEC"
  --fetch-size "$FETCH_SIZE"
  --ingest-dashcam-meta
  --lon-auto-west
  --allow-latlon-swap
  --geo-bbox "$GEO_BBOX"
  --meta-fps "$META_FPS"
  --meta-downsample-sec "$META_DOWNSAMPLE_SEC"
  --meta-max-speed-mph "$META_MAX_SPEED_MPH"
  --meta-min-keep-ratio "$META_MIN_KEEP_RATIO"
  --meta-skip-when-bad
  --log-level "$LOG_LEVEL"
)

# Limits/flags
if [[ "$LIMIT" -gt 0 ]]; then
  CMD+=( --limit "$LIMIT" )
fi
if [[ "$DRY_RUN" == "1" ]]; then
  CMD+=( --dry-run )
fi
if [[ "$FORCE" == "1" ]]; then
  CMD+=( --force )
fi

# Make sure the venv + script exist
if [[ ! -x "$PY" ]]; then
  echo "Python not found at $PY. Set VENV or PY env var correctly." >&2
  exit 2
fi
if [[ ! -f "$SCRIPT" ]]; then
  echo "Script not found: $SCRIPT" >&2
  exit 2
fi

echo "==== Ingest starting @ $(date) ===="
echo "Logging to: $LOG_FILE"
echo "Running: ${CMD[*]}"
echo

# Nicer I/O scheduling + CPU niceness, continuous logging
# Remove 'nohup' if you prefer foreground only.
$IONICE $NICE nohup "${CMD[@]}" \
  > >(tee -a "$LOG_FILE") \
  2> >(tee -a "$LOG_FILE" >&2)

echo
echo "==== Ingest finished @ $(date) ===="
echo "Log: $LOG_FILE"
