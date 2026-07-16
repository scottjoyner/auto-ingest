#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/backfill_audio_day.sh YYYY-MM-DD [--dry-run] [--force] [--limit N] [--print-roots]

Backfill exactly one calendar day of audio/transcription data from both NAS mounts.
This intentionally does NOT enable dashcam metadata ingestion; run dashcam separately.

Examples:
  scripts/backfill_audio_day.sh 2026-05-31 --dry-run --limit 5
  scripts/backfill_audio_day.sh 2026-05-31
  scripts/backfill_audio_day.sh 2026-05-01 --force --limit 100
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi
ORIG_ARGS=("$@")

DAY="$1"
shift
if [[ ! "$DAY" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  echo "DAY must be YYYY-MM-DD, got: $DAY" >&2
  exit 2
fi

DRY_RUN=0
FORCE=0
LIMIT=""
PRINT_ROOTS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --force) FORCE=1; shift ;;
    --limit)
      [[ $# -ge 2 ]] || { echo "--limit requires a number" >&2; exit 2; }
      LIMIT="$2"; shift 2 ;;
    --print-roots) PRINT_ROOTS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

YEAR="${DAY:0:4}"
MONTH="${DAY:5:2}"
DD="${DAY:8:2}"
REL="$YEAR/$MONTH/$DD"

cd "$(dirname "$0")/.."

# If running inside Compose, prefer the stable dual-NAS container mounts.
# If running on the host, resolve the fileserver root from config (mount-aware),
# falling back to the legacy NAS4 path for backward compatibility.
if [[ -d /nas1 || -d /nas2 ]]; then
  CANDIDATE_BASES=(/nas1 /nas2)
else
  CANDIDATE_BASES=(
    "$(python3 - <<'PY'
from auto_ingest_config import get_fileserver_root
print(get_fileserver_root())
PY
)"
    /media/scott/NAS4/fileserver
  )
fi

roots=()
for base in "${CANDIDATE_BASES[@]}"; do
  for sub in "audio/$REL" "audio/transcriptions/$REL"; do
    p="$base/$sub"
    [[ -d "$p" ]] && roots+=("$p")
  done
  # Some legacy audio files sit flat under audio/ with date-like keys. Include the
  # top-level audio directory only for dry-run root discovery is deliberately not
  # done here, because it would explode a one-day batch into a full-corpus scan.
done

if [[ ${#roots[@]} -eq 0 ]]; then
  echo "No audio/transcription day directories found for $DAY under NAS1 or NAS2." >&2
  echo "Checked relative path: audio/$REL and audio/transcriptions/$REL" >&2
  exit 1
fi

SCAN_ROOTS_JOINED="$(IFS=,; echo "${roots[*]}")"
export SCAN_ROOTS="$SCAN_ROOTS_JOINED"
export RTTM_DIRS="$SCAN_ROOTS_JOINED"
export DASHCAM_ROOT="/nonexistent/dashcam-disabled-for-audio-day-backfill"
export OLD_DASHCAM_ROOT="/nonexistent/dashcam-disabled-for-audio-day-backfill"
export LOCAL_TZ="${LOCAL_TZ:-America/New_York}"
export EMBED_MODEL_NAME="${EMBED_MODEL_NAME:-sentence-transformers/all-MiniLM-L6-v2}"
export EMBED_DIM="${EMBED_DIM:-384}"
export EMBED_BATCH="${EMBED_BATCH:-32}"
export NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
export NEO4J_USER="${NEO4J_USER:-neo4j}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:-${NEO4J_PASSWORD_DEFAULT:-knowledge_graph_2026}}"
export NEO4J_DB="${NEO4J_DB:-neo4j}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"

if [[ "$PRINT_ROOTS" == "1" ]]; then
  echo "DAY=$DAY"
  echo "SCAN_ROOTS=$SCAN_ROOTS"
  echo "RTTM_DIRS=$RTTM_DIRS"
  exit 0
fi

VENV="${VENV:-./.venv}"
if [[ -x "$VENV/bin/python3" ]]; then
  PY="$VENV/bin/python3"
else
  PY="python3"
fi

if [[ ! -d /nas1 && ! -d /nas2 ]]; then
  if ! "$PY" - <<'PY' >/dev/null 2>&1
import torch, transformers, neo4j
PY
  then
    printf -v quoted_args '%q ' "${ORIG_ARGS[@]}"
    echo "Local Python does not have the auto-ingest ML dependencies (torch/transformers/neo4j)." >&2
    echo "Run this batch in the Compose image instead:" >&2
    echo "  docker compose run --rm --no-deps ingest-service bash -lc '/app/scripts/backfill_audio_day.sh ${quoted_args}'" >&2
    exit 3
  fi
fi

LOG_DIR="${LOG_DIR:-./logs/audio-day-backfill}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/audio_${DAY}_$(date +'%Y%m%d_%H%M%S').log"
LOCK_FILE="${LOCK_FILE:-/tmp/auto_ingest_audio_day_${DAY}.lock}"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another audio day backfill is running for $DAY (lock: $LOCK_FILE)." >&2
  exit 1
fi

cmd=("$PY" ./ingest_transcriptsv5_3.py
  --tx-seg-batch "${TX_SEG_BATCH:-120}"
  --tx-utt-batch "${TX_UTT_BATCH:-120}"
  --tx-edge-batch "${TX_EDGE_BATCH:-300}"
  --tx-ent-batch "${TX_ENT_BATCH:-300}"
  --tx-loc-batch "${TX_LOC_BATCH:-500}"
  --tx-timeout-sec "${TX_TIMEOUT_SEC:-120}"
  --fetch-size "${FETCH_SIZE:-100}"
  --log-level "$LOG_LEVEL"
)
[[ -n "$LIMIT" ]] && cmd+=(--limit "$LIMIT")
[[ "$DRY_RUN" == "1" ]] && cmd+=(--dry-run)
[[ "$FORCE" == "1" ]] && cmd+=(--force)

echo "==== Audio day backfill starting @ $(date) ====" | tee -a "$LOG_FILE"
echo "DAY=$DAY" | tee -a "$LOG_FILE"
echo "SCAN_ROOTS=$SCAN_ROOTS" | tee -a "$LOG_FILE"
echo "Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Running: ${cmd[*]}" | tee -a "$LOG_FILE"

# Keep the job gentle by default. Override IONICE_CMD/NICE_CMD to empty strings
# if you need foreground-priority debugging.
read -r -a ionice_cmd <<< "${IONICE_CMD:-ionice -c2 -n7}"
read -r -a nice_cmd <<< "${NICE_CMD:-nice -n 10}"
"${ionice_cmd[@]}" "${nice_cmd[@]}" "${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"

echo "==== Audio day backfill finished @ $(date) ====" | tee -a "$LOG_FILE"
