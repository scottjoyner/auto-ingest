#!/bin/bash
# Compress dashcam footage (dashcam _F/_R/_FR .MP4) into a mirrored portable tree
# on NAS5 at /fileserver/dashcam/compressed/YYYY/MM/DD/<file>.mp4
#
# Reads source from the fileserver dashcam root (SSD_4TB/fileserver -> NAS5) and
# writes HEVC/H.264 re-encodes. Resumable: existing/verified outputs are skipped,
# so re-launching after a crash/cont/Ctrl-C continues where it left off.
set -uo pipefail
cd "$(dirname "$0")"

export NEO4J_URI="${NEO4J_URI:-$(python3 -c 'import auto_ingest_config as c; print(c.get_neo4j_config()["uri"])')}"
export NEO4J_USER="${NEO4J_USER:-$(python3 -c 'import auto_ingest_config as c; print(c.get_neo4j_config()["user"])')}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:-$(python3 -c 'import auto_ingest_config as c; print(c.get_neo4j_config()["password"])')}"
export NEO4J_DB="${NEO4J_DB:-neo4j}"

FILESERVER_ROOT="${FILESERVER_ROOT:-$(python3 -c 'import auto_ingest_config as c; print(c.get_fileserver_root())')}"
COMPRESSED_ROOT="${COMPRESSED_ROOT:-$FILESERVER_ROOT/dashcam/compressed}"
INPUT_ROOT="${INPUT_ROOT:-$FILESERVER_ROOT/dashcam}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$COMPRESSED_ROOT}"
WORKERS="${WORKERS:-$(nproc)}"
CRF="${CRF:-26}"
LIMIT="${LIMIT:-0}"   # 0 = no limit (process everything)
VAAPI="${VAAPI:-0}"   # 1 = use VAAPI hardware encode (AMD/Intel GPU)
VAAPI_DEVICE="${VAAPI_DEVICE:-/dev/dri/renderD128}"
VAAPI_BITRATE="${VAAPI_BITRATE:-}"

case "$VAAPI" in
  0|false|FALSE|no|NO|off|OFF) VAAPI_ENABLED=0 ;;
  1|true|TRUE|yes|YES|on|ON) VAAPI_ENABLED=1 ;;
  *)
    echo "ERROR: VAAPI must be 0/1 or a boolean value, got: $VAAPI" >&2
    exit 2
    ;;
esac

echo "=== dashcam compression ==="
echo "input : $INPUT_ROOT"
echo "output: $OUTPUT_ROOT"
echo "workers: $WORKERS  crf: $CRF  limit: $LIMIT"
echo "vaapi : $VAAPI_ENABLED (device $VAAPI_DEVICE${VAAPI_BITRATE:+ bitrate $VAAPI_BITRATE})"

args=(
  --input-root "$INPUT_ROOT"
  --output-root "$OUTPUT_ROOT"
  --workers "$WORKERS"
  --crf "$CRF"
  --max-width 1280
  --fps 30
  --audio-k 96
  --order newest
)

if [[ "$VAAPI_ENABLED" == "1" ]]; then
  args+=(--vaapi --vaapi-device "$VAAPI_DEVICE")
  if [[ -n "$VAAPI_BITRATE" ]]; then
    args+=(--vaapi-bitrate "$VAAPI_BITRATE")
  fi
fi
if [[ "$LIMIT" != "0" ]]; then
  args+=(--limit "$LIMIT")
fi

python3 -u compress_dashcam2.py "${args[@]}" "$@"

echo "Compression pass complete. Output mirrored at: $OUTPUT_ROOT"
