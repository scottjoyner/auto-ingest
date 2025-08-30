#!/usr/bin/env bash
set -euo pipefail

PYTHON_EXEC="${PYTHON_EXEC:-/home/deathstar/git/video-automation/.venv/bin/python}"
SCRIPT_PATH="${SCRIPT_PATH:-/home/deathstar/git/video-automation/vector_search.py}"

if [[ ! -x "$PYTHON_EXEC" ]]; then
  echo "❌ PYTHON_EXEC not found/executable: $PYTHON_EXEC" >&2; exit 1
fi
if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "❌ vector_search.py not found: $SCRIPT_PATH" >&2; exit 1
fi

echo "Seed choice:"
echo "  1) by frame-id"
echo "  2) by file_key + frame number"
read -r -p "Select [1/2] (default: 1): " CHOICE
CHOICE=${CHOICE:-1}

FRAME_ID=""
FILE_KEY=""
FRAME_NO=""

if [[ "$CHOICE" == "2" ]]; then
  read -r -p "📁 file_key: " FILE_KEY
  read -r -p "🎞  frame number: " FRAME_NO
  if [[ -z "${FILE_KEY// }" || -z "${FRAME_NO// }" ]]; then
    echo "❌ file_key and frame number required." >&2; exit 1
  fi
else
  read -r -p "🆔 frame-id: " FRAME_ID
  if [[ -z "${FRAME_ID// }" ]]; then
    echo "❌ frame-id required." >&2; exit 1
  fi
fi

read -r -p "🔢 Top-K results (default: 20): " TOPK
TOPK=${TOPK:-20}

read -r -p "➕ Include the seed itself? (y/N): " INCLUDE_SEED
read -r -p "📄 Output JSON? (y/N): " AS_JSON
read -r -p "🧾 Also write CSV file path (or blank to skip): " CSV_PATH

cmd=( "$PYTHON_EXEC" "$SCRIPT_PATH" "similar-frames" "--topk" "$TOPK" )
if [[ -n "$FRAME_ID" ]]; then
  cmd+=( "--frame-id" "$FRAME_ID" )
else
  cmd+=( "--file-key" "$FILE_KEY" "--frame" "$FRAME_NO" )
fi
[[ "$INCLUDE_SEED" =~ ^[Yy]$ ]] && cmd+=( "--include-seed" )
[[ "$AS_JSON" =~ ^[Yy]$ ]] && cmd+=( "--json" )
[[ -n "${CSV_PATH// }" ]] && cmd+=( "--csv" "$CSV_PATH" )

echo "▶️ ${cmd[*]}"
"${cmd[@]}"
