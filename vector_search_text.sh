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

read -r -p "🔍 Enter text query: " QUERY
if [[ -z "${QUERY// }" ]]; then
  echo "❌ Query cannot be empty." >&2; exit 1
fi

read -r -p "🎯 Target [utterance | segment | transcription] (default: utterance): " TARGET
TARGET=${TARGET:-utterance}

read -r -p "🔢 Top-K results (default: 10): " TOPK
TOPK=${TOPK:-10}

read -r -p "⏱ Window minutes for PhoneLog snap (default: 10): " WIN_MINS
WIN_MINS=${WIN_MINS:-10}

read -r -p "🧠 Include embeddings? (y/N): " INCLUDE_EMB
read -r -p "📄 Output JSON? (y/N): " AS_JSON
read -r -p "🧾 Enter CSV file path (leave blank to skip): " CSV_PATH

cmd=( "$PYTHON_EXEC" "$SCRIPT_PATH" "search-text" "--q" "$QUERY" "--target" "$TARGET" "--topk" "$TOPK" "--win-mins" "$WIN_MINS" )
[[ "$INCLUDE_EMB" =~ ^[Yy]$ ]] && cmd+=( "--include-emb" )
[[ "$AS_JSON" =~ ^[Yy]$ ]] && cmd+=( "--json" )
[[ -n "${CSV_PATH// }" ]] && cmd+=( "--csv" "$CSV_PATH" )

echo "▶️ ${cmd[*]}"
"${cmd[@]}"
