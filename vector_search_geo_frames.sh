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

read -r -p "🧭 Latitude: " LAT
read -r -p "🧭 Longitude: " LON
read -r -p "📏 Radius (meters, default: 200): " RADIUS
RADIUS=${RADIUS:-200}

read -r -p "⏱ Start time (ISO8601 or epochMillis; blank=none): " START
read -r -p "⏱ End time (ISO8601 or epochMillis; blank=none): " END

read -r -p "🚗 Min MPH (blank=none): " MIN_MPH
read -r -p "🚗 Max MPH (blank=none): " MAX_MPH

read -r -p "🔢 Limit (default: 100): " LIMIT
LIMIT=${LIMIT:-100}

read -r -p "📄 Output JSON? (y/N): " AS_JSON
read -r -p "🧾 Also write CSV file path (or blank to skip): " CSV_PATH

cmd=( "$PYTHON_EXEC" "$SCRIPT_PATH" "geo-frames"
      "--lat" "$LAT" "--lon" "$LON" "--radius-m" "$RADIUS"
      "--limit" "$LIMIT" )

[[ -n "${START// }"  ]] && cmd+=( "--start" "$START" )
[[ -n "${END// }"    ]] && cmd+=( "--end" "$END" )
[[ -n "${MIN_MPH// }" ]] && cmd+=( "--min-mph" "$MIN_MPH" )
[[ -n "${MAX_MPH// }" ]] && cmd+=( "--max-mph" "$MAX_MPH" )
[[ "$AS_JSON" =~ ^[Yy]$ ]] && cmd+=( "--json" )
[[ -n "${CSV_PATH// }" ]] && cmd+=( "--csv" "$CSV_PATH" )

echo "▶️ ${cmd[*]}"
"${cmd[@]}"
