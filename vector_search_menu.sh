#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PS3=$'\n'"Select an action: "
select opt in "Text semantic search" "Similar frames" "Geo frames" "Quit"; do
  case "$REPLY" in
    1) exec "$here/vector_search_text.sh" ;;
    2) exec "$here/vector_search_similar_frames.sh" ;;
    3) exec "$here/vector_search_geo_frames.sh" ;;
    4) echo "Bye!"; exit 0 ;;
    *) echo "Invalid option";;
  esac
done
