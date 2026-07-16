#!/bin/bash
# run_worker.sh — background worker that processes queued auto-ingest work on
# IDLE machines. Launch with:
#     nohup ./run_worker.sh > logs/worker.log 2>&1 &
# Stop gracefully with:  touch worker.stop
#
# Each cycle, only if the machine is idle (1-min load average below threshold),
# it:
#   1) advances global speaker linking  (resumable via linker_state.json)
#   2) advances dashcam compression      (resumable; skips finished files)
#   3) turns new compressed clips into TikTok shorts (worker_content.py)
#   4) pulls in + processes the Nextcloud iPhone filestore (movies/pictures) into
#      the graph + content, when NEXTCLOUD_URL is set (resumable via nc_state)
# Then it sleeps and repeats. Because every stage is resumable, an idle machine
# can chip away at the whole corpus a little at a time.
set -uo pipefail
cd "$(dirname "$0")"

export NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
export NEO4J_USER="${NEO4J_USER:-neo4j}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:-${NEO4J_PASSWORD_DEFAULT:-knowledge_graph_2026}}"
export NEO4J_DB="${NEO4J_DB:-neo4j}"

IDLE_LOAD="${IDLE_LOAD:-$(awk "BEGIN{printf \"%.2f\", $(nproc)*0.6}")}"  # loadavg(1m) ceiling
LINK_CHUNK="${LINK_CHUNK:-200}"        # speakers per linking cycle
SLEEP="${SLEEP:-120}"                   # seconds between cycles
LINK_STATE="./linker_state.json"
CONTENT="${CONTENT:-1}"                 # set 0 to skip content generation
NEXTCLOUD_URL="${NEXTCLOUD_URL:-}"      # set to WebDAV base to enable Nextcloud ingest
NEXTCLOUD_USER="${NEXTCLOUD_USER:-admin}"
NEXTCLOUD_PASS="${NEXTCLOUD_PASS:-}"
# fall back to config.yaml nextcloud: block if env not set
if [ -z "${NEXTCLOUD_URL:-}" ]; then
  NEXTCLOUD_URL="$(python3 -c "import auto_ingest_config as c; print(c.get_nextcloud_webdav()[0] or '')" 2>/dev/null)"
  NEXTCLOUD_USER="$(python3 -c "import auto_ingest_config as c; print(c.get_nextcloud_webdav()[1] or 'admin')" 2>/dev/null)"
fi
if [ -z "${NEXTCLOUD_PASS:-}" ]; then
  NEXTCLOUD_PASS="$(python3 -c "import auto_ingest_config as c; print(c.get_nextcloud_webdav()[2] or '')" 2>/dev/null)"
fi
NC_LIMIT="${NC_LIMIT:-40}"              # media files per Nextcloud cycle
NC_STATE="./nc_ingest_state.json"
NC_ROOT="${NC_ROOT:-Photos}"            # subfolder under the Nextcloud user's files

idle() {
  local load
  load=$(awk '{print $1}' /proc/loadavg)
  awk "BEGIN{exit !($load < $IDLE_LOAD)}"
}

mkdir -p logs
echo "$(date) worker started (idle_load<=$IDLE_LOAD, link_chunk=$LINK_CHUNK, sleep=${SLEEP}s)"

while [ ! -f worker.stop ]; do
  if ! idle; then
    echo "$(date) not idle (loadavg $(cat /proc/loadavg | cut -d' ' -f1) >= $IDLE_LOAD); sleeping"
    sleep "$SLEEP"
    continue
  fi

  # 1) global speaker linking (monotonic via state file)
  echo "$(date) >>> speaker linking (chunk $LINK_CHUNK)"
  python3 bin/auto-ingest link-speakers --faiss --state-file "$LINK_STATE" --max-speakers "$LINK_CHUNK" || true

  # 2) dashcam compression (resumable; existing/verified outputs skipped)
  echo "$(date) >>> dashcam compression"
  ./run_compress_dashcam.sh || true

  # 3) content generation from new compressed clips
  if [ "${CONTENT:-1}" = "1" ]; then
    echo "$(date) >>> content generation"
    python3 worker_content.py --state ./content_state.json --limit 5 || true
  fi

  # 4) Nextcloud iPhone filestore ingest (only if configured)
  if [ -n "${NEXTCLOUD_URL:-}" ]; then
    echo "$(date) >>> Nextcloud ingest ($NC_ROOT, limit $NC_LIMIT)"
    python3 bin/auto-ingest ingest \
      --nextcloud-url "${NEXTCLOUD_URL%/}/remote.php/dav/files/${NEXTCLOUD_USER}/${NC_ROOT}" \
      --nextcloud-user "$NEXTCLOUD_USER" --nextcloud-pass "$NEXTCLOUD_PASS" \
      --source nextcloud --kind all --slideshow --limit "$NC_LIMIT" \
      --state "$NC_STATE" || true
  fi

  echo "$(date) cycle complete; sleeping ${SLEEP}s"
  sleep "$SLEEP"
done

echo "$(date) worker.stop present; exiting"
rm -f worker.stop
