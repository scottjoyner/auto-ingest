#!/bin/bash
# run_ingest_daily.sh — recurring auto-ingest for the LOCAL DR lane.
#
# Runs the idempotent transcript ingest against the canonical merged graph on
# this machine (bolt://127.0.0.1:7687). Guarded by a flock so a scheduled pass
# never stacks on a manually-started full ingest, and skips while the machine
# is busy (loadavg above a ceiling). New RTTMs are picked up automatically
# because the ingest SKIPs up-to-date keys and processes only what's missing.
#
# Install in cron with:
#   0 3 * * * /home/deathstar/git/auto-ingest/run_ingest_daily.sh >> /home/deathstar/git/auto-ingest/logs/daily_ingest.log 2>&1
set -uo pipefail
cd "$(dirname "$0")"

LOCK_FILE="/tmp/auto_ingest_daily.lock"
IDLE_LOAD="${IDLE_LOAD:-$(awk "BEGIN{printf \"%.2f\", $(nproc)*0.6}")}"   # loadavg(1m) ceiling
SLEEP="${SLEEP:-60}"          # seconds to wait for the lock before giving up
MAX_WAIT="${MAX_WAIT:-1200}"  # max seconds to wait for the lock total

mkdir -p logs

# Refuse to start while the load is already high (e.g. a manual full ingest or
# the diarization sweep is running).
load=$(awk '{print $1}' /proc/loadavg)
if awk "BEGIN{exit !($load >= $IDLE_LOAD)}"; then
  echo "$(date) SKIP: loadavg $load >= $IDLE_LOAD (busy); not starting a scheduled pass"
  exit 0
fi

# flock guard: only one ingest pass at a time (against any lock holder).
exec 9>"$LOCK_FILE"
if ! flock -w "$MAX_WAIT" 9; then
  echo "$(date) SKIP: another ingest pass holds the lock; giving up after ${MAX_WAIT}s"
  exit 0
fi

echo "$(date) daily ingest pass starting (uri=bolt://127.0.0.1:7687)"
./run_ingest_all.sh --log-level INFO
rc=$?
echo "$(date) daily ingest pass finished rc=$rc"
exit $rc