#!/usr/bin/env bash
# Watches the enrich_phonelog_place backfill; when PhoneLog->AT_PLACE stops growing,
# runs enrich_scott_unify.py (rebuilds :PlaceHour) and verifies past-week location data.
# Run detached: nohup bash finish_location_backfill.sh > logs/backfill_watcher.log 2>&1 &
set -u
CD="/media/scott/SSD_4TB/hermes-home/auto-ingest"
LOG="$CD/logs/backfill_watcher.log"
PW="knowledge_graph_2026"
CY="docker exec neo4j cypher-shell -u neo4j -p $PW"
ATTACHED() { $CY "MATCH (p:PhoneLog)-[:AT_PLACE]->() RETURN count(*) AS c" 2>/dev/null | awk 'NR==2{print $1}'; }
RUNNING() { pgrep -f enrich_phonelog_place.py >/dev/null && echo yes || echo no; }
PLACEHOUR() { $CY "MATCH (ph:PlaceHour) RETURN count(ph) AS c" 2>/dev/null | awk 'NR==2{print $1}'; }

echo "[$(date)] watcher start. AT_PLACE attached=$(ATTACHED)" >>"$LOG"
prev=$(ATTACHED); stable=0
while true; do
  sleep 300
  cur=$(ATTACHED)
  running=$(RUNNING)
  echo "[$(date)] attached=$cur running=$running" >>"$LOG"
  if [ "$running" = "no" ]; then
    # job exited; confirm count is stable (not just between batches)
    if [ "$cur" = "$prev" ]; then
      stable=$((stable+1)); else stable=0; prev=$cur
    fi
    if [ "$stable" -ge 2 ]; then
      echo "[$(date)] phonelog_place DONE at $cur. Running enrich_scott_unify (PlaceHour rebuild)..." >>"$LOG"
      cd "$CD" && .venv/bin/python scripts/enrich_scott_unify.py >>"$LOG" 2>&1
      echo "[$(date)] scott_unify done. PlaceHour count=$(PLACEHOUR)" >>"$LOG"
      # verify past-week
      week=$($CY "MATCH (ph:PlaceHour) WHERE ph.hourBucket >= timestamp().epochMillis - 7*86400000 RETURN count(ph) AS c" 2>/dev/null | awk 'NR==2{print $1}')
      echo "[$(date)] past-week PlaceHour buckets=$week" >>"$LOG"
      echo "[$(date)] BACKFILL COMPLETE" >>"$LOG"
      break
    fi
  else
    prev=$cur; stable=0
  fi
done
