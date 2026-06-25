#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  echo "Usage: scripts/qa_audio_day.sh YYYY-MM-DD" >&2
  exit 2
fi
DAY="$1"
YEAR="${DAY:0:4}"
MONTH="${DAY:5:2}"
DD="${DAY:8:2}"
REL="$YEAR/$MONTH/$DD"
KEY_PREFIX="${YEAR}_${MONTH}${DD}"

cd "$(dirname "$0")/.."

echo "==== Filesystem coverage for $DAY ===="
for base in /media/scott/NAS1/fileserver /media/scott/NAS2/fileserver; do
  for sub in "audio/$REL" "audio/transcriptions/$REL"; do
    p="$base/$sub"
    if [[ -d "$p" ]]; then
      printf '%s\n' "$p"
      find "$p" -type f \( -iname '*.mp3' -o -iname '*.wav' -o -iname '*.m4a' -o -iname '*.flac' -o -iname '*.webm' -o -iname '*_transcription.txt' -o -iname '*_transcription.csv' -o -iname '*_speakers.rttm' \) | wc -l | awk '{print "  relevant_files=" $1}'
    fi
  done
done

QUERY="
MATCH (t:Transcription)
WHERE t.key STARTS WITH '$KEY_PREFIX'
  AND (coalesce(t.source_media,'') CONTAINS '/audio/'
       OR coalesce(t.source_json,'') CONTAINS '/audio/'
       OR coalesce(t.source_csv,'') CONTAINS '/audio/')
OPTIONAL MATCH (t)-[:HAS_SEGMENT]->(s:Segment)
OPTIONAL MATCH (t)-[:HAS_UTTERANCE]->(u:Utterance)
RETURN count(DISTINCT t) AS transcriptions,
       count(DISTINCT s) AS segments,
       sum(CASE WHEN s IS NOT NULL AND s.embedding IS NULL THEN 1 ELSE 0 END) AS segments_missing_embedding,
       count(DISTINCT u) AS utterances,
       sum(CASE WHEN u IS NOT NULL AND u.embedding IS NULL THEN 1 ELSE 0 END) AS utterances_missing_embedding;
"
export QA_AUDIO_DAY_QUERY="$QUERY"

echo
echo "==== Neo4j audio embedding coverage for key prefix $KEY_PREFIX ===="
if command -v docker >/dev/null 2>&1 && docker ps --format '{{.Names}}' | grep -qx neo4j; then
  docker exec -i neo4j cypher-shell -u "${NEO4J_USER:-neo4j}" -p "${NEO4J_PASSWORD:-knowledge_graph_2026}" -d "${NEO4J_DB:-neo4j}" "$QUERY"
else
  python3 - <<'PY'
from neo4j import GraphDatabase
import os
query = os.environ['QA_AUDIO_DAY_QUERY']
uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
user = os.environ.get('NEO4J_USER', 'neo4j')
pw = os.environ.get('NEO4J_PASSWORD', 'knowledge_graph_2026')
db = os.environ.get('NEO4J_DB', 'neo4j')
with GraphDatabase.driver(uri, auth=(user, pw)) as driver:
    with driver.session(database=db) as session:
        rec = session.run(query).single()
        print('transcriptions, segments, segments_missing_embedding, utterances, utterances_missing_embedding')
        print(', '.join(str(rec[k]) for k in rec.keys()))
PY
fi
