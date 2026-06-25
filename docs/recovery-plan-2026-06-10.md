# Neo4j Recovery Plan: PhoneLog and Dashcam

> For Hermes: follow this document literally. Do not improvise. If a prerequisite is missing, stop and report it before touching Neo4j.

## Goal
Recover the missing PhoneLog and dashcam data into Neo4j while preserving the currently recovered graph and avoiding duplicate or destructive writes.

## Current facts
- The live Neo4j graph has already been partially recovered.
- Audio/transcription recovery is in progress and should be treated as a separate track.
- PhoneLog appears to have a real gap in the live DB.
- Dashcam coverage is incomplete, especially for 2026-05 and 2026-06.
- PhoneLog may only be recoverable if an external backup or source export exists.
- If no source exists, document the loss explicitly and stop PhoneLog recovery.

## Operating rules
1. Do not overwrite the live DB.
2. Do not restore a dump directly over the current graph without a safety backup.
3. Do not run full-corpus FORCE jobs.
4. Work in small, verifiable batches.
5. After each batch, run QA before continuing.
6. If a step fails, stop and report the exact failure.
7. Keep commands copy-pasteable for a simpler model or human operator.

## Known repositories and paths
- Auto-ingest repo: `/home/scott/git/auto-ingest`
- Neo4j backups: `/media/scott/NAS2/fileserver/neo4j-bkps`
- Possible alternate backup root: `/media/scott/NAS1/fileserver/neo4j-bkps`
- PhoneLog service repo: `/home/scott/docker-compose/phonelog/`
- Current ingest state note: `/home/scott/git/auto-ingest/docs/current-ingest-state-2026-06-10.md`

## What must be checked before recovery starts
### A. Backup inventory
We need to know what Neo4j backup artifacts exist and which are valid candidates.

Commands:
```bash
for root in \
  /media/scott/NAS1/fileserver/neo4j-bkps \
  /media/scott/NAS2/fileserver/neo4j-bkps; do
  [ -d "$root" ] || continue
  echo "== $root =="
  du -h "$root"/*.backup 2>/dev/null | sort -rh | head -20
  ls -lt "$root" 2>/dev/null | head -20
  echo
 done
```

Decision:
- Choose the newest valid backup only as a candidate.
- Do not assume the newest file contains everything we need.

### B. PhoneLog source inventory
We need to confirm whether PhoneLog data exists anywhere outside the live DB.

Search targets:
- any phone-call/SMS export source
- any archival backup in the SB / storage-box area if present
- any legacy import files or dump artifacts
- the phonelog service repository and its config

Commands:
```bash
cd /home/scott/docker-compose/phonelog
find . -maxdepth 3 -type f | sort
```

If backup/source files exist elsewhere, inventory them by date and size before touching Neo4j.

### C. Dashcam corpus inventory
We need to confirm the filesystem corpus vs Neo4j coverage.

Known corpus path:
- `/media/scott/NAS2/fileserver/dashcam`

Verify monthly file presence and compare to DB counts before any ingest run.

## Recovery plan overview

### Phase 1: Safety and decision point
Objective: determine whether PhoneLog is recoverable at all, and lock in a safe plan for dashcam.

Tasks:
1. Inventory backups.
2. Inventory PhoneLog source/archive locations.
3. Confirm dashcam corpus roots and current DB gaps.
4. If PhoneLog source is absent, mark PhoneLog as unrecoverable from current material and stop that track.

Output of this phase:
- list of valid Neo4j backup candidates
- list of PhoneLog source candidates, or explicit note that none exist
- dashcam coverage summary

### Phase 2: PhoneLog recovery, only if source exists
Objective: backfill only the missing PhoneLog range using stable IDs and idempotent writes.

Required properties:
- stable unique ID per event
- timestamp
- type (`call`, `sms`, `location`)
- payload fields

Plan:
1. Extract the exact missing date range.
2. Load only that range from source data.
3. MERGE by stable ID.
4. Preserve existing live values on MATCH unless a newer source value is clearly authoritative.
5. QA the timestamp range after each chunk.

Chunking rule:
- Use day-sized chunks first.
- If a day is too large, split into hour-sized chunks.

QA checks:
- node count by day
- min/max timestamp
- duplicate ID check
- spot-check a few records against source files

Stop conditions:
- no source data found
- schema mismatch that would require guessing
- duplicate behavior not understood

### Phase 3: Dashcam recovery
Objective: recover missing dashcam clips, embeddings, and frames from NAS2.

Plan:
1. Compare filesystem dashcam keys to `DashcamClip.key`.
2. Start with missing months 2026-05 and 2026-06.
3. Run dashcam-specific ingestion only.
4. Keep the batch size small enough to verify progress.
5. After each batch, QA clip counts and embedding/frame coverage.

Expected target labels:
- `DashcamClip`
- `DashcamEmbedding`
- `Frame`

QA checks:
- FS keys vs DB keys
- month-by-month counts
- missing embedding counts
- orphaned frame/metadata rows

Stop conditions:
- ingestion scans the wrong root
- counts do not move in the expected direction
- embeddings/frames are missing unexpectedly

### Phase 4: Final verification and backup
Objective: prove the recovery is stable and create a fresh backup.

Checks:
- PhoneLog min/max timestamp and gap status
- Dashcam month coverage for 2026-05 and 2026-06
- major label counts
- no obvious embedding gaps in the recovered ranges

Then:
- take a fresh Neo4j backup
- store it in the documented backup root
- record the exact backup filename in the current-state note

## Execution order
1. Backup inventory
2. PhoneLog source inventory
3. Dashcam corpus inventory
4. PhoneLog decision: recover or declare lost
5. PhoneLog backfill if possible
6. Dashcam backfill
7. QA both tracks
8. Fresh backup

## Exact verification commands to use
### Neo4j smoke test
```bash
docker exec neo4j cypher-shell -u neo4j -p knowledge_graph_2026 "RETURN 1 AS ok"
```

### PhoneLog timestamp coverage
```cypher
MATCH (p:PhoneLog)
RETURN count(p) AS c, min(p.timestamp) AS min_ts, max(p.timestamp) AS max_ts;
```

### Dashcam month coverage
```cypher
MATCH (d:DashcamClip)
RETURN substring(d.key, 0, 7) AS month, count(*) AS c
ORDER BY month;
```

### Dashcam embedding completeness
```cypher
MATCH (d:DashcamClip)
OPTIONAL MATCH (d)-[:HAS_EMBEDDING]->(e:DashcamEmbedding)
RETURN count(d) AS clips, count(e) AS embeddings;
```

## Documentation rule for the operator
After each phase, update the current-state note with:
- what was checked
- what was found
- what is still missing
- what command should be run next
- what not to do

## If PhoneLog is unrecoverable
If the source is not present in the SB/backup/archive material, document the gap clearly and do not invent a recovery path.
The correct outcome in that case is:
- PhoneLog gap remains acknowledged
- dashcam recovery still proceeds
- current state note records that PhoneLog cannot be reconstructed from available material

## Minimum acceptable output from Hermes
When executing this plan, Hermes must always report:
- the exact command run
- the exact path checked
- the exact counts observed
- the next safe step

No vague summaries.
