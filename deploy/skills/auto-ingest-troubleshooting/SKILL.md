---
name: auto-ingest-troubleshooting
description: Use when troubleshooting the auto-ingest system — container issues, Neo4j connection problems, job queue failures, or data pipeline errors. Covers known issues, fixes, and diagnostic commands.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [auto-ingest, troubleshooting, debugging, docker, neo4j, queue]
    related_skills: [auto-ingest-docker, auto-ingest-job-management]
---

# Auto-Ingest Troubleshooting

## Quick Diagnostics

```bash
cd /home/scott/git/auto-ingest

# 1. Check container status
docker compose ps

# 2. Check API health
curl http://localhost:8766/api/health
curl http://localhost:8766/api/status

# 3. Check Neo4j connectivity
docker exec neo4j cypher-shell -u neo4j -p knowledge_graph_2026 "RETURN 1"

# 4. Check Neo4j data counts
docker exec neo4j cypher-shell -u neo4j -p knowledge_graph_2026 \
  "MATCH (n) RETURN labels(n) AS label, count(*) AS cnt ORDER BY cnt DESC"

# 5. Check worker logs
docker compose logs --tail=100 ingest-worker

# 6. Check ingest-service logs
docker compose logs --tail=100 ingest-service

# 7. Check cron logs (inside container)
docker compose exec ingest-cron tail -n 50 /var/log/cron-ingest.log
docker compose exec content-cron tail -n 50 /var/log/cron-content.log

# 8. Check sync logs
docker compose exec auto-sync-service cat /var/log/sync.log

# 9. Check job queue
ls -lah /nas/drop/ /nas/drop/claimed/ /nas/drop/done/ /nas/drop/failed/
```

## Known Issues and Fixes

### 1. Neo4j Connection Failure: "Cannot resolve address host.docker.internal:7687"
**Cause**: On Linux Docker, `host.docker.internal` does not resolve automatically.
**Fix**: Added `extra_hosts: ["host.docker.internal:host-gateway"]` to all services that need host access (ingest-service, ingest-worker, sync-service, job-api). Already configured in compose file.

### 2. libGL / libvpx Errors in Worker
**Cause**: ffmpeg requires GL libraries not present in slim images.
**Fix**: Dockerfile installs `libgl1 libglib2.0-0` during build. Already configured.

### 3. Legacy Drop Sync Fails: "Legacy drop root not found"
**Cause**: `/nas/fileserver/incoming/deathstar` doesn't exist on the host.
**Fix**: Created the directory and updated `sync_from_legacy_drop.sh` to exit gracefully when the directory is missing. Already fixed.

### 4. No Jobs Processed
**Diagnosis**:
```bash
# Check if drop directory exists and is mounted
ls -lah /nas/drop/
# Check worker logs
docker compose logs --tail=50 ingest-worker
# Check if worker is running
docker compose ps | grep ingest-worker
```
**Fix**: Verify `/nas/drop/` is mounted in the worker container and has correct permissions.

### 5. Duplicate Work / Overlapping Runs
**Cause**: Multiple ingest runs executing simultaneously.
**Fix**: `run_ingest_all.sh` uses flock-based guard (`/tmp/ingest_transcripts.lock`). Verify the lock file exists and is not stale.

### 6. Neo4j Password Mismatch
**Symptom**: Connection refused or authentication errors.
**Cause**: docker-compose uses `knowledge_graph_2026` but legacy app uses `livelongandprosper`.
**Fix**: Always use `knowledge_graph_2026` for docker-compose services. The docker-compose file sets `NEO4J_PASSWORD=knowledge_graph_2026`.

### 7. Cron Jobs Not Running
**Diagnosis**:
```bash
# Check cron is running inside container
docker compose exec ingest-cron cat /var/log/cron.log
# Check cron output logs
docker compose exec ingest-cron cat /var/log/cron-ingest.log
# Verify crontab is installed
docker compose exec ingest-cron crontab -l
```
**Fix**: Recreate cron service after cron file edits:
```bash
docker compose up -d --force-recreate ingest-cron content-cron
```

### 8. Content OS Status Returns Empty
**Diagnosis**:
```bash
docker compose logs --tail=50 content-service
```
**Fix**: Content OS may have no active runs. Run `content-os init` if the folder tree doesn't exist.

### 9. Sync Service Not Syncing Anything
**Diagnosis**:
```bash
docker compose exec auto-sync-service cat /var/log/sync.log
```
**Fix**: Legacy drop directory `/nas/fileserver/incoming/deathstar` may be empty or not mounted. Sync runs every 10 minutes — wait for the next cycle or trigger manually.

## Diagnostic Commands by Service

### Ingest Worker
```bash
# Check if worker is processing
docker compose logs --tail=30 ingest-worker | grep -E "Claimed|Completed|Failed"

# Check queue movement
watch -n 5 'ls -lah /nas/drop/ /nas/drop/claimed/ /nas/drop/done/ /nas/drop/failed/'

# Check worker claims a job
docker compose logs --tail=100 ingest-worker | grep "Claimed job"
```

### Ingest Service
```bash
# Check last completed ingest
docker compose logs --tail=200 ingest-service | grep "END INGEST"

# Check Neo4j connection in service logs
docker compose logs --tail=500 ingest-service | grep -E "Neo4j|connection|ERROR"
```

### Sync Service
```bash
# Check sync status
docker compose exec auto-sync-service cat /var/log/sync.log

# Trigger manual sync
docker compose run --rm sync-service
```

### Neo4j
```bash
# Check health
docker exec neo4j cypher-shell -u neo4j -p knowledge_graph_2026 "RETURN 1"

# Check specific node counts
docker exec neo4j cypher-shell -u neo4j -p knowledge_graph_2026 \
  "MATCH (n:Transcription) RETURN count(*)"
docker exec neo4j cypher-shell -u neo4j -p knowledge_graph_2026 \
  "MATCH (n:DashcamEmbedding) RETURN count(*)"
docker exec neo4j cypher-shell -u neo4j -p knowledge_graph_2026 \
  "MATCH (n:PhoneLog) RETURN count(*)"

# Check relationships
docker exec neo4j cypher-shell -u neo4j -p knowledge_graph_2026 \
  "MATCH ()-[r]->() RETURN type(r), count(*) ORDER BY count(*) DESC LIMIT 20"
```

## Verification Checklist After Fixes

- [ ] All containers are up: `docker compose ps`
- [ ] API responds: `curl http://localhost:8766/api/health`
- [ ] Neo4j is healthy: `docker exec neo4j cypher-shell -u neo4j -p knowledge_graph_2026 "RETURN 1"`
- [ ] Worker processes jobs: enqueue a test job and verify completion
- [ ] Cron runs on schedule: check `/var/log/cron*.log` inside containers
- [ ] Sync runs without errors: check `/var/log/sync.log`
- [ ] Data appears in Neo4j: verify node counts increase after ingest
