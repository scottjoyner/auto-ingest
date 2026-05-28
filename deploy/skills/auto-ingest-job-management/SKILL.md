---
name: auto-ingest-job-management
description: Use when enqueueing jobs, managing the distributed worker queue, or triggering ingest runs for the auto-ingest system. Covers API endpoints, create_job.sh, and job file format.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [auto-ingest, jobs, queue, api, distributed, worker]
    related_skills: [auto-ingest-docker, auto-ingest-troubleshooting]
---

# Auto-Ingest Job Management

## Overview

Auto-Ingest uses a NAS-drop queue system for distributed job processing. Jobs are created as executable `.job` files in `/nas/drop/`, claimed by workers via atomic `mv`, and moved to `done/` or `failed/`. A REST API (`job-api` on port 8766) provides HTTP endpoints for enqueueing and monitoring.

## Job Queue Protocol

```
/nas/drop/
  *.job          # Pending jobs — workers poll for these
  claimed/       # Currently being processed (hostname.PID suffix)
  done/          # Successfully completed
  failed/        # Failed jobs (move to failed/ for debugging)
```

Worker claim loop (every 30s):
1. Scan `/nas/drop/*.job`
2. Atomically `mv` to `claimed/<name>.<hostname>.<pid>`
3. Execute the `.job` script
4. Move to `done/` (success) or `failed/` (error)

## Job Types

| Kind | Description | Default SCAN_ROOTS |
|------|-------------|-------------------|
| `audio` | Audio transcription ingest | `/nas/S/audio,/nas/fileserver/audio/transcriptions` |
| `dashcam` | Dashcam video ingest | `/nas/fileserver/dashcam` |
| `bodycam` | Bodycam video ingest | `/nas/fileserver/bodycam` |
| `all` | All data types | All SCAN_ROOTS |

## HTTP API (port 8766)

### Enqueue a Job
```bash
curl -X POST http://localhost:8766/api/enqueue \
  -H 'Content-Type: application/json' \
  -d '{"kind": "dashcam"}'
```

Response:
```json
{
  "status": "queued",
  "job_file": "/nas/drop/20260522_004231_dashcam.job",
  "kind": "dashcam",
  "message": "Job created. Worker will pick it up automatically."
}
```

### Check Queue Status
```bash
curl http://localhost:8766/api/status
```

Response:
```json
{
  "drop_root": "/nas/drop",
  "pending_jobs": 1,
  "claimed_jobs": 0,
  "done_jobs": 5,
  "failed_jobs": 2,
  "pending_files": ["20260522_004231_dashcam.job"],
  "failed_files": ["20260521_213132_all.job.failed"]
}
```

### Health Check
```bash
curl http://localhost:8766/api/health
```

### One-Shot Run (bypasses queue)
```bash
curl -X POST http://localhost:8766/api/run \
  -H 'Content-Type: application/json' \
  -d '{"type": "ingest"}'

curl -X POST http://localhost:8766/api/run \
  -H 'Content-Type: application/json' \
  -d '{"type": "sync"}'
```

## Shell Script (create_job.sh)

```bash
cd /home/scott/git/auto-ingest

# Enqueue jobs by data type
./deploy/create_job.sh audio
./deploy/create_job.sh dashcam
./deploy/create_job.sh bodycam
./deploy/create_job.sh all

# Verify queue
ls -lah /nas/drop/*.job
```

## Job File Format

```bash
#!/usr/bin/env bash
set -euo pipefail
cd /app
SCAN_ROOTS="/nas/fileserver/dashcam" DASHCAM_ROOT="/nas/fileserver/dashcam" /usr/bin/env bash run_ingest_all.sh
```

## Manual Job Management

### Move a failed job for debugging
```bash
mv /nas/drop/failed/20260521_213132_all.job.failed /nas/drop/20260522_debug.job
chmod +x /nas/drop/20260522_debug.job
# Worker will pick it up on next poll (~30s)
```

### Clear failed jobs
```bash
# Review first, then clear
ls -lah /nas/drop/failed/
# rm /nas/drop/failed/*  # only after reviewing
```

### Check worker logs for a specific job
```bash
# Find the log file
ls -lah /home/scott/git/auto-ingest/logs/ingest_*.log | tail -5

# View the latest log
tail -100 /home/scott/git/auto-ingest/logs/ingest_20260522_004235.log
```

## Verification Checklist

- [ ] Enqueued job appears in `/nas/drop/*.job`
- [ ] Worker picks up job within 30s (check `/nas/drop/claimed/`)
- [ ] Job completes and appears in `/nas/drop/done/`
- [ ] Neo4j shows new data after ingest completes
- [ ] No duplicate processing (flock guard prevents overlaps)
