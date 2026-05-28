---
name: auto-ingest-docker
description: Use when deploying, managing, or troubleshooting the auto-ingest Docker services (ingest, worker, sync, content, cron, job-api, neo4j). Covers build, start, stop, logs, and health checks.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [docker, auto-ingest, deployment, containerization, orchestration]
    related_skills: [auto-ingest-job-management, auto-ingest-troubleshooting]
---

# Auto-Ingest Docker Services

## Overview

Auto-Ingest runs as 7 Docker containers orchestrated by `docker-compose.yml` at `/home/scott/git/auto-ingest/`. The system ingests dashcam, audio, and bodycam recordings into a Neo4j knowledge graph via distributed workers and a job queue.

## Service Topology

| Service | Port | Poll | Purpose |
|---------|------|------|---------|
| job-api | 8766 | — | HTTP API for enqueueing jobs, checking status |
| ingest-service | — | 5 min | Runs `run_ingest_all.sh` continuously |
| ingest-worker | — | 30s | Claims `.job` files from `/nas/drop/` |
| sync-service | — | 10 min | Syncs legacy drop from deathstar |
| content-service | — | 30 min | Content OS CLI status loop |
| ingest-cron | — | 5 min | Scheduled ingest cron |
| content-cron | — | 30 min | Scheduled content cron |
| neo4j | 7474/7687 | — | Graph database (healthy, 20M+ nodes) |

## Quick Commands

```bash
cd /home/scott/git/auto-ingest

# Build image
docker compose build

# Start all services
docker compose up -d

# Start specific services
docker compose up -d ingest-service ingest-worker sync-service ingest-cron content-cron content-service job-api

# Stop all
docker compose down

# View logs
docker compose logs -f ingest-worker
docker compose logs -f ingest-service
docker compose logs -f sync-service
docker compose logs --tail=50 ingest-cron   # cron logs to /var/log/cron*.log inside container

# Rebuild and restart
docker compose up -d --build

# Check status
docker compose ps
curl http://localhost:8766/api/health
curl http://localhost:8766/api/status
```

## Environment Configuration

Copy and edit `.env` from the example:

```bash
cp deploy/path_profiles.env.example .env
# Edit paths, Neo4j credentials, etc.
```

Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| NAS_ROOT | /media/scott/NAS1 | NAS mount point (mounted as /nas in containers) |
| DROP_ROOT | /nas/drop | Job queue directory |
| NEO4J_URI | bolt://localhost:7687 | Neo4j Bolt URI |
| NEO4J_USER | neo4j | Neo4j username |
| NEO4J_PASSWORD | knowledge_graph_2026 | Neo4j password |
| SCAN_ROOTS | (comma-separated) | Directories to scan for transcripts |
| DASHCAM_ROOT | /nas/fileserver/dashcam | Dashcam root |
| AUDIO_ROOT | /nas/S/audio | Audio root |
| BODYCAM_ROOT | /nas/fileserver/bodycam | Bodycam root |
| TRANSCRIPT_ROOT | /nas/fileserver/audio/transcriptions | Transcripts root |
| LEGACY_DROP_ROOT | /nas/fileserver/incoming/deathstar | Legacy drop source |
| CONTENT_OS_LLM_BASE_URL | http://localhost:1234/v1 | LLM endpoint for content OS |
| CONTENT_OS_LLM_API_KEY | lm-studio | LLM API key |
| CONTENT_OS_LLM_MODEL | local-model | LLM model name |

## Key Files Reference

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Service definitions, volumes, networks |
| `Dockerfile` | Container image (python:3.11-slim + ffmpeg + Neo4j driver) |
| `deploy/job_trigger_api.py` | HTTP API server |
| `deploy/worker_ingest.sh` | Distributed worker claim loop |
| `deploy/sync_from_legacy_drop.sh` | Legacy drop sync script |
| `deploy/create_job.sh` | Helper to create .job files |
| `deploy/start-cron.sh` | Cron daemon starter |
| `deploy/cron/ingest.crontab` | Ingest schedule (*/5 * * * *) |
| `deploy/cron/content_generation.crontab` | Content schedule (*/30 * * * *) |
| `deploy/path_profiles.env.example` | Environment template |
| `run_ingest_all.sh` | Main ingest runner (flock-guarded) |
| `ingest_transcriptsv5_3.py` | Python ingest script |

## Container Paths

All containers mount `/nas` (from host `NAS_ROOT`). Inside containers:

- `/nas/drop/` — Job queue
- `/nas/fileserver/` — Data roots (audio, dashcam, bodycam, transcriptions)
- `/app/` — Auto-ingest repo
- `/var/log/` — Cron and service logs (inside container)

## Verification Checklist

- [ ] `docker compose ps` shows all 8 services up
- [ ] `curl http://localhost:8766/api/health` returns `{"status": "ok"}`
- [ ] `curl http://localhost:8766/api/status` shows queue counts
- [ ] Neo4j is healthy: `docker exec neo4j cypher-shell -u neo4j -p knowledge_graph_2026 "RETURN 1"` returns `1`
- [ ] Worker processes jobs: enqueue a test job and verify it appears in `/nas/drop/done/`
