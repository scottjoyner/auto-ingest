# Auto-Ingest System

Containerized pipeline that ingests dashcam, audio, and bodycam recordings into a Neo4j knowledge graph. Features distributed workers, a NAS-drop job queue, REST API for triggering jobs, and scheduled cron workers.

## Documentation

| Document | Description |
|----------|-------------|
| [System Design](docs/system_design.md) | Architecture, service topology, data model, deployment guide |
| [Deployment Runbook](docs/deployment_runbook.md) | File-by-file operational guide |
| [Content OS Architecture](docs/architecture.md) | Content workflow engine design |
| [Docker Skill](deploy/skills/auto-ingest-docker/SKILL.md) | Deploy, manage, and monitor Docker services |
| [Job Management Skill](deploy/skills/auto-ingest-job-management/SKILL.md) | Enqueue jobs, manage queue, API endpoints |
| [Troubleshooting Skill](deploy/skills/auto-ingest-troubleshooting/SKILL.md) | Known issues, diagnostics, fixes |

## Quick Start

```bash
cd /home/scott/git/auto-ingest

# Configure environment
cp deploy/path_profiles.env.example .env
# Edit .env for your host

# Build and start all services
docker compose up -d --build

# Verify
docker compose ps
curl http://localhost:8766/api/health
curl http://localhost:8766/api/status
```

### Machine-agnostic CLI (new primary entrypoint)

`bin/auto-ingest` is the preferred local entrypoint. It is machine-agnostic: all
paths/credentials come from `config.yaml` + env (no hardcoded paths/creds). It
auto-detects compute via `auto_ingest/backend.py` (CUDA / ROCm / MLX / ONNX) and
routes torch/faiss/YOLO device selection accordingly.

```bash
bin/auto-ingest run-all            # copy→diarize→ingest→reconcile→classify→link→yolo
bin/auto-ingest link-speakers      # global speaker linking
bin/auto-ingest caps               # probe hardware/backend profile
bin/auto-ingest status             # graph / linkage status
bin/auto-ingest ingest             # iPhone media -> graph
bin/auto-ingest whoami             # anchor the "me" speaker
bin/auto-ingest tiktok             # vertical shorts
bin/auto-ingest worker             # idle-gated background worker
```

> `run_all_optimized.sh` is a **deprecated** shim over `bin/auto-ingest run-all`.

## Trigger Jobs

```bash
# Via HTTP API
curl -X POST http://localhost:8766/api/enqueue \
  -H 'Content-Type: application/json' \
  -d '{"kind": "dashcam"}'

# Via shell script
./deploy/create_job.sh dashcam
./deploy/create_job.sh audio
./deploy/create_job.sh bodycam
./deploy/create_job.sh all

# Check status
curl http://localhost:8766/api/status
```

## Services

| Service | Port | Poll | Purpose |
|---------|------|------|---------|
| job-api | 8766 | — | HTTP API for enqueueing jobs |
| ingest-service | — | 5 min | Runs `run_ingest_all.sh` continuously |
| ingest-worker | — | 30s | Claims `.job` files from `/nas/drop/` |
| sync-service | — | daily | Syncs legacy drop + all top-level deathstar 8TB roots |
| content-service | — | 30 min | Content OS CLI status loop |
| ingest-cron | — | 5 min | Scheduled ingest cron |
| content-cron | — | 30 min | Scheduled content cron |
| neo4j | 7474/7687 | — | Graph database (20M+ nodes) |

## Architecture

```
+-------------------+     +------------------+     +-----------------+
|  Job Trigger API  |     |  Ingest Service  |     |  Sync Service   |
|  (HTTP on :8766)  |     |  (loop 5 min)    |     |  (daily sync)    |
+-------------------+     +------------------+     +-----------------+
         |                        |                        |
         v                        v                        v
+-------------------+     +------------------+     +-----------------+
|  .job Queue       |<----|  Ingest Worker   |<----|  Legacy Drop     |
|  /nas/drop/       |     |  (loop 30s)      |     |  /incoming/      |
|   claimed/        |     |                  |     |   deathstar/     |
|   done/           |     |                  |     +-----------------+
|   failed/         |     |                  |
+-------------------+     |                  |
                            |                  |
                            v                  v
                     +------------------+     +-----------------+
                     |  Neo4j Graph DB  |     |  Content OS     |
                     |  :7687 (:7474)   |     |  (cron 30 min)  |
                     +------------------+     +-----------------+
```

## Data Model

Core Neo4j node types:

| Label | Count | Description |
|-------|-------|-------------|
| PhoneLog | 20M | Phone call/SMS records |
| DashcamEmbedding | 4.2M | Dashcam video embeddings |
| YOLODetection | 4.1M | Vehicle/object detections |
| Frame | 3.7M | Video frames |
| Utterance | 420K | Speech utterances |
| Segment | 361K | Transcript segments |
| Speaker | 233K | Speaker entities |
| Transcription | 64K | Transcription records |

## Key Files

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Service definitions |
| `Dockerfile` | Container image |
| `deploy/job_trigger_api.py` | HTTP API server |
| `deploy/worker_ingest.sh` | Distributed worker |
| `deploy/sync_from_legacy_drop.sh` | Legacy sync |
| `deploy/create_job.sh` | Job creation helper |
| `deploy/start-cron.sh` | Cron daemon starter |
| `deploy/cron/ingest.crontab` | Ingest schedule |
| `deploy/cron/content_generation.crontab` | Content schedule |
| `deploy/path_profiles.env.example` | Environment template |
| `run_ingest_all.sh` | Main ingest runner |
| `ingest_transcriptsv5_3.py` | Python ingest script |
| `bin/auto-ingest` | Machine-agnostic CLI (primary entrypoint); `run-all` chains copy→diarize→ingest→reconcile→classify→link→yolo. All config from `config.yaml` + env. |
| `run_all_optimized.sh` | Deprecated shim over `bin/auto-ingest run-all` |
| `auto_ingest/backend.py` | Auto-detects CUDA/ROCm/MLX/ONNX compute; routes torch/faiss/YOLO device selection |

## Management

```bash
cd /home/scott/git/auto-ingest

# Start/stop
docker compose up -d          # start all
docker compose down            # stop all
docker compose up -d --build  # rebuild and start

# Logs
docker compose logs -f ingest-worker
docker compose logs -f ingest-service
docker compose logs -f sync-service

# Neo4j
docker exec neo4j cypher-shell -u neo4j -p knowledge_graph_2026 "RETURN 1"
docker exec neo4j cypher-shell -u neo4j -p knowledge_graph_2026 \
  "MATCH (n) RETURN labels(n) AS label, count(*) AS cnt ORDER BY cnt DESC LIMIT 10"

# Queue
ls -lah /nas/drop/ /nas/drop/claimed/ /nas/drop/done/ /nas/drop/failed/
```

## Troubleshooting

See [Troubleshooting Skill](deploy/skills/auto-ingest-troubleshooting/SKILL.md) for:
- Neo4j connection failures
- libGL/libvpx errors
- Legacy drop sync issues
- Job queue problems
- Cron job failures
- Diagnostic commands
