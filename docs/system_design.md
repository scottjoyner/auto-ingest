# Auto-Ingest System Design

## Overview

Auto-Ingest is a containerized pipeline that ingests dashcam, audio, and bodycam recordings into a Neo4j knowledge graph. It features distributed workers, a job queue via NAS drop directory, a REST API for triggering jobs, and scheduled cron workers.

## Architecture

```
+-------------------+     +------------------+     +-----------------+
|  Job Trigger API  |     |  Ingest Service  |     |  Sync Service   |
|  (HTTP on :8766)  |     |  (loop 5 min)    |     |  (loop 10 min)   |
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

Data Flow:
  Source Files -> Sync -> /nas/fileserver/{audio,dashcam,bodycam}
                        -> Ingest -> Neo4j (PhoneLog, DashcamEmbedding, Frame, etc.)
                        -> Content OS (for content creation workflow)
```

## Services

### 1. Job Trigger API (job-api)
- **Port**: 8766 (host) -> 8765 (container)
- **Purpose**: REST API for enqueueing jobs and checking queue status
- **Endpoints**:
  - `GET /api/health` - Health check
  - `GET /api/status` - Queue status (pending, claimed, done, failed counts)
  - `POST /api/enqueue` - Enqueue a job (body: `{kind: "audio"|"dashcam"|"bodycam"|"all", scan_roots?}`)
  - `POST /api/run` - One-shot run (body: `{type: "ingest"|"sync"}`)
- **Key files**: `deploy/job_trigger_api.py`

### 2. Ingest Service (ingest-service)
- **Purpose**: Continuous loop that runs `run_ingest_all.sh` every 5 minutes
- **Mechanism**: Uses flock-based concurrency guard (`/tmp/ingest_transcripts.lock`)
- **Key files**: `run_ingest_all.sh`, `ingest_transcriptsv5_3.py`
- **Environment**: `SCAN_ROOTS`, `DASHCAM_ROOT`, `NEO4J_*`, `EMBED_MODEL_NAME`, `MODEL_PREF`, `LOCAL_TZ`

### 3. Ingest Worker (ingest-worker)
- **Purpose**: Distributed worker that claims `.job` files from NAS drop queue
- **Poll interval**: Every 30 seconds
- **Queue protocol**: Atomic `mv` to `claimed/`, then `done/` or `failed/`
- **Key files**: `deploy/worker_ingest.sh`

### 4. Sync Service (sync-service)
- **Purpose**: Syncs legacy drop from deathstar into canonical roots
- **Poll interval**: Every 10 minutes
- **Source**: `/nas/fileserver/incoming/deathstar/{audio,dashcam,bodycam,transcripts,yolo,metadata}`
- **Destination**: `/nas/fileserver/{audio,dashcam,bodycam,transcriptions,yolo,metadata}`
- **Key files**: `deploy/sync_from_legacy_drop.sh`

### 5. Content Service (content-service)
- **Purpose**: Runs Content OS CLI for content workflow monitoring
- **Poll interval**: Every 30 minutes
- **Key files**: `content_os/cli.py`, `content_os/core.py`

### 6. Ingest Cron (ingest-cron)
- **Purpose**: Scheduled ingest trigger (every 5 minutes)
- **Mechanism**: Runs crontab inside container, executes sync + ingest
- **Key files**: `deploy/start-cron.sh`, `deploy/cron/ingest.crontab`

### 7. Content Cron (content-cron)
- **Purpose**: Scheduled content workflow batch (every 30 minutes)
- **Key files**: `deploy/start-cron.sh`, `deploy/cron/content_generation.crontab`

### 8. Neo4j (neo4j)
- **Ports**: 7474 (HTTP), 7687 (Bolt)
- **Version**: neo4j:5.24-enterprise
- **Authentication**: neo4j / knowledge_graph_2026
- **Data**: 20M+ nodes, 50M+ relationships

## Data Model (Neo4j)

### Core Node Types
| Label | Count | Description |
|-------|-------|-------------|
| PhoneLog | 20M | Phone call/SMS records |
| DashcamEmbedding | 4.2M | Dashcam video embeddings |
| YOLODetection | 4.1M | Vehicle/object detections |
| Frame | 3.7M | Video frames |
| Utterance | 420K | Speech utterances |
| Segment | 361K | Transcript segments |
| Speaker | 233K | Speaker entities |
| PriceBar | 218K | Trading price bars |
| Summary | 179K | Content summaries |
| DashcamClip | 71K | Dashcam clips |
| Transcription | 64K | Transcription records |
| AudioSegment | 46K | Audio segments |
| Filing | 44K | Legal filings |
| AudioFile | 38K | Audio files |
| Entity | 14K | General entities |
| Link | 9K | Links between entities |

### Key Relationships
- `Transcription` -[:HAS_SEGMENT]-> `Segment`
- `Segment` -[:HAS_EMBEDDING]-> `DashcamEmbedding`
- `Frame` -[:HAS_DETECTION]-> `YOLODetection`
- `Transcription` -[:HAS_UTTERANCE]-> `Utterance`
- `Utterance` -[:HAS_SPEAKER]-> `Speaker`

## Job Queue System

### Drop Directory Structure
```
/nas/drop/
  *.job          # Pending jobs (created by API or create_job.sh)
  claimed/       # Currently being processed
  done/          # Successfully completed
  failed/        # Failed jobs
```

### Job File Format
```bash
#!/usr/bin/env bash
set -euo pipefail
cd /app
SCAN_ROOTS="/nas/fileserver/dashcam" DASHCAM_ROOT="/nas/fileserver/dashcam" /usr/bin/env bash run_ingest_all.sh
```

### Job Types
| Kind | Description | Default SCAN_ROOTS |
|------|-------------|-------------------|
| `audio` | Audio transcription ingest | `/nas/S/audio,/nas/fileserver/audio/transcriptions` |
| `dashcam` | Dashcam video ingest | `/nas/fileserver/dashcam` |
| `bodycam` | Bodycam video ingest | `/nas/fileserver/bodycam` |
| `all` | All data types | All SCAN_ROOTS |

## Deployment

### Prerequisites
- Docker + Docker Compose
- NAS mounted at `/media/scott/NAS1` (or configured via `.env`)
- Neo4j running on host (ports 7474, 7687 exposed)

### Quick Start
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

### Triggering Jobs
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
```

### Key Configuration (in .env or compose)
| Variable | Default | Description |
|----------|---------|-------------|
| NAS_ROOT | /media/scott/NAS1 | NAS mount point |
| DROP_ROOT | /nas/drop | Job queue directory |
| NEO4J_URI | bolt://localhost:7687 | Neo4j connection |
| NEO4J_USER | neo4j | Neo4j username |
| NEO4J_PASSWORD | knowledge_graph_2026 | Neo4j password |
| SCAN_ROOTS | (comma-separated) | Directories to scan |
| DASHCAM_ROOT | /nas/fileserver/dashcam | Dashcam root |
| AUDIO_ROOT | /nas/S/audio | Audio root |
| BODYCAM_ROOT | /nas/fileserver/bodycam | Bodycam root |
| TRANSCRIPT_ROOT | /nas/fileserver/audio/transcriptions | Transcripts root |
| LEGACY_DROP_ROOT | /nas/fileserver/incoming/deathstar | Legacy drop source |

## Troubleshooting

### Common Issues

1. **Neo4j connection fails**: On Linux Docker, use `host.docker.internal:host-gateway` in `extra_hosts` (already configured in compose)

2. **libGL errors**: ffmpeg requires libGL1 and libglib2.0-0 (already installed in Dockerfile)

3. **Legacy drop sync fails**: Directory `/nas/fileserver/incoming/deathstar` may not exist — it's now handled gracefully

4. **No jobs processed**: Check `/nas/drop/` permissions and that worker container has the mount

5. **Duplicate work**: The flock guard in `run_ingest_all.sh` prevents overlapping runs

### Log Locations
- Ingest worker: `./logs/ingest_YYYYMMDD_HHMMSS.log`
- Ingest cron: `/var/log/cron-ingest.log` (inside container)
- Sync: `/var/log/sync.log` (inside container)
- Cron: `/var/log/cron.log` (inside container)
