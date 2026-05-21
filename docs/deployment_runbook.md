# Auto-Ingest Deployment Runbook

This guide explains **what each deployment file does**, how to configure it, and how to run the services reliably across hosts.

## 1) File-by-file operational guide

### `Dockerfile`
Purpose:
- Builds a reusable runtime image for ingestion scripts and Content OS CLI workflows.

How it works:
1. Uses `python:3.11-slim`.
2. Installs OS-level runtime packages (`cron`, `tzdata`, certs).
3. Installs project package + requirements.
4. Defaults to `python -m content_os --help`.

Operator actions:
- Rebuild after Python dependency or script changes:
  ```bash
  docker compose build --no-cache
  ```

---

### `docker-compose.yml`
Purpose:
- Defines service topology for long-running and scheduled ingestion/content workers.

Services:
- `ingest-service`: runs `run_ingest_all.sh` directly.
- `ingest-worker`: claims and executes queued `.job` files from NAS drop.
- `content-service`: runs Content OS CLI.
- `ingest-cron`: cron-driven ingest trigger.
- `content-cron`: cron-driven content trigger.

Operator actions:
- Start all:
  ```bash
  docker compose up -d
  ```
- Start only distributed ingest workers:
  ```bash
  docker compose up -d ingest-worker ingest-cron
  ```
- View live logs:
  ```bash
  docker compose logs -f ingest-worker ingest-cron
  ```

---

### `deploy/path_profiles.env.example`
Purpose:
- Template for host-specific path and Neo4j config.

Operator actions:
1. Copy to `.env`:
   ```bash
   cp deploy/path_profiles.env.example .env
   ```
2. Edit for host reality:
   - `NAS_ROOT`
   - `AUDIO_ROOT` (S drive target)
   - `DASHCAM_ROOT` / `BODYCAM_ROOT`
   - `NEO4J_URI`, credentials
3. Validate env interpolation:
   ```bash
   docker compose config
   ```

---

### `deploy/worker_ingest.sh`
Purpose:
- Distributed worker claim loop for NAS queue jobs.

How it works:
1. Scans `${DROP_ROOT}` for `*.job`.
2. Atomically `mv` claims to `claimed/`.
3. Executes claimed script.
4. Moves completed jobs to `done/`; failures to `failed/`.

Operator actions:
- Confirm worker queue paths exist and are shared across hosts.
- Ensure `.job` payloads are executable scripts with `set -euo pipefail`.

---

### `deploy/sync_from_legacy_drop.sh`
Purpose:
- Moves/copies legacy drop content from deathstar into canonical x1-370 roots before ingest.

How it works:
1. Reads `LEGACY_DROP_ROOT` (default `/nas/fileserver/incoming/deathstar`).
2. Rsyncs folders (`audio`, `dashcam`, `bodycam`, `transcripts`, `yolo`, `metadata`).
3. Copies sidecar transcript/metadata file types (`.srt`, `.vtt`, `.txt`, `.json`, `.csv`).

Operator actions:
- Ensure SMB/NAS path from deathstar is mounted on x1-370 under NAS1.
- Run standalone for verification:
  ```bash
  docker compose run --rm sync-service
  ```

---

### `deploy/create_job.sh`
Purpose:
- Helper to enqueue jobs by data type (`audio`, `dashcam`, `bodycam`, `all`).

Operator actions:
- Create jobs:
  ```bash
  ./deploy/create_job.sh audio
  ./deploy/create_job.sh dashcam
  ./deploy/create_job.sh bodycam
  ./deploy/create_job.sh all
  ```
- Verify queue:
  ```bash
  ls -lah /nas/drop/*.job
  ```

---

### `deploy/start-cron.sh`
Purpose:
- Installs selected crontab and runs cron foreground in container.

Operator actions:
- Swap cron schedule by setting `CRON_FILE` in compose service.
- Recreate cron service after cron file edits:
  ```bash
  docker compose up -d --force-recreate ingest-cron content-cron
  ```

---

### `deploy/cron/ingest.crontab`
Purpose:
- Ingest schedule trigger.

Current behavior:
- Runs every 5 minutes.

Operator actions:
- Tune cadence by editing cron expression.
- Keep script lock behavior enabled in `run_ingest_all.sh`.

---

### `deploy/cron/content_generation.crontab`
Purpose:
- Content workflow schedule trigger.

Current behavior:
- Runs every 30 minutes.

Operator actions:
- Adjust cadence based on publishing and review workload.

---

### `README.md`
Purpose:
- High-level project and deployment overview.

Operator actions:
- Treat `README.md` as entry-point summary.
- Use this runbook for detailed execution + operations.

## 2) Standard deployment procedure

1. Configure host env:
   ```bash
   cp deploy/path_profiles.env.example .env
   # edit .env
   ```
2. Build image:
   ```bash
   docker compose build
   ```
3. Start services:
   ```bash
   docker compose up -d sync-service ingest-service ingest-worker content-service ingest-cron content-cron
   ```
4. Verify:
   ```bash
   docker compose ps
   docker compose logs --tail=100 ingest-service ingest-worker ingest-cron
   ```

## 3) Multi-host distributed rollout

On each worker host:
1. Mount the same NAS share path used by `NAS_ROOT`.
2. Sync repo and `.env`.
3. Start at least `ingest-worker`.

Dispatcher host:
1. Generate jobs with `deploy/create_job.sh`.
2. Monitor `drop/claimed`, `drop/done`, `drop/failed`.

## 4) Operational checks and troubleshooting

### Health checks
- Containers running:
  ```bash
  docker compose ps
  ```
- Queue movement:
  ```bash
  ls -lah /nas/drop /nas/drop/claimed /nas/drop/done /nas/drop/failed
  ```
- Cron output:
  ```bash
  docker compose exec ingest-cron tail -n 200 /var/log/cron-ingest.log
  docker compose exec content-cron tail -n 200 /var/log/cron-content.log
  ```

### Common issues
1. **No jobs processed**
   - Check `${DROP_ROOT}` mapping and permissions.
2. **Duplicate work risk**
   - Ensure job claim path is same shared NAS and `mv` is atomic on that filesystem.
3. **Neo4j connection failures**
   - Verify `.env` endpoint and credentials after DB migration.
4. **Missing input files**
   - Reconcile path map between S-drive migration and NAS1 legacy roots.

## 5) Change management recommendations

- Treat `.env` as host profile; keep a versioned `.env.example` only.
- Change one schedule at a time and observe queue pressure for 24h.
- Keep ingestion and content services independent so one pipeline cannot starve the other.
