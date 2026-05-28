# Birdcam Deployment Guide (Developer)

## Purpose
This guide explains how to deploy the **birdcam integration module** as a separate service within the auto-ingest platform, with Neo4j as the authoritative metadata store and local disk for media artifacts.

## Architecture Summary
- Birdcam runtime: `birdcam/` package.
- API/worker entrypoint: `python -m birdcam.cli`.
- Container build: `Dockerfile.birdcam`.
- Compose stack: `docker-compose.birdcam.yml`.
- Canonical metadata store: Neo4j (`DetectionEvent`, `Detection`, `Clip`, etc.).
- Local durability fallback: SQLite outbox (`GraphOutbox`) for failed graph writes.
- Media storage: mounted SSD path, clips/thumbnails/metadata files on filesystem.

## Prerequisites
- Docker + Docker Compose.
- Reachable Neo4j (remote Tailscale or local dev profile).
- Camera stream URL (RTSP/HTTP) or local test file.
- SSD storage mount (recommended), e.g. `/mnt/ssd/birdcam`.

## Key Files
- `Dockerfile.birdcam`
- `docker-compose.birdcam.yml`
- `config.example.yaml`
- `birdcam/config.py`
- `birdcam/graph/cypher.py`
- `birdcam/graph/repository.py`
- `birdcam/graph/outbox.py`
- `birdcam/worker.py`
- `birdcam/api.py`

## Configuration
Create a real runtime config from `config.example.yaml`.

### Required values
- `camera_id`
- `stream_url`
- `storage_root`
- `model_name_or_path`
- `detection_class`
- `confidence_threshold`
- `pre_roll_seconds` / `post_roll_seconds`
- `event_merge_seconds` / `cooldown_seconds`
- `sqlite_path` (used for outbox/state, not canonical detections)
- `neo4j.*`

### Environment variables
Set via shell/CI/compose:
- `NEO4J_URI`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`

## Deployment Paths

### Option A: Connect to existing Neo4j (recommended)
1. Set Neo4j env vars.
2. Ensure `config.yaml` points to mounted storage path.
3. Build and run:
   ```bash
   docker compose -f docker-compose.birdcam.yml up --build -d
   ```
4. Initialize schema (one-time, per DB):
   ```bash
   docker compose -f docker-compose.birdcam.yml exec birdcam \
     python -m birdcam.cli graph init-schema --config /config/config.yaml
   ```
5. Health check:
   ```bash
   docker compose -f docker-compose.birdcam.yml exec birdcam \
     python -m birdcam.cli graph check --config /config/config.yaml
   ```

### Option B: Local dev Neo4j profile
```bash
docker compose -f docker-compose.birdcam.yml --profile dev-neo4j up --build -d
```
Use matching credentials from compose (`neo4j/change-me`) unless overridden.

## Operational Commands
- Start worker:
  ```bash
  python -m birdcam.cli run --config config.yaml
  ```
- Process local file:
  ```bash
  python -m birdcam.cli detect-file --input sample.mp4 --config config.yaml
  ```
- Start API:
  ```bash
  python -m birdcam.cli api --config config.yaml --host 0.0.0.0 --port 8000
  ```
- Replay outbox:
  ```bash
  python -m birdcam.cli graph replay-outbox --config config.yaml
  ```
- List events:
  ```bash
  python -m birdcam.cli graph list-events --config config.yaml
  ```

## Data and Retention
- Files are stored locally under `storage_root`.
- Neo4j stores metadata/relationships.
- Retention deletes old files and marks clips deleted in Neo4j with deletion metadata.

## Troubleshooting
- **Neo4j auth/connection failure**: verify URI/credentials/network; outbox should queue failed writes.
- **No events in graph**: run `graph check`, inspect worker logs, replay outbox.
- **No clips saved**: verify storage mount writable and pre/post-roll settings.
- **Model unavailable**: confirm model path and runtime dependencies.

## Verification Checklist
- [ ] `graph init-schema` executed successfully.
- [ ] Camera node upserted.
- [ ] DetectionEvent + Detection nodes appear.
- [ ] Clip/Thumbnail relationships created after finalization.
- [ ] Outbox replay works after simulated Neo4j outage.

## Suggested Cypher Queries
```cypher
MATCH (c:Camera)-[:PRODUCED_EVENT]->(e:DetectionEvent)
RETURN c.camera_id, e.event_id, e.start_time, e.confidence_max
ORDER BY e.start_epoch_ms DESC
LIMIT 50;
```

```cypher
MATCH (e:DetectionEvent)-[:HAS_CLIP]->(clip:Clip)
RETURN e.event_id, clip.path, clip.size_bytes, clip.deleted
ORDER BY e.start_epoch_ms DESC;
```
