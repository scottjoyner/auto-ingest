# Auto-Ingest System Design

## Overview

Auto-Ingest is a containerized ingestion and enrichment system for dashcam, audio, bodycam, personal media, and knowledge-graph data. It combines a machine-agnostic CLI, NAS-backed jobs/claims, recurring service loops, scheduled cron entrypoints, and an externally configured Neo4j database.

The unauthenticated HTTP job-trigger API previously documented on port `8766` is retired and is not part of the current Compose topology.

## Current architecture

```text
                     +-----------------------+
                     |   bin/auto-ingest     |
                     | canonical local CLI   |
                     +-----------+-----------+
                                 |
                                 v
+-------------+       +-----------------------+       +------------------+
| Source data | ----> | intake / job / claim  | ----> | stage execution  |
+-------------+       +-----------------------+       +---------+--------+
                                                                  |
                              +-----------------------------------+----------------------+
                              |                                   |                      |
                              v                                   v                      v
                       +-------------+                      +-------------+       +-------------+
                       | filesystem  |                      |   Neo4j     |       | content /   |
                       | artifacts   |                      | external DB |       | downstream  |
                       +-------------+                      +-------------+       +-------------+

Recurring execution surfaces:
  ingest-service  -> main ingest loop
  ingest-worker   -> NAS-backed job worker
  sync-service    -> legacy/canonical storage sync
  content-service -> Content OS monitoring/workflow loop
  ingest-cron     -> scheduled ingest path
  content-cron    -> scheduled content path
```

A hardening objective is to converge the CLI, local workers, fleet workers, and scheduled entrypoints on the same stage executor and state-transition contracts.

## Compose services

### `ingest-service`

Runs `run_ingest_all.sh` in a five-minute loop. A flock-based guard reduces overlap, but long-term scheduling should converge to a single canonical scheduler/executor path.

### `ingest-worker`

Claims durable jobs from the configured drop root and executes them. Job claims are expected to be atomic/resumable and to end in explicit success/failure state.

### `sync-service`

Syncs legacy input locations into canonical roots. Filesystem operations should remain bounded and fail closed when required mounts are absent.

### `content-service`

Runs Content OS workflow/status checks independently of ingest execution.

### `ingest-cron`

Runs scheduled ingest work via `deploy/start-cron.sh` and `deploy/cron/ingest.crontab`.

### `content-cron`

Runs scheduled content work via `deploy/start-cron.sh` and `deploy/cron/content_generation.crontab`.

## External Neo4j

Neo4j is not defined as a service in `docker-compose.yml`. Runtime code receives the graph endpoint through configuration:

```text
NEO4J_URI
NEO4J_USER
NEO4J_PASSWORD
NEO4J_DB
```

The shared default URI is `bolt://localhost:7687`. Fleet-specific hosts or Tailscale addresses must be supplied by an explicit machine profile or environment variable rather than embedded in shared fallback code.

## Canonical execution model

Target job lifecycle:

```text
DISCOVERED -> READY -> CLAIMED -> RUNNING -> VALIDATING -> DONE
                         |           |
                         +-> RETRYABLE
                         +-> BLOCKED
                         +-> QUARANTINED
                         +-> DEAD
```

Each job/stage should persist enough state to answer:

- what input was processed;
- which stage/version processed it;
- which worker owned the lease;
- how many attempts occurred;
- when the stage started/finished;
- what outputs/provenance were produced;
- whether the operation can be retried safely;
- what error class/message caused a stop.

## Durable job queue

The existing NAS-backed queue is still a supported transport:

```text
<drop-root>/
  *.job
  claimed/
  done/
  failed/
```

Jobs can be created with:

```bash
./deploy/create_job.sh dashcam
./deploy/create_job.sh audio
./deploy/create_job.sh bodycam
./deploy/create_job.sh all
```

Graph-backed claim helpers are exposed through the CLI where appropriate:

```bash
bin/auto-ingest claims
bin/auto-ingest reap
```

The retired HTTP trigger service must not be reintroduced without authentication, authorization, request validation, and an explicit threat model.

## Data domains

The production graph contains multiple high-volume domains, including:

- PhoneLog and spatial/location records;
- dashcam frames, detections, and embeddings;
- transcription, segments, and utterances;
- speakers/voiceprints and identity anchors;
- media, trips, places, and personal recall;
- papers, concepts, entities, and knowledge bridges;
- ingest jobs, claims, stage state, and provenance.

Do not rely on hard-coded cardinalities in documentation for migration planning. Query live counts during preflight.

## Migration and backfill contract

Because graph mutations may touch millions of records, production migrations should follow these rules:

1. report preflight cardinalities;
2. support dry-run where feasible;
3. validate source data before mutation;
4. use bounded batch/transaction sizes;
5. be idempotent and resumable;
6. return non-zero on fatal failure;
7. verify post-run state;
8. document backup/recovery expectations for destructive changes.

`scripts/migrate_phonelog_spatial.py` is the first migration covered by a real Neo4j CI contract test. It validates coordinate ranges, caps batch sizes, supports dry-run, and can resume after interruption.

## Testing architecture

Tests are divided with pytest markers:

- `unit` — no external services or model downloads;
- `integration` — real external service, currently Neo4j;
- `e2e` — full pipeline contract tests;
- `ml` — runtime/model smoke tests;
- `destructive` — isolated migration/backfill mutation tests.

CI currently has distinct gates for:

- lint + lightweight tests;
- configuration portability/secret regression checks;
- Compose interpolation and expected service topology;
- real Neo4j integration tests;
- production Docker image build and entrypoint smoke test.

## Deployment

```bash
cd /home/scott/git/auto-ingest
cp deploy/path_profiles.env.example .env
# edit host-specific paths and credentials

docker compose config
docker compose up -d --build
docker compose ps
./deploy/manage.sh health
```

Neo4j must already be reachable at the configured endpoint.

## Operational risks under active hardening

The main system-level risks are no longer missing feature capability; they are drift and coordination risks:

- overlapping daemon and cron scheduling;
- machine-specific assumptions leaking into shared configuration;
- legacy scripts bypassing canonical stage contracts;
- migrations running against very large graph cardinalities;
- optional ML dependencies behaving differently across hosts;
- insufficient observability around retries, quarantine, and last-success state.

See `docs/PRODUCTION_HARDENING_PLAN.md` for the active remediation plan.
