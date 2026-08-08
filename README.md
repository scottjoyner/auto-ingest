# Auto-Ingest

Auto-Ingest is the ingestion and enrichment pipeline for dashcam, audio, bodycam, personal media, and knowledge-graph data. It writes into an externally configured Neo4j database and supports local CLI execution, NAS-backed distributed workers, scheduled jobs, media enrichment, speaker linking, and downstream content workflows.

The primary goal of the current hardening work is to make those paths deterministic, resumable, observable, and safe to operate against a large production graph.

## Canonical entrypoint

`bin/auto-ingest` is the preferred machine-agnostic CLI. Paths and credentials are resolved through `config.yaml` plus environment variables; compute selection is handled by `auto_ingest/backend.py`.

```bash
bin/auto-ingest run-all
bin/auto-ingest link-speakers
bin/auto-ingest caps
bin/auto-ingest status
bin/auto-ingest ingest
bin/auto-ingest whoami
bin/auto-ingest tiktok
bin/auto-ingest worker
bin/auto-ingest claims
bin/auto-ingest reap
```

`run_all_optimized.sh` is a deprecated compatibility shim over `bin/auto-ingest run-all`.

## Runtime topology

The current `docker-compose.yml` contains six application services:

| Service | Schedule | Purpose |
|---|---:|---|
| `ingest-service` | loop / 5 min | Runs the main ingestion pipeline |
| `ingest-worker` | loop / 30 sec | Claims and processes NAS-backed jobs |
| `sync-service` | loop / 10 min | Syncs legacy input locations into canonical storage |
| `content-service` | loop / 30 min | Runs Content OS status/workflow checks |
| `ingest-cron` | cron | Scheduled ingestion path |
| `content-cron` | cron | Scheduled content workflow path |

Neo4j is **not** provisioned by this Compose file. Set `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, and `NEO4J_DB` to the target graph service.

The deprecated unauthenticated HTTP job-trigger API has been removed. Do not depend on port `8766` or `/api/enqueue`; use the CLI, `deploy/create_job.sh`, or the claim/job workflow instead.

## Quick start

```bash
cd /home/scott/git/auto-ingest
cp deploy/path_profiles.env.example .env
# edit .env for this host

docker compose config
docker compose up -d --build
docker compose ps
```

Inspect health and logs with:

```bash
./deploy/manage.sh health
docker compose logs --tail=200 ingest-service
docker compose logs --tail=200 ingest-worker
```

## Job flow

Distributed ingestion uses durable job/claim semantics rather than an unauthenticated trigger API.

```bash
./deploy/create_job.sh dashcam
./deploy/create_job.sh audio
./deploy/create_job.sh bodycam
./deploy/create_job.sh all

bin/auto-ingest claims
bin/auto-ingest reap
```

Target lifecycle for hardening work:

```text
DISCOVERED -> READY -> CLAIMED -> RUNNING -> VALIDATING -> DONE
                         |           |
                         +-> RETRYABLE / BLOCKED / QUARANTINED / DEAD
```

Execution paths should converge on the same stage contracts so a CLI run, local worker, fleet worker, and scheduled run produce equivalent state transitions and provenance.

## Production safety

High-volume graph and filesystem operations must be treated as recoverable migrations rather than ad-hoc scripts. New migration/backfill work should provide, where applicable:

- dry-run or preflight cardinality reporting;
- bounded transaction/batch sizes;
- checkpoint/resume behavior;
- idempotency;
- validation of source data before writes;
- explicit failure reporting and non-zero exit codes;
- post-run verification;
- backup/recovery guidance for destructive operations.

For example, the PhoneLog spatial migration can be checked before mutation:

```bash
python scripts/migrate_phonelog_spatial.py --dry-run
python scripts/migrate_phonelog_spatial.py --batch-size 5000
```

The migration rejects oversized batches, skips invalid coordinates, and is safe to resume.

## Testing

Tests are separated into execution layers with pytest markers:

```bash
# fast, dependency-light tests
python -m pytest tests -m "not integration and not ml and not e2e and not destructive"

# real service tests require their service-specific environment
python -m pytest tests/integration -m integration
```

CI validates:

- Ruff and secret/configuration regression guards;
- fast unit/compatibility tests;
- `docker compose config` and expected service topology;
- a real Neo4j service with isolated migration fixtures;
- production Docker image build and entrypoint smoke execution.

See `docs/PRODUCTION_HARDENING_PLAN.md` for the active hardening roadmap.

## Configuration

Shared fallback configuration must remain machine-agnostic. In particular, the default Neo4j URI is local:

```text
bolt://localhost:7687
```

Fleet-specific Tailscale hosts/addresses belong in explicit machine profiles or environment variables, not shared runtime defaults.

Important environment variables include:

```text
NEO4J_URI
NEO4J_USER
NEO4J_PASSWORD
NEO4J_DB
FILESERVER_ROOT
HOT_STORAGE_ROOT
COLD_STORAGE_ROOT
SCAN_ROOTS
DASHCAM_ROOT
AUDIO_ROOT
BODYCAM_ROOT
TRANSCRIPT_ROOT
DROP_ROOT
```

## Storage and data model

The production graph is large, so queries and migrations must be bounded. Representative graph domains include:

- PhoneLog/location data;
- dashcam/frame/YOLO detections and embeddings;
- transcription/segment/utterance data;
- speaker identity and voiceprint linkage;
- media, trips, places, and personal recall;
- papers, concepts, entities, and knowledge bridges;
- ingest jobs, claims, stages, and provenance.

Do not infer current production cardinalities from documentation; use live diagnostics before migration or capacity decisions.

## Key files

| Path | Purpose |
|---|---|
| `bin/auto-ingest` | primary CLI |
| `auto_ingest_config.py` | shared configuration resolution |
| `auto_ingest/backend.py` | compute/backend detection |
| `docker-compose.yml` | application service topology |
| `Dockerfile` | production image |
| `deploy/worker_ingest.sh` | distributed NAS worker |
| `deploy/create_job.sh` | durable job creation helper |
| `deploy/start-cron.sh` | cron runner |
| `scripts/migrate_phonelog_spatial.py` | bounded PhoneLog spatial migration |
| `tests/integration/` | real external-service contract tests |
| `docs/PRODUCTION_HARDENING_PLAN.md` | active hardening plan |

## Documentation

- `docs/system_design.md` — system architecture and data model
- `docs/deployment_runbook.md` — operational deployment guidance
- `docs/architecture.md` — Content OS architecture
- `docs/PRODUCTION_HARDENING_PLAN.md` — current reliability/test/recovery work
- `deploy/skills/auto-ingest-troubleshooting/SKILL.md` — operational troubleshooting

## Current hardening direction

The repository has substantial capability but historically accumulated multiple execution paths, legacy scripts, and production assumptions. The hardening branch is deliberately prioritizing safety and contracts over new features: real integration tests, Docker validation, architecture reconciliation, migration safety, scheduler convergence, observability, and recovery behavior.
