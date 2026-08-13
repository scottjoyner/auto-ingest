# Production Hardening Plan

Status: active hardening branch

This branch intentionally prioritizes reliability, recoverability, testability, and operational clarity over new ingestion features.

## Goals

1. Prevent configuration and secret regressions.
2. Establish real Neo4j integration testing with deterministic fixtures.
3. Verify Docker images and Compose configuration in CI.
4. Define safe contracts for migrations, backfills, and destructive operations.
5. Converge ingestion onto one canonical job/stage execution model.
6. Make failures observable, resumable, and diagnosable.
7. Reconcile documentation with the actual deployed architecture.
8. Move known technical debt out of prose and into tracked GitHub work.

## P0 — Production safety contracts

- Remove machine-specific Neo4j/Tailscale addresses from shared fallback configuration.
- Add CI checks that reject hard-coded fleet IPs in shared runtime defaults.
- Extend secret scanning beyond Neo4j password literals.
- Require explicit dry-run support for migrations/backfills where practical.
- Add bounded batch, checkpoint/resume, and abort-threshold conventions for graph mutations.
- Add preflight cardinality reporting before high-volume graph writes/deletes.
- Define a destructive-operation safety helper used by migration/backfill scripts.

## P0 — CI and integration testing

Create a layered test matrix:

- `unit`: no network, no database, no model downloads.
- `integration`: ephemeral Neo4j with deterministic fixture graph.
- `e2e`: synthetic media/job through the canonical ingest path.
- `ml`: optional model/runtime import and inference smoke tests.
- `destructive`: migration/backfill idempotency and recovery behavior.

CI should additionally validate:

- `docker compose config`.
- Docker image build.
- service healthchecks.
- schema/index creation.
- PhoneLog spatial migration idempotency.
- configuration precedence and portability.

## P0 — Architecture drift

- Remove stale job-trigger API references from docs and runbooks.
- Reconcile README service topology with `docker-compose.yml`.
- Decide on one scheduler for recurring ingestion rather than overlapping daemon + cron paths.
- Make CLI, local worker, fleet worker, and scheduled execution invoke the same stage executor.

## Canonical job lifecycle

Target states:

`DISCOVERED -> READY -> CLAIMED -> RUNNING -> VALIDATING -> DONE`

Failure/recovery states:

`RETRYABLE`, `BLOCKED`, `QUARANTINED`, `DEAD`

Each job/stage should record at minimum:

- stable job id
- source identity/hash
- stage name + stage version
- owner/worker
- lease expiry
- attempt count
- start/end timestamps
- input/output cardinality
- error class/message
- provenance/output references

## P1 — Neo4j contract testing

Use an ephemeral Neo4j service in CI and seed a small fixture graph containing representative nodes/relationships for:

- PhoneLog spatial data
- transcription/segment/utterance
- speakers and anchors
- media and embeddings metadata
- trips/places/location links
- jobs/claims/stages

Tests should validate schema creation, query compatibility, migration idempotency, retries, and transaction batching without depending on production data.

## P1 — Observability and recovery

- Structured logs with job/stage identifiers.
- Health checks that validate useful work, not just process existence.
- Failure counters and last-success timestamps.
- Retry/quarantine visibility.
- A recovery runbook for interrupted ingestion and graph migrations.
- Explicit backup/preflight guidance before destructive graph changes.

## P1 — Code organization

Progressively move runtime logic toward clear package boundaries:

- `core/`: config, jobs, stages, state, errors, logging
- `graph/`: schema, migrations, repositories
- `media/`: audio/video/image ingestion
- `speech/`: transcription, diarization, speaker identity
- `vision/`: YOLO and visual embeddings
- `knowledge/`: entities, concepts, papers, Signal bridges
- `personal/`: recall and timeline
- `fleet/`: discovery, dispatch, claims
- `content/`: planning, rendering, publishing
- `ops/`: health, metrics, recovery

Shell scripts should become thin operational wrappers rather than owning pipeline business logic.

## Initial branch checklist

- [ ] Fix Neo4j shared fallback regression.
- [ ] Add forbidden fleet-IP/config regression test.
- [ ] Reconcile README with current Compose topology.
- [ ] Add Compose validation to CI.
- [ ] Add Docker build smoke job.
- [ ] Add ephemeral Neo4j integration job.
- [ ] Add deterministic graph fixture.
- [ ] Test PhoneLog migration against fixture data.
- [ ] Introduce pytest markers (`unit`, `integration`, `e2e`, `ml`, `destructive`).
- [ ] Audit Ruff exclusions and convert them to tracked debt.
- [ ] Evaluate stale PR #1 against current architecture.
- [ ] Create GitHub issues for remaining hardening work.

## Merge criteria

This branch should not be promoted from draft until:

1. CI protects the critical configuration/safety invariants.
2. A real Neo4j integration suite passes.
3. Docker/Compose are validated automatically.
4. documentation describes the actual runtime topology.
5. destructive graph operations have documented and tested safety behavior.
6. remaining gaps are represented by explicit GitHub issues rather than only prose.
