# Offline Swarm Compute Architecture — auto-ingest Integration Plan

_Last updated: 2026-05-26_

## Role of this repo

auto-ingest is the data-plane and worker layer of the offline swarm. It should scan, normalize, transcribe, detect, summarize, package, and persist local artifacts. It should report state and outcomes to AssistX, while keeping binary payloads on local/NAS storage.

The global control plane should live in `scottjoyner/auto-assist`. Sophia should own voice/auth edge behavior. auto-ingest should own local data processing capabilities and publish normalized events/results.

## Current baseline

This repo currently includes:

- Content OS local-first workflow with approval gates.
- Local source adapters for notes, transcripts, PDFs, CSV/JSON metadata, and manifests.
- Containerized ingest service, ingest worker, content service, ingest cron, and content cron.
- NAS/drop queue worker pattern with atomic claim/done/failed folders.
- Data-family configuration for audio, dashcam, bodycam, legacy drops, and shared mounts.
- Legacy drop sync flow for `deathstar-XPS-8920` into canonical x1-370/NAS paths.
- Birdcam integration with API, worker, Neo4j graph repository, local outbox replay, clip storage, and tests.

## Target auto-ingest responsibilities

1. Provide reusable worker capabilities for the swarm:
   - `ingest.scan_files`
   - `ingest.transcribe_audio`
   - `ingest.normalize_transcript`
   - `ingest.extract_media_metadata`
   - `vision.detect_objects`
   - `vision.detect_birds`
   - `content.scan_inputs`
   - `content.generate_brief`
   - `content.verify_draft`
2. Keep local files/NAS as the binary artifact source of truth.
3. Store durable metadata, detections, transcript references, and artifact records in Neo4j.
4. Use SQLite only for local operational state such as outbox/cache/claim bookkeeping.
5. Emit normalized events to AssistX for every meaningful state transition.
6. Accept tasks from AssistX by capability while preserving NAS drop queue compatibility.

## Contract with AssistX

Every worker should eventually support the AssistX task lifecycle:

- poll available work by capability.
- claim a task with a lease.
- heartbeat while running.
- emit artifact metadata as outputs are produced.
- complete/fail with structured results.
- replay outbox state when AssistX/Neo4j was unavailable.

Example event envelope:

```json
{
  "event_id": "uuid-or-deterministic-key",
  "event_type": "ingest.transcript.completed",
  "source_repo": "auto-ingest",
  "source_service": "ingest-worker",
  "node_id": "registered-swarm-node-id",
  "occurred_at": "ISO-8601",
  "idempotency_key": "stable replay key",
  "subject": {
    "kind": "file",
    "id": "sha256-or-canonical-path"
  },
  "payload": {
    "source_path": "storage-root-relative-path",
    "duration_seconds": 0,
    "language": "en",
    "model": "local-model-name"
  },
  "artifact_refs": [
    {
      "kind": "transcript",
      "uri": "file:///nas/...",
      "sha256": "optional"
    }
  ],
  "privacy": {
    "pii": true,
    "retention_class": "private-local"
  }
}
```

## Worker registration contract

Each auto-ingest worker should advertise:

```yaml
node_id: x1-370
service: auto-ingest-worker
capabilities:
  - ingest.scan_files
  - ingest.transcribe_audio
  - vision.detect_objects
storage_roots:
  - name: nas1
    container_path: /nas
    host_path: /media/scott/NAS1
health:
  last_seen_at: ISO-8601
  current_task_id: optional
  queue_backlog: number
```

## Next implementation phases

### Phase 1 — Shared event/outbox client

- Add an AssistX event client with retry and local outbox.
- Reuse birdcam outbox lessons, but make the event shape generic.
- Emit events for file seen, file normalized, transcript complete, detection created, clip created, and job failed.

### Phase 2 — Capability advertisement

- Add `auto-ingest capabilities` CLI command.
- Output JSON/YAML describing current host capabilities, mounts, and enabled modules.
- Add optional `POST /api/swarm/register` call to AssistX.

### Phase 3 — AssistX task polling mode

- Keep NAS drop queue support.
- Add AssistX polling mode for tasks by capability.
- Map AssistX tasks to existing scripts/jobs:
  - `run_ingest_all.sh`
  - `deploy/create_job.sh audio|dashcam|bodycam`
  - Content OS commands
  - birdcam worker actions

### Phase 4 — Artifact normalization

- Standardize artifact IDs, paths, sha256, retention classes, and producer task links.
- Prefer storage-root-relative paths plus optional host/container resolved paths.
- Mark protected/evidence-grade artifacts so cleanup jobs cannot prune them.

### Phase 5 — Model and compute integration

- Report local compute profile: CPU threads, GPU availability, memory, storage mount health.
- Let AssistX route STT/vision/content jobs based on capabilities and benchmarks.
- Add job profiles for CPU-only, GPU-preferred, and IO-heavy workloads.

### Phase 6 — End-to-end swarm demo

- AssistX dispatches a transcript or birdcam/vision job.
- auto-ingest claims and runs it.
- Outputs are written to NAS/local storage.
- Metadata and event trace are persisted in Neo4j.
- AssistX dashboard shows task, worker, artifacts, and graph links.

## Design-decision questions for Scott

1. Should NAS drop jobs remain the primary queue for heavy ingest, or become a fallback behind AssistX task polling?
2. Which machine should be the first auto-ingest production worker: x1-370, deathstar-XPS-8920, mini-pc-22, or another host?
3. Which data family should be the first swarm demo: audio, dashcam, bodycam, birdcam, or Content OS?
4. Should artifact paths be stored as absolute host paths, container paths, storage-root-relative paths, or all three?
5. Which artifacts should be `protected` by default?
6. Should transcript output skip logic remain artifact-exists based, not mtime/freshness based?
7. Should birdcam stay fully separate under `birdcam/`, or should shared worker/outbox utilities move into a common package?
8. Should auto-ingest write directly to Neo4j for all outputs, or emit events to AssistX and let AssistX perform graph reconciliation?
9. Should Content OS be treated as a swarm worker capability, or remain a separate local CLI workflow for now?
10. What is the first minimum working end-to-end acceptance test for this repo?

## Immediate next docs after answers

- `docs/swarm_contracts/auto_ingest_event_contract.md`
- `docs/swarm_contracts/artifact_contract.md`
- `docs/swarm_contracts/worker_capabilities.md`
- `docs/swarm_contracts/assistx_task_polling.md`
