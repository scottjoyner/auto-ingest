# Birdcam Integration Skill

## When to use this skill
Use this skill when a task involves the `birdcam/` integration, including:
- camera detection worker behavior
- Neo4j graph persistence
- outbox replay/reliability behavior
- birdcam API/CLI/runtime deployment
- birdcam-only docker/compose operations

## Scope boundaries
- Birdcam is an additive integration and must not replace core auto-ingest services.
- Prefer editing birdcam-specific artifacts:
  - `birdcam/**`
  - `Dockerfile.birdcam`
  - `docker-compose.birdcam.yml`
  - `config.example.yaml` (birdcam fields)
  - `docs/birdcam_deployment_guide.md`
- Avoid overwriting root platform deployment defaults unless explicitly requested.

## Architectural rules
1. Neo4j is authoritative for camera/event/detection/clip graph metadata.
2. SQLite may be used only for local operational state (e.g., outbox/cache).
3. Filesystem remains source of truth for MP4/JPG binary payloads.
4. Writes should be idempotent where possible (`MERGE` semantics).
5. On Neo4j failure, enqueue write operations in outbox and replay later.

## Common commands
- Init schema:
  - `python -m birdcam.cli graph init-schema --config config.yaml`
- Graph health:
  - `python -m birdcam.cli graph check --config config.yaml`
- Replay outbox:
  - `python -m birdcam.cli graph replay-outbox --config config.yaml`
- Run worker:
  - `python -m birdcam.cli run --config config.yaml`
- File-mode test:
  - `python -m birdcam.cli detect-file --input sample.mp4 --config config.yaml`
- Run API:
  - `python -m birdcam.cli api --config config.yaml`

## Implementation checklist for agents
- [ ] Keep birdcam integration separate from core platform services.
- [ ] Add/adjust tests under `tests/test_*birdcam*` and graph tests.
- [ ] Verify Neo4j schema constraints/indexes if graph model changes.
- [ ] Ensure outbox logic still works for transient graph failures.
- [ ] Update deployment guide for any operational command/config changes.

## Handoff expectations
In final responses, include:
- files changed
- commands executed
- test results and limitations
- any migration or rollout cautions
