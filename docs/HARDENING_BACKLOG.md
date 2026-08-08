# Hardening execution backlog

This file mirrors the GitHub issue backlog at a high level so production work remains visible in-repo. GitHub issues are the authoritative work queue.

## P0

### Scheduler and executor convergence

- choose one canonical recurring scheduler;
- make daemon, cron, CLI, and fleet paths call the same stage executor;
- persist stage/job transitions consistently;
- prove duplicate claims cannot execute the same stage concurrently.

### Destructive-operation framework

- shared preflight/cardinality helpers;
- transaction/batch safety caps;
- dry-run conventions;
- checkpoint/resume metadata;
- explicit post-run verification;
- backup acknowledgement for destructive operations.

### Recovery and observability

- structured job/stage logs;
- last-success and failure counters;
- retry/quarantine visibility;
- health checks that prove useful work rather than process existence;
- interrupted-run recovery procedure.

## P1

### ML/runtime validation

- separate dependency lane for torch/Whisper/CLIP/YOLO;
- prohibit import-time model downloads;
- smoke-test CPU and optional accelerator runtime selection;
- document host-specific optional dependencies.

### Legacy and lint debt retirement

- eliminate Ruff exclusions incrementally;
- archive or remove superseded scripts;
- ensure deprecated commands fail with clear guidance;
- reduce shell-script business logic to thin wrappers.

### E2E fixture pipeline

- tiny deterministic audio/media fixture;
- one canonical ingest job from READY through DONE;
- assert graph nodes, artifact provenance, and stage state;
- test retry/resume from an injected mid-pipeline failure.

## Exit criteria for production-hardening phase

- real Neo4j integration tests are required and green;
- Docker and Compose gates are required and green;
- configuration regressions are blocked automatically;
- all destructive graph migrations follow the safety contract;
- one canonical execution/state model is used across runtime surfaces;
- remaining debt is tracked explicitly rather than hidden in prose or comments.
