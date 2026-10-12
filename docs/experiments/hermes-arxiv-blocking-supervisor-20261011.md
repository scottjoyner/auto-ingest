# Staged Hermes arXiv blocking supervisor — experiment only (October 11, 2026)

**Not installed, not connected to Hermes, not production admitted.** This draft stacks on [Auto-Ingest PR #22](https://github.com/scottjoyner/auto-ingest/pull/22), whose 87 source-only tests and full repository CI passed. Tracks [P0 Hermes detached arXiv execution #25](https://github.com/scottjoyner/auto-ingest/issues/25).

## Physical problem: scheduler launch `ok` is not child completion

Read-only authorized Fleet Commander inspection of xwing's active Hermes cron registry found three enabled `no_agent=true` scripts, one enabled agent-driven weekly job, and a real script `kg-arxiv-ingest.py` under the local Hermes scripts directory. The live arXiv wrapper calls `subprocess.Popen(..., start_new_session=True)` and does not wait for the child. Its scheduler last status is `ok`, even though actual child completion, exit status and source binding have not been observed. Two other scripts use blocking `subprocess.call`, and none supplies a signed per-user model execution receipt. Script prompt/contents, private paths, tokens and command arguments were not copied into the repo.

## Narrow experimental improvement

`scripts/hermes_arxiv_supervisor_experiment.py` is a **library without a CLI or auto-execute entrypoint**. An independently reviewed Hermes adapter and explicit operator change plan would still be required before use. It is deliberately NOT wired into `~/.hermes/scripts`, `~/.hermes/cron/jobs.json`, crontab or production files.

A future authorized adapter could call `supervise()`, which validates an independently supplied full SHA-256 of the **exact arXiv caller script** and **matching Nomic v1.5/768 registry**, verifies the shared embedding client file and Python interpreter exist, takes a nonblocking per-host `fcntl.flock` to reject overlapping invocations, then starts a child under a separate process session. It uses a **synchronous `wait(timeout)`** and propagates success/nonzero return status; on timeout it sends TERM to the child process group, then KILL if necessary, and waits for reaping. It suppresses stdout/stderr instead of exposing model prompt or secret output and returns only sanitized status codes.

**Trust caveats:** Caller-supplied file hashes merely check consistency with those inputs: they are **NOT cryptographically reviewed operator/source approvals**. The local file lock works only on one host/filesystem context and is NOT PostgreSQL admission, cross-node fencing or replay authority. The wrapper does **NOT authenticate Deathstar clients** or prove a GGUF, owner-bound worker execution or a completed Neo4j transaction. Child exit 0 remains an ordinary process result, not an ingestion/source-custody receipt. Since process descendants can have their own groups, process-group termination is not a general proof that every descendant exited.

### Synthetic-only evidence

`tests/test_hermes_arxiv_supervisor_experiment.py`: **15 isolated tests** write harmless temporary Python children under `/tmp` and use the local Python runtime, never a real model or graph. They cover successful waited completion, nonzero exit, bounded timeout/reap, overlapping lock denial without starting a child, digest mismatch, wrong/missing registry metadata, absent shared client, symlink refusal, invalid budgets, stdout/stderr suppression, lock release and no CLI. Their artifacts are disposable. Dedicated hosted workflow `hermes-arxiv-supervisor-experiment.yml` enforces 15 tests, Ruff, and a static no-authority/no-autoexecute contract.

### Explicit operator gates before any deployment

1. Confirm xwing Hermes' `no_agent` executor resolves the relative wrapper basename under `~/.hermes/scripts`, with exact interpreter, job owner and checkout path, and capture source SHA witness. Do not infer all behavior from the scheduler registry.
2. Review and authorize an authenticated app-layer Deathstar gateway and independent owner-bound receipt ([Auto-Ingest #24](https://github.com/scottjoyner/auto-ingest/issues/24)); tokenless Tailnet synthetic HTTP200 is insufficient.
3. Deploy PR #21 Nomic/768 resolver, PR #22 four callers, and matching registry together to the genuinely used shared checkout; address the secondary divergent home copy under rollback, and verify no job double-run.
4. Independently authorize a **bounded test window** for changing only the arXiv Hermes wrapper. Integrate/test real blocking status propagation and timeout with a disposable/mock graph and authenticated provider before any live ingestion; maintain a full rollback of original wrapper/cron.
5. Prove negative offline/error/timeout behavior and proper scheduler failure classification; verify job command/path SHA, process start and true child exit with a durable trace/receipt, then check graph/cursor consistency.
6. Contain the exposed Beelink fallback AND its reboot/15-min watchdog only after the alternate embedding path is authenticated and healthy ([AssistX #319](https://github.com/scottjoyner/auto-assist/issues/319)). Keep signed Fleet generation renewal and three Neo4j uniqueness constraints separately NO-GO.

**No production Hermes job, script or service was changed by this staged experiment.**
