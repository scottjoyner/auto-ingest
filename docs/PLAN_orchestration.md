# PLAN: Canonical Runtime Model (O-G1 / O-G3)

> **Scope:** Decide the canonical runtime between `docker-compose.yml` (7
> services) and the `bin/auto-ingest` CLI + shell scripts, and decide the fate
> of `job_trigger_api.py`. **Document-only deliverable** — no compose services
> or running scripts are deleted or modified here (deletion is risky/visible).
>
> **Status:** Decision made and documented. `job_trigger_api.py` marked
> `# DEPRECATED` (header comment only; not deleted).

---

## 1. The overlap problem

`docker-compose.yml` defines 7 services that **duplicate** functionality the
`bin/auto-ingest` CLI and shell scripts already provide:

| Compose service | What it runs | Overlaps with |
|-----------------|--------------|---------------|
| `ingest-service` | `run_ingest_all.sh` loop | `bin/auto-ingest run-all`, `runall.sh` |
| `ingest-worker` | `worker_ingest.sh` (drop queue) | `deploy/worker_ingest.sh` (the real claim loop) |
| `sync-service` | `sync_from_legacy_drop.sh` loop | `cron/content_generation.crontab`, `runall.sh` |
| `content-service` | `content-os status` loop | `bin/auto-ingest shorts ...` |
| `ingest-cron` | `start-cron.sh` (ingest.crontab) | `ingest-service` (both ingest!) |
| `content-cron` | `start-cron.sh` (content_generation.crontab) | `sync-service` / `content-service` |
| `job-api` | `job_trigger_api.py` (HTTP enqueue/run) | `deploy/create_job.sh` |

Running both the CLI path and the compose path on one host causes **double
ingest** (the most expensive, graph-locking stage) and fights over the same
Neo4j write locks (the backfill deadlock noted in DEEP_DIVE §6.9).

---

## 2. Decision: **Compose for prod, `bin/auto-ingest` for dev**

* **Production / always-on boxes:** the `docker-compose` services are the
  intended long-lived runtime (restart: unless-stopped, health via loops).
  Keep them as the canonical *deployed* model.
* **Development / one-shot / interactive:** the `bin/auto-ingest` CLI is the
  canonical *operator* entrypoint (machine-agnostic, env-driven, no hardcoded
  paths). Use it to run a single stage, a linking pass, or `status`/`caps`.
* **Rule of thumb:** the CLI is what a *human* types; compose is what *runs
  unattended*. They must not both drive the same ingest stage on the same host
  simultaneously.

### Service disposition (documented, NOT deleted)

| Service | Disposition | Note |
|---------|-------------|------|
| `ingest-service` | **KEEP (canonical ingest)** | Primary prod ingest loop. |
| `ingest-worker` | **KEEP (canonical distributed)** | The claim-loop worker; will absorb `ingest_claim` (see PLAN_distributed_ingest.md). |
| `sync-service` | **KEEP** | Legacy-drop sync; no CLI equivalent. |
| `content-service` | **DEPRECATE (duplicate)** | `content-os status` is a no-op status loop already covered by `bin/auto-ingest shorts`. Mark dead; do not schedule both it *and* `content-cron`. |
| `ingest-cron` | **DEPRECATE (duplicate of ingest-service)** | Both run ingest. Pick ONE: keep `ingest-service` for continuous, drop `ingest-cron` to avoid double ingest. Documented as redundant. |
| `content-cron` | **KEEP** | The real content batch schedule. |
| `job-api` | **DEPRECATE (stub-grade, unauthenticated)** | See §3. |

> These are *disposition labels*, not code changes. Deleting services from
> `docker-compose.yml` is a separate, visible change that should be reviewed
> before landing.

---

## 3. `job_trigger_api.py` — keep or cut?

**Finding:** It is a stub-grade HTTP server that (a) enqueues `.job` files via
`create_job.sh` semantics and (b) runs ingest/sync *inline* via
`subprocess.run`. It has **no authentication** (its own SECURITY TODO admits
this = RCE risk) and **no caller** in the repo — nothing POSTs to it.

**Decision: keep the file but mark it `# DEPRECATED`.** Rationale:
* It is the only HTTP enqueue surface and may be wired later to the
  `ingest_claim` protocol (§2 of PLAN_distributed_ingest.md).
* Deleting it is visible/risky; the gap doc only requires a decision + safe
  deprecation note, not removal.
* The inline `subprocess.run` "run" endpoints are the dangerous part and should
  be disabled in any real deployment (behind auth, or simply never started).

Action taken this pass: added a `# DEPRECATED` header comment to the file.
No behavior changed, no endpoints removed.

---

## 4. Safe deprecation notes (for the follow-up PR that actually edits compose)

1. Before removing `ingest-cron`/`content-service`, confirm no host relies on
   them (grep cron/.env for their container names).
2. If both `ingest-service` and `ingest-cron` are present, the safest immediate
   fix is to **stop `ingest-cron`** (not delete) to stop double-ingest, leaving
   the file intact for review.
3. `job-api` should be excluded from any prod compose profile until its auth
   story (Caddy + bearer/mTLS, per its own TODO) is implemented.
4. Keep `bin/auto-ingest` as the dev/operator path; document that compose is
   prod-only so contributors know which to edit.

---

## 5. What was actually changed this pass

* `deploy/job_trigger_api.py`: added `# DEPRECATED` header comment (no logic
  change).
* This document, recording the decision.

No compose services were deleted or modified. No running scripts were touched.
