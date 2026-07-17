# PLAN: Distributed Ingest — Claim Protocol vs. AssistX Fleet

> **Scope of this doc:** Decide the approach for the distributed processing
> gaps **G1 (knowledge_map sync), G2 (vault), G4 (preprocessing manifest),
> G5 (cleaned intermediate), G6 (distributed claim)**. This is a *decision +
> light first slice*, not a full implementation. The heavy wiring is explicitly
> left to a follow-up (see §6).
>
> **Status:** Decision made; coordination slice implemented
> (`auto_ingest/ingest_claim.py`); remainder design-only.
> **Constraint honored:** Nothing in this doc or its implemented slice writes
> to the live graph beyond index-backed, paged, read-only queries, and the
> existing live-upload guard (`publish_guard.py`) is untouched.

---

## 1. The two candidate approaches

### A. Local `/nas/drop` claim protocol (Neo4j `IngestJob` owner/TTL + flock)
Rehabilitate the **already-shipped** file-based drop queue
(`deploy/worker_ingest.sh`, `deploy/create_job.sh`, `deploy/job_trigger_api.py`)
and layer the DEEP_DIVE §3.3 coordination model on top:

1. Coordinator writes an `IngestJob` node (manifest) with `owner=""`.
2. Any worker: `claim(key, owner, ttl)` → conditional `SET owner=<host>`
   guarded by `owner="" OR claimed_at < now-ttl`; plus a `flock` on the `.job`
   file so the file move and the graph claim cannot disagree.
3. Worker copies source → local fast storage, processes with best-fit backend,
   writes a **cleaned JSON/CSV** intermediate + embeds.
4. Worker commits metadata back (`graph_written=done`, `owner=""`) and moves the
   `.job` to `done/`.

### B. Adopt `fleet_batch.py` → AssistX HTTP and write the node-agent consumer
`fleet_batch.py` already POSTs capability-tagged `READY` Tasks to an external
`assistx:8000` (`POST /api/tasks`). The missing piece is the node agent that
polls by capability and runs them. Auto-ingest would become a *task producer*
only; execution lives in AssistX.

---

## 2. Decision: **Approach A** (local `/nas/drop` claim protocol)

**Justification**

1. **The file-based queue already works.** `worker_ingest.sh` claims via atomic
   `mv` and segregates `claimed/`/`done/`/`failed/`. Approach A builds on a
   running primitive; Approach B requires standing up an *external* AssistX
   service that does not exist in this repo and is not operated here.
2. **No new auth/network surface.** `job_trigger_api.py` already carries a
   SECURITY TODO warning it has no auth (RCE risk). Approach B multiplies that
   surface (every node agent needs AssistX creds + network egress). Approach A
   stays on the existing Tailscale + NAS mount trust boundary.
3. **Keeps the graph as the source of truth for state.** `IngestJob` nodes
   give a queryable, resumable manifest (G4) that the file queue alone lacks;
   `list_claims()` answers "what is each box doing right now" in one indexed
   query. AssistX would duplicate that state in a foreign system.
4. **Smallest blast radius.** The slice implemented here (`claim`/`release`/
   `list_claims`) is read-mostly and index-backed; it cannot OOM the 21M-node
   graph. Approach B's node-agent is a long-running process on every box.
5. **Reuses hardware routing already built.** `auto_ingest.backend` +
   `bin/auto-ingest caps` (K-N4, done) select the per-host backend; the worker
   loop just needs to consult it when picking which `.job` to run. AssistX
   would need that logic re-implemented in its agent.

**Why not B (for now):** AssistX is the cleaner long-term "swarm" story, but it
is an *external dependency* with no deployment here, an unauthenticated API, and
no consumer. Adopting it is a larger, riskier bet than finishing the protocol we
already half-built. Recommendation: revisit B only if a real AssistX deployment
materializes; until then, A is canonical.

---

## 3. Implemented slice (this pass)

`auto_ingest/ingest_claim.py` — coordination only:

| Function | Behavior | Graph safety |
|----------|----------|--------------|
| `claim(driver, key, owner, ttl_sec)` | Conditional `MERGE IngestJob` + guarded `SET owner/claimed_at`. Returns `True` only if *this* call won. Atomic race via single Cypher statement. | Index-backed on `IngestJob(key)` / `IngestJob(owner)`; no full-scan. |
| `release(driver, key, owner)` | Clears owner **only if** currently owned by `owner`. | Single node match; no side effects on others. |
| `list_claims(driver, ttl_sec, limit)` | Returns active (non-expired) claims, newest-first, **paged**. | Read-only; `LIMIT`-bounded; never aggregates the whole graph. |

Tests: `tests/test_ingest_claim.py` (6 tests, fake driver, no live DB).

**Prerequisite (not yet created — follow-up):** indexes
`CREATE INDEX ingestjob_key FOR (j:IngestJob) ON (j.key)` and
`CREATE INDEX ingestjob_owner FOR (j:IngestJob) ON (j.owner)` so the conditional
`SET` and `list_claims` stay OOM-safe at 21M nodes. These are one-line,
online, read-only DDL and safe to run; they are simply out of scope for this
slice.

---

## 4. File / function changes specified (follow-up, NOT done here)

| Area | File | Change |
|------|------|--------|
| Coordinator | `deploy/create_job.sh` (or new `auto_ingest/ingest_coord.py`) | After writing the `.job` file, also `MERGE (j:IngestJob{key}) SET j.owner=''`. |
| Worker | `deploy/worker_ingest.sh` | After `mv` claim, call `ingest_claim.claim(key, host, ttl)`; on success set `flock`; on finish call `release` + move to `done/`; reaper cron calls `list_claims` to spot expired owners. |
| Intermediate | new `auto_ingest/ingest_write.py` | Write cleaned JSON/CSV per key; dynamic `UNWIND $rows` import (G5). |
| Vault (G2) | `scripts/knowledge_sync_all.sh` | Replace 4 hardcoded IPs with Tailscale hostnames + `config.yaml` discovery list. |
| knowledge_map (G1) | new `auto_ingest/knowledge_map/__init__.py` | Thin wrapper around existing `knowledge_harvest*.py` so `python3 -m knowledge_map` resolves (no rewrite). |
| Manifest status | `auto_ingest/ingest_claim.py` | Extend `IngestJob` with stage properties (`copied/transcribed/...`) + a `status` enum; `bin/auto-ingest status` already queries the graph and can show it. |

---

## 5. Out of scope / explicitly left as design-only

* The actual copy→process→commit worker loop (§4 `worker_ingest.sh` changes).
* The cleaned-intermediate writer and dynamic-tx importer (G5).
* Vault consolidation + `knowledge_map` wrapper (G1/G2).
* Any write against the live graph (this slice reads only when `list_claims`
  is called by an operator; `claim`/`release` are only invoked by the worker
  loop which is not yet wired).

## 6. Next action

Wire `ingest_claim` into `worker_ingest.sh` + create the two indexes, then add
the intermediate writer. Until then, `ingest_claim` is safe to import and unit-
test but is **not** yet invoked by any running service.
