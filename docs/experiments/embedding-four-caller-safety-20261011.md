# Auto-Ingest four-caller embedding safety integration — 2026-10-11

**Status: DRAFT ONLY, stacked on [Auto-Ingest PR #21](https://github.com/scottjoyner/auto-ingest/pull/21). No live embedding requests, privileged graph writes, cron changes, registry deployment, model loads, firewalls, NAS writes, routing changes or approvals.**

## Discovered installed deployment gap

An isolated read-only scan of the installed Auto-Ingest scripts at x1-370 (under the installed Auto-Ingest scripts directory, not a comprehensive whole-fleet filesystem search) found four known Nomic-768 embedding callers:

| Existing script | Prior behavior | Draft behavior |
| --- | --- | --- |
| `backfill_paper_768.py` | Calls a resolver, **but falls back to a legacy hardcoded provider URL** on discovery failure. | Uses only the pinned provider helper; fails before writing a paper vector if no eligible provider. |
| `enrich_bridge_semantic.py` | Calls a hardcoded URL via `requests.post`; batches concepts in 100 and summaries in 64. | Pinned, shape-checked client; batches at `MAX_BATCH=16` before writing vectors/semantic relationships. |
| `signal_kg_bridge.py` | Embeds Signal message text through a direct hardcoded URL. | Uses pinned client; on embedding failure returns `None` rather than falling back or writing a malformed vector. **Existing raw-message ingestion can still proceed without the vector.** |
| `arxiv_kg_bridge.py` | Direct hardcoded endpoint. | Uses pinned client; on embedding failure returns `None` instead of a malformed vector. **Existing paper ingestion can still proceed without the vector.** |

Other installed scripts or remote hosts may have further hardcoded clients; this is **not** an exhaustive dependency inventory. The current Fleet Commander command environment had neither `EMBEDDING_URL` nor `EMBEDDING_REGISTRY` set, but `crontab -l` was unavailable from that execution context and therefore scheduled job environments **remain unverified**.

## Shared `scripts/embedding_client.py` acceptance

- Use `scripts.get_embedding_endpoint(model="text-embedding-nomic-embed-text-v1.5", fail=False)` only, relying on PR #21's installed model-family + `embedding_dimensions=768` requirements. No hardcoded emergency fallback.
- Require bounded 1–16 nonblank input strings, each ≤8,000 characters; exact configured model and 768-dimensional output contract; timeout 1–120 seconds.
- Only after successful GET-based provider selection may the client issue a POST to the selected `/v1/embeddings` endpoint. This is **real inference if called by production**, so the code has **not** been executed against live providers.
- Reject redirects and environment proxies, non-200 HTTP, malformed JSON, response bytes >1 MiB, wrong output count, absent/duplicate/misordered indices, missing or nonnumeric/boolean/nonfinite coordinates, wrong length and zero vector. Return vectors in original input order, independent of provider row order.
- Do not log private prompts, inference responses or internal endpoint addresses on failure; the public `EmbeddingUnavailable` exception uses generic error messages.
- Even a numerically valid vector is **not** a cryptographic proof of model GGUF digest, per-user authorization, or worker-specific execution. The currently exposed Beelink endpoint remains a [P0 issue #319](https://github.com/scottjoyner/auto-assist/issues/319); a listed-model or HTTP-200 response does not make it safe.

## Evidence and release acceptance

`tests/test_embedding_client_guard.py` performs **22 local-only tests** using mocked HTTP and selector calls, including model/vector validation and structural checks that each of the four known callers uses the central function without raw endpoint constants. The parent [PR #21](https://github.com/scottjoyner/auto-ingest/pull/21) contributes **27 additional** model-family, dimensionality and failover tests. No actual POST is sent in either suite.

**Before release**, verify:
1. Current cron/service job owners, imports, working directories, `EMBEDDING_URL`/`EMBEDDING_REGISTRY` overrides, and any embedding callers on other hosts. Do not silently skip xwing/other NAS-sourced scripts.
2. Provision and stage the pinned registry alongside PR #21 and this code; older override-only installations with no registry will **fail closed**. Test both expected availability and safe offline denial.
3. Authorize one bounded, synthetic, resource-admitted Deathstar embedding call with an authenticated owner and verified `nomic-embed-text-v1.5` identity, output dimension 768 and finite, nonzero vector. Read-only Deathstar `/v1/models` previously self-reported Nomic-768, healthy and four slots; that's **not** execution proof.
4. Test all four scripts in disposable mode with mocked Neo4j writes first, including timeout/refusal/invalid embedding to ensure graph model and 384/768 dimensions remain separated; perform no live historical batch mutation until the rollback plan and operators approve.
5. Coordinate [AssistX direct listener audit PR #320](https://github.com/scottjoyner/auto-assist/pull/320) / [P0 #319](https://github.com/scottjoyner/auto-assist/issues/319): after alternate client acceptance, harden **both** active Beelink listener and reboot/15-minute embedding watchdog. Negative unauthenticated Tailnet checks + positive intended caller checks must pass. Separately review the second, dormant inference watchdog without assuming active exposure.
6. Keep independent fleet signed-source, owner-bound request receipts, three Neo4j uniqueness constraints and canonical generation 643 expiry gates **NO-GO**. Neither this client nor successful CI grants inference admission or canonical promotion.

### Staged rollback

No live files changed. These are new GitHub commits on a branch stacked on PR #21. During a future **authorized** staged deployment, snapshot the installed caller scripts and registry, deploy the registry/model checks and client together, test negative refusal before scheduled jobs, and restore the prior agreed good package on failure. Do **not** use the unauthenticated Beelink URL as a silent emergency bypass; disable affected embedding-dependent tasks rather than writing unverified vectors.
