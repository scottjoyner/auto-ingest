# Auto-Ingest embedding endpoint failover correction — 2026-10-11

**Status: source-only, DRAFT release; no production registry edits, cron changes, model inferences, forced model loads, router promotions, credentials or listener/firewall mutations.**

## Physical production problem, read-only

The installed Auto-Ingest endpoint registry on x1-370 was checked with bounded, **GET-only** `/v1/models` calls against the seven **already-configured** Tailnet entries. Exact host URLs and internal IPs are excluded from this report.

| Priority | Registry role | /v1/models result at checkpoint | Listed embedding-model candidate |
|---:|---|---|---|
| 1 | x1-370 primary | **HTTP 200, empty `data: []`** | No |
| 2 | Destroyer standby | Unreachable | Unobserved |
| 3 | Deathstar standby | **HTTP 200, one model** | Yes, model ID resembles configured embedding family |
| 4 | Lenovo standby | Unreachable | Unobserved |
| 5 | Joyner standby | Unreachable | Unobserved |
| 6 | Beelink standby | **HTTP 200, one model** | Yes, model ID resembles configured embedding family |
| 9 | Xwing legacy tunnel | Unreachable | Unobserved |

`scripts/get_embedding_endpoint.py` currently defines `_healthy` as **HTTP 200 only**, then returns the first endpoint by `priority`. Under the observed state, it accepts x1-370's **empty model inventory** and may direct an embedding task to a server with no model. A health response or listed model is **not** proof of LOADED residency, exact model digest, authenticated owner authorization, worker receipt, or successful embeddings.

The existing `embedding_endpoints.json` explicitly lists Beelink as the lower-priority fallback. It is therefore unsafe to assume the exposed Beelink embedding listener is unused; the [direct listener P0 #319](https://github.com/scottjoyner/auto-assist/issues/319) and [draft physical listener audit PR #320](https://github.com/scottjoyner/auto-assist/pull/320) must preserve this consumer or migrate it to an authenticated equivalent.

Other code consumers in this repository include `scripts/backfill_paper_768.py` and additional graph/semantic enrichment tools. `scripts/get_embedding_endpoint.py` also permits an `EMBEDDING_URL` single-host override and a configurable `EMBEDDING_REGISTRY`; operators must check **which installed instance and environment overrides active scheduled jobs actually use** before promotion.

## Smallest source-only bugfix

- Require a bounded, valid OpenAI-compatible `GET /v1/models` **JSON `data` list** containing at least one nonblank `id`. Reject 200 with empty list, HTML, malformed JSON, missing schema, unbounded body, non-200 HTTP, and unreachable providers.
- Preserve existing configured **priority order**, environment override support with required model/dimension pins, and fail/return-None modes. Skip malformed registry rows and bad priorities fail-closed rather than crash.
- Never probe `POST /v1/embeddings`, auto-load a model, emit prompts, read credentials, change service bindings, create Neo4j evidence or alter any canonical routing authority.
- **Never conflate a nonempty inventory with actual resident model evidence.** This is a necessary listing validation, not sufficient admission. A provider can advertise another model or lie about model availability; authenticated exact model identity and a genuinely completed embedding receipt still require separate approved checks.

## Physical model match, dimensionality and override hardening (2026-10-11)

A second **read-only** model inspection on configured Deathstar verified `nomic-embed-text-v1.5` from the listed model-ID basename (which has a local path prefix), `meta.n_embd=768`, `meta.n_ctx=2048`, HTTP-200 `/health` with status `ok`, `/props.is_sleeping=false`, and four reported slots. The existing Beelink embedding endpoint also advertises the same model family and 768-dimensional model metadata. This is inventory, model identity **by self-reported metadata**, and health evidence only — **NOT** completed embeddings, exact GGUF SHA verification, authenticated owner binding, per-request worker attestation or usable capacity admission.

The original model-list-only check would still accept a server listing the **wrong** model. The draft now fixes this: its GET-only `/v1/models` verifier additionally requires the canonical registry model family (e.g. `text-embedding-nomic-embed-text-v1.5`) and exact dimension `meta.n_embd=768`. It accepts a valid quantized GGUF filename with a path prefix, but rejects different model families/versions, misleading prefix aliases, absent or invalid dimension metadata, and nonmatching dimensions.

The **draft GitHub registry** adds `embedding_dimensions: 768`. The **installed x1 registry was NOT changed**. Both the normal ordered-selector path and the `EMBEDDING_URL` single-host override now require an installed, readable registry with model and dimension pins; an unpinned override is refused. **This changes behavior for deployments that previously used an override without a registry**, so review and stage the registry alongside the source during an authorized deployment, never push code alone. A custom `model` argument still needs an approved compatible dimension and source/owner authority outside this helper.

A bounded GET-only priority check against the **installed 7-node registry** using the proposed model-family and dimension rules rejected x1's empty inventory and unreachable Destroyer and accepted **Deathstar priority 3**. It did not write the installed registry, make any embeddings, or prove the candidate can replace a private authenticated client.

New negative coverage: **27 offline tests** (previously 15), including wrong family/version, spoofed prefix, wrong/missing dimensions, malformed registry, override without pins, override with correct pins, and the explicit empty-primary → correct Deathstar selection. All passed on x1 with strict Ruff using exact branch source `43b55cac95eb0664df1db5d5c31da8b09e7d6359`. Hosted CI must be checked independently at the final head.

## Staged acceptance

1. **Hosted exact-head source-only CI**: 27 isolated tests cover HTTP 200 empty primary, offline second provider, valid third provider, malformed/empty model IDs, invalid schemas, oversized responses, single-host override, malformed registry, all-unavailable and GET-only behavior. Ruff and static source checks must pass.
2. **Read-only installed-state reconciliation**: check that the active cron jobs import this resolver and do not override it with a different `EMBEDDING_URL`; confirm the installed registry's model family against each endpoint's actual advertised ID without logging Tailnet URLs or private data.
3. **Operator-authorized model correctness**: ensure Deathstar genuinely serves the expected embedding dimensionality and signature, with a single bounded, authorized, synthetic embedding canary, legitimate resource grant and privacy-compatible result custody.
4. **Contain the exposed lower-priority Beelink server** only after active consumers are enumerated and authenticated alternate access/failover behavior is proven across a watchdog interval and reboot-equivalent test. The Beelink embedding watchdog runs at reboot and every 15 minutes; changing only the active process could recreate a wildcard listener. The separate inference watchdog also has wildcard launch text but **no independently observed second active listener**, and must remain labeled a latent risk.
5. **Separate signed fleet release gates:** source/process execution attestation, authenticated owner receipts, independent signer/operator keys, 3 Neo4j uniqueness constraints and genuinely approved new canonical generation. No direct auto-admission from registry health.

**Rollback:** Source-only changes are isolated in a dedicated draft branch. Do not merge until callers, registry/model identity compatibility and fallback behavior have been reviewed. If an approved release later regresses, revert the resolver commit and use the independently confirmed authorized fallback; do not make a live exposed listener the sole emergency path.

Links: [Auto-Ingest embedding registry](https://github.com/scottjoyner/auto-ingest/blob/main/embedding_endpoints.json), [resolver](https://github.com/scottjoyner/auto-ingest/blob/main/scripts/get_embedding_endpoint.py), [Fleet direct listener security P0](https://github.com/scottjoyner/auto-assist/issues/319).
