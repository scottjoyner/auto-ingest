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
- Preserve existing configured **priority order**, explicit environment override and fail/return-None modes. Skip malformed registry rows and bad priorities fail-closed rather than crash.
- Never probe `POST /v1/embeddings`, auto-load a model, emit prompts, read credentials, change service bindings, create Neo4j evidence or alter any canonical routing authority.
- **Never conflate a nonempty inventory with actual resident model evidence.** This is a necessary listing validation, not sufficient admission. A provider can advertise another model or lie about model availability; authenticated exact model identity and a genuinely completed embedding receipt still require separate approved checks.

## Staged acceptance

1. **Hosted exact-head source-only CI**: 15 isolated tests cover HTTP 200 empty primary, offline second provider, valid third provider, malformed/empty model IDs, invalid schemas, oversized responses, single-host override, malformed registry, all-unavailable and GET-only behavior. Ruff and static source checks must pass.
2. **Read-only installed-state reconciliation**: check that the active cron jobs import this resolver and do not override it with a different `EMBEDDING_URL`; confirm the installed registry's model family against each endpoint's actual advertised ID without logging Tailnet URLs or private data.
3. **Operator-authorized model correctness**: ensure Deathstar genuinely serves the expected embedding dimensionality and signature, with a single bounded, authorized, synthetic embedding canary, legitimate resource grant and privacy-compatible result custody.
4. **Contain the exposed lower-priority Beelink server** only after active consumers are enumerated and authenticated alternate access/failover behavior is proven across a watchdog interval and reboot-equivalent test. The Beelink embedding watchdog runs at reboot and every 15 minutes; changing only the active process could recreate a wildcard listener. The separate inference watchdog also has wildcard launch text but **no independently observed second active listener**, and must remain labeled a latent risk.
5. **Separate signed fleet release gates:** source/process execution attestation, authenticated owner receipts, independent signer/operator keys, 3 Neo4j uniqueness constraints and genuinely approved new canonical generation. No direct auto-admission from registry health.

**Rollback:** Source-only changes are isolated in a dedicated draft branch. Do not merge until callers, registry/model identity compatibility and fallback behavior have been reviewed. If an approved release later regresses, revert the resolver commit and use the independently confirmed authorized fallback; do not make a live exposed listener the sole emergency path.

Links: [Auto-Ingest embedding registry](https://github.com/scottjoyner/auto-ingest/blob/main/embedding_endpoints.json), [resolver](https://github.com/scottjoyner/auto-ingest/blob/main/scripts/get_embedding_endpoint.py), [Fleet direct listener security P0](https://github.com/scottjoyner/auto-assist/issues/319).
