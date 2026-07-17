# Auto-Ingest — Master Gap Analysis (2026-07-17)

> **Status:** Findings only. Nothing here was executed against the live graph except two read-only checks (Scott-anchor count, publish dry-run guard).
> **Companion:** `docs/DEEP_DIVE_2026-07-14.md` (prior deep-dive, gaps G1–G13). This doc re-verifies those and adds four new domain deep-dives (shorts/render/persona, publish/analytics/brand, KG/ingest/speaker, services/deploy/ops).
> **Method:** Four parallel subagents read current code + existing docs and reported gaps with `file:line` evidence. Findings were spot-checked by the lead agent.

---

## 0.1 Implementation status (2026-07-17 session)

Completed this session (verified green — 176 tests pass on system python + `.venv`, ruff clean on changed files):
- **P-G1** live-upload guard (`publish_guard.py` + `require_live_mode`, wired into `cli` + `uploader.process_queue`; `safe_to_run` added to all 3 auth modules; 4 tests).
- **O-G7** secrets hygiene (`.env`/`.env-ssd` removed from git index + ignored; placeholder in `.env.example`; secrets linter added).
- **K-N1** Scott anchor re-merged → exactly 1 `is_me` GlobalSpeaker; `speaker_health.py` + `check_anchor_health.py` + 6 tests.
- **O-G12** CI workflow (`.github/workflows/ci.yml`: ruff + pytest on system + venv/moviepy; excludes pre-existing legacy-linker debt).
- **K-N4** backend routing wired into both linkers (`recommended_torch_device`).
- **S-G16/S-G17** legacy `shorts_builder` dep removed from package; shared `captions.py`/`compose.py` (171→176 tests pass).
- **S-G1/G4/G7/S-G24** persona two-pass fixed: threads width/height/bitrate, single canonical audio (narration or base B-roll, never video-as-audio), avatar-exists guard, faststart encode; + test now generates a real clip (was a silent skip).
- **S-G18** `db_retry.py` resilience helper; applied to planner mining + curator + content_miner (4 tests).
- **S-G9/G10** content_miner mines real myth/fact contrast + resilient driver=None; default `plan` CLI now passes the live driver so real content is mined.
- **P-G3** dry-run now prints full caption + uses_thumbnail for all 3 platforms.
- **P-G10** A/B variant → uploader override wired end-to-end (plan_report + live upload) + test; fixed `youtube_shorts` thumbnail variant gap.
- **P-G7** `metrics predict` sub-action closes the virality→prediction loop.
- **S-G14** persona test no longer silently skips (synthetic clip).
- **K-N2/K-N5** `bin/run` symlink → `auto-ingest`; archived `auto.py` + `autogen_coder_team_*` (clearly-redundant legacy).

Still open (deferred): K-N3 (distributed claim), O-G6 (config unification), O-G1/G3 (orchestration canonicalization), P-G8/G9/G12 (metrics schema versioning, feedback actions, brand --check), P-G5 (persona in schedule), G1/G2/G4/G5/G6 (knowledge_map sync, vault, manifest, outbox), G11/N6/N7/N8/N9/N11 (speaker promotion, MCP tools, knowledge_map refs, doc paths), S-G17 residual (addressed: legacy generators marked deprecated, canonical = compose.py).

## 0. TL;DR

The system has grown enormously since the 2026-07-14 deep-dive. The **shorts / publishing / persona** stack was built almost entirely *after* that doc, so it was never gap-analyzed. The KG/speaker work from that doc is **mostly landed but has regressed in one important place** (the Scott anchor fragmented back to 4 nodes), and the headline **distributed-processing** goal was quietly dropped (replaced by an unwired external HTTP call).

### Top priorities (do these first)
1. **[BLOCKER] Publishing live-upload guard.** `publish upload` is dry-run by default but has **no hard `AUTO_INGEST_LIVE` enforcement** — a stray `--no-dry-run` + token = real post, violating the "publishing held" constraint. (Publish-G1)
2. **[HIGH] Re-merge the Scott anchor.** Live graph has **4 `GlobalSpeaker`s with `is_me=true`** (should be 1). Run `whoami --merge` + add a health-check. (KG-N1)
3. **[HIGH] Stand up CI** (ruff + typecheck + tests). There is **no `.github/workflows` at all**. (Ops-G12)
4. **[HIGH] Wire backend routing into the linker.** Detection exists (`backend.py`) but the linker/YOLO/whisper still pick device the legacy way; on this AMD HX370 box everything runs CPU-only. (KG-N4)
5. **[HIGH] Consolidate the render stack.** `auto_ingest/shorts/render.py` imports a 1376-line legacy `shorts_builder.py`; three caption renderers exist. (Shorts-G16/G17)
6. **[HIGH] Persona two-pass audio/param bug.** `compose_with_persona` ignores width/height/bitrate, double-encodes audio, drops B-roll ambient audio, and can pass a *video* as the talking-head audio source. (Shorts-G1/G4/G7)

---

## 1. Status of the 2026-07-14 gaps (G1–G13)

Re-verified against current code + live read-only graph inspection.

| Gap | 2026-07-14 | Status now | Evidence |
|-----|-----------|-----------|----------|
| G1 knowledge_map sync engine missing | OPEN | **OPEN** | `find . -name "knowledge_map*"` → none; `scripts/knowledge_sync_all.sh:73` + `knowledge_sync_handler.py:132` still call `python3 -m knowledge_map` → ModuleNotFoundError |
| G2 Vault fragmented / hardcoded IPs | OPEN | **OPEN** | `scripts/knowledge_sync_all.sh` still SSH-pushes to hardcoded hosts; `config.yaml:41` 3 vault paths; no single-vault service |
| G3 No single `run` command | — | **PARTIAL** | `bin/auto-ingest` exists (`caps/status/link-speakers/whoami/run-all/...`) but not symlinked as `bin/run`; only `aliases.auto-ingest` (`ai`) |
| G4 No preprocessing manifest | OPEN | **OPEN** | `MATCH (n:IngestJob)` → **0 nodes** live; `status` prints aggregates, not per-source stages |
| G5 No cleaned intermediate artifact | OPEN | **OPEN** | scripts write directly to Neo4j; `outbox.py` opt-in and uncalled |
| G6 No distributed claim protocol | OPEN | **OPEN (regressed intent)** | `fleet_batch.py` posts to external `assistx:8000` (not `/nas/drop`); `outbox.py`/`events.py` unused shims; claim protocol never built |
| G7 No HW-aware backend routing | — | **DETECTION ONLY** | `backend.py` probes CUDA/ROCm/MLX/ONNX + `caps` prints it, but linker/YOLO/whisper don't route by it; HX370 resolves to CPU "onnx" |
| G8 No "me"/Scott anchor | — | **RESOLVED + REGRESSED** | anchor persists (`is_me` live) **but 4 fragments**, not 1 canonical (see N1) |
| G9 DashcamClip→Trip link | — | **RESOLVED** | `IN_TRIP` = 72,788 edges live; `Trip`=1,430 |
| G10 Password drift | — | **RESOLVED** | `livelongandprosper` only in docs now; 55 files use `knowledge_graph_2026` |
| G11 GlobalSpeaker mostly tentative | — | **PARTIAL** | confirmed 429 → **445**; still ~98% tentative; `--rank-and-label` never fleet-run |
| G12 SSD read/write in embedding build | — | **PARTIAL** | ffmpeg pipe fallback + `emb_cache.sqlite` added; heavy backfills still need container pause (N8) |
| G13 Segment→clip attribution | — | **RESOLVED (structural)** | `FOR_CLIP`=108,166; `Segment.clip_key` on 368,141/368,161; `IN_CLIP`=0 |

**Net:** G9, G10, G13 structurally resolved. G8 resolved-then-regressed (N1). G7 half-done. G1/G2/G4/G5/G6 still open — and G6's *intent* (distributed claim) was effectively abandoned.

---

## 2. New gaps by domain

### 2.1 Shorts / Render / Persona  (`auto_ingest/shorts`, `shorts_builder.py`, `scripts/run_narrated_mix.py`)
Full report: see Shorts subagent. Highest-severity:

| ID | Sev | Gap | Evidence | Fix |
|----|-----|-----|----------|-----|
| S-G1 | high | `compose_with_persona` ignores `width/height/bitrate/profile` from `render_short`; hardcodes 1080×1920 + 6M | `persona.py:171-217` | thread params through / composite avatar in one pass |
| S-G4 | high | Persona pass **double-encodes audio** and drops B-roll ambient audio; base already muxed TTS | `persona.py:214-217`, `shorts_builder.py:1281-1284` | single canonical audio path |
| S-G7 | high | `make_talking_head(audio or broll_mp4, ...)` passes *video* as audio if TTS failed | `persona.py:201` | require `audio` non-None |
| S-G5 | high | Real talking-head is a stub: no model-path env, no driver checklist | `persona.py:111-144` | add `PERSONA_ONNX_PATH` + `docs/brand/talking_head_setup.md` |
| S-G9 | high | `myth_fact_for_topic` builds a *fake* generic myth; `reveal_twist` naive word-list scoring; silent template fallback | `content_miner.py:22-84` | mine real misconception framings |
| S-G10 | high | Default `plan` CLI passes `driver=None` → real content **never mined**; falls back to templates | `planner.py:110-114`, `cli.py:69-72` | pass `driver` in `_cmd_plan` or document template-only |
| S-G16 | high | Legacy `shorts_builder.py` (1376 lines) is a parallel renderer; `render.py:51` imports `compose_scripted_short` from it | `render.py:51` | migrate + retire legacy import |
| S-G17 | med | Three caption renderers (`shorts_builder.py`, `smart_shorts.py`, `tiktok_shorts.py`) | `smart_shorts.py:143-209` | one shared renderer module |
| S-G18 | high | No `TransientError`/OOM retry in `planner`/`curator` (only in `run_narrated_mix.py`) | `planner.py:344-394` | port retry wrapper into package |
| S-G14 | med | CI runs system python (no moviepy) → all render/persona code **untested in CI**; split-venv story undocumented | `ci.yml`, verified `python3` no moviepy | add `.venv` test matrix + tests/README |
| S-G15 | med | Only persona integration test skips unless `/tmp/opencode/smoke.mp4` exists (never generated) | `test_persona.py:53-65` | generate synthetic clip in fixture |
| S-G24 | low | Missing avatar asset crashes persona render (no guard) | `persona.py:158` | add `Path.exists()` guard |

### 2.2 Publish / Analytics / Brand  (`publish.py`, `uploader.py`, `scheduling.py`, `metrics.py`, `feedback.py`, `abtest.py`, `docs/brand`)
Full report: see Publish subagent.

| ID | Sev | Gap | Evidence | Fix |
|----|-----|-----|----------|-----|
| P-G1 | **blocker** | `publish upload` has **no hard live guard**; dry-run is default but `--no-dry-run` + creds = real post | `cli.py:300-302,545`; `publish.py` branches on dry_run only | `assert_safe_to_publish()` gated on `AUTO_INGEST_LIVE=1` + creds |
| P-G2 | high | Auth modules have no preemptive `safe_to_run()` precheck | `yt_auth.py`/`tiktok_auth.py`/`instagram_auth.py` | central `auth.safe_to_run()` consulted by CLI |
| P-G3 | high | Dry-run prints YT payload fully but omits IG/TikTok schedule/thumbnail/caption fields | `publish.py` dry-run branch | render all 3 platforms' full payloads |
| P-G7 | high | No virality-scorer → prediction loop; `pred_vs_actual MAE` is placeholder | `metrics.py` | wire `virality.score_short` into `metrics.predict()` |
| P-G10 | high | A/B variant→uploader override **not verified end-to-end** | `abtest.py` + `uploader.py` | integration test: ab pick → override |
| P-G12 | high | `brand_manifest.json` palette vs generated `avatar.png`/`banner_youtube.png` unverified | `gen_brand_assets.py` | `gen_brand_assets.py --check` consistency test |
| P-G5 | med | Schedule slots don't carry `persona_source`/render flag | `scheduling.py` | embed persona source + `NEEDS_RENDER` per entry |
| P-G4/G6 | med/low | Warm-up ramp off-by-one risk; timezone/hashtag rotation untested | `scheduling.py:90`, `test_schedule_cli.py` | boundary + rotation unit tests |
| P-G8/G9 | med | `metrics.jsonl`/`ab_variants.jsonl` no schema `_v`; `feedback_report` no action thresholds | `metrics.py`, `feedback.py` | add version + action rules |
| P-G13 | med | Missing IG profile / TikTok avatar assets | `docs/brand/` | generate + document |
| P-G14 | med | `publish_strategy.md`/`account_runbook.md` drift vs actual `cli.py` | cross-check | reconcile docs to CLI |
| P-G15 | med | publish path imports persona/tts at top-level (unneeded GPU/ONNX load) | `publish.py`/`cli.py` | lazy-import |

### 2.3 Knowledge Graph / Ingest / Speaker / Diarization
Full report: see KG subagent (includes the G1–G13 re-verification above).

| ID | Sev | Gap | Evidence | Fix |
|----|-----|-----|----------|-----|
| K-N1 | high | **Scott anchor fragmented to 4 `is_me` GlobalSpeakers** (should be 1) | live: `MATCH (g:GlobalSpeaker{is_me:true})` → 4; `anchor_speaker_me.py:95-113` merge logic exists but not applied | run `whoami --merge`; add health-check asserting exactly 1 |
| K-N3 | high | Distributed claim protocol abandoned; `fleet_batch.py` → external `assistx:8000` unwired; `outbox.py`/`events.py` unused | `fleet_batch.py:18,89`; `outbox.py` no caller | implement `/nas/drop` claim (IngestJob owner/TTL) or adopt AssistX + node agent |
| K-N4 | high | Backend routing detected but not consumed by linker/YOLO/whisper; HX370 CPU-only | `backend.py`, `link_global_speakers.py` | wire `backend_info()` into device selection; verify ROCm torch |
| K-N5 | med | Legacy scripts contradict CLI (`runall.sh`, `auto.py`, `autogen_*.py`, `run_ingest_all.sh`) | root scripts still live | archive to `archive/`; make `run-all` sole orchestrator |
| K-N6 | med | MCP/brain partial: missing `find_speaker`/`when_did`/`who_was_with`; `Mention`=0 (pattern extraction never run) | `scott_graph_mcp.py` (6 tools), live `Mention`=0 | add per-trip speaker tools + Entity/Mention extractor pass |
| K-N7 | med | `knowledge_map` still referenced by live scripts → runtime ModuleNotFoundError | `knowledge_sync_all.sh:73`, `knowledge_sync_handler.py:132` | build wrapper or repoint to `knowledge_harvest*.py` |
| K-N8 | med | Heavy backfills require `docker stop auto-ingest-service/worker` → not portable to other boxes | `run_full_speaker_link.sh:13` | detect container presence; or use Neo4j causal locking |
| K-N9 | med | Doc-path drift: `docs/UNIFICATION.md`/`docs/KG_INGEST_DEPLOY.md` don't exist (files at repo root); `LLD.md` still describes `knowledge_map` as real | root `UNIFICATION.md`, `KG_INGEST_DEPLOY.md` | RESOLVED 2026-07-17: files relocated to `docs/` via git mv; HLD.md relinked. `knowledge_map` is now a thin wrapper (see `auto_ingest/knowledge_map.py`, K-N7). |
| K-N2 | med | No `bin/run` symlink (G3 "same command everywhere" unmet) | `ls bin/run` → none | symlink `bin/run → auto-ingest` |
| K-N10 | low | No resumable per-source state beyond linker (`run-all` stages lack resume) | `bin/auto-ingest` | extend IngestJob manifest to all stages |
| K-N11 | low/med | `whoami --merge` not auto-run on every link pass → fragments regrow | `run_full_speaker_link.sh:30` | auto-merge at end of every link pass |

### 2.4 Services / Deployment / Orchestration / Observability / Config
Full report: see Ops subagent.

| ID | Sev | Gap | Evidence | Fix |
|----|-----|-----|----------|-----|
| O-G7 | **blocker** | Secrets in plaintext repo-tracked `.env*` (Neo4j bolt pwd, Tailscale, API keys) | `.env*`, `config*.yaml` | git-ignore `.env`; secret store / injected env only |
| O-G12 | high | **No CI/CD** (`.github/` has no `workflows/`) | `ls .github/` | add GH Actions: ruff + typecheck + tests |
| O-G6 | high | Config spread across ≥5 files with overlapping keys → drift | `config.yaml`,`.env`,`.env.example`,`.env-ssd`,`auto_ingest_config.py` | single loader |
| O-G1/G15 | high | Compose 7 services overlap `bin/auto-ingest` + shell scripts; no canonical runtime | `docker-compose.yml`, `scripts/runall.sh` | pick compose(prod)/CLI(dev); mark dead services |
| O-G3/G4 | high | No real claim protocol / job queue; dispatch cron-only; `job_trigger_api.py` stub-grade | `deploy/job_trigger_api.py`, grep "claim" → 0 | implement queue or document cron-only + inter-job guards |
| O-G8 | high | No log rotation; no compose healthchecks; no pipeline metrics | `logs/`, `docker-compose.yml` | RotatingFileHandler + logrotate + healthchecks + metrics |
| O-G9 | med | No alerting on Neo4j OOM/TransientError/down | `docs/DEEP_DIVE` | Neo4j watchdog alert |
| O-G13 | high | Env kills bg jobs; no documented `setsid`/tmux/systemd pattern for 20–24h speaker link / moviepy renders | operator context | standardize detached/supervised execution |
| O-G10/G11 | med | Runbook + birdcam guide likely lag scripts; no turnkey worker-box recipe | `docs/deployment_runbook.md`, `docs/birdcam_deployment_guide.md` | regenerate from scripts; add bootstrap checklist |
| O-G14 | med | NAS mount fragility (`fix-s-drive-mount.sh`), in-flight migration, SSD wear unprotected | `deploy/fix-s-drive-mount.sh` | mount-health watchdog; SSD-protect I/O plan |
| O-G5 | med | Content vs ingest cron coupling undocumented (content may run before ingest done) | `deploy/cron/*.crontab` | explicit dependency/guard |

---

## 3. Unified prioritized backlog

### P0 — Blockers (DONE ✅)
- **P-G1** Hard live-upload guard ✅ (`AUTO_INGEST_LIVE=1` + creds required; default refuse).
- **O-G7** Purge/protect secrets from repo (git-ignore `.env`; never commit real creds).

### P1 — High leverage / correctness (DONE ✅ except noted)
- **K-N1** Re-merge Scott anchor ✅ (`whoami --merge`) + assert exactly 1 `is_me` GlobalSpeaker.
- **O-G12** Stand up CI ✅
- **K-N4** Wire `backend.py` routing into linker/YOLO/whisper ✅
- **S-G1/S-G4/S-G7** Fix persona two-pass ✅
- **S-G16/S-G17** Consolidate renderers ✅
- **K-N3** Decide + implement a real distributed claim protocol (PENDING — out of scope this pass)
- **O-G6** Unify config into one loader (PENDING — out of scope this pass)

### P2 — Completeness / safety
- **S-G9/S-G10** Ground `myth_fact`/`reveal` in real graph text AND pass `driver` in default `plan` CLI.
- **P-G7** Wire virality scorer → `metrics.predict()` feedback loop.
- **P-G10** Integration test: A/B variant → uploader override (end-to-end).
- **P-G3** Full 3-platform dry-run payload rendering.
- **S-G18** Port Neo4j OOM/`TransientError` retry into `planner`/`curator`.
- **O-G8/O-G13** Log rotation + healthchecks + pipeline metrics; documented detached-job pattern.
- **K-N5/K-N2** Archive legacy scripts; symlink `bin/run`.
- **O-G1/G3** Pick canonical orchestration; retire dead compose services / stub API.

### P3 — Hardening / docs / nice-to-have
- **P-G12** `gen_brand_assets.py --check` (manifest vs assets).
- **P-G5** Persona source + `NEEDS_RENDER` in schedule slots.
- **S-G14/S-G15** CI `.venv` test matrix; generate synthetic smoke clip for persona test.
- **K-N6** Extend MCP tools to per-trip speaker queries + run Entity/Mention extractor.
- **K-N7/K-N9** Fix `knowledge_map` references; relocate doc paths; doc-lint.
- **P-G13** Generate IG/TikTok-specific brand assets.
- **P-G14** Reconcile `publish_strategy.md`/`account_runbook.md` to actual CLI.
- **K-N8/K-N11** Decouple backfills from container names; auto-merge on every link pass.
- **O-G10/G11/G14/G5** Runbook accuracy; birdcam cross-check; NAS watchdog; content/ingest ordering.

---

## 4. Notes / spot-checks performed by lead agent
- Live read-only: `MATCH (g:GlobalSpeaker{is_me:true})` → **4** (confirms K-N1 regression).
- `publish upload` default is `--dry-run` (`cli.py:300`) but no `AUTO_INGEST_LIVE` enforcement on the live path → confirms P-G1 is a *safe-by-default but not hard-guarded* blocker.
- `python3` (system) has no `moviepy`; `.venv` has `moviepy 1.0.3`; `.venv-tts` has `TTS` → confirms S-G14 split-venv CI gap.

## 5. Recommended next action
Start a **P0 + P1** tracking issue set. Concretely, the single highest-value first move is **O-G12 (CI) + P-G1 (live guard)** because they are cheap and prevent both regressions and an accidental real post, after which the render/persona consolidation (S-G1/G4/G7/G16) and the Scott re-merge (K-N1) are the next clean wins.
