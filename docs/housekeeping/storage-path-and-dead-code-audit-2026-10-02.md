# Storage-Path & Dead-Code Audit — 2026-10-02

> **Status:** CURRENT REFERENCE for path authority, entrypoint status, and the
> classified legacy-storage inventory. Produced by a static-only housekeeping pass
> (no services started/stopped, no jobs run, no media touched).
> **Scope:** `scottjoyner/auto-ingest` only. Branch: `housekeeping/storage-path-audit`.

---

## 1. Active architecture as proven by repository evidence

| Question | Evidence-based answer | Source(s) |
|---|---|---|
| Canonical configuration layer | `config.yaml` (committed, per-`machine_paths`) + environment overrides; `.env` (copied from `deploy/path_profiles.env.example`) is the container/host contract | `config.yaml`, `deploy/path_profiles.env.example`, `auto_ingest_config.py` |
| Canonical path resolver | `auto_ingest_config.py` (`get_fileserver_root`, `get_audio_root`, `get_dashcam_root`, `get_bodycam_root`, `get_cold_root`, `get_hot_root`). `auto_ingest/_config.py` is a thin re-export shim | `auto_ingest_config.py`, `auto_ingest/_config.py` |
| Supported env vars | `FILESERVER_ROOT`, `HOT_STORAGE_ROOT`, `COLD_STORAGE_ROOT`, `AUDIO_ROOT`, `DASHCAM_ROOT`, `BODYCAM_ROOT`, `TRANSCRIPT_ROOT`, `DROP_ROOT`, `NAS_ROOT`, `SCAN_ROOTS`, `NEXTCLOUD_ROOT`, `NEO4J_*`, `TIKTOK_OUT_ROOT`, `SHORTS_PUBLISH_QUEUE`, `SHORTS_OUT_ROOT` | `deploy/path_profiles.env.example`, `auto_ingest_config.py` |
| Current `/nas` contract | `/nas` is the **container-side** canonical mount: `docker-compose.yml` binds `${NAS_ROOT}:/nas`; `DROP_ROOT=/nas/drop`. Host-side paths vary per host and come from `.env` | `docker-compose.yml`, `deploy/path_profiles.env.example`, `docs/system_design.md` |
| Active CLI entrypoint | `bin/auto-ingest` (also `bin/run` → symlink) | `README.md`, `bin/auto-ingest` |
| Worker / job queue contract | `.job` files in `${DROP_ROOT}` (`/nas/drop`), claimed by `deploy/worker_ingest.sh` every 30 s | `docker-compose.yml`, `deploy/create_job.sh` |
| Legacy/deprecated entrypoints | `run_all_optimized.sh` (shim), `runall.sh` (shim), `deploy/job_trigger_api.py` (stub), `batch_update_paths.py` (one-shot codemod) | see §4 |

### Recorded disagreements (NOT silently resolved)

1. **NAS4 vs NAS5.** `config.yaml` and `deploy/path_profiles.env.example` say **NAS5**;
   `.env.example` and `config.example.yaml` said **NAS4** (now aligned to NAS5 in this
   pass, with an inline note). `config.example.yaml` also still carried `/mnt/8TB_2025`
   and Scott-specific `/home/scott/*` vault paths — retained, they are example content.
2. **Host path vs container path.** `deploy/path_profiles.env.example` sets
   `DASHCAM_ROOT=/media/scott/NAS5/dashcam` (host path) while `docs/system_design.md`
   documents `DASHCAM_ROOT=/nas/fileserver/dashcam` (container path). Compose injects
   `DASHCAM_ROOT` into the container where only `/nas` exists. **Unresolved on purpose:**
   changing it could redirect existing data writes. Flagged for operator decision.
3. **Legacy alias mounts.** Compose still binds `NAS1_ROOT`/`NAS2_ROOT` to `/nas1` and
   `/nas2`; `scripts/backfill_audio_day.sh` still consumes `/nas1`/`/nas2`. Not removed:
   removing the mounts would break that script inside containers.
4. **Vault canonical path.** `config.yaml` sets `mirror_vault_path` *and*
   `canonical_vault_path` to `/media/scott/NAS5/shared-knowledge`, while
   `scripts/sync_knowledge_to_nas.sh` defaulted to a `NAS4/shared-knowledge-mirror`.
   The script default was aligned to the config value; the `shared-knowledge-mirror`
   variant is retained only as the "revert to" comment.

---

## 2. Path-resolution precedence (documented contract)

```
explicit environment variable
  → canonical resolver (auto_ingest_config, config.yaml machine_paths)
    → FAIL CLOSED (RuntimeError / required env) when storage is unconfigured
```

Removed in this pass: silent probing of *other* machines' roots
(`/mnt/8TB_2025/...`, `/media/scott/SSD_4TB/...`, `/media/scott/NAS5/...`) from the
Python resolver, and per-host hardcoded defaults in active shell/Python callers.
A legacy directory that merely *exists* on a host can no longer win.

---

## 3. Duplicate-configuration-authority map

| Caller | Resolver | Old fallback | Env override | Risk (before) | Recommended authority |
|---|---|---|---|---|---|
| `auto_ingest_config.get_fileserver_root` | self | `/mnt/8TB_2025`, `/media/scott/SSD_4TB` | `FILESERVER_ROOT` | legacy mount wins silently | env → config.yaml → fail closed (**fixed**) |
| `auto_ingest_config.get_cold_root` | self | `/media/scott/NAS5` | `COLD_STORAGE_ROOT` | cold writes to NAS3-era tree | env → config.yaml → fail closed (**fixed**) |
| `auto_ingest_config.get_audio_root` | self | 3 legacy mounts | `AUDIO_ROOT` | audio could land on SSD/8TB mirror | env → hot root → fail closed (**fixed**) |
| `auto_ingest_config.get_dashcam_root`/`get_bodycam_root` | self | `/media/scott/NAS5/...` | `DASHCAM_ROOT`/`BODYCAM_ROOT` | dashcam/cold split | env → fileserver root → fail closed (**fixed**) |
| `auto_ingest/_config.py` inline fallback | inline copy | `/media/scott/SSD_4TB/...` | same env vars | **second authority** with different answers | env-only + fail closed (**fixed**) |
| `deploy/manage.sh` (health) | shell | `/media/scott/NAS3` | `NAS_ROOT` | reported health for a dead mount | `NAS_ROOT` required (**fixed**) |
| `deploy/job_trigger_api.py` | env | `/media/scott/NAS4` | `NAS_ROOT` | stale host default | env only (**fixed**) |
| `process_usb_device.py` | literals | `/media/scott/SSD_4TB/fileserver`, `/media/scott/data/git/...` | — | wrong checkout, baked creds | `auto_ingest_config` + repo-relative (**fixed**) |
| `transcribe_usb_batch.py` | literals | `/media/scott/SSD_4TB/audio`, `/media/scott/data/git/...` | — | writes into stale roots | env + repo-relative (**fixed**) |
| `worker_content.py` | env/literal | `/media/scott/SSD_4TB/tiktok_shorts` | `TIKTOK_OUT_ROOT` | shorts written to legacy SSD | env required (**fixed**) |
| `bulk_ingest_dashcam.sh` | literal | `/media/scott/NAS2/fileserver/dashcam` | `DASHCAM_ROOT` | docker `-v` of a dead mount | env required (**fixed**) |
| `scripts/ingest_supervisor.py` | literal `-v` | `/mnt/8TB_2025/fileserver/dashcam` | `DASHCAM_ROOT` | container mounts wrong tree | resolver + env (**fixed**) |
| `scripts/knowledge_sync_handler.py` | literal | `/media/scott/NAS2/fileserver/shared-knowledge` | `KNOWLEDGE_MIRROR_PATH` | vault writes to NAS2 | env required (**fixed**) |
| `scripts/get_embedding_endpoint.py` | literal registry path | `.../SSD_4TB/hermes-home/auto-ingest/...` | `EMBEDDING_REGISTRY` | cross-repo checkout dependency | repo-relative (**fixed**) |
| `finish_location_backfill.sh`, `install-signal-bridge-cron.sh` | literal repo path | `.../SSD_4TB/hermes-home/auto-ingest` | — | breaks off that host | script-relative (**fixed**) |
| `auto_ingest/shorts/{publish,uploader,cli}.py` | literal queue/out roots | `/media/scott/NAS5/fileserver/...` | `SHORTS_PUBLISH_QUEUE`, `SHORTS_OUT_ROOT` | publish queue on one host only | env required (**fixed**) |
| `ingest_media.py` | env + literal | `/media/scott/SSD_4TB/{photos,nextcloud,media_thumbs,tiktok_shorts,nextcloud_stage}` | `NEXTCLOUD_ROOT`, `MEDIA_THUMB_ROOT`, `TIKTOK_OUT_ROOT`, `NEXTCLOUD_STAGE_ROOT` | **not changed** — see §7 | next slice |
| `scripts/sync_hermes_data.sh`, `scripts/setup_shared_hermes.sh`, `scripts/shared_hermes_handler.py` | literals (`/media/scott/S/...`, `~/.hermes`) | — | — | legacy sync lane | next slice |
| `deploy/fix-s-drive-mount.sh`, `scripts/run_backfill_x1.sh`, `scripts/run_watch_x1.sh`, `scripts/setup_knowledge_cron.sh`, `scripts/enrich_pipeline.py`, `scripts/knowledge_harvest_v2.py` | literal host/home/venv paths | — | — | host-specific operator scripts | next slice (mark deprecated) |

---

## 4. Entrypoint classification

| Entrypoint | Class | Action taken |
|---|---|---|
| `bin/auto-ingest` (+ `bin/run`) | CANONICAL | none |
| `run_all_optimized.sh` | COMPATIBILITY_SHIM | already `exec`s `bin/auto-ingest run-all` |
| `runall.sh` | DEPRECATED (compatibility shim) | loud deprecation header; removed `cd ~/git/auto-ingest` (script-relative) |
| `deploy/job_trigger_api.py` | DEPRECATED (unauthenticated stub) | pre-existing header; removed NAS4 default |
| `batch_update_paths.py` | DEPRECATED / dangerous | loud header, **dry-run by default** (`--apply` to write), repo-relative `REPO` |
| `dashcam_copy.sh`, `bodycam_copy.sh`, `audio_copy.sh` | SPECIALIZED_CURRENT (called by `bin/auto-ingest run-all` + `runall.sh`) | none (mount guards retained) |
| `run_pipeline.sh`, `run_worker.sh`, `run_ingest_all.sh`, `run_ingest_daily.sh` | SPECIALIZED_CURRENT | `run_pipeline.sh` `OUT_ROOT` now required |
| `process_usb_device.py`, `transcribe_usb_batch.py` | SPECIALIZED_CURRENT | env/repo-relative paths |
| `deploy/worker_ingest.sh`, `deploy/create_job.sh` | CANONICAL (queue) | none |
| `install-signal-bridge-cron.sh`, `scripts/setup_knowledge_cron.sh`, `scripts/setup_shared_hermes.sh` | CRON INSTALLERS — **cannot be proven unused from the repo** (external crontabs) | repo-relative paths only; **not deleted** |
| `scripts/run_watch_x1.sh`, `scripts/run_backfill_x1.sh`, `scripts/run_bodycam_2nd_x1.sh`, `scripts/run_bodycam_all_xwing.sh`, `scripts/run_dashcam_frames_x1.sh`, `scripts/run_research_mini12_xwing.sh` | DEPRECATED host-specific runners (nohup/setsid) | not deleted (may be invoked externally) |
| `dashcam_monitor.py`, `dashcam_salvage.py`, `whisper_audio_chunked.py`, `parse_vehicel_detections.py`, `save_detected_files.py`, `yolo_vehicle_detction.py`, `add_audio.sh`, `move_files.py` | SPECIALIZED / ONE-OFF | legacy write targets made env-required |

---

## 5. Cross-repository checkout dependencies

- `auto_ingest/events.py` documented `…/home_scott_git_auto-assist/src/assistx/contracts/`
  as "canonical source of truth" in a docstring while the code already imports
  `assistx.contracts.*` and falls back locally. **Fixed**: docstring now names the
  package contract, not a checkout path. Fallback behavior unchanged.
- `scripts/get_embedding_endpoint.py` read its registry from another host's checkout
  path. **Fixed**: defaults to the repo-relative `embedding_endpoints.json`
  (`EMBEDDING_REGISTRY` still overrides).
- `finish_location_backfill.sh`, `install-signal-bridge-cron.sh`, `scripts/run_narrated_mix.py`
  no longer reference `…/hermes-home/auto-ingest`.
- Remaining physical-checkout references live only in **docs** (historical) — see §6.

---

## 6. Historical documentation (retained, labeled)

| Document | Label added |
|---|---|
| `docs/DEEP_DIVE_2026-07-14.md` | HISTORICAL / SUPERSEDED banner |
| `docs/deathstar-cli-storage-migration.md` | HISTORICAL / SUPERSEDED banner (NAS3 era) |
| `docs/current-ingest-state-2026-06-10.md` | HISTORICAL SNAPSHOT banner |
| `docs/KG_INGEST_DEPLOY.md` | HISTORICAL DEPLOY NOTE banner |
| `docs/recovery-plan-2026-06-10.md`, `docs/OFFLINE_SWARM_INTEGRATION_PLAN.md`, `LLD.md`, `DASHCAM_VISION_PIPELINE.md`, `notes.md`, `KNOWLEDGE_MAP_SUMMARY.md` | not relabeled (design/history text, no deployment instruction) |
| `docs/system_design.md`, `deploy/skills/auto-ingest-docker/SKILL.md` | **current** docs: removed the `NAS_ROOT=/media/scott/NAS1` default (no longer exists) |

---

## 7. Legacy-path inventory (classified)

Legend: `ACTIVE_RUNTIME` / `ACTIVE_DEFAULT` (were) fixed in this pass; the rest are
intentionally retained.

### ACTIVE (resolved)

`deploy/manage.sh`, `deploy/job_trigger_api.py`, `auto_ingest_config.py`,
`auto_ingest/_config.py`, `auto_ingest/events.py`,
`auto_ingest/shorts/{cli,publish,uploader}.py`, `process_usb_device.py`,
`transcribe_usb_batch.py`, `worker_content.py`, `dashcam_monitor.py`,
`dashcam_salvage.py`, `dashcam_frame_vision.py`, `run_pipeline.sh`,
`mount_nextcloud.sh`, `runall.sh`, `bulk_ingest_dashcam.sh`, `run_dashcam_vision.sh`,
`add_audio.sh`, `finish_location_backfill.sh`, `install-signal-bridge-cron.sh`,
`batch_update_paths.py`, `scripts/get_embedding_endpoint.py`,
`scripts/knowledge_sync_handler.py`, `scripts/sync_knowledge_to_nas.sh`,
`scripts/backfill_audio_day.sh`, `scripts/qa_audio_day.sh`, `scripts/anchor_speaker_me.py`,
`scripts/run_narrated_mix.py`, `scripts/ingest_supervisor.py`,
`parse_vehicel_detections.py`, `save_detected_files.py`, `yolo_vehicle_detction.py`.

### RETAINED — classified, deliberately unchanged

| Class | Locations |
|---|---|
| EXAMPLE | `config.yaml`, `config.example.yaml`, `.env.example`, `deploy/path_profiles.env.example`, `embedding_endpoints.json` |
| ACTIVE_DEFAULT (current contract, not legacy) | `docker-compose.yml` (`NAS_ROOT`→`/nas`, legacy `/nas1`,`/nas2` aliases), `deploy/sync_from_legacy_drop.sh` (`/nas/...` container contract), `deploy/create_job.sh` (`/nas/drop`), `scripts/backfill_audio_day.sh` (`/nas1`,`/nas2`) |
| SPECIALIZED_CURRENT | `ingest_media.py` (SSD_4TB output roots — changing them could redirect media writes), `dashcam_copy.sh` / `bodycam_copy.sh` (NAS5 mount guards + `findmnt` auto-discovery), `run_ingest_all.sh` (comment), `scripts/_smoke_render_shorts.py` (dev-only host remap) |
| DEPRECATED_SCRIPT | `runall.sh`, `batch_update_paths.py`, `deploy/job_trigger_api.py` (loud deprecation header) + banner added to `deploy/fix-s-drive-mount.sh`, `scripts/run_backfill_x1.sh`, `scripts/run_bodycam_2nd_x1.sh`, `scripts/run_bodycam_all_xwing.sh`, `scripts/run_dashcam_frames_x1.sh`, `scripts/run_research_mini12_xwing.sh`, `scripts/run_watch_x1.sh`, `scripts/enrich_pipeline.py` (`/home/scott/git/hermes-agent/venv`), `scripts/knowledge_harvest_v2.py` (`/media/scott/S/neo4j`). Banners state *host-specific legacy helper* and point at `bin/auto-ingest`; they do **not** assert removal (external cron/`nohup` invocation cannot be proven from the repo) |
| HISTORICAL_DOC | `docs/current-ingest-state-2026-06-10.md`, `docs/deathstar-cli-storage-migration.md`, `docs/DEEP_DIVE_2026-07-14.md`, `docs/KG_INGEST_DEPLOY.md`, `docs/recovery-plan-2026-06-10.md`, `docs/OFFLINE_SWARM_INTEGRATION_PLAN.md`, `docs/UNIFICATION.md`, `LLD.md`, `DASHCAM_VISION_PIPELINE.md` |
| ARCHIVE | `archive/**`, `plans_narrated/**` (narrated plan data, untouched) |
| UNKNOWN | `scripts/sync_hermes_data.sh`, `scripts/setup_shared_hermes.sh`, `scripts/shared_hermes_handler.py` (`/media/scott/S/shared-hermes`, `~/.hermes` symlinks) — legacy sync lane whose consumer is not provable from this repo; banner added, behavior untouched. Also `scripts/_smoke_render_shorts.py` (dev-only `/mnt/8TB_2025`→SSD remap) |

### Current-era paths that are NOT defects

`/media/scott/NAS5` and `/media/scott/SSD_4TB` remain the **current** documented
host mounts in `config.yaml` / `deploy/path_profiles.env.example`; `/nas` is the
current **container** contract. NAS1/NAS2/NAS3/NAS4 and `/mnt/8TBHDD` are legacy.

---

## 8. Dangerous-housekeeping findings (statically inspected)

| Finding | Location | Action |
|---|---|---|
| Hardcoded default password in 14 active scripts (pre-existing lint failure on `main`) | `link_entities_crosskg.py`, `link_entities_dedupe.py`, `dashcam_monitor.py`, `embed_ab_eval.py`, `dashcam_link.py`, `graph_communities.py`, `crosskg_query.py`, `dashcam_ask.py`, `dashcam_salvage.py`, `embed_coverage.py`, `community_summaries.py`, `run_pipeline.py`, `run_dashcam_vision.sh`, `scripts/run_research_mini12_xwing.sh` | routed through `NEO4J_PASSWORD_DEFAULT` chain; `tools/lint_no_hardcoded_secrets.py` now passes (**fixed**) |
| Repo-wide in-place codemod | `batch_update_paths.py` | dry-run by default (**fixed**) |
| Automatic mount discovery (`findmnt` scraping) | `dashcam_copy.sh`, `bodycam_copy.sh` | **not changed** — behavior-critical copy source selection |
| Cron installers writing live crontabs | `install-signal-bridge-cron.sh`, `scripts/setup_knowledge_cron.sh`, `scripts/setup_shared_hermes.sh` | **not executed, not deleted**; status banners added, `install-signal-bridge-cron.sh` made repo-relative |
| `nohup`/`setsid` background workers | `scripts/run_dashcam_frames_x1.sh` and `finish_location_backfill.sh` header | **not changed** (documented; may be started by operators) |
| Directory-exists == correct-storage assumption | `auto_ingest_config._first_existing_path` (kept as a helper) | no longer used for root selection (**fixed**) |
| Embedded private IPs | `dashcam_frame_vision.py --vision-url http://100.85.64.117:1234/...`, `docs/*` | **not changed** (service discovery is out of scope) |
| `find`/hash recursion over media trees | not present in active code | — |

**Not performed by this campaign:** credential rotation, service start/stop, cron or
systemd changes, media copy/move/delete, Neo4j mutation, job claiming, custody,
routing, or schema changes.

---

## 9. Guard test

`tests/test_storage_contract.py` pins the resolver contract (env precedence,
fail-closed, legacy NAS3/SSD_4TB directories cannot win, generic-host config, 
determinism) **and** fails if a listed ACTIVE file reintroduces a legacy literal
(NAS1/NAS2/NAS3/NAS4, `/mnt/8TBHDD`, `home_scott_`, `/media/scott/data/git`,
`…/SSD_4TB/hermes-home`). Extend `ACTIVE_FILES` when new files are promoted to active.