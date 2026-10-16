# Auto-Ingest — October 2026 fleet storage and SD-card custody

**Status:** implementation + read-only validation; NO card source release, NO bulk transfer, NO deployment authorization. Source of truth for SD ingestion is `auto_ingest/custody`, not a pile of ad-hoc `rsync`/NAS recovery commands. Tracking issue: [#27](https://github.com/scottjoyner/auto-ingest/issues/27).

## Observed fleet roles

| Host | October role | Rule |
| --- | --- | --- |
| `x1-370` | SD-card reader and campaign orchestrator; local SSD for small manifests | Avoid heavyweight inference and broad NAS walks while ingesting. Three observed vfat cards are mounted **rw**, hence **not release-ready**. |
| Beelink / NAS5 | Durable Btrfs-backed media and NAS/archive custody | A share being mounted does not independently prove that a source file was archived. Never conflate `/nas` namespace from two hosts. |
| `deathstar` | Historical dashcam source and cold/recovery receipts | Preserve SIGSTOPPED recovery workers and independent receipts until their own reconciliation closes. |
| `xwing` | GPU video analysis / detection and Hermes graph jobs | Compute must consume already verified NAS files; not an SD deletion oracle. Active Hermes jobs require exact-checkout binding. |
| Joyner / other fleet | Optional inference and receipt witnesses | Do not infer mount names or access from the x1 host profile. Verify each host separately. |

## Path namespace (load-bearing)

On **x1**, `/nas` is the CIFS share `//192.168.1.202/fileserver` at the time of the October 16 observation. Therefore x1 `/nas/archive/dashcam` refers to Beelink physical `/nas/fileserver/archive/dashcam`. x1 `/nas/fileserver/dashcam/sdcard-ingest` refers to another, nested tree and is **not automatically canonical**. The historical `/nas/dashcam`, `/nas/dashcam-evac`, `/nas/recovery/dashcam-sd-ingest`, and `/nas/fileserver/dashcam/sdcard-ingest` are **read/reference namespaces** until reconciled by content and provenance. Different hosts must not reuse x1 literal paths.

`deploy/fleet-profiles/x1-370-custody.env.example` illustrates the x1 paths for a *future* operator-configured deployment, without passwords or an asserted mount device. It is **not** activated automatically. `deploy/path_profiles.env.example` is a June legacy reference and contains obsolete `/media/scott/NAS5` paths. Do not use it unchanged for the current fleet.

## October 16 mounted SD snapshot

| UUID | Mount on x1 | MP4 names | MP4 logical bytes | Staging disposition after current patch |
| --- | --- | ---: | ---: | --- |
| `69BE-B963` | `/media/scott/69BE-B963` | 1,979 | 286,030,060,168 | 1,979 unique media targets |
| `68FC-E499` | `/media/scott/68FC-E499` | 609 | 84,778,680,320 | 609 unique targets; prior 260 collision names repaired |
| `5140-6932` | `/media/scott/UNTITLED` | 1,108 | 161,145,622,528 | 1,105 unique; two pre-2000 clock reviews; one AppleDouble metadata |

The 3,693 stageable target paths have **zero collisions both within and across the three cards**. This is a *filename-plan* assertion, not a transfer or checksum receipt. All 4,268 original files remain in custody. Historical evidence: `~/knowledge/20-Projects/NAS5-RECONCILIATION-INVENTORY-2026-10-16.md`, the same-day continuation and `/home/scott/nas-reconciliation-audit-20261016`.

## UUID-bound campaign handoff

The physical 4,268-file source ledger is now bridged into three real auto-ingest campaign bundles, with source observations, no destination assertion, and no release eligibility. See [frozen ledger to custody campaigns](SD_FROZEN_LEDGER_TO_CUSTODY_CAMPAIGNS_2026-10.md). This is intentionally separate from ingestion hash/copy/verification stages.

## Read-only fleet preflight

```bash
cd /home/scott/git/auto-ingest
python3 -m auto_ingest.custody.fleet_preflight \
  --card /media/scott/69BE-B963 \
  --card /media/scott/68FC-E499 \
  --card /media/scott/UNTITLED
```

Optional `--destination-root /nas/archive/dashcam` probes exact proposed file paths and sizes **only**; it does not hash, transfer, or delete, and must never be treated as proof of custody. Avoid that path-probe flag during NAS pressure. Status fields state the evidence scope explicitly. By design there is no `--execute` option in the preflight.

## Required lifecycle and safety gates

1. Verify source UUID and device and obtain a stable, read-only source observation; avoid active writers. Inventory *all* source objects, including out-of-scope, AppleDouble, and clock-anomaly entries, without discarding any bytes.
2. Anti-join source UUID+digest receipts against independent NAS evidence indexes. Do not assume a matching filename, size, Neo4j row, or previous `rsync` exit status proves destination custody.
3. Resolve cross-card collision and time-anomaly quarantines. Publish exact immutable source->staged destination mapping. No staging plan may overwrite an existing unverified target.
4. Use campaign `hash`, `stage`, `preflight`, `execute` and `verify` with authorized explicit copy windows; store SHA-256 digests, immutable destination identity, and append-only ledgers outside the source card.
5. Before any removal, require `SAFE_TO_RELEASE`, coherent source+destination+staged ledgers, independent full SHA-256 destination verification, a reviewed dry-run proposal and durable audit path. Use `custody release-source --mp4-only` to **propose** the exact MP4 set; `--execute` is a separately authorized destructive action, never part of this PR.
6. When separately authorized and gate-open, release only recorded MP4 source files. The custody unlink function preserves `DCIM/` and all nested directory scaffolding; CSV, JPG and other sidecars remain. Post-release confirm the manifest and source counts.

Other recovery workstreams (Deathstar, Beelink source retirement, missing scanner receipts, 2-TB hash) remain independently on **NO-GO**. Passing SD custody tests cannot override those controls. This change intentionally does **not** deploy Docker, modify production graph service bindings, start old cron jobs, mount filesystems, or touch SD/NAS media bytes.
