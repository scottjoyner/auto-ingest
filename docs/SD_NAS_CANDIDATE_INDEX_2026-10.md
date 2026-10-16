# SD-to-NAS candidate index — October 16, 2026

**Authority:** auto-ingest custody campaigns; **status:** manifest-only, no current destination SHA proof, no media copy and NO source release. This stacks on PRs #28 and #29; issue #27 remains the end-to-end tracker.

## What the index proves (and does not prove)

`python3 -m auto_ingest.custody.nas_evidence_index` consumes the pinned October 13 frozen 4,268-file source JSONL plus *four small* read-only NAS candidate/historical witness manifests. Source UUID and mount provenance come from `auto-ingest`'s stored campaign records rather than hard-coded x1 card mount strings. An explicit `--staging-root` identifies the x1-visible `/nas` archive subtree, which is **not** interchangeable with Beelink's physical `/nas/fileserver` namespace.

The index joins by exact **(card UUID, source-relative path, byte length)** and demands unique NAS candidate path ownership across all source objects. Malformed paths, mismatched source UUID/path, duplicate NAS destination ownership, inconsistent previous SHA reports, or a changing input manifest refuse the run. Historical SHA witnesses remain **point-in-time receipts only**. Every output row carries `current_nas_sha_verified=false` and `source_release_allowed=false`. Absence from these *scoped* candidate manifests is **not proof of NAS loss**. No archive payload bytes or card media bytes are read.

## Measured index, physical x1-370

| All source objects | Files | Logical bytes |
| --- | ---: | ---: |
| Existing historical source→NAS SHA witness; recheck live bytes | 20 | 310,241,245 |
| NAS path+size candidate only; verify SHA before custody | 498 | 9,434,540,206 |
| No candidate in scanned manifests; broader discovery needed | 3,750 | 522,634,433,419 |
| **Total** | **4,268** | **532,379,214,870** |

| MP4-named entries specifically | Files | Logical bytes |
| --- | ---: | ---: |
| Historical source→NAS SHA witness to recheck | 3 | 301,993,984 |
| Existing path+size candidate, not content-verified | 74 | 9,070,641,152 |
| No indexed archive candidate (not proven lost) | 3,619 | 522,581,727,880 |
| **MP4 total** | **3,696** | **531,954,363,016** |

There are **3,693 valid unique MP4 staging filenames**; two 1999 clock-reset recordings require review and one AppleDouble `._` file is filesystem metadata, not recorder media. These three are never silently counted as deletable.

The durable machine outputs live **outside NAS and the cards**, on x1's OS disk:

- `/home/scott/nas-reconciliation-audit-20261016/auto-ingest-nas-candidate-index-v1-20261016.jsonl` — 4,268 object-level rows, SHA-256 `736a784b398ff4e5a62355d9b043276c30ad8719b5c80ecb3e47588deef40f92`.
- `/home/scott/nas-reconciliation-audit-20261016/auto-ingest-nas-candidate-index-v1-20261016-summary.json` — bounded counts plus all input hashes.

Input source ledger SHA-256: `0a2213f72c921e6443c6753660109b5d681e3a50a1751d7cf2638f0f80e8afa5`.

## Safe replay command

From the checkout, with `audit=/home/scott/nas-reconciliation-audit-20261016` and `ledger=/media/scott/SSD_4TB/hermes-home/FLEET-STATE/sdcard-ingest/20261012/20261013-tranche17-independent-2573/complete-4268-source-2573-verified-custody.jsonl`:

```bash
python3 -m auto_ingest.custody.nas_evidence_index \
  --frozen-ledger "$ledger" \
  --expected-sha256 0a2213f72c921e6443c6753660109b5d681e3a50a1751d7cf2638f0f80e8afa5 \
  --campaign-root "$audit/auto-ingest-campaigns-v1" \
  --staging-csv "$audit/sd-5140-nas-staging-source-1554-metadata-20261016.csv" \
  --alternate-csv "$audit/sd-cards-68FC-69BE-alternate-root-candidates-20261016.csv" \
  --staging-witness "$audit/sd-5140-nas-staging-bounded-18-hash-witness-20261016.json" \
  --alternate-witness "$audit/sd-two-alternate-content-hash-witness-20261016.json" \
  --staging-root /nas/recovery/dashcam-sd-ingest/SDCARD-20261010-51406932/source
```

This only prints bounded statistics and a sample. Optional `--out PATH` writes a new **exclusive** JSONL report on the OS disk; it will not overwrite existing files and refuses NAS, source-card and campaign directories. It is **not** an ingestion execution or source retirement command.

## Next safety gates

1. Register more candidate indexes for historical `deathstar`, NAS5 recovery, and direct date folders while preserving existing NO-GO and ownership boundaries. Prefer cheap manifest joins before remote I/O.
2. Obtain bounded, fresh independent NAS SHA readback for the **3 previously SHA-matched MP4s**, then for the other **74 size-only MP4 candidates**. Check exact NAS path, device identity, stable content, source/independent digest and archive permanence. Do not auto-promote any historical witness to the current custody ledger.
3. Classify 3,619 unindexed MP4s by source UUID and original clock metadata; search additional roots and existing off-host receipt manifests, then stage/copy only independently proven gaps.
4. Only after source quiescence and hardware RO observation, full authoritative source + NAS digest ledgers and `SAFE_TO_RELEASE`, allow a separately approved exact-file `release-source --mp4-only` proposal. Preserve `DCIM/` and the rest of the original filesystem tree.

## Controlled current content comparison

The candidate index itself never hashes media. For a deliberately bounded source-versus-destination recheck of existing candidate MP4s, see [bounded live media probe](SD_BOUNDED_LIVE_MEDIA_PROBE_2026-10.md). It has independent mount/read-budget checks, and its receipts explicitly cannot open the source-release gate.

## Additional date-folder candidate discovery

[Date-folder NAS reconciliation](SD_DATE_ARCHIVE_RECONCILIATION_2026-10.md) adds a bounded metadata-only candidate finder for a separate `deathstar/8TB_2025` archive subtree. It found 1,136 additional MP4 paths, including all 1,074 previously unindexed `UNTITLED` MP4 names. These are not current SHA custody proofs.
