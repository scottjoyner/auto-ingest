# Modern fleet: freeze-to-campaign SD handoff — October 16, 2026

**State:** metadata-only campaigns created on x1-370 OS disk. Source release remains **NO-GO**. Extends PR #28 and issue #27 without changing Beelink/Deathstar recovery holds.

## Why this bridge exists

The October 13 frozen source ledger accounts for all **4,268** original files on the three SD cards and contains **2,573** historical off-host hash qualifications. The bridge preserves these as historical provenance, never as current NAS destination verification or authorization to release a source. It never reads media payload bytes.

The new module auto_ingest.custody.fleet_campaigns performs:
1. SHA-256 of the **small frozen JSONL manifest**, checked against an independently pinned digest.
2. Exact /proc/mounts and /dev/disk/by-uuid source mount identity comparison. No volume-label guesses.
3. Full source pathname+size census, including all hidden and non-MP4 entries. Missing, changed, new and unsafe/symlink entries close the bootstrap gate.

The default CLI is dry-run JSON. The explicit --apply creates only new campaign.json and frozen-source-assessment.json on the independent OS disk, never NAS bytes or source changes. It refuses overwrites and disallows output under NAS/SD/removable mounts. It deliberately leaves every destination unresolved, writes no hash/copy/verification ledgers and sets no source custody release eligibility.

## Live x1-370 command

Frozen JSONL path:
    /media/scott/SSD_4TB/hermes-home/FLEET-STATE/sdcard-ingest/20261012/20261013-tranche17-independent-2573/complete-4268-source-2573-verified-custody.jsonl

Pinned SHA256:
    0a2213f72c921e6443c6753660109b5d681e3a50a1751d7cf2638f0f80e8afa5

From the auto-ingest checkout, run:

    python3 -m auto_ingest.custody.fleet_campaigns \
      --frozen-ledger /media/scott/SSD_4TB/hermes-home/FLEET-STATE/sdcard-ingest/20261012/20261013-tranche17-independent-2573/complete-4268-source-2573-verified-custody.jsonl \
      --expected-sha256 0a2213f72c921e6443c6753660109b5d681e3a50a1751d7cf2638f0f80e8afa5 \
      --card 69BE-B963=/media/scott/69BE-B963 \
      --card 68FC-E499=/media/scott/68FC-E499 \
      --card 5140-6932=/media/scott/UNTITLED \
      --output-root /home/scott/nas-reconciliation-audit-20261016/auto-ingest-campaigns-v1

This was run in dry-run mode and then **once** with --apply for metadata-only creation. Do not rerun --apply on the nonempty output directory.

## Physical results

| FAT UUID | Frozen files | Live name+size equal | Historical off-host qualified | Missing/new/changed |
|---|---:|---:|---:|---:|
| 69BE-B963 | 1,998 | 1,998 | 923 | 0 |
| 68FC-E499 | 716 | 716 | 716 | 0 |
| 5140-6932 | 1,554 | 1,554 | 934 | 0 |
| **Total** | **4,268** | **4,268** | **2,573** | **0** |

The bundles live in:

    /home/scott/nas-reconciliation-audit-20261016/auto-ingest-campaigns-v1/{69BE-B963,68FC-E499,5140-6932}

Readback shows each is state **DISCOVERED**, source read_only=false (the cards really are mounted rw), destination host_path=null, and source_release_allowed=false. No active hash/copy/verify/release ledgers were created. The card bytes and NAS bytes were untouched.

## Next independently gated transitions

1. Reconcile alternate NAS archive/staging copies by source UUID + digest using indexed manifests rather than 500+ GB of unnecessary rescanning.
2. Verify the x1 CIFS /nas to Beelink physical /nas/fileserver mapping and independently declare destination authority, Btrfs storage and snapshot retention.
3. Quiesce writers, obtain a read-only source observation and create fresh scoped source digest evidence in the repository custody model.
4. Copy only proven gaps, persist source/destination SHA-256 verified ledgers, preserve the 1999 clock/reset quarantine and all sidecars.
5. Review the full MP4-only **dry-run** file proposal, and separately authorize source unlink only when all gates pass. Preserve DCIM directory scaffolding.

**Not a release receipt:** path/size agreement, previous source SHA, available NAS space, or passing CI do not establish current destination custody.

## Next manifest-only archive anti-join

See [SD/NAS candidate index](SD_NAS_CANDIDATE_INDEX_2026-10.md). It narrows the first re-verification target set without changing any campaign states or source-release policy.
