# Offline NAS SHA verification against qualified xwing source receipts

Status: read-only, historical-source comparison only; source-release HOLD. This work follows draft PRs #28, #29, #30, #32, #33, and #34.

## Context

At 2026-10-16 13:59 EDT the three FAT card UUIDs 69BE-B963, 68FC-E499 and 5140-6932 are visible to the block layer but UNMOUNTED. No source-card payloads may be read until separately authorized mount/quiescence checks. The x1 CIFS NAS remained accessible.

The prior 46-object readiness report found every indexed candidate NAS MP4 path and byte size present, with zero media bytes read. The frozen October 13 source ledger identifies **18 of those 46** with both a full SHA-256 source digest and a positive xwing local NVMe independent full-hash verification witness. Merely having a digest string is not adequate; the positive witness is required.

## Measured results

| Item | Count or size |
| --- | ---: |
| Qualified offhost source-digest matches | **18 of 18** |
| Records on card 5140-6932 | **12** |
| Records on card 68FC-E499 | **6** |
| NAS contents freshly SHA-256 rehashed | **2,747,269,120 bytes** |
| Indexed NAS candidates still without comparable qualified source SHA | **28** |
| Earlier current SD-to-NAS SHA pairs, not rehashed now | **28** |
| Source-card payload bytes read this phase | **0** |
| Source-release authorizations | **0** |

Five individually capped NAS read-only batch reports contain the digest, candidate path, qualified historical source digest, pre/post destination stability, and a categorical false release flag. The reports are not source-custody ledgers and are not Beelink Btrfs physical-storage attestation.
## Reproduce

Frozen source manifest:
 /media/scott/SSD_4TB/hermes-home/FLEET-STATE/sdcard-ingest/20261012/20261013-tranche17-independent-2573/complete-4268-source-2573-verified-custody.jsonl

Expected frozen source manifest SHA-256:
 0a2213f72c921e6443c6753660109b5d681e3a50a1751d7cf2638f0f80e8afa5

Offline readiness input:
 /home/scott/nas-reconciliation-audit-20261016/auto-ingest-offline-remaining-46-nas-metadata-20261016.json

Run from the checked-out auto-ingest repo using an UNUSED output filename:

    python3 -m auto_ingest.custody.offline_nas_sha \
      --frozen /media/scott/SSD_4TB/hermes-home/FLEET-STATE/sdcard-ingest/20261012/20261013-tranche17-independent-2573/complete-4268-source-2573-verified-custody.jsonl \
      --expected-frozen-sha256 0a2213f72c921e6443c6753660109b5d681e3a50a1751d7cf2638f0f80e8afa5 \
      --readiness /home/scott/nas-reconciliation-audit-20261016/auto-ingest-offline-remaining-46-nas-metadata-20261016.json \
      --max-files 4 --max-nas-bytes 650000000 \
      --out /home/scott/nas-reconciliation-audit-20261016/new-unused-output.json

Repeat --prior-report /full/path/to/prior-batch.json for each previously verified NAS-only batch. Previous evidence is accepted for skipping only after pinning the source ledger and readiness report, candidate UUID/path/byte count, historical SHA, observed NAS SHA, and exact no-source-read/no-release flags. This is a throughput optimization, not independent live-source evidence.

Stored on x1 OS-local disk:
- auto-ingest-offline-nas-only-qualified-sha-batch01-20261016.json through batch05-20261016.json
- auto-ingest-offline-nas-18-xwing-qualified-sha-summary-20261016.json, containing all 18 verified historic-source comparisons plus SHA-256 checksums of the five batch reports.

## Safety gates

The verification command reads only selected NAS files (O_NOFOLLOW final-file read with pre/post stable stat and bounded size); it never mounts, copies, writes media, deletes, or reads source SD data. Output uses a new file on OS-local storage, not NAS. The exact expected ledger SHA is explicitly provided by the operator, making the code portable to future campaigns.

The other 28 candidates lack qualified source-hash evidence; a hash of NAS alone cannot prove SD equivalence. Beelink physical backing-store identity and snapshots/retention, source quiescence and deliberate RO remount, and file-by-file destination custody are still missing. The 3,619 videos with no candidate in scoped indexes are not globally confirmed absent. Preserve all original files, DCIM directories and sidecars. NO-GO on any source release.
