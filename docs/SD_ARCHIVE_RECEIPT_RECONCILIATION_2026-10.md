# Three-card raw SD archive: exact receipt reconciliation

As of October 16, the supervised x1-370 user service `auto-ingest-sd-nas-copy-20261016.service` is making bounded, verified copies of all three FAT SD cards into Beelink's NAS. **This is not an authorization to erase any SD original**. The archive preserves the full source-relative folder layout, including JPG, CSV, AppleDouble, quarantine anomalies and FAT metadata.

## Authoritative inputs

The frozen source ledger is `/media/scott/SSD_4TB/hermes-home/FLEET-STATE/sdcard-ingest/20261012/20261013-tranche17-independent-2573/complete-4268-source-2573-verified-custody.jsonl`.

Its SHA-256 is `0a2213f72c921e6443c6753660109b5d681e3a50a1751d7cf2638f0f80e8afa5`, representing **4,268 source files and 532,379,214,870 bytes** across UUIDs 69BE-B963, 68FC-E499 and 5140-6932.

Per-file SHA-256 transfer receipts are fsynced under `/home/scott/nas-reconciliation-audit-20261016/sdcopy-*.jsonl`. Each receipt states exact card UUID, source-relative path, source size, source hash, actual NAS hash and pathname, source/destination stability flags and `source_release_authorized=false`.

The actual NAS destination of a source key is always `/nas/recovery/dashcam-sd-ingest/SDCARD-20261016-<UUID>/source/<original-source-relative-path>`.

## Exact inventory reconciliation

`python3 -m scripts.sd_archive_receipt_audit` independently checks that the entire frozen ledger is checksum-identical, contains only valid, nonduplicated source identities and has exactly the expected 4,268 entries. It reads each per-file copy receipt, rejects duplicate identities, invalid paths/UUIDs, wrong destination namespace, mismatching byte count, unequal/invalid SHA-256 digests, unstable source or destination flags and any unexpected release authorization.

Example (read-only on source and NAS, no media bytes read):

    python3 -m scripts.sd_archive_receipt_audit \
      --frozen /media/scott/SSD_4TB/hermes-home/FLEET-STATE/sdcard-ingest/20261012/20261013-tranche17-independent-2573/complete-4268-source-2573-verified-custody.jsonl \
      --expected-sha256 0a2213f72c921e6443c6753660109b5d681e3a50a1751d7cf2638f0f80e8afa5 \
      --audit-root /home/scott/nas-reconciliation-audit-20261016 \
      --out /home/scott/nas-reconciliation-audit-20261016/new-unique-audit-summary.json

The output includes SHA-256 pins for each receipt file, counts and missing bytes by card. An incomplete copy exits code 3. A complete *receipt* census exits code 0 but **still explicitly denies SD deletion**.

## Separate gates before deletion

1. Finish all 4,268 raw copies with stable matching source and NAS digests, and stop/quiesce the SD writers.
2. Re-read/reverify the actual NAS destination inventory and independently prove the Beelink filesystem identity plus snapshot or retained second-copy protection. An empty `/nas/snapshots` directory is NOT proof of backup.
3. Reconcile the exact final source-to-destination inventory and source mount quiescence. The three existing custody campaigns are currently BLOCKED: source mounts/declarations RW, destination identity unresolved, missing full hash/copy/destination release ledgers. Do not modify policy to pretend those observations are satisfied.
4. Use the audited, explicitly authorized `auto_ingest.custody.release_source` gate *only after it passes all requirements*; never use `rm -rf`, formatting, a blind `rsync --remove-source-files` or an MP4-only cleanup that discards sidecars. All source objects, including quarantined timestamp anomalies and metadata, require their own verified preservation.

The receipt auditor performs **zero source/NAS payload reads** and cannot by itself prove current NAS bytes have not changed since their recorded SHA matches; it does not create release ledgers, authorize deletion, alter a mount, or write on NAS.
