# Three-card NAS raw preservation copy — October 16, 2026

## Intent and authority

The owner explicitly authorized copying the contents of all three SD cards to Beelink NAS and releasing the originals **after verified custody**. These are separate operations. This phase implements the copy and SHA-256 verification portion, but never deletes, truncates, renames or alters SD-card originals. Source release still requires the normal fail-closed custody gate and storage-retention evidence.

The authoritative read-only manifest is the frozen 4,268-object ledger:
`/media/scott/SSD_4TB/hermes-home/FLEET-STATE/sdcard-ingest/20261012/20261013-tranche17-independent-2573/complete-4268-source-2573-verified-custody.jsonl`

Pinned SHA-256:
`0a2213f72c921e6443c6753660109b5d681e3a50a1751d7cf2638f0f80e8afa5`

The latest physical x1-370 census agrees with all 4,268 object names and sizes across card UUIDs 69BE-B963, 68FC-E499 and 5140-6932. No extra/missing/changed-size entries were found. All source mounts are still **rw**. The NAS CIFS mount is `//192.168.1.202/fileserver` at `/nas`; direct SSH to `BEELINK-RYZEN-7-MINI-PC` confirmed `/nas/fileserver` is on `/dev/sdd2 btrfs` and about 39 TB was free.

### Exact raw-archive namespaces

- `/nas/recovery/dashcam-sd-ingest/SDCARD-20261016-69BE-B963/source/`
- `/nas/recovery/dashcam-sd-ingest/SDCARD-20261016-68FC-E499/source/`
- `/nas/recovery/dashcam-sd-ingest/SDCARD-20261016-5140-6932/source/`

**These preserve every source-relative path exactly**, including `DCIM/Movie`, `DCIM/Photo`, CSV/JPG sidecars, quarantined date anomalies, AppleDouble files and any non-media filesystem metadata. Nothing is silently flattened into a date directory. Existing older staging/archive paths stay untouched.
## Guarded copy operator command

The new `scripts/sd_verified_transfer.py` module uses the pinned frozen file list, a fresh UUID-bound source census, the exact source block-device mount identity, and the NAS CIFS share identity before copy. It selects a bounded set of files by bytes and file count, optionally prioritizing MP4s. It calls `rsync` with `--ignore-existing --fsync --partial-dir=.custody-partial`, then **independently** SHA-256 hashes both source and NAS destination, requiring equal digests and stable file metadata. Any mismatch stops the pass, preserving both files for investigation.

Run under `python3 -m scripts.sd_verified_transfer` from the repository root. Example read-only plan, with no files transferred until `--apply` is supplied:

    python3 -m scripts.sd_verified_transfer \
      --manifest /media/scott/SSD_4TB/hermes-home/FLEET-STATE/sdcard-ingest/20261012/20261013-tranche17-independent-2573/complete-4268-source-2573-verified-custody.jsonl \
      --expected-sha256 0a2213f72c921e6443c6753660109b5d681e3a50a1751d7cf2638f0f80e8afa5 \
      --uuid 68FC-E499 --source /media/scott/68FC-E499 \
      --nas /nas --nas-share //192.168.1.202/fileserver \
      --max-files 20 --max-bytes 1600000000 --min-videos 10 \
      --audit-out /home/scott/nas-reconciliation-audit-20261016/sdcopy-new-unique-name.jsonl

To copy only this bounded set, add `--apply`. To continue without re-reading already verified copies, repeat `--prior-audit /absolute/path/to/previous-audit.jsonl` for each earlier completed pass from the same card. The prior receipt is accepted only if UUID, source relative key, byte count, intended NAS destination, SHA equality, stable metadata, no-release status and destination path/size remain consistent.

The script creates **new** OS-local audit JSONL files and fsyncs each row. A completed row is a *point-in-time verified copy*, **not** an authoritative deletion receipt. It does not overwrite an existing NAS file: an existing destination is independently hashed and compared rather than replaced. It never removes source files.
## Source release is a separate final gate

The 532,379,214,870-byte frozen corpus comprises 4,268 originals. A few hundred pilot and bounded-batch copies are not complete archive custody. Never use a partial audit as authority to erase an SD card. A successful transfer must eventually reconcile every file (including sidecars and clock-reset anomalies) against verified destination content, with source mounted/quiescent, a full copy ledger, complete destination verification and storage identity proof.

The existing `auto_ingest.custody.release_source` command is the only approved source-removal path. It currently refuses all three campaigns because: source mount declarations are rw, destination is not declared in the campaign, and complete hash/copy/destination release ledgers do not exist. The Beelink Btrfs backing filesystem is confirmed but snapshot/retention policy is not yet independently attested. Do not bypass or weaken these controls just to make deletion possible.

Once complete, separately validate the proper read-only producer-quiescence phase and the audited transition needed for MP4-only source cleanup while keeping `DCIM` directories/sidecars intact until everything else has a suitable retention plan. **Do not format cards, run recursive delete, or delete files based only on filenames/sizes.**

## Supervised full-corpus transfer

The companion script `python3 -m scripts.sd_transfer_campaign` runs the same SHA-256-verified copier in sequential batches. It uses a **single-writer OS-local flock**, stops on source UUID drift, source metadata census changes, NAS identity drift, copy/hash mismatch, low free space (20 GB extra reserve), or audit collision. It creates a distinct, fsynced JSONL audit file per card/batch and reuses strictly matching older receipts on restart. It never deletes source content and never claims that the full campaign has finished unless the frozen manifest has zero unaccounted items for all three cards. Network copy writes are rate-capped with `rsync --bwlimit=50000` (~50 MB/s maximum each sequential transfer) plus low-priority scheduling.

The intended command, from a pinned Git checkout on x1:

    python3 -m scripts.sd_transfer_campaign \
      --manifest /media/scott/SSD_4TB/hermes-home/FLEET-STATE/sdcard-ingest/20261012/20261013-tranche17-independent-2573/complete-4268-source-2573-verified-custody.jsonl \
      --expected-sha256 0a2213f72c921e6443c6753660109b5d681e3a50a1751d7cf2638f0f80e8afa5 \
      --audit-root /home/scott/nas-reconciliation-audit-20261016 \
      --nas /nas --nas-share //192.168.1.202/fileserver \
      --max-files 100 --max-bytes 8000000000 --max-batches-per-card 100 \
      --report-out /home/scott/nas-reconciliation-audit-20261016/sd-copy-full-corpus-final-report-20261016.json

A completed per-file receipt has equal SHA-256 digests, stable source/destination metadata and `source_release_authorized=false`. The supervisor will not silently skip a file lacking such a receipt, and it refuses to overwrite existing destination copies. If a batch is interrupted, earlier fsynced rows are preserved; the next run uses a new batch ID rather than replacing any audit.

**Operator controls:** inspect `journalctl --user -u auto-ingest-sd-nas-copy-20261016`, `systemctl --user status auto-ingest-sd-nas-copy-20261016`, or stop the NAS transfer with `systemctl --user stop auto-ingest-sd-nas-copy-20261016`. Stop does not remove any source data, completed destination files or audit records. If the job halts, investigate the first reported blocker and run fresh preflight before resuming.

**Important:** finishing the copy is NOT completion of source release. Existing `release_source` gates remain closed until whole-inventory custody, destination identity, retention/snapshot witness, and permitted source mount transition are audited. No recursive deletion or formatting is authorized by this supervisor.
