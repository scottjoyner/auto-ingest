# SD-to-NAS bounded full-SHA media probe (October 16, 2026)

**Research only; source release HOLD.** This document adds a controlled current source-versus-NAS hash comparison to the manifest-only candidate index in PR #31. The existing SD ingestion implementation, not a stand-alone rsync, owns future copy/verify/release state.

## Invariants

- Inputs are the existing 4,268-row, SHA-256 pinned candidate index, the three UUID-bound auto-ingest campaign records and an explicitly observed CIFS destination mount. x1 has **both** systemd autofs and CIFS at `/nas`; this tool requires the actual CIFS backing mount **and the explicitly pinned share source** and does not mistake autofs for durable storage.
- Select only **MP4 files with exactly one size-matching candidate**, valid staging identity and zero current destination verification claims. Selection is sorted by byte count for a bounded first proof. File access uses read-only file descriptors, `openat`, `O_NOFOLLOW`, checks on storage device/regular-file status, and stable inode/size/timestamps before versus after full SHA-256 reads.
- **Explicit** `--allow-writable-source-observation` is needed while the physical cards remain mounted rw. Its result is marked **research only**, not a quiescent/read-only source or safe-to-release assertion.
- Hard limits: at most 8 files and 512 MiB of combined source+destination reads per invocation. Default caps are smaller. A file's total read cost is 2x its size. No batch copy, mount changes, corpus walk, source unlink or custody-ledger promotion.
- Exclusive JSON receipt is written to x1 OS-disk only; no overwrite of existing proof. The result never sets `source_release_allowed=true`, `custody_evidence_imported=true`, or `source_quiescence_attested=true`.

## Reproduction (read-only, bounded)

Run from the `auto-ingest` checkout:

```bash
python3 -m auto_ingest.custody.bounded_media_probe \
  --index /home/scott/nas-reconciliation-audit-20261016/auto-ingest-nas-candidate-index-v1-20261016.jsonl \
  --expected-index-sha256 736a784b398ff4e5a62355d9b043276c30ad8719b5c80ecb3e47588deef40f92 \
  --campaign-root /home/scott/nas-reconciliation-audit-20261016/auto-ingest-campaigns-v1 \
  --destination-root /nas --destination-filesystem cifs \
  --expected-destination-device //192.168.1.202/fileserver \
  --max-files 4 --max-read-bytes 128000000 --max-file-bytes 64000000 \
  --allow-writable-source-observation \
  --out /home/scott/nas-reconciliation-audit-20261016/auto-ingest-live-source-nas-4-mp4-research-20261016.json
```

The example output path has **already been used**; for future unique runs choose a new receipt path. Never overwrite an existing evidence JSON. To continue without duplicate reads, repeat `--exclude-report PATH` and `--expected-exclude-sha256 DIGEST` in the same order for every successful prior batch. Exclusions are strictly SHA-pinned and apply only to stable, matching research receipts for the same index. No exclusion promotes source release.

The explicit x1 backing device pin is `//192.168.1.202/fileserver`; `/nas` also shows systemd autofs in the mount table, and autofs by itself is **not** valid destination proof.

## Physical x1 result

Four small June 27, 2026 front/rear MP4s on `5140-6932` were read from their source card and current NAS recovery paths: **4/4 complete SHA-256 matches**, **117,964,800 total bytes read**, zero media writes or deletions. The full immutable local research receipt SHA256 is:

`29827c13742010ede625ea3dd770ea3f52eaaa60058bde0ae05f5877987177a4`

Two additional non-overlapping four-object research runs have now passed:

- Batch 2: 4/4 full source/NAS SHA matches, 180,748,288 bytes read, excludes batch 1, receipt `auto-ingest-live-source-nas-next-4-mp4-research-20261016.json`, pinned SHA `5c5d429576905ca2538d59e93ddad05f8b3fc75ef669a2981135af503fcc6abb`.
- Batch 3: 4/4 full source/NAS SHA matches, 356,515,840 bytes read, excludes both earlier receipts, includes one recording on `69BE-B963`, receipt `auto-ingest-live-source-nas-third-4-mp4-research-20261016.json`, pinned SHA `7dee3d5cb1382ac252be3fa5c6ff46c2a402350f7d0600f54f98025eb7220fb8`.

**Cumulative:** 12 distinct MP4 files with matching current source+NAS digests, 655,228,928 combined bytes read. From the original 74 size-only MP4 candidates, 62 remain without this live comparison. These observations are never automatically imported into the source-release ledger.

This is stronger than a filename+size match. It is **weaker** than release authority because the SD source was mounted rw, no separate producer quiescence was attested, independent long-term NAS authority/snapshot policy was not established, and the actual custody source/destination verification ledgers have not been admitted.

## Remaining gates

A second, independently governed verifier must establish source RO/quiescence, a stable source UUID, live destination device/pool identity and permanence, matching durable full SHA-256 entries for *every* original to be removed, and a fully reviewed exact-file MP4-only release plan. Do not remove the card MP4s or reset/format DCIM while those gates remain blocked. Next batch should avoid rereading these four, rather than treating a duplicate read as new coverage.
