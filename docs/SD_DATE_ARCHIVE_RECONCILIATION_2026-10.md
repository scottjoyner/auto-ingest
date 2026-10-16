# Date-folder NAS reconciliation for currently unindexed SD recordings

**2026-10-16 — READ ONLY, SOURCE RELEASE HOLD.** This extends the scoped SD-to-NAS anti-join in PR #31. It is not a source/destination digest ledger and does not authorize source retirement.

## Root discovery

The first candidate index found 74 path/size-only MP4 matches, three earlier NAS SHA matches, and 3,619 MP4-named source objects with no candidate **in those two limited indexes**. An additional NAS subtree under the x1 CIFS mount is:

`/nas/archive/dashcam/deathstar/8TB_2025/YYYY/MM/DD/RECORDING.MP4`

A 270-item stratified initial check found 99 metadata candidates, justifying one full **bounded exact-filename/date path** census. No recursion, broad NAS scan, source-card media read, hashing NAS video payloads, copying, unlinking or graph writes were used. The program checks the pinned CIFS share source separately from x1 systemd autofs at `/nas`.

## Reproducible operator command

From the `auto-ingest` checkout:

```bash
python3 -m auto_ingest.custody.date_archive_probe \
  --index /home/scott/nas-reconciliation-audit-20261016/auto-ingest-nas-candidate-index-v1-20261016.jsonl \
  --expected-index-sha256 736a784b398ff4e5a62355d9b043276c30ad8719b5c80ecb3e47588deef40f92 \
  --archive-root /nas/archive/dashcam/deathstar/8TB_2025 \
  --expected-nas-device //192.168.1.202/fileserver \
  --max-probes 5000 \
  --out /home/scott/nas-reconciliation-audit-20261016/auto-ingest-date-root-8TB_2025-candidates-v2-20261016.jsonl
```

The output path is already occupied by immutable evidence; replay with a *new* OS-local filename. CLI output is JSON statistics; `--out` creates exclusively and never overwrites an existing receipt. The complete recorded result has SHA-256:

`65863ed6f268b7ccbad27939f4f22910f256f2e954d13a8100566af567163b51`

## Physical findings

The exact-date match was attempted for all **3,619 previously unindexed MP4 names**. It found **1,136 additional filename-and-size matches** across 3,619 bounded NAS path checks, with zero duplicate NAS destination claims:

| FAT UUID | Additional metadata candidate MP4s | In the original 77-candidate group | Total MP4 names on card | MP4s with an indexed NAS path now |
| --- | ---: | ---: | ---: | ---: |
| 5140-6932 (`UNTITLED`) | 1,074 | 34 | 1,108 | **1,108/1,108** |
| 68FC-E499 | 62 | 13 | 609 | 75/609 |
| 69BE-B963 | 0 | 30 | 1,979 | 30/1,979 |

The original candidate group includes three historical SHA-matched MP4s; do not double count it against this newly discovered group. The `UNTITLED` match count **includes** timestamp anomalies and AppleDouble `._` filesystem metadata. Preserve those, as well as 424 CSV and 17 JPG sidecars and all directory structure, until explicitly reconciled. Historical source hashes and equal byte lengths are not sufficient proof of NAS data permanence or a current SHA match.

This discovery materially changes the transfer plan: **do not bulk copy the 161.1 GB of `UNTITLED` recordings until the already-existing alternative archive copies have been content-verified and their destination identities established**. The other cards still need further candidate discovery across nested archives and receipt indexes; `2,483` unindexed MP4s had no exact date-path match in this specific root.

## Required gates before any source-card reuse

1. Verify source file identity/UUID and a quiescent/read-only observation, and compute or independently attest authoritative SHA-256 for every original planned for removal.
2. Verify the destination file's live SHA-256 with a separately pinned, durable NAS storage identity and a stable readback. Existing date-index files are **only candidate locations**.
3. Verify all non-MP4 sidecars and clock resets as necessary, and reconcile duplicate variants. Preserve the authoritative operator audit trail and append-only custody ledgers.
4. Obtain the full exact-file `release-source --mp4-only` plan and independently accepted `SAFE_TO_RELEASE` authority; delete only fully verified MP4s with audit, preserving `DCIM/` and sidecars.
5. Stop readers, flush and cleanly unmount/eject the specific card before physical removal. Until then, leave physical cards in place or keep them intact as unreleased originals, not as reusable empty media.
