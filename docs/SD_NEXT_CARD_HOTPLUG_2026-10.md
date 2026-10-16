# UUID-bound next-card discovery for x1-370 (October 16, 2026)

**Non-mutating / pre-ingest only.** Preserve the three original SDs under their existing unmerged custody campaigns. Their source release state is `DISCOVERED` / blocked even after successful physical unmount. The cards **may be physically removed and stored intact**, but are not approved for recording, format or erasure.

At approximately 14:01 EDT on October 16, x1-370 still showed the original, cleanly unmounted UUIDs `69BE-B963`, `68FC-E499`, `5140-6932`, plus one empty removable USB reader. No new SD cards had been inserted. The original audited handoff receipt is on local x1 storage:

`/home/scott/nas-reconciliation-audit-20261016/sd-card-originals-clean-unmount-handoff-20261016.json`

Frozen receipt SHA-256: `f9c22200e1517e07a29ad2c076ea564f59d32d44a2931d0b00146c94bcb18d17`.

## After inserting new cards

From the checked out `auto-ingest` repo, use:

```bash
python3 -m auto_ingest.custody.next_card_discovery \
 --prior-handoff /home/scott/nas-reconciliation-audit-20261016/sd-card-originals-clean-unmount-handoff-20261016.json \
 --expected-handoff-sha256 f9c22200e1517e07a29ad2c076ea564f59d32d44a2931d0b00146c94bcb18d17
```

This reads **only lsblk device metadata**, not the card file contents, and matches old cards by **filesystem UUID**, never unstable `/dev/sdX` slots. It rejects cloned/repeated current UUIDs, malformed old evidence, unknown USB filesystem layouts and duplicate source claims. No automatic mounting, unmounting, copying, hashing, deleting, NAS mutation, or data graph mutation occurs. Optional `--out /home/scott/.../unique-next-card-report.json` writes an exclusive read-only discovery receipt on the OS volume.

An observed candidate is **not automatically ready for source-file enumeration**: for each newly inserted UUID, independently qualify the real kernel filesystem mount and USB device-to-UUID mapping and preferably mount read-only before using the existing `fleet_preflight`/`fleet_campaigns` inventory tools. Reject cards which match one of the old UUIDs unless specifically continuing its existing custody campaign. If the device is unmounted, it is only a candidate awaiting an operator-reviewed mount step.

Continue with source name/size census, capacity and clock-anomaly checks; then stage only what is proven missing and verify with the existing hash gates. Do not overwrite a canonical archive destination, format source media, delete MP4s, or erase original cards without independent complete SHA-256 readback plus separately authorized per-file `SAFE_TO_RELEASE` evidence.
