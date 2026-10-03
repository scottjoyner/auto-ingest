# SD-card campaign custody — deterministic state for card ingest

> **Status:** implemented. Read-only by default. Preservation-first.
> **Why:** Hermes should never have to reconstruct "what happened to this card?"
> by reading logs, `find` output and ad-hoc shell.

An SD card that is pulled from a dashcam has one job left: get every byte to the
canonical destination, prove it, and only then become releasable. Until now that
proof lived in log files and in an operator's head. This document and the
`auto_ingest.custody` package replace that with a derived state, an evidence
contract, a read-only planner and a fail-closed release gate.

---

## 1. The questions this answers

| question | answered by |
| --- | --- |
| What exactly has happened to this card? | derived `state` + `reasons` |
| What is verified? | `hash.verified`, `destination.verified_files/bytes` |
| What is missing? | `reconciliation.source_only`, `copy.in_flight_files` |
| Can this card safely resume? | `plan.safe_to_resume` + `plan.actions` |
| What would the resume plan do? | `plan.actions[]` (a description, never an execution) |
| Has destination custody been proven? | `state ∈ {VERIFIED, SAFE_TO_RELEASE}` |
| Can the source ever be released? | `source_release_allowed` / `release.allowed` |

---

## 2. The state model

States are **derived**, never declared. There is no setter; `derive_state()`
is a total pure function of `(campaign, evidence, policy)`. A `state` key in an
evidence file is ignored and reported in `ignored_declared_fields`.

```
DISCOVERED          card seen, nothing trustworthy observed yet
SOURCE_VERIFIED     source inventoried + inventory verified, hashing not begun
HASHING             hashing underway, coverage short of the inventory
HASH_COMPLETE       hashing complete, no copy plan recorded yet
COPY_PENDING        copy plan recorded, copy has not started
COPYING             copy started and a worker is still driving it
COPY_COMPLETE       copy result attested, destination verification not started
VERIFYING           destination verification underway, incomplete
VERIFIED            destination custody proven; release gate still closed
RECONCILE_REQUIRED  evidence is ambiguous or disagrees with the destination
BLOCKED             evidence contradicts itself; a human must intervene
SAFE_TO_RELEASE     custody proven AND every release condition satisfied
```

Precedence (first match wins) lives in `auto_ingest/custody/machine.py`:

| # | condition | state |
| --- | --- | --- |
| 0 | arithmetic contradiction, fatal error, destination identity conflict | `BLOCKED` |
| 1 | nothing observed at all | `DISCOVERED` |
| 2 | inventory incomplete, work underway | `HASHING` |
| 3 | inventory incomplete, nothing underway | `DISCOVERED` |
| 4 | inventory complete + verified, hashing not started | `SOURCE_VERIFIED` |
| 5 | hash coverage short of the inventory | `HASHING` |
| 6 | hashing complete, no copy plan | `HASH_COMPLETE` |
| 7 | hashing complete, copy planned, not started | `COPY_PENDING` |
| 8 | copy started, result not attestable, worker idle | `RECONCILE_REQUIRED` |
| 9 | copy started, worker still alive | `COPYING` |
| 10 | copy attested, verification not started | `COPY_COMPLETE` |
| 11 | verification underway | `VERIFYING` |
| 12 | source-only / mismatched (destination-only only under strict scope) | `RECONCILE_REQUIRED` |
| 13 | custody proven, gate closed | `VERIFIED` |
| 14 | custody proven, gate open | `SAFE_TO_RELEASE` |

Two distinctions worth stating explicitly, because they are where hand-rolled
scripts usually go wrong:

* **`HASH_COMPLETE` vs `COPY_PENDING`** — `HASH_COMPLETE` means hashing is done
  and *no copy plan has been recorded*; `COPY_PENDING` means a plan exists and
  the copy has not started. Both are pre-copy states.
* **A stopped worker is a fact about a process, not a verdict about the data.**
  It only distinguishes "copy still in flight" (`COPYING`) from "copy result
  cannot be attested" (`RECONCILE_REQUIRED`). On its own it never blocks and
  never releases. `tests/test_custody_state_machine.py` pins all three cases.

---

## 3. Campaign identity

Identity is anchored on the **card**, not on where it was mounted:

```
card_key     = sha256(device | filesystem_uuid | serial)[:16]
campaign_id  = "sdcard-<card_key>-<sha256(created_at)[:8]>"
```

Comparison order is most-authoritative-first: `filesystem_uuid` → `serial` →
`device`. A matching UUID is enough even when the card moved to another USB slot
(the device node changes); a *different* UUID never matches; and when a stored
record carries a UUID the current observation does not, reuse is refused because
sameness is unprovable.

This is what stops a second unformatted card at `/media/scott/UNTITLED` from
inheriting CARD-01's evidence — every such card shares the label, and the label
is never treated as identity.

---

## 4. Evidence contract

`CampaignEvidence` is the bounded summary. Per-file detail lives in
`ledgers/*.jsonl` next to the bundle, read by `auto_ingest.custody.ledger`.

| block | fields |
| --- | --- |
| `inventory` | `discovered_files/bytes`, `complete`, `verified`, `roots` |
| `hash` | `verified_files/bytes`, `errors`, `complete`, `algorithm`, `exemptions`, `last_checkpoint` |
| `copy` | `planned{files,bytes}`, `completed{files,bytes}`, `started`, `result_complete`, `interrupted`, `ledger_complete` |
| `destination` | `verified_files/bytes`, `failures`, `verification_started/complete`, `unverified_present_files`, `observed_identity` |
| `reconciliation` | `source_only`, `destination_only`, `mismatched` |
| `worker` | `identity`, `status`, `started_at/stopped_at`, `last_checkpoint` |
| `errors` | `unresolved`, `fatal`, `summaries`, `witness` |

Boundedness: every free-form list (`roots`, `exemptions`, `error_summary`,
`summaries`) is capped at `policy.max_summary_entries` (default 20). The ledger
reader returns counts plus a capped error sample — never the record list.

---

## 5. Canonical destination abstraction

Three separate concepts, never collapsed:

| concept | field | meaning |
| --- | --- | --- |
| logical custody destination | `destination.logical` | host-independent name + relative path |
| resolved host path | `destination.host_path` | where it lives on *this* host right now |
| storage identity evidence | `destination.identity` | fs UUID / device / label / type |

Resolution order: `CUSTODY_DESTINATION_ROOT` → `custody.destination_root` in
`config.yaml`. **No historical destination is embedded in core logic** — with
neither configured, `host_path` is `None`, which blocks release instead of
silently picking `NAS3`, `NAS5`, `SSD_4TB` or any other mount. The current
runtime architecture is free to export `/nas`; the repository contract simply
does not assume it.

Tests assert the absence of those literals from every module in the package.

---

## 6. Resume planner (read-only)

`plan_resume()` returns a description. It has no filesystem, network, database
or subprocess access (asserted by an AST test), never copies, never deletes, and
always reports:

```json
{
  "safe_to_resume": true,
  "current_state": "RECONCILE_REQUIRED",
  "next_phase": "destination_reconciliation",
  "next_safe_action": "reconcile_destination",
  "source_mutation_allowed": false,
  "source_deletion_allowed": false,
  "requires_operator_authorization": true,
  "actions": [
    {"operation": "verify_existing_destination", "mutates_destination": false, "...": "..."},
    {"operation": "copy_objects", "excludes_verified": true, "gated_by": ["act-..."]},
    {"operation": "verify_destination", "excludes_verified": true}
  ]
}
```

**Verification before re-copy.** After an interrupted copy the destination may
already hold most of the payload. The plan therefore always emits
`verify_existing_destination` first, and every copy action carries
`excludes_verified=true` with an estimate that already subtracts the verified
objects. Copying a byte-identical file again is wasted I/O and pointless churn.

**Idempotent task identity.** `action_id` is a digest of the action's own
content and `plan_fingerprint` digests the whole action set, so `inspect → plan
→ inspect → plan` against unchanged evidence yields byte-identical output and no
duplicate task ids.

---

## 7. Fail-closed release gate

`SAFE_TO_RELEASE` requires *every* condition below. The gate never
short-circuits, so the blocker list is a complete explanation.

```
source read-only                     destination resolved
destination mounted                  destination identity proven
source inventory complete            hash evidence complete
hash gaps only policy-declared       copy result attested complete
copy ledger complete                 destination verification complete
mismatches = 0                       missing destination = 0
unresolved errors = 0                copy plan covers the required scope
operator witness (if policy requires)
```

`required_scope` defaults to `all_inventory`: every inventoried object must be
accounted for, so a copy plan that quietly dropped objects is a blocker, not a
pass. `hashed_set` is available as a weaker, explicit opt-in.

Hash exemptions only count when `custody.policy.declared_hash_exemptions`
declares the pattern; an exemption appearing only in evidence is a blocker
(`hash_exemptions_not_declared`).

---

## 8. The interrupted-copy case

```
hashing completed
copy may have partially occurred
worker stopped
destination ledger incomplete
        │
        ▼
RECONCILE_REQUIRED      (not COPY_PENDING, never SAFE_TO_RELEASE)
```

Reasons reported: `copy_started`, `copy_result_not_attestable`,
`copy_interrupted`, `worker_stopped`, `destination_ledger_incomplete`.

---

## 9. Commands

All read-only by default. `--require-release` exits 3 when the gate is closed,
which makes the commands usable as a CI/Hermes gate.

```bash
auto-ingest custody status  --bundle /path/to/campaign          # human
auto-ingest custody status  --bundle /path/to/campaign --json   # machine
auto-ingest custody plan    --bundle /path/to/campaign --json
auto-ingest custody verify  --bundle /path/to/campaign          # describes; --execute is refused
auto-ingest custody import  --bundle /path/to/campaign --evidence ev.json   # validate only
auto-ingest custody import  --bundle /path/to/campaign --evidence ev.json --apply
```

Observed hardware can be cross-checked inline:

```bash
auto-ingest custody status --bundle ... --json \
  --observed-uuid "$(...)" --observed-device /dev/sdb1 --observed-label UNTITLED
```

`observed_card_matches_campaign: false` means a different card is sitting where
this campaign's card used to be.

### plan vs execute

`status`, `plan`, `verify` and `import` (without `--apply`) never execute
anything. `verify` deliberately has no execution path: `custody verify --execute`
returns exit 3 and a refusal message, because destination verification is an
operator-authorized step performed by an executor outside this package. There is
no implicit execution as a side effect of reading state.

---

## 10. Operator loop

```
observe physical source        # lsblk / blkid, read-only; mount read-only
  → update/import evidence     # auto-ingest custody import ... --apply
  → custody status             # what state is this card in?
  → custody plan               # what would the next safe operation be?
  → operator-authorized execution
  → custody verify             # describe/confirm the verification set
  → custody status             # has custody been proven?
```

Hermes can then answer every question in the table in §1 without reconstructing
anything from logs.

---

## 11. CARD-01 (latest observed)

Fixture: `tests/fixtures/custody/card-01/` (static JSON, not wired to the card).

| fact | value |
| --- | --- |
| source | `/media/scott/UNTITLED`, read-only |
| hashes verified | 67,644 |
| copy | incomplete |
| destination verification | 0 files / 0 bytes |
| source deletion | none |
| worker | stopped |

Derived: **`RECONCILE_REQUIRED`**, `source_release_allowed: false`,
`next_safe_action: reconcile_destination`. Not `COPY_PENDING` (a copy was
attempted), certainly not `SAFE_TO_RELEASE`.

---

## 12. Legacy watcher disposition

Audit of repository paths referring to the old autonomous watchers:

| path / literal | class | disposition |
| --- | --- | --- |
| `sdcard-nas-ingest-watcher` | **DEAD / UNTRACKED** | zero occurrences in the repo or in any of its 20 refs; it lives only at host/system level |
| `untitled-sd-ingest` | **DEAD / UNTRACKED** | same — no code, no unit, no cron, no udev rule in version control |
| `/media/scott/UNTITLED` | HISTORICAL PATH | appears in the fixture as recorded evidence only; custody never opens it |
| `dashcam_copy.sh`, `bodycam_copy.sh`, `audio_copy.sh` | **MANUAL_ONLY** | human-run copy scripts; nothing schedules them. **Not reactivated, not referenced by custody** |
| `runall.sh` copy chain | MANUAL_ONLY | chains the three copy scripts above; unchanged |
| `scripts/ingest_supervisor.py` | MANUAL_ONLY | no `.timer`/`.service` ships for it; unshipped scheduler |
| `scripts/sync_cache_to_nas5.sh`, `sync_knowledge_to_nas.sh` | MANUAL_ONLY | rsync mirrors; not invoked by custody |
| `NAS3` / `NAS5` / `SSD_4TB` literals | HISTORICAL PATH | config-only; **not embedded in custody core**, asserted by tests |
| `deploy/watchdog.py`, `KnowledgeVaultWatcher` (`HLD.md`, `LLD.md`) | DEAD_REFERENCE | design-doc references to a file that does not exist |
| `systemd/*.service` | ACTIVE (unrelated) | Neo4j health / re-embed / dashcam-vision over NAS archives; never removable media |

**The custody state machine does not depend on any of them.** It has no udev
rule, no `.path`/`.timer` unit, no cron entry, no inotify/watchdog import and no
autonomous trigger: it cannot start itself, and it derives its state purely from
a campaign bundle on disk. `tests/test_custody_legacy_watchers.py` asserts this
in code — including that this slice adds no `.rules`, `.path`, `.timer` or
`.mount` file.

Disposition: **do not reactivate.** If a card needs ingesting, an operator
observes it, imports evidence, reads `custody status` / `custody plan`, and
authorizes execution explicitly.

---

## 13. Safety properties (and where they are proven)

| property | where proven |
| --- | --- |
| state is derived, not declared | `test_custody_state_machine.py::test_declared_state_is_ignored`, `test_derive_state_signature_rejects_a_state_argument` |
| release is fail-closed | `test_custody_release_gate.py` (18 cases) |
| plan never mutates the source | `test_custody_planner.py::test_no_action_ever_mutates_the_source` |
| status/plan cause zero filesystem mutation | `test_custody_readonly.py` (CPython audit hook in a subprocess) |
| no duplicate task generation | `test_custody_idempotency.py` |
| a different card cannot inherit state | `test_custody_identity.py` |
| JSON serialisation is deterministic | `test_custody_idempotency.py::test_status_json_keys_are_sorted` |
| custody does not depend on legacy watchers | `test_custody_legacy_watchers.py` |
| the fixture is not wired to the card | `test_custody_fixture_card01.py` |