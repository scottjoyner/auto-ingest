# CARD-01 fixture

The latest observed CARD-01 situation, frozen as data:

| fact | value |
| --- | --- |
| source | `/media/scott/UNTITLED`, read-only |
| hashes verified | 67,644 |
| copy | incomplete (started, interrupted, ledger incomplete) |
| destination verification | 0 files / 0 bytes |
| source deletion | none |
| worker | stopped |
| replacement worker | not started |

Derived state (see `tests/test_custody_fixture_card01.py`): `RECONCILE_REQUIRED`,
`source_release_allowed = false`.

This fixture is **not** connected to the physical card. `campaign.json` and
`evidence.json` are static JSON; nothing in the custody package opens the card,
mounts it, or reads `/media/scott/UNTITLED`. `ledgers/destination.jsonl` is a
three-row *sample* of an incomplete destination ledger (all rows `pending`,
zero verified) — it exists so the ledger reader has something to aggregate, not
to represent 67,644 objects. Per-file detail for a real campaign lives in the
campaign's own `ledgers/` directory, not here.

## The byte counts are measured, not estimated

Measured 2026-10-05 by walking the card read-only:

```sh
find /media/scott/UNTITLED -type f -printf '%s\n' | awk '{n++; s+=$1} END {print n, s}'
#   67644 94278672670
```

The file count in `evidence.json` matches that walk exactly, which is how this
fixture came to represent this card. The byte count originally did not: it read
`412885402112` (412.9 GB), which is **larger than the entire volume** (255.8 GB)
and 4.38x the sum of the file sizes. Nothing on the card measures that way - not
apparent size (94.3 GB) and not allocated size (95.7 GB, i.e. `st_blocks * 512`)
- so it was an estimate frozen into the fixture, not an observation.

It never caused a wrong decision here, because the destination has ~60 TB free,
so both figures pass the capacity gate. But a fixture that overstates a card by
4.4x is a bad rehearsal: any future test or capacity estimate built on it starts
from a fiction. Corrected to the measured `94278672670`.

Derived state is unaffected: it is computed from counters and completeness flags,
never from the absolute byte value.
