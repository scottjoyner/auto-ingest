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