# Auto-ingest merge readiness — 2026-11-28 bounded review

Evidence-only note recorded under `FLEET_LAUNCH_HOLD` (SWAP_RESERVE_LOW /
IO_PRESSURE). No push, merge, gate change, heavy test, or bulk sweep was run.

## Exact heads

| Ref | Commit |
|---|---|
| Canonical checkout (`.../auto-ingest`, `integration/round7-auto-ingest-20261126`) | `f24b26465f56bab0eadce977ed67b91a410125b7` |
| `origin/integration/round7-auto-ingest-20261126` | same commit (local/remote 0/0) |
| `origin/main` | `177288cf0e19dcb30511c1d90a41d82caa483db1` |
| Relationship | integration is **9 ahead, 0 behind** `origin/main` |

Working tree was clean before and after this review.

## Unmerged series (`origin/main..HEAD`)

1. `2b778b63` custody: import pinned NAS metadata observations without release authority
2. `13383a63` custody: fail closed on casefold-colliding source or destination paths
3. `89d20773` docs(custody): record NAS casefold collision fail-closed rule
4. `9f74cf5b` docs(auto-ingest): record PR #20 supersession of pipeline-staging
5. `b81b9e11` fix(content_os): accept UTF-8 BOM in CsvAdapter.extract
6. `b237e9a9` custody: accept UTF-8 BOM on digest-pinned NAS inventory CSVs
7. `871e05fc` ingest: accept UTF-8 BOM on CSV intermediate readers
8. `7c244d86` merge of `agent/csv-adapter-bom-r35-20261126`
9. `f24b2646` merge of `agent/bom-forward-port-r36-20261126`

## Duplicate-work warning

The repo has **140 pre-existing registered worktrees** (141 with this review
worktree): `auto-ingest-r9..r60`, `round34..round60`, and `sweep*` families.
`sweep46-auto-ingest-20261126-a` already sits on the same `f24b2646` head.
Treat the r/round/sweep branches as unmerged candidates, not as extra work to
re-derive. In particular the three BOM fixes already ship regression tests in
`tests/test_adapters.py`, `tests/test_custody_external_inventory.py`, and
`tests/test_ingest_intermediate.py`; do not add duplicate BOM sweeps.

## Assessment and next actions

- The round7 integration branch is the only branch carrying this BOM/custody
  series, and it is not on `main`. It is intact at exact head `f24b2646`.
- Merge readiness: **operator review first; no automatic merge**. Conflict and
  CI status were **not** evaluated (guard HOLD, no heavy test or merge simulation).
- Next: review the nine commits above, use `integration/round7-auto-ingest-20261126`
  as the base for any further round7 work, and run canonical CI only after the
  launch guard clears. Do not push or merge from this review worktree.
