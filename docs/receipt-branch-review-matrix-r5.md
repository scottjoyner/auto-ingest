# Receipt-branch review matrix — Round 5, Session 3 (2026-11-26)

**Scope:** read-only inventory of the eight Neo4j receipt / signed-publication
safety branches carried forward from
`/home/scott/opencode-reviews/session4-autoingest-assistx-20261126.md` §1.3 C4.
**No merge, push, PR, fetch-that-updates-working-files, checkout switch, rebase,
reset, or commit on any of these branches was performed by this session.** All
tips remain unmerged into `origin/main` (`git merge-base --is-ancestor` = false
for all eight) and all listed worktrees were clean at inspection time
(`git status --porcelain` empty).

## Method (read-only commands only)

- `git ls-remote --heads origin` — remote tip hashes.
- `git branch -vv --list …`, `git worktree list` — local refs, upstream state,
  worktree paths.
- `git log -1 --format=…`, `git log --grep='capture uncommitted work'`,
  `git merge-base`, `git rev-list --count`, `git status --porcelain` — dates,
  sweep-commit presence, divergence, cleanliness.

No `git fetch` was run; no index, worktree or ref was modified.

## Two corrections to the Session 4 framing

1. `git ls-remote --heads origin` exposes **six** remote heads in this family,
   not eight. The other two branches in the Session 4 list
   (`neo4j-signing-no-ancestor-symlink`, `neo4j-signed-publication-durability`)
   are **local-only** — they have no remote head. They are included below for
   completeness and marked accordingly.
2. Session 4 recorded two remote branches touched by the 2026-11-25 "capture
   uncommitted work (2026-11-25 sweep)" commit (`0cc721a3`, `7f37aae3`).
   There are actually **three**: the `wip/…public-receipt-verify…` tip
   `44fac7f1` carries the same sweep commit as its tip. All three sweep commits
   were authored within the same ~3-second window on 2026-11-25 08:43 ET.

## Matrix

| # | Branch | Remote / local | Tip hash | Last date (tip) | Worktree path | 2026-11-25 sweep commit? | Recommended disposition (review only) |
|---|---|---|---|---|---|---|---|
| 1 | `agent/neo4j-receipt-identity-hardening-20261027` | **remote** + local tracking (0/0) | `0cc721a3` | 2026-11-25 | `/home/scott/git/auto-ingest-receipt-identity-hardening-20261027` | **Yes** — tip itself | **Review, do not merge.** Safety prototype; the tip only captured uncommitted WIP, so re-read the full diff and verify the hardening claim before any PR. |
| 2 | `agent/neo4j-receipt-stage-order-20261027` | **remote** + local tracking (0/0) | `7f37aae3` | 2026-11-25 | `/home/scott/git/auto-ingest-receipt-stage-order-20261027` | **Yes** — tip itself | **Review, do not merge.** Same caveat as #1: sweep commit is a WIP capture, not a reviewed change. |
| 3 | `agent/neo4j-signed-durability-review-20261027` | **remote** only (no local branch of this name) | `bf649ed1` | 2026-10-27 | detached HEAD at `bf649ed1`: `/home/scott/git/auto-ingest-signed-ancestor-published-verify-20261027` | No | **Keep as review baseline.** This is the common ancestor of both owner-adapter lines (#4); do not close until those are reconciled. |
| 4 | `agent/neo4j-owner-producer-adapter-review-20261027` | **remote** only; related local branch `agent/neo4j-owner-producer-adapter-20261027` @ `7069b51c` (no upstream) | `fee8eeff` | 2026-10-27 | detached HEAD at `fee8eeff`: `/home/scott/git/auto-ingest-owner-published-verify-20261027`; local sibling at `/home/scott/git/auto-ingest-owner-adapter-20261027` | No | **Review / reconcile.** Divergent siblings off `bf649ed1`: remote has 4 commits not in the local branch, local has 1 not in remote. Owner must pick the canonical line before a PR. |
| 5 | `agent/neo4j-backup-receipt-verifier-review-20261027` | **remote** only; related local branches `agent/incident227-backup-receipt-verifier-20261027` @ `02066993` and `agent/incident227-backup-receipt-review-clean-20261027` @ `0690ed4a` (latter tracks `origin/main`, 1/1) | `424f25f6` | 2026-10-27 | detached HEAD at `424f25f6`: `/home/scott/git/auto-ingest-public-stage-order-verify-20261027`; locals at `/home/scott/git/auto-ingest-backup-receipt-verifier-20261027` and `/home/scott/git/auto-ingest-backup-receipt-clean-20261027` | No | **Review, then close candidate.** Furthest divergence of the set: from base `924b22c6`, remote is 13 commits and `02066993` is 29. Needs an explicit owner decision on the canonical verifier line. |
| 6 | `wip/20261125-auto-ingest-public-receipt-verify-20261027-` | **remote** + local tracking (0/0) | `44fac7f1` | 2026-11-25 | `/home/scott/git/auto-ingest-public-receipt-verify-20261027` | **Yes** — tip itself | **Review then likely close.** Explicitly `wip/` and dated sweep; the captured WIP is not a reviewed change. Do not merge without owner sign-off. |
| 7 | `agent/neo4j-signing-no-ancestor-symlink-20261027` | **local only** — no remote head | `7d6a3995` | 2026-10-27 | `/home/scott/git/auto-ingest-signed-noancestor-20261027` | No | **Review / keep.** Safety prototype (reject ancestor symlink redirects in signed receipt publication). Consider publishing for review, but no push was made by this session. |
| 8 | `agent/neo4j-signed-publication-durability-20261027` | **local only** — no remote head | `3272e791` | 2026-10-27 | `/home/scott/git/auto-ingest-signed-publication-durability-20261027` | No | **Review / keep.** Rollback-on-publication-sync-failure prototype. Keep until the durability question is settled. |

## Additional related refs (context, not part of the eight)

- `agent/neo4j-signed-receipt-review-20261027` @ `807c728f` (local worktree
  `/home/scott/git/auto-ingest-signed-receipt-20261027`) — independently scoped
  signed archive byte readback; no remote head of that exact name.
- Detached verification worktrees `031c1be7`
  (`/home/scott/git/auto-ingest-public-receipt-race-verified-20261027`) and
  `63262e45` (`/home/scott/git/auto-ingest-signed-published-verify-20261027`).
- `agent/incident227-*` backup-receipt lines are listed under #5 above.

## Bottom line

Six remote heads, two local-only branches, eight tips, **zero merged** into
`origin/main`, **zero swept by this session**. Three tips carry the 2026-11-25
WIP-capture commit. All dispositions above are recommendations only; no
merge/push/PR was performed, and ownership of `main` and of these receipt
branches remains an owner decision.
