# Paste-ready PR descriptions — pushed auto-ingest branches (round 7, 2026-11-26)

Prepared by Session R7-4. Documentation only; no PR was opened, no push/merge was
performed, and **no push or merge to `main` is authorized** (INCIDENT-173 P0 hold;
owner gate). These are titles/bodies for an owner to paste later, if and when the
gate is released. All hashes and validation counts below were verified from local
`git log`/`git diff` and from `/home/scott/FLEET-ITERATION-STATUS-20261126.md`
(rounds 1, 2, 4, 6) plus `/home/scott/opencode-reviews/round5-session3-autoingest-20261126.md`;
nothing is invented.

Repo: auto-ingest (git dir `/media/scott/SSD_4TB/hermes-home/auto-ingest/.git`).
Both branches are stacked; open/merge in the order below, never into `main`
without explicit owner authorization.

---

## 1. Custody-landing branch

**Branch:** `agent/custody-external-inventory-forward-20261126`
**HEAD:** `9f74cf5b92854abb543b1ab3fe220e6a7d685f34`
**Base recorded at:** `origin/main` `177288cf0e19dcb30511c1d90a41d82caa483db1`
**Commits (oldest → newest, from `git log`):**
- `2b778b63db2b4c498c09d4a3549c5e81839fc541` — custody: import pinned NAS metadata observations without release authority
- `13383a635b743498a0094874d467ee653e02ef57` — custody: fail closed on casefold-colliding source or destination paths
- `89d20773713b3fbd324e388aa7d4d307cc833d28` — docs(custody): record NAS casefold collision fail-closed rule
- `9f74cf5b92854abb543b1ab3fe220e6a7d685f34` — docs(auto-ingest): record PR #20 supersession of pipeline-staging

**Diff vs base:** 4 files, +410/−0 — `auto_ingest/custody/external_inventory.py`
(+175), `tests/test_custody_external_inventory.py` (+168),
`docs/nas-external-inventory-observations.md` (+61), `README.md` (+6).

### Paste-ready title

```
custody: land external-inventory forward-port and casefold fail-closed guard
```

### Paste-ready body

```
## What changed

Brings forward the three residual custody commits from the parked
`feat/pipeline-staging` branch that were not part of PR #20's squash merge
(PR #20 `0d7f21d9`, 2026-11-25, merged 25 of the 28 branch commits):

- `auto_ingest/custody/external_inventory.py` — imports *pinned* NAS external
  inventory observations; no release authority is asserted and no live `/nas`
  access is performed at runtime.
- `reject_casefold_collision` — fails closed when a source or destination path
  collides under casefolding, so a case-variant path cannot alias another.
- `docs/nas-external-inventory-observations.md` — records the observations and
  the casefold fail-closed rule.
- `README.md` — records that PR #20 supersedes `feat/pipeline-staging` and that
  these residual commits are being brought forward on a new branch.

Verified base: `origin/main` @ `177288cf`. HEAD: `9f74cf5b`
(`2b778b63` → `13383a63` → `89d20773` → `9f74cf5b`). Four files, +410/−0.

## Why

The Oct-16 feature branch was never switched off and looked "stalled", but the
pipeline work itself landed via PR #20. Only the three NAS external-inventory /
casefold commits were genuinely unmerged. This branch closes that real gap
without re-landing any already-merged code and without touching live state.

## Tests / validation (recorded during the landing work)

- Focused custody tests: **16 passed / 16** (`tests/test_custody_external_inventory.py`
  and `tests/test_custody_reconcile.py`; command from the round-1 review plan:
  `pytest -q tests/test_custody_external_inventory.py tests/test_custody_reconcile.py`).
- Default lightweight suite: `python3 -m pytest -q tests/` →
  **1157 passed, 8 skipped**.
- `python3 -m pytest -q auto_ingest/tests/` → **51 passed**.
- No heavy ML or NAS-touching tests were run; no live staging destination was used.

## Risks / notes

- The imported inventory is *pinned observations*, not live truth; a future
  refresh needs its own owner-authorized provenance step.
- The casefold guard can reject paths that previously passed silently; that is
  the intended fail-closed behavior, but downstream callers must handle the new
  rejection path.
- This branch is based on the `origin/main` recorded in the review (`177288cf`);
  no rebase/merge was performed against any newer tip.
- Owner-gated cleanup (`scripts/create_hello.txt`) is deliberately NOT included.

## Authorization

**No push or merge to `main` is authorized.** This text is a paste-ready draft
only; no PR was opened and no branch was published by this session. Do not merge
until the INCIDENT-173 hold / owner gate is explicitly released.
```

---

## 2. Dependency-pinning branch (stacked on branch 1)

**Branch:** `agent/dependency-pinning-r5-20261126`
**HEAD:** `72e7e1a3606e5fd871f8ad861a77ea0eef159ce9`
**Base:** `9f74cf5b92854abb543b1ab3fe220e6a7d685f34` (the custody-landing branch above)
**Commit:** `72e7e1a3` — build(deps): align optional extras with requirements.txt pins; add receipt-branch matrix
**Diff vs base:** 4 files, +211/−14 — `pyproject.toml` (+33/−14),
`.github/workflows/ci.yml` (+6), `scripts/check_dependency_pins.py` (+121, new),
`docs/receipt-branch-review-matrix-r5.md` (+65, new).

### Paste-ready title

```
build(deps): align pyproject extras with requirements.txt pins; add CI drift guard
```

### Paste-ready body

```
## What changed

- `pyproject.toml`: optional extras now use the exact versions pinned in
  `requirements.txt` (transformers 4.49.0, speechbrain 1.0.2, faster-whisper
  1.2.0, pyannote.audio 3.1.1, faiss-cpu 1.12.0, librosa 0.11.0, soundfile
  0.13.1, ultralytics 8.3.94, opencv-python 4.10.0.84, pandas 2.2.2, moviepy
  1.0.3, Flask 3.0.3, PyYAML 6.0.1, requests 2.32.3, pytest 8.4.1,
  pytest-cov 6.2.1). `torch`/`torchaudio`/`torchvision` CPU pins were kept
  unchanged; `sentence-transformers`, `pydantic` and `ruff` were deliberately
  left lower-bounded because they have no authoritative pin in
  `requirements.txt`.
- `scripts/check_dependency_pins.py`: stdlib-only CI guard (manual TOML-extras
  parser, PEP 503 name normalization) that fails on a missing, non-`==`, or
  mismatched pin; reports unpinned extras as skipped.
- `.github/workflows/ci.yml`: one new step, "Dependency pin drift check
  (pyproject extras vs requirements.txt)", after the secret-scan step.
  Install strategy and all other steps are unchanged (workflow now has 9 steps).
- `docs/receipt-branch-review-matrix-r5.md`: read-only inventory of the eight
  receipt/identity branches named in the round-1 review (6 remote heads,
  2 local-only; three tips carry the 2026-11-25 sweep commit; none are ancestors
  of `origin/main`). Review-only recommendations; no merges.

## Why

Optional installs could silently resolve to versions different from the
manifest used by the pinned install path. Making the extras match the manifest
and adding a drift gate keeps CI and optional installs reproducible without
inventing versions that the manifest does not specify.

## Tests / validation (recorded)

- `python3 -m pytest -q tests/` → **1159 passed, 9 skipped** (no failures).
  The known-green baseline was 1157 passed / 8 skipped; identical tracked test
  counts and only non-test files changed, so the +2/+1 delta is environmental.
- `python3 -m pytest -q auto_ingest/tests/` → **51 passed**.
- `python3 scripts/check_dependency_pins.py` → **PASS**: 20 overlapping extras
  checked, 3 skipped (no `requirements.txt` pin).
- Negative check against the pre-change `pyproject.toml` → **16 drifts, exit 1**
  (guard demonstrably catches the original loose specs).
- `tomllib` parse of `pyproject.toml` OK (extras: api, content, dashcam, dev,
  ingest); PyYAML parse of `ci.yml` OK (9 steps).

## Risks / notes

- This branch is **stacked** on `agent/custody-external-inventory-forward-20261126`
  (`9f74cf5b`); a PR against `main` would also show those four custody commits.
  Set the PR base to the custody branch, or land that branch first.
- The checker parses TOML extras line-by-line to remain Python 3.10 compatible
  without `tomli`; unusual future formatting could need a parser update.
- Pins are not upgrades: a pinned version that breaks an optional feature needs
  an owner decision, not a silent bump.
- No runtime/production path was changed; no install/network step was run.

## Authorization

**No push or merge to `main` is authorized.** This text is a paste-ready draft
only; no PR was opened and no branch was published by this session. Do not merge
until the INCIDENT-173 hold / owner gate is explicitly released.
```

---

## Provenance of the verification claims

- `git log` on both branches confirmed every short/full hash and commit subject
  listed above; `git diff --stat` confirmed the file/count summaries.
- Status doc (round 1) records the custody branch at `9f74cf5b` with
  "16/16 focused custody tests; `tests/` 1157 passed / 8 skipped;
  `auto_ingest/tests/` 51 passed".
- Status doc (round 5) records the pinning branch at `72e7e1a3` with
  "1159/9 skipped + 51 passed" and the receipt-branch matrix.
- `round5-session3-autoingest-20261126.md` records the exact commands, the
  checker result (20 checked/3 skipped), and the 16-drift negative test.
- This session did not re-run the full suites (launch hold:
  `FLEET_LAUNCH_HOLD RAM_LOW,SWAP_RESERVE_LOW,IO_PRESSURE,MEMORY_PRESSURE`);
  results are quoted, not fabricated.
