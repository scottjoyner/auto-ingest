# External inventory custody runbook (R9)

**Status:** documentation only. This runbook describes a read-only observation
importer. It grants no release authority, reads no media payloads, writes no
custody state, and authorizes no `/nas` traversal. It is derived solely from
`docs/nas-external-inventory-observations.md`,
`auto_ingest/custody/external_inventory.py`, and
`tests/test_custody_external_inventory.py` in this repository; no external paths
or credentials are introduced here.

## 1. Scope and rule

Large NAS inventories are **not custody receipts**. A filename-and-size match is
not a cryptographic identity, and a missing canonical path does not prove
absence from other stores or an archive. Operator-safe observations belong
beside the existing `auto_ingest.custody` state machine, **not inside the
campaign's proof-bearing destination ledgers**.

The importer is a **historical observation snapshot**, not continuous
monitoring. Before planning any write, independently check the current source
identity, physical NAS identity, hash receipts, and conflicting active writers.

## 2. Running the read-only inventory

Entry point: `auto_ingest.custody.external_inventory.observe_metadata_csv`.

The caller supplies an immutable, frozen CSV snapshot plus an independently
pinned SHA-256 and exact row count. The importer validates the pin *before*
parsing, reads the file bytes once, digests those same bytes, and parses from
memory (so there is no read-digest/reopen-parse race):

```python
from auto_ingest.custody.external_inventory import observe_metadata_csv

result = observe_metadata_csv(
    "/home/scott/nas-reconciliation-audit-20261016/sd-exact-path-fd-safe-repro-20261016.csv",
    expected_sha256="e0026bdc3975f1f0f7fd5db529edb65105918c900dc89655e9c2a1ffc822d484",
    expected_rows=4268,
)
assert result.destination_verified_files == 0
assert not result.source_release_allowed
```

The absolute CSV path above is the **historical example already recorded** in
`docs/nas-external-inventory-observations.md`. It is an illustration of a pinned
input, not an instruction to traverse any mount and not a claim that the file is
currently present.

Admission checks (all fail closed with `ExternalInventoryError`):

- `expected_rows` must satisfy `1 <= expected_rows <= max_rows <= 100_000`;
  `max_rows` defaults to `6000`. The SHA-256 pin must be exactly 64 lowercase
  hex characters.
- The source must be an existing **regular file and not a symlink**, at most
  `16 * 1024 * 1024` bytes.
- The file must be UTF-8 and its digest must equal the pinned SHA-256.
- Required CSV columns: `card_uuid`, `source_relative_path`,
  `candidate_relative_path`, `expected_bytes`, `metadata_status`,
  `observed_bytes`, `nas_payload_sha256_verified_today`,
  `source_release_authorized`.
- Bounded `metadata_status` values only: `EXACT_NAME_ABSENT`,
  `NAME_AND_SIZE_ONLY`, `SIZE_MISMATCH`, `NOT_REGULAR_FILE`,
  `SYMLINK_NOT_ACCEPTED`, `SYMLINK_PARENT_NOT_ACCEPTED`, `PARENT_NOT_DIRECTORY`,
  `IO_UNKNOWN`, `PERMISSION_UNKNOWN`, `INVALID_RELATIVE_PATH`.
- Sizes must be non-negative; `NAME_AND_SIZE_ONLY` requires
  `observed_bytes == expected_bytes`, and `SIZE_MISMATCH` requires an observed
  value different from the expected size. Duplicate source identities,
  truncated rows, and a final row count different from `expected_rows` are
  refused.

The returned `Observation.to_dict()` counts rows by status and by
`card_uuid|status`. It reports only metadata: `meta_only=True`,
`destination_verified_files=0`, `source_release_allowed=False`, and
`import_as_custody_receipt_allowed=False`. The importer never reads media
payloads, never generates hash-equality proof, never writes campaign bundles or
ledgers, and never mutates a custody state.

## 3. What "no release authority" means

"No release authority" is a hard property of both the input contract and the
output object, not a policy note:

- The CSV columns `nas_payload_sha256_verified_today` and
  `source_release_authorized` must be the literal string `"False"` on **every**
  row. Any other value raises `ExternalInventoryError` ("this importer cannot
  accept release or hash proof assertions"). A `metadata_status` of `VERIFIED`
  is likewise unsupported.
- The result constants are fixed: `destination_verified_files=0`,
  `source_release_allowed=False`, `import_as_custody_receipt_allowed=False`.
  No observation can raise them.
- Therefore no source release, archival cleanup, or transfer is authorized by
  this importer. Hash equality and destination verification must come from the
  existing independent custody verification flow (`custody hash`, `custody
  verify`) and the fail-closed release gate in
  `auto_ingest/custody/release.py` (`evaluate_release`); the canonical policy is
  `docs/sd-card-campaign-custody.md`, section 7 ("Fail-closed release gate").
- Research-vault repository directory and file queues are a **distinct**
  dataset. Do not import them as SD-card campaigns: they need their own typed
  contract without implicit source-release semantics.

## 4. NAS casefold fail-closed rule

A case-sensitive SSD can hold both `Fig1.png` and `fig1.png`, while a
case-insensitive (SMB/NAS) destination may collapse them into one name. The
importer rejects such portable path aliases rather than silently accepting an
ambiguous manifest.

For every row, paths are normalized with
`unicodedata.normalize("NFC", path.replace("\\", "/")).casefold()`:

- **Source paths:** keyed by `(card_uuid, folded_source_path)`. Two source
  paths that collapse to the same folded name **within the same card** raise
  `ExternalInventoryError("source path casefold collision cannot be safely
  imported")`.
- **Destination candidates:** keyed by `folded candidate_relative_path` across
  the whole import. Two candidate paths that collapse to the same folded name
  — even from different cards — raise `ExternalInventoryError("destination
  path casefold collision cannot be safely imported")`.
- Exact duplicate source identities (same card, byte-identical path string) are
  refused independently of the fold.
- Different card namespaces and different destination namespaces may share a
  basename; that is accepted, because neither side collapses to one portable
  name.

This guard comes from the observed 2026-10-16 NAS5 research-vault collision:
the SSD held `Fig1.png` and `fig1.png` with different SHA-256 payloads, but a
single NAS `Fig1.png` held only the lowercase variant's data. The importer
**fails closed** on that ambiguity; it does not pick a winner.

## 5. Exact test commands

Run from the repository root. The tests use synthetic temporary CSVs only — no
NAS mount, no network, no credentials, and no live state.

Isolated importer module (16 tests):

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q tests/test_custody_external_inventory.py
```

Focused custody pair (importer plus reconciliation):

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q tests/test_custody_external_inventory.py tests/test_custody_reconcile.py
```

Default lightweight suite (do **not** substitute a broad or ML-heavy run):

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q
```

`pyproject.toml` sets `testpaths = ["tests", "auto_ingest"]` and an `addopts`
filter that ignores `auto_ingest/ingest`, `auto_ingest/diarize`,
`auto_ingest/dashcam`, and `auto_ingest/content`, which import torch,
transformers, ultralytics, and similar ML dependencies. Do not run the heavy
ML/NAS-touching tests without an independently admitted staging destination.

## 6. Source references (repo-relative)

- `docs/nas-external-inventory-observations.md` — original observation record
  and the read-only example.
- `auto_ingest/custody/external_inventory.py` — importer, validation, and the
  casefold collision guards.
- `tests/test_custody_external_inventory.py` — 16 offline tests, including the
  two casefold fail-closed cases and the distinct-namespace acceptance case.
- `auto_ingest/custody/release.py` and `docs/sd-card-campaign-custody.md` — the
  independent verification and release authority this importer cannot replace.
