# APPLIED (slice 22B/5, 2026-11-28) — static verification only; NOT executed.
#
# Guard context: FLEET_LAUNCH_HOLD SWAP_RESERVE_LOW (instantaneous io avg10=0.31).
# No builds, test execution, servers, network, coord writes, resets, force, or
# gate changes; static verification only (py_compile, git diff --check,
# read-back). No production enablement. This module is inert until the guard is
# cleared and the repo owner signs off; the test suite was NOT run.
#
# Rule source (read-only `git show`): commit
#   13383a635b743498a0094874d467ee653e02ef57
#   "custody: fail closed on casefold-colliding source or destination paths"
#   - source: auto_ingest/custody/external_inventory.py
#     observe_metadata_csv(path, *, expected_sha256, expected_rows, max_rows=6000)
#     -> Observation(rows, source_sha256, by_status, by_card_and_status, ...)
#     raises ExternalInventoryError (a ValueError) on collisions
#   - commit's test file: tests/test_custody_external_inventory.py (pytest style)
# Confirmed at the applied base 069ad7fc720a2fe9fb2e646fe000778b8c024d1e that
# observe_metadata_csv still exists with the same signature, the same two
# casefold-collision error messages, and the same Observation fields. No import
# or path adaptation was required; both package __init__.py files are present.
#
# Applied as a NEW standalone unittest module at:
#   tests/test_custody_external_inventory_casefold.py
#   (kept separate from the commit's pytest-style tests/test_custody_external_inventory.py)
#
# Hermeticity: every manifest is an injected in-memory list of path rows. The
# CSV snapshot is built in io.StringIO; `external_inventory.Path` is replaced
# with an in-memory object; builtins.open and socket.socket are tripwired to
# fail. No real filesystem, no temp files, no NAS, no network, no CSV on disk.
# The REAL production function observe_metadata_csv is invoked, so this pins
# shipped behavior rather than a reimplementation of the rule.
#
# POST-GUARD VERIFICATION COMMAND (NOT run now; repo's own .venv env):
#   cd /home/scott/git/auto-ingest
#   .venv/bin/python3 -m unittest tests/test_custody_external_inventory_casefold.py -v
#   (equivalent dotted form: .venv/bin/python3 -m unittest tests.test_custody_external_inventory_casefold -v)
#
# Verification actually done on this applied file:
#   python3 -m py_compile .../tests/test_custody_external_inventory_casefold.py
#   git diff --check
#   read-back
#   -> PASS. unittest/pytest were NOT executed.
"""Hermetic unittest module for commit 13383a63's casefold-collision custody rule.

The recorded guard (in observe_metadata_csv) folds each source path with
NFC + backslash-to-slash + casefold and keys it by (card_uuid, folded_source);
it folds each candidate destination path the same way and keys it globally.
A repeat in either set fails closed. Source aliases are therefore card-scoped
and destination aliases are global; the two namespaces are independent.

Four scenarios are pinned below, using only injected in-memory path lists:
  1. source casefold collision  -> ExternalInventoryError (fail closed)
  2. destination casefold collision across cards -> fail closed
  3. cross source/destination casefold aliases -> fail closed
  4. non-colliding control (same basename, distinct card + destination
     namespace) -> succeeds, still meta-only and non-releasable
"""

from __future__ import annotations

import csv
import io
import socket
import unittest
from hashlib import sha256
from types import SimpleNamespace
from unittest import mock

from auto_ingest.custody import external_inventory
from auto_ingest.custody.external_inventory import ExternalInventoryError

# Exact column order used by commit 13383a63's test helper.
FIELDS = [
    "card_uuid",
    "source_relative_path",
    "candidate_relative_path",
    "expected_bytes",
    "metadata_status",
    "observed_bytes",
    "nas_payload_sha256_verified_today",
    "source_release_authorized",
]

# NAME_AND_SIZE_ONLY requires observed_bytes == expected_bytes.
DEFAULT_ROW = {
    "card_uuid": "CARD-1",
    "source_relative_path": "DCIM/a.MP4",
    "candidate_relative_path": "2026/10/16/a.MP4",
    "expected_bytes": "12",
    "metadata_status": "NAME_AND_SIZE_ONLY",
    "observed_bytes": "12",
    "nas_payload_sha256_verified_today": "False",
    "source_release_authorized": "False",
}


class _MemoryPath:
    """Minimal pathlib.Path stand-in backed purely by in-memory bytes."""

    def __init__(self, raw: bytes) -> None:
        self._raw = raw

    def is_file(self) -> bool:
        return True

    def is_symlink(self) -> bool:
        return False

    def stat(self) -> SimpleNamespace:
        return SimpleNamespace(st_size=len(self._raw))

    def read_bytes(self) -> bytes:
        return self._raw


def build_snapshot(rows: list[dict[str, str]]) -> tuple[bytes, str]:
    """Render injected path rows to an in-memory CSV byte snapshot + its digest."""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS)
    writer.writeheader()
    for row in rows:
        full = dict(DEFAULT_ROW)
        full.update(row)
        writer.writerow(full)
    raw = stream.getvalue().encode("utf-8")
    return raw, sha256(raw).hexdigest()


def observe_rows(rows: list[dict[str, str]]):
    """Call the real observe_metadata_csv with no filesystem/NAS/network access."""
    raw, digest = build_snapshot(rows)
    memory_path = _MemoryPath(raw)
    with (
        mock.patch.object(external_inventory, "Path", lambda _ignored: memory_path),
        mock.patch(
            "builtins.open",
            side_effect=AssertionError("hermetic test: builtins.open is forbidden"),
        ),
        mock.patch(
            "socket.socket",
            side_effect=AssertionError("hermetic test: network is forbidden"),
        ),
    ):
        return external_inventory.observe_metadata_csv(
            "in-memory://custody-casefold.csv",
            expected_sha256=digest,
            expected_rows=len(rows),
        )


class CasefoldCollisionCustodyTests(unittest.TestCase):
    """Pin the fail-closed behavior of commit 13383a63 without touching a disk."""

    def test_source_casefold_collision_fails_closed(self) -> None:
        # Same card, two source paths that are distinct on a case-sensitive SSD
        # but identical after NFC + casefold. Mirrors the commit's source test.
        rows = [
            {
                "card_uuid": "CARD-1",
                "source_relative_path": "DCIM/Fig1.png",
                "candidate_relative_path": "A/Fig1.png",
            },
            {
                "card_uuid": "CARD-1",
                "source_relative_path": "DCIM/fig1.png",
                "candidate_relative_path": "B/fig1.png",
            },
        ]
        with self.assertRaisesRegex(
            ExternalInventoryError, "source path casefold collision"
        ):
            observe_rows(rows)

    def test_destination_casefold_collision_fails_closed_across_cards(self) -> None:
        # Destination namespace is global: case-variant destinations on two
        # different cards must still fail closed. Distinct source paths here.
        rows = [
            {
                "card_uuid": "CARD-1",
                "source_relative_path": "A/one.mp4",
                "candidate_relative_path": "a/Fig1.png",
            },
            {
                "card_uuid": "CARD-2",
                "source_relative_path": "B/two.mp4",
                "candidate_relative_path": "a/fig1.png",
            },
        ]
        with self.assertRaisesRegex(
            ExternalInventoryError, "destination path casefold collision"
        ):
            observe_rows(rows)

    def test_cross_source_destination_casefold_collision_fails_closed(self) -> None:
        # Cross case: case-variant aliases appear in BOTH injected lists. Source
        # aliases are card-scoped (CARD-1 vs CARD-2, so that half is allowed),
        # while the destination aliases collide globally. The guard must fail
        # closed whichever namespace check fires first; assert the shared
        # "casefold collision" wording so the pin survives check reordering.
        rows = [
            {
                "card_uuid": "CARD-1",
                "source_relative_path": "DCIM/Fig1.png",
                "candidate_relative_path": "a/Fig1.png",
            },
            {
                "card_uuid": "CARD-2",
                "source_relative_path": "DCIM/fig1.png",
                "candidate_relative_path": "A/fig1.png",
            },
        ]
        with self.assertRaisesRegex(ExternalInventoryError, "casefold collision"):
            observe_rows(rows)

    def test_non_colliding_control_succeeds(self) -> None:
        # Same basename differs only by case, but card UUID and destination
        # namespace both differ, so neither folded key repeats. This is the
        # commit's accepted control and must still be metadata-only.
        rows = [
            {
                "card_uuid": "CARD-1",
                "source_relative_path": "DCIM/FILE.MP4",
                "candidate_relative_path": "card1/FILE.MP4",
            },
            {
                "card_uuid": "CARD-2",
                "source_relative_path": "DCIM/file.mp4",
                "candidate_relative_path": "card2/file.mp4",
            },
        ]
        result = observe_rows(rows)
        self.assertEqual(result.rows, 2)
        self.assertTrue(result.meta_only)
        self.assertFalse(result.source_release_allowed)
        self.assertFalse(result.import_as_custody_receipt_allowed)
        self.assertEqual(result.by_status, {"NAME_AND_SIZE_ONLY": 2})


if __name__ == "__main__":
    unittest.main()
