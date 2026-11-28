# APPLIED — validated 14-test BOM draft placed at tests/test_bom_acceptance.py.
#
# Static-only verification; NOT execution-verified.
#   - `python3 -m py_compile` -> OK (bytecode redirected to scratch so the
#     worktree stays clean); `git diff --check` -> OK; full read-back -> OK.
#   - unittest/pytest was NOT run. No pass/fail claim is made here.
#
# HOLD context: guard `FLEET_LAUNCH_HOLD SWAP_RESERVE_LOW,IO_PRESSURE,MEMORY_PRESSURE`
# (2026-11-28, x1-370) forbids test execution, builds, servers and CI in this
# slice. This file was placed and statically checked only; no production code
# was changed. Post-guard execution is deferred to a later slice / node.
#
# Provenance / source of truth (read-only `git show` inspection only):
#   Applied on branch agent/autoingest-bom-apply-20261128, base
#   9efb7715e649f02c728c448b264b85591b4ad55e (contains the earlier casefold
#   test, deacc2e0). The draft was reviewed against auto-ingest HEAD
#   f24b26465f56bab0eadce977ed67b91a410125b7 on
#   integration/round7-auto-ingest-20261126. All signatures/strings were
#   re-confirmed against current HEAD 9efb7715 with NO adaptation required.
#   Source commits:
#     b81b9e112c534f0c8a31339df3318dd44844a6e0 -> content_os/adapters.py
#         CsvAdapter.extract / read_limited / file_sha256 / extract_source
#     b237e9a927e8968c3d63fc9b91e773aeb54616e8 ->
#         auto_ingest/custody/external_inventory.py observe_metadata_csv /
#         ExternalInventoryError (subclass of ValueError) / Observation
#     871e05fc35dc8cecdf6c8340421a27c1ba241698 ->
#         auto_ingest/ingest_write.py read_intermediate and
#         auto_ingest/ingest_import.py _load_rows / import_intermediate
#   (The upstream regression tests for these commits live in
#    tests/test_adapters.py, tests/test_custody_external_inventory.py and
#    tests/test_ingest_intermediate.py. This is a single consolidated,
#    filesystem-free variant; nothing was copied into those files.)
#
# Post-guard unittest command (DO NOT RUN under the current HOLD):
#   <cd to the repo root> && \
#     python3 -m unittest discover -s tests -p 'test_bom_acceptance.py' -v
#   (No tests/__init__.py exists, so `discover -s tests` is the correct form.
#    Needs repo root on sys.path for `content_os` / `auto_ingest` imports;
#    running with `cd` into the repo root provides that.)
#
# Hermeticity: every byte stream is injected in memory. `read_limited` and
# `file_sha256` are replaced for CsvAdapter; `Path` inside the custody module is
# replaced by an in-memory stand-in; `builtins.open` / `os.path.exists` are
# replaced for the intermediate readers. A denial guard makes any accidental
# real pathlib filesystem access raise immediately. No tmp_path, no NAS, no
# network, no database.

"""Hermetic unittest file for UTF-8 BOM acceptance in auto-ingest (applied from a statically validated draft; not execution-verified)."""

from __future__ import annotations

import csv
import io
import unittest
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

from auto_ingest import ingest_import, ingest_write
from auto_ingest.custody import external_inventory as ei
from auto_ingest.custody.external_inventory import ExternalInventoryError
from content_os import adapters

UTF8_BOM = b"\xef\xbb\xbf"          # the only BOM these commits accept
UTF16LE_BOM = b"\xff\xfe"           # BOM-like bytes that are NOT a UTF-8 BOM
INLINE_BOM = b"\xef\xbb\xbf"        # U+FEFF encoded inside a data field


# ---------------------------------------------------------------------------
# Hermeticity helpers
# ---------------------------------------------------------------------------

@contextmanager
def _deny_real_path_io():
    """Any real pathlib filesystem call inside the test is a draft bug."""
    def denied(*_a, **_kw):
        raise AssertionError("real filesystem access attempted in hermetic test")

    with (
        patch.object(Path, "read_bytes", denied),
        patch.object(Path, "open", denied),
        patch.object(Path, "stat", denied),
        patch.object(Path, "is_file", denied),
        patch.object(Path, "is_symlink", denied),
    ):
        yield


class _MemoryReadLimited:
    """Stand-in for content_os.adapters.read_limited over injected bytes.

    Mirrors production semantics exactly: plain ``utf-8`` decode with
    ``errors="replace"`` (so non-UTF-8 bytes become U+FFFD, never a raised
    UnicodeDecodeError in this adapter).
    """

    def __init__(self, data: bytes):
        self.data = data

    def __call__(self, path, limit=250_000):
        return self.data.decode("utf-8", errors="replace"), False


def _extract_csv(data: bytes):
    """Run the real CsvAdapter.extract/extract_source with injected bytes."""
    virtual = Path("notes.csv")  # suffix only; never opened
    with (
        _deny_real_path_io(),
        patch.object(adapters, "read_limited", _MemoryReadLimited(data)),
        patch.object(adapters, "file_sha256", lambda _p: sha256(data).hexdigest()),
    ):
        return adapters.extract_source(virtual)


# ---------------------------------------------------------------------------
# b81b9e11 — CsvAdapter.extract
# ---------------------------------------------------------------------------

class CsvAdapterBomTests(unittest.TestCase):
    def test_bom_prefixed_header_parses_identically_to_bare_header(self):
        """BOM header -> same columns/rows/text as a bare header; digest differs."""
        bare_bytes = b"id,notes\r\n1,Useful proof\r\n2,Second proof\r\n"
        bare = _extract_csv(bare_bytes)
        bom = _extract_csv(UTF8_BOM + bare_bytes)

        self.assertEqual(bom.metadata["columns"], ["id", "notes"])
        self.assertEqual(bom.metadata["columns"], bare.metadata["columns"])
        self.assertEqual(bom.metadata["rows_read"], bare.metadata["rows_read"])
        self.assertEqual(bom.metadata["sample_rows"], bare.metadata["sample_rows"])
        self.assertEqual(bom.title, bare.title)
        self.assertEqual(bom.text, bare.text)
        self.assertNotIn("\ufeff", bom.text)
        self.assertIn("Useful proof", bom.text)
        self.assertIn("Second proof", bom.text)
        # BOM is stripped for parsing only; byte digest still covers it.
        self.assertEqual(bare.sha256, sha256(bare_bytes).hexdigest())
        self.assertEqual(bom.sha256, sha256(UTF8_BOM + bare_bytes).hexdigest())
        self.assertNotEqual(bom.sha256, bare.sha256)

    def test_bom_in_data_field_is_preserved(self):
        """removeprefix() strips only a leading BOM; an in-field BOM survives."""
        doc = _extract_csv(UTF8_BOM + b"id,notes\r\n1,keep" + INLINE_BOM + b"mid\r\n")
        self.assertEqual(doc.metadata["columns"], ["id", "notes"])
        self.assertEqual(doc.text, "keep\ufeffmid")

    def test_non_utf8_bom_like_bytes_are_not_normalized_as_utf8_bom(self):
        """UTF-16 BOM-like bytes are NOT a UTF-8 BOM and stay replacement chars.

        CsvAdapter uses errors="replace", so this path does not raise; the
        fail-closed assertions for undecodable bytes live in the custody and
        intermediate-reader classes below (strict utf-8-sig decode).
        """
        doc = _extract_csv(UTF16LE_BOM + b"id,notes\r\n1,Useful proof\r\n")
        self.assertEqual(doc.metadata["columns"], ["\ufffd\ufffdid", "notes"])
        self.assertNotIn("id", doc.metadata["columns"])
        self.assertIn("Useful proof", doc.text)


# ---------------------------------------------------------------------------
# b237e9a9 — digest-pinned NAS inventory CSV
# ---------------------------------------------------------------------------

INV_FIELDS = [
    "card_uuid", "source_relative_path", "candidate_relative_path",
    "expected_bytes", "metadata_status", "observed_bytes",
    "nas_payload_sha256_verified_today", "source_release_authorized",
]


def _inventory_csv(*, bom: bool = False, **row_updates) -> bytes:
    """Build a valid one-row observation CSV entirely in memory."""
    row = {
        "card_uuid": "CARD-1",
        "source_relative_path": "DCIM/a.MP4",
        "candidate_relative_path": "2026/10/16/a.MP4",
        "expected_bytes": "12",
        "metadata_status": "NAME_AND_SIZE_ONLY",
        "observed_bytes": "12",
        "nas_payload_sha256_verified_today": "False",
        "source_release_authorized": "False",
    }
    row.update(row_updates)
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=INV_FIELDS)
    writer.writeheader()
    writer.writerow(row)
    raw = buf.getvalue().encode("utf-8")
    return UTF8_BOM + raw if bom else raw


class _MemoryStat:
    def __init__(self, size: int):
        self.st_size = size


class _MemoryPath:
    """In-memory stand-in for pathlib.Path in external_inventory."""

    def __init__(self, data: bytes):
        self._data = data

    def is_file(self):
        return True

    def is_symlink(self):
        return False

    def stat(self):
        return _MemoryStat(len(self._data))

    def read_bytes(self):
        return self._data


def _observe(data: bytes, *, pin: str | None = None, expected_rows: int = 1):
    """Call the real observe_metadata_csv with fully injected bytes."""
    with (
        _deny_real_path_io(),
        patch.object(ei, "Path", lambda _p: _MemoryPath(data)),
    ):
        return ei.observe_metadata_csv(
            "virtual://inventory.csv",
            expected_sha256=sha256(data).hexdigest() if pin is None else pin,
            expected_rows=expected_rows,
        )


class CustodyBomTests(unittest.TestCase):
    def test_bom_prefixed_header_parses_identically_to_bare_header(self):
        bare = _observe(_inventory_csv())
        bom = _observe(_inventory_csv(bom=True))
        self.assertEqual(bom.rows, bare.rows)
        self.assertEqual(bom.by_status, bare.by_status)
        self.assertEqual(bom.by_card_and_status, bare.by_card_and_status)
        self.assertEqual(bom.to_dict()["by_status"], {"NAME_AND_SIZE_ONLY": 1})

    def test_digest_pinned_csv_with_bom_still_validates(self):
        """Pin is over BOM-inclusive bytes; BOM is consumed only for decoding."""
        data = _inventory_csv(bom=True)
        pin = sha256(data).hexdigest()
        result = _observe(data, pin=pin)
        self.assertEqual(result.rows, 1)
        self.assertEqual(result.source_sha256, pin)
        self.assertFalse(result.source_release_allowed)
        self.assertFalse(result.import_as_custody_receipt_allowed)
        self.assertEqual(result.meta_only, True)
        self.assertEqual(result.destination_verified_files, 0)

    def test_dropping_the_bom_fails_the_pin_fail_closed(self):
        data = _inventory_csv(bom=True)
        bare_pin = sha256(_inventory_csv()).hexdigest()
        with self.assertRaisesRegex(ExternalInventoryError, "digest mismatch"):
            _observe(data, pin=bare_pin)

    def test_non_utf8_bom_like_bytes_fail_closed(self):
        """UTF-16 BOM-like bytes are not utf-8-sig-decodable -> hard error."""
        data = UTF16LE_BOM + _inventory_csv()
        with self.assertRaisesRegex(ExternalInventoryError, "not UTF-8"):
            _observe(data)

    def test_bom_inside_data_field_is_preserved_not_rejected(self):
        data = _inventory_csv(bom=True, source_relative_path="DCIM/" + "\ufeff" + "a.MP4")
        result = _observe(data)
        self.assertEqual(result.rows, 1)
        self.assertEqual(result.to_dict()["by_status"], {"NAME_AND_SIZE_ONLY": 1})

    def test_bom_mid_header_is_rejected_not_silently_stripped(self):
        """utf-8-sig strips only a LEADING BOM; an embedded one breaks the pin check."""
        data = _inventory_csv(bom=True).replace(b"card_uuid", b"c" + INLINE_BOM + b"ard_uuid", 1)
        with self.assertRaisesRegex(ExternalInventoryError, "required column missing"):
            _observe(data)


# ---------------------------------------------------------------------------
# 871e05fc — ingest_write.read_intermediate / ingest_import._load_rows
# ---------------------------------------------------------------------------

class _MemoryOpen:
    """open() replacement: decode injected bytes with the codec production asks for."""

    def __init__(self, blobs: dict[str, bytes]):
        self.blobs = blobs
        self.calls: list[dict] = []

    def __call__(self, path, mode="r", *, encoding=None, newline=None, **kwargs):
        self.calls.append({
            "path": path, "mode": mode, "encoding": encoding, "newline": newline,
        })
        codec = encoding if encoding is not None else "utf-8"
        text = self.blobs[path].decode(codec)  # strict: BOM-like invalid bytes raise
        return io.StringIO(text, newline=newline if newline is not None else "\n")


@contextmanager
def _memory_intermediate(blobs: dict[str, bytes]):
    opener = _MemoryOpen(blobs)
    with patch("builtins.open", opener), patch("os.path.exists", return_value=True):
        yield opener


class _Call:
    __slots__ = ("query", "params")

    def __init__(self, query, params):
        self.query = query
        self.params = params


class _FakeSession:
    def __init__(self, log):
        self._log = log

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def run(self, query, **params):
        self._log.append(_Call(query, dict(params)))


class _FakeDriver:
    """Pure in-memory Neo4j stand-in; no driver import, no network."""

    def __init__(self):
        self.calls: list[_Call] = []

    def session(self):
        return _FakeSession(self.calls)


class IntermediateBomTests(unittest.TestCase):
    def test_read_intermediate_bom_header_matches_bare_header(self):
        path = ingest_write._path("/virtual", "k", "metadata", "csv")  # pure, no I/O
        header = b"id,mph\r\na1,55\r\n"
        with _memory_intermediate({path: header}) as bare_opener:
            bare = ingest_write.read_intermediate("/virtual", "k", "metadata", fmt="csv")
        with _memory_intermediate({path: UTF8_BOM + header}) as bom_opener:
            bom = ingest_write.read_intermediate("/virtual", "k", "metadata", fmt="csv")

        self.assertEqual(bom, [{"id": "a1", "mph": "55"}])
        self.assertEqual(bom, bare)
        self.assertNotIn("\ufeffid", bom[0])
        self.assertEqual(bare_opener.calls[0]["encoding"], "utf-8-sig")
        self.assertEqual(bom_opener.calls[0]["encoding"], "utf-8-sig")

    def test_load_rows_bom_header_resolves_key_field(self):
        path = "/virtual/k.segments.csv"
        with _memory_intermediate({path: UTF8_BOM + b"id,v\r\na1,7\r\n"}) as opener:
            rows = ingest_import._load_rows(path)

        self.assertEqual(rows, [{"id": "a1", "v": "7"}])
        self.assertEqual(opener.calls[0]["encoding"], "utf-8-sig")
        self.assertEqual(ingest_import._split_row(rows[0], "id"), ("a1", {"v": "7"}))

    def test_import_intermediate_bom_csv_uses_clean_key(self):
        path = "/virtual/k.segments.csv"
        driver = _FakeDriver()
        with _memory_intermediate({path: UTF8_BOM + b"id,v\r\na1,7\r\n"}) as opener:
            count = ingest_import.import_intermediate(
                driver, path, label="Segment", key_field="id"
            )

        self.assertEqual(count, 1)
        self.assertEqual(len(driver.calls), 1)
        self.assertIn("UNWIND", driver.calls[0].query)
        self.assertEqual(driver.calls[0].params["rows"], [
            {"key": "a1", "props": {"v": "7"}}
        ])
        self.assertEqual(opener.calls[0]["encoding"], "utf-8-sig")

    def test_bom_in_data_field_is_preserved_in_intermediate(self):
        path = "/virtual/k.segments.csv"
        with _memory_intermediate(
            {path: UTF8_BOM + b"id,v\r\na1,keep" + INLINE_BOM + b"mid\r\n"}
        ):
            rows = ingest_import._load_rows(path)
        self.assertEqual(rows, [{"id": "a1", "v": "keep\ufeffmid"}])

    def test_non_utf8_bom_like_bytes_fail_closed_in_intermediate(self):
        path = "/virtual/k.segments.csv"
        data = UTF16LE_BOM + b"id,v\r\na1,7\r\n"
        with _memory_intermediate({path: data}) as opener:
            with self.assertRaises(UnicodeDecodeError):
                ingest_import._load_rows(path)
        # Both strict codecs reject it; the BOM cases above pin utf-8-sig.
        self.assertIn(opener.calls[0]["encoding"], ("utf-8-sig", "utf-8"))


if __name__ == "__main__":
    unittest.main()
