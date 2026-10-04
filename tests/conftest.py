import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402


@pytest.fixture
def hermetic_mounts(tmp_path, monkeypatch):
    """Point the custody CLI at a synthetic mount table.

    The CLI reads /proc/mounts, so without this any test asserting on
    `source_present` or `source_read_only_observed` was really asserting that
    *this developer's SD card is currently plugged in and mounted read-only*.
    That passed locally and failed on CI, where no card exists - so the suite
    could not distinguish a regression from a missing peripheral, and the helper
    module's claim to be hermetic was false.

    Request it (or mark a module with `usefixtures`) and the source end becomes a
    fact the test controls. The real parser and observation logic still run -
    only their input changes.
    """
    from custody_helpers import patch_mount_table, synthetic_mounts

    table = synthetic_mounts(tmp_path / "mounts")
    patch_mount_table(monkeypatch, table)
    return table


@pytest.fixture
def real_mounts(monkeypatch):
    """Undo `hermetic_mounts` and read the genuine kernel mount table.

    For the one test whose subject IS the real /proc/mounts reader.
    """
    from custody_helpers import patch_mount_table

    from auto_ingest.custody.mounts import MOUNTS_PATH

    patch_mount_table(monkeypatch, MOUNTS_PATH)
    return MOUNTS_PATH
