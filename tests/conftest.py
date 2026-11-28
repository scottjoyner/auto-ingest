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


@pytest.fixture(autouse=True)
def hermetic_campaign_locks(tmp_path_factory, monkeypatch):
    """Keep every test out of the shared production campaign lock directory.

    ``CUSTODY_LOCK_ROOT`` defaults to ``/nas/custody-locks``, which is shared
    across machines and holds a marker for any campaign currently copying. Left
    alone, a test that shells out to the CLI observes live production state: it is
    refused as ``destination_locked_by_another_campaign`` while a real campaign
    runs and passes when none does.

    That is worse than a flaky test, because it is invisible in CI - which has no
    campaign running - and shows up only on a workstation mid-copy, where it reads
    as a product bug. Two tests failed exactly this way during a live 91 GB
    campaign; both passed in isolation and in CI.

    Autouse and session-wide rather than part of ``hermetic_mounts``, because the
    tests that shell out to the real CLI do not request that fixture and were the
    ones that broke.
    """
    root = tmp_path_factory.mktemp("custody-locks")
    monkeypatch.setenv("CUSTODY_LOCK_ROOT", str(root))
    return root
