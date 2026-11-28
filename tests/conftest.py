import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402


def _stub_optional_ml_stack_at_import_time():
    """Make top-level test modules importable without the optional ML stack.

    CI deliberately installs no torch/transformers (see .github/workflows/ci.yml
    "lightweight test deps (no ML stack)"). The BOM regression tests import
    auto_ingest.ingest.transcripts, which imports torch and transformers at
    module scope. Fixtures run too late for that: pytest imports test modules
    during collection, before any fixture is set up. Install a MagicMock tree
    here, at conftest import time, ONLY when the real packages are absent, so a
    developer or CI job with the real stack still exercises the real thing.
    """
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        pass
    else:
        return

    from unittest import mock

    torch_names = (
        "torch", "torch.cuda", "torch.backends", "torch.backends.cuda",
        "torch.backends.mps", "torch.nn", "torch.nn.functional", "torch.fx",
    )
    nodes = {name: mock.MagicMock(name=name) for name in torch_names}
    root = nodes["torch"]
    for attr, dotted in (("cuda", "torch.cuda"), ("backends", "torch.backends"),
                         ("nn", "torch.nn"), ("fx", "torch.fx")):
        setattr(root, attr, nodes[dotted])
    for name, node in nodes.items():
        sys.modules.setdefault(name, node)

    transformers_stub = sys.modules.setdefault("transformers", mock.MagicMock(name="transformers"))
    transformers_stub.AutoTokenizer = mock.MagicMock(name="AutoTokenizer")
    transformers_stub.AutoModel = mock.MagicMock(name="AutoModel")


_stub_optional_ml_stack_at_import_time()


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
