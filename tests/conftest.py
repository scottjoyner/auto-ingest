import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def explicit_machine_profile_context(request, monkeypatch):
    """Make host assumptions explicit in legacy config tests.

    Production config now treats unmatched hosts as the generic `any` profile.
    `tests/test_config.py` builds an x1-370-only fixture and is specifically
    asserting that profile, so identify that test module as x1-370 instead of
    restoring unsafe arbitrary-first-profile fallback behavior.
    """
    if Path(str(request.fspath)).name == "test_config.py":
        import auto_ingest_config

        monkeypatch.setattr(
            auto_ingest_config.socket,
            "gethostname",
            lambda: "x1-370-ci",
        )
