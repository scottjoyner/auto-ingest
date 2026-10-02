import importlib.machinery
import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "tools" / "dashcam-card-ingest"
loader = importlib.machinery.SourceFileLoader("dashcam_card_ingest", str(SCRIPT))
spec = importlib.util.spec_from_loader(loader.name, loader)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_verified_destination_totals_use_copied_record_status():
    inventory = [
        {"path": "clip.mp4", "kind": "file", "size": 123},
        {"path": "DCIM", "kind": "directory", "size": 0},
    ]
    copied_records = [
        {"path": "clip.mp4", "kind": "file", "size": 123, "status": "VERIFIED"},
        {"path": "DCIM", "kind": "directory", "size": 0, "status": "NOT_APPLICABLE"},
    ]

    assert module.verified_destination_totals(inventory, copied_records) == (1, 123)
