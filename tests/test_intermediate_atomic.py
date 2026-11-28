"""Replayed intermediates must survive interrupted replacement."""
from pathlib import Path

import pytest

from auto_ingest import ingest_write


@pytest.mark.parametrize("fmt", ["json", "csv"])
def test_interrupted_replacement_preserves_artifact(tmp_path, monkeypatch, fmt):
    old = [{"id": "old", "text": "valid"}]
    path = Path(ingest_write.write_intermediate(str(tmp_path), "key", "segments", old, fmt=fmt))
    path.chmod(0o640)
    before = path.read_bytes()

    def interrupted(*args, **kwargs):
        raise OSError("injected serialization failure")

    with monkeypatch.context() as patch:
        if fmt == "json":
            patch.setattr(ingest_write.json, "dump", interrupted)
        else:
            patch.setattr(ingest_write.csv.DictWriter, "writerow", interrupted)
        with pytest.raises(OSError, match="injected"):
            ingest_write.write_intermediate(str(tmp_path), "key", "segments",
                                            [{"id": "new"}], fmt=fmt)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]
    assert ingest_write.read_intermediate(str(tmp_path), "key", "segments", fmt=fmt) == old
    replacement = [{"id": "new", "text": "complete"}]
    ingest_write.write_intermediate(str(tmp_path), "key", "segments", replacement, fmt=fmt)
    assert ingest_write.read_intermediate(str(tmp_path), "key", "segments", fmt=fmt) == replacement
    assert path.stat().st_mode & 0o777 == 0o640
