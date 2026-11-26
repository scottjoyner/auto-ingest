"""Do not count a submitted batch before its result confirms completion."""
import pytest

from auto_ingest.ingest_import import import_intermediate
from auto_ingest.ingest_write import write_intermediate


class DeferredDriver:
    def __init__(self, fail_at=None):
        self.submitted = 0
        self.completed = 0
        self.fail_at = fail_at

    def session(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, *, rows):
        assert self.submitted == self.completed, "previous result not consumed"
        self.submitted += 1
        return self

    def consume(self):
        if self.submitted == self.fail_at:
            raise RuntimeError("deferred transaction failure")
        self.completed += 1


@pytest.mark.parametrize("fail_at", [None, 1, 2, 3])
def test_each_batch_consumed_before_progress(tmp_path, fail_at):
    path = write_intermediate(str(tmp_path), "fixture", "segments",
                              [{"id": str(i)} for i in range(3)])
    driver = DeferredDriver(fail_at)
    if fail_at is None:
        assert import_intermediate(driver, path, batch=1, key_field="id") == 3
        assert driver.submitted == driver.completed == 3
    else:
        with pytest.raises(RuntimeError, match="deferred transaction failure"):
            import_intermediate(driver, path, batch=1, key_field="id")
        assert driver.submitted == fail_at
        assert driver.completed == fail_at - 1
