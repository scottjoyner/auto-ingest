"""Interrupted ledger serialization must preserve the last good checkpoint."""
import ast
import json
from pathlib import Path

import pytest


def test_interrupted_ledger_write_preserves_checkpoint(tmp_path, monkeypatch):
    # Load only the persistence function, not module-level credentials or scheduling.
    source = Path(__file__).resolve().parents[1] / "scripts" / "ingest_supervisor.py"
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == "save_ledger")
    target = tmp_path / "ledger.json"
    previous = {"2026/01/01": {"status": "ok", "attempts": 1}}
    target.write_text(json.dumps(previous))
    before = target.read_bytes()
    namespace = {"LEDGER": str(target), "json": json}
    exec(compile(ast.Module(body=[function], type_ignores=[]),
                 str(source), "exec"), namespace)

    def interrupted(obj, fp, **kwargs):
        fp.write('{"partial":')
        raise OSError("injected serialization interruption")

    with monkeypatch.context() as patch:
        patch.setattr(json, "dump", interrupted)
        with pytest.raises(OSError, match="injected"):
            namespace["save_ledger"]({"next": {"status": "fail"}})
    assert target.read_bytes() == before
    assert list(tmp_path.iterdir()) == [target]
    replacement = {"2026/01/01": {"status": "fail", "attempts": 2}}
    namespace["save_ledger"](replacement)
    assert json.loads(target.read_text()) == replacement
