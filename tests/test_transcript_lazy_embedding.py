"""Hermetic regression tests for lazy HuggingFace model loading.

Closes PRODUCTION_READINESS_GAP_ANALYSIS.md gap #4 ("HuggingFace Model Download
Dependencies"): a plain ``import auto_ingest.ingest.transcripts`` used to call
``AutoTokenizer.from_pretrained`` / ``AutoModel.from_pretrained`` at module
scope, so CLI --help, the shim smoke test and discovery-contract checks could
reach out to HuggingFace (and require auth / a cached model). The model must
now load on first use via ``get_embed_models()``.

No test downloads or loads a real model: loaders are faked or guarded to fail
loudly if called. Heavy ML imports are skipped where that stack is absent.
"""
import ast
import contextlib
import importlib
import sys
import types
from pathlib import Path

import pytest

MODULE = "auto_ingest.ingest.transcripts"
SOURCE = Path(__file__).resolve().parents[1] / "auto_ingest" / "ingest" / "transcripts.py"


def test_no_module_scope_from_pretrained():
    """Dependency-free guard: no from_pretrained() call may execute on import."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    parents = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "from_pretrained"):
            continue
        enclosing = parents.get(node)
        while enclosing is not None and not isinstance(
            enclosing, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            enclosing = parents.get(enclosing)
        assert enclosing is not None, "from_pretrained() at module scope loads a model on import"


@contextlib.contextmanager
def _fresh_transcripts():
    """Import transcripts anew, restoring sys.modules and the package attr after."""
    import auto_ingest.ingest as pkg

    had_module = MODULE in sys.modules
    old_module = sys.modules.get(MODULE)
    had_attr = hasattr(pkg, "transcripts")
    old_attr = getattr(pkg, "transcripts", None)
    sys.modules.pop(MODULE, None)
    try:
        yield importlib.import_module(MODULE)
    finally:
        if had_module:
            sys.modules[MODULE] = old_module
        else:
            sys.modules.pop(MODULE, None)
        if had_attr:
            pkg.transcripts = old_attr
        else:
            delattr(pkg, "transcripts")


def test_import_does_not_call_from_pretrained(monkeypatch):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("a model was loaded during import")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", forbidden)
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", forbidden)

    with _fresh_transcripts() as fresh:
        assert fresh.emb_tokenizer is None
        assert fresh.emb_model is None
    assert calls == []


@pytest.fixture
def tx():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    return importlib.import_module(MODULE)


def test_lazy_load_is_cached_and_configured(tx, monkeypatch):
    calls = []

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, name):
            calls.append(("tokenizer", name))
            return f"tokenizer:{name}"

    class FakeModel:
        @classmethod
        def from_pretrained(cls, name):
            return cls(name)

        def __init__(self, name):
            calls.append(("model", name))
            self.device = None
            self.evaluated = False

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.evaluated = True
            return self

    monkeypatch.setattr(tx, "AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr(tx, "AutoModel", FakeModel)
    monkeypatch.setattr(tx, "emb_tokenizer", None)
    monkeypatch.setattr(tx, "emb_model", None)

    tok1, model1 = tx.get_embed_models()
    tok2, model2 = tx.get_embed_models()

    assert tok1 == tok2 == f"tokenizer:{tx.EMBED_MODEL_NAME}"
    assert model1 is model2
    assert calls == [
        ("tokenizer", tx.EMBED_MODEL_NAME),
        ("model", tx.EMBED_MODEL_NAME),
    ]
    assert model1.device == tx.DEVICE
    assert model1.evaluated is True


def test_prepopulated_models_are_not_replaced(tx, monkeypatch):
    """Existing tests that monkeypatch the globals keep working."""
    calls = []

    class Exploding:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append(True)
            raise AssertionError("should not load when globals are already set")

    monkeypatch.setattr(tx, "AutoTokenizer", Exploding)
    monkeypatch.setattr(tx, "AutoModel", Exploding)
    sentinel_tok, sentinel_model = object(), object()
    monkeypatch.setattr(tx, "emb_tokenizer", sentinel_tok)
    monkeypatch.setattr(tx, "emb_model", sentinel_model)

    assert tx.get_embed_models() == (sentinel_tok, sentinel_model)
    assert calls == []


def test_embed_texts_uses_lazy_accessor(tx, monkeypatch):
    import torch

    class FakeTokenizer:
        def __call__(self, batch, **kwargs):
            n = len(batch)
            ones = torch.ones((n, 2), dtype=torch.long)
            return {"input_ids": ones, "attention_mask": ones}

    class FakeModel:
        def __call__(self, **enc):
            n = enc["input_ids"].shape[0]
            return types.SimpleNamespace(last_hidden_state=torch.ones((n, 2, 4)))

    calls = []

    def fake_get():
        calls.append(True)
        return FakeTokenizer(), FakeModel()

    monkeypatch.setattr(tx, "get_embed_models", fake_get)

    vectors = tx.embed_texts(["a", "b"], batch_size=1)

    assert len(vectors) == 2
    assert all(len(v) == 4 for v in vectors)
    assert len(calls) == 1
