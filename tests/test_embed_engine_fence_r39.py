"""Pin the ``auto_ingest.embed.load_embed_model`` engine-dispatch contract (r39).

``engine`` is a *data-affecting* choice: mixing torch and ONNX vectors under one
vector property silently diverges retrieval recall.  The CLI callers validate
their ``--engine`` via argparse ``choices``, but the ingest path
(``auto_ingest/ingest/transcripts.py::embed_extra_texts``) takes the engine from
``EMBED_EXTRA_ENGINE`` / ``EMBED_ENGINE`` with **no** validation.  Historically
``load_embed_model`` used ``if engine == "onnx": … else: EmbedModel``, so any
typo/case variant silently produced torch vectors while claiming an ONNX
engine.

These tests stub both model classes, so no weights, network or inference are
loaded.  Only the dispatch decision is exercised.
"""
import pytest

import auto_ingest.embed as embed_mod


class _TorchProbe:
    kind = "torch"

    def __init__(self, name):
        self.name = name


class _OnnxProbe:
    kind = "onnx"

    def __init__(self, name):
        self.name = name


@pytest.fixture(autouse=True)
def _stub_models(monkeypatch):
    """Replace the real model classes; no HF/ONNX load can occur."""
    monkeypatch.setattr(embed_mod, "EmbedModel", _TorchProbe)
    monkeypatch.setattr(embed_mod, "OrtEmbedModel", _OnnxProbe)


@pytest.mark.parametrize(
    "bad_engine",
    ["onnxruntime", "ONNX", "onnx ", "", None, "torch2", "onnx-v2"],
)
def test_unknown_engine_fails_closed(bad_engine):
    """Any engine outside the documented set must be rejected, not coerced."""
    with pytest.raises(ValueError, match="engine"):
        embed_mod.load_embed_model("some/model", engine=bad_engine)


def test_explicit_torch_engine_still_selects_torch():
    model = embed_mod.load_embed_model("some/model", engine="torch")
    assert isinstance(model, _TorchProbe)
    assert model.name == "some/model"


def test_explicit_onnx_engine_still_selects_onnx():
    model = embed_mod.load_embed_model("some/model", engine="onnx")
    assert isinstance(model, _OnnxProbe)
    assert model.name == "some/model"


def test_default_engine_is_torch():
    model = embed_mod.load_embed_model("some/model")
    assert isinstance(model, _TorchProbe)
