"""Tests for the optional-dependency adapters and identity helpers, with fakes (no heavy deps)."""

import sys
import types

import pytest

from ef.ops import segmenter_identity
from ef.reranking import cross_encoder_reranker
from ef.segmenter_adapters import imbed_segmenter
from ef.segmenters import FunctionSegmenter


def _fake_module(monkeypatch, name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


def test_cross_encoder_reranker_scores_pairs_with_the_model(monkeypatch):
    seen = {}

    class CrossEncoder:
        def __init__(self, name, device=None):
            seen["init"] = (name, device)

        def predict(self, pairs, batch_size=None, **kw):
            seen["pairs"], seen["batch"], seen["kw"] = pairs, batch_size, kw
            return [len(b) for _, b in pairs]

    _fake_module(monkeypatch, "sentence_transformers", CrossEncoder=CrossEncoder)
    rr = cross_encoder_reranker("m", device="cpu", batch_size=4)
    assert rr("q", [{"text": "ab"}, {"text": "abc"}]) == [2.0, 3.0]
    assert seen["init"] == ("m", "cpu")
    assert seen["pairs"] == [("q", "ab"), ("q", "abc")] and seen["batch"] == 4
    assert rr("q", []) == []
    assert rr.model_name == "m"


def test_imbed_segmenter_resolves_a_registered_name(monkeypatch):
    registry = {"words": str.split}
    pkg = _fake_module(monkeypatch, "imbed")
    comp = _fake_module(monkeypatch, "imbed.components")
    seg = _fake_module(monkeypatch, "imbed.components.segmentation", segmenters=registry)
    pkg.components, comp.segmentation = comp, seg
    assert isinstance(imbed_segmenter("words"), FunctionSegmenter)
    with pytest.raises(ValueError, match="Unknown imbed segmenter 'nope'"):
        imbed_segmenter("nope")


def test_segmenter_identity_of_function_and_anonymous_segmenters():
    def named(text):
        return [text]

    ident = segmenter_identity(FunctionSegmenter(named))
    assert ident.startswith("function:") and ident.endswith("named")
    assert segmenter_identity(FunctionSegmenter(named)) == ident


def test_segmenter_identity_of_plain_callables_and_objects():
    def plain(text):
        return [text]

    class Obj:
        def __call__(self, text):
            return [text]

    assert segmenter_identity(plain).endswith("plain")
    assert segmenter_identity(Obj()) == "Obj"
