"""Tests for the Embedder protocol and the deterministic StubEmbedder."""

from __future__ import annotations

import math

import pytest

from patha.models.embedder import Embedder, StubEmbedder


def test_stub_conforms_to_protocol():
    assert isinstance(StubEmbedder(), Embedder)


def test_stub_produces_correct_shape():
    emb = StubEmbedder(dim=32)
    out = emb.embed(["hello", "world", "foo"])
    assert len(out) == 3
    assert all(len(v) == 32 for v in out)


def test_stub_vectors_are_unit_normalized():
    emb = StubEmbedder(dim=64)
    for v in emb.embed(["alpha", "beta", "gamma with some extra text"]):
        norm = math.sqrt(sum(x * x for x in v))
        assert math.isclose(norm, 1.0, abs_tol=1e-9)


def test_stub_is_deterministic():
    a = StubEmbedder(dim=32).embed(["hello world"])
    b = StubEmbedder(dim=32).embed(["hello world"])
    assert a == b


def test_different_inputs_yield_different_vectors():
    emb = StubEmbedder(dim=64)
    v1, v2 = emb.embed(["hello", "hello!"])
    assert v1 != v2


def test_same_input_always_yields_identical_vector():
    """This is the property ingest + search tests rely on for exact-text lookup."""
    emb = StubEmbedder(dim=48)
    target = "The quick brown fox."
    v_initial = emb.embed([target])[0]
    v_later = emb.embed(["unrelated", target, "more unrelated"])[1]
    assert v_initial == v_later


def test_zero_dim_rejected():
    with pytest.raises(ValueError):
        StubEmbedder(dim=0)
    with pytest.raises(ValueError):
        StubEmbedder(dim=-5)


def test_empty_batch_returns_empty():
    assert StubEmbedder().embed([]) == []


def test_dim_attribute_exposed():
    emb = StubEmbedder(dim=128)
    assert emb.dim == 128
