"""Tests for the Store protocol and InMemoryStore implementation."""

from __future__ import annotations

import math

import pytest

from patha.chunking.views import VIEW_NAMES
from patha.indexing.store import InMemoryStore, Store, cosine_similarity


def _make_row(chunk_id: str, vec: list[float]) -> dict:
    return {
        "chunk_id": chunk_id,
        "session_id": "s1",
        "turn_idx": 0,
        "prop_idx": int(chunk_id.split("p")[-1]),
        "text": chunk_id,
        "speaker": None,
        "timestamp": None,
        "entities": [],
        "views": {name: {"text": chunk_id, "embedding": vec} for name in VIEW_NAMES},
    }


class TestCosineSimilarity:
    def test_identical_vectors_are_one(self):
        v = [1.0, 0.0, 0.0]
        assert math.isclose(cosine_similarity(v, v), 1.0)

    def test_orthogonal_vectors_are_zero(self):
        assert math.isclose(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_opposite_vectors_are_minus_one(self):
        assert math.isclose(cosine_similarity([1.0, 0.0], [-1.0, 0.0]), -1.0)

    def test_zero_vector_returns_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError, match="dimension mismatch"):
            cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])


class TestInMemoryStore:
    def test_conforms_to_protocol(self):
        assert isinstance(InMemoryStore(), Store)

    def test_empty_store_has_count_zero(self):
        store = InMemoryStore()
        assert store.count() == 0
        assert list(store.all_rows()) == []

    def test_upsert_and_get(self):
        store = InMemoryStore()
        store.upsert([_make_row("s1#t0#p0", [1.0, 0.0])])
        assert store.count() == 1
        row = store.get("s1#t0#p0")
        assert row is not None
        assert row["chunk_id"] == "s1#t0#p0"

    def test_get_missing_returns_none(self):
        assert InMemoryStore().get("nope") is None

    def test_upsert_replaces_existing(self):
        store = InMemoryStore()
        store.upsert([_make_row("cid", [1.0, 0.0])])
        store.upsert([_make_row("cid", [0.0, 1.0])])
        assert store.count() == 1
        assert store.get("cid")["views"]["v1"]["embedding"] == [0.0, 1.0]

    def test_search_returns_top_k_sorted(self):
        store = InMemoryStore()
        store.upsert([
            _make_row("s1#t0#p0", [1.0, 0.0, 0.0]),
            _make_row("s1#t0#p1", [0.0, 1.0, 0.0]),
            _make_row("s1#t0#p2", [0.7071, 0.7071, 0.0]),
        ])
        results = store.search_view("v1", [1.0, 0.0, 0.0], k=3)
        assert len(results) == 3
        assert results[0][0] == "s1#t0#p0"
        assert math.isclose(results[0][1], 1.0, abs_tol=1e-6)
        assert results[1][0] == "s1#t0#p2"  # 45-deg cos ~0.707
        assert results[2][0] == "s1#t0#p1"  # orthogonal

    def test_search_k_larger_than_corpus_returns_all(self):
        store = InMemoryStore()
        store.upsert([_make_row("a", [1.0, 0.0]), _make_row("b", [0.0, 1.0])])
        results = store.search_view("v1", [1.0, 0.0], k=100)
        assert len(results) == 2

    def test_search_zero_k_returns_empty(self):
        store = InMemoryStore()
        store.upsert([_make_row("a", [1.0, 0.0])])
        assert store.search_view("v1", [1.0, 0.0], k=0) == []

    def test_all_rows_preserves_insertion_order(self):
        store = InMemoryStore()
        store.upsert([_make_row(f"c{i}", [float(i), 0.0]) for i in range(5)])
        order = [row["chunk_id"] for row in store.all_rows()]
        assert order == ["c0", "c1", "c2", "c3", "c4"]
