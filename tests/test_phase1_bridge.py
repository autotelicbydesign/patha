"""Tests for the Phase 1 ↔ Phase 2 bridge.

These tests use stub embedders so they don't download real models. They
verify the bridge builds indexes correctly and returns belief-store
proposition ids (not Phase-1 chunk ids).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from patha.belief.store import BeliefStore
from patha.models.embedder import StubEmbedder
from patha.phase1_bridge import (
    build_phase1_indexes,
    build_phase1_retriever,
)


def _populate(store: BeliefStore) -> None:
    for i, text in enumerate([
        "I love sushi every week",
        "I am avoiding raw fish on my doctor's advice",
        "my rent is 1500",
        "I prefer tea in the mornings",
    ]):
        store.add(
            proposition=text,
            asserted_at=datetime(2024, 1, i + 1),
            asserted_in_session=f"s{i}",
            source_proposition_id=f"src-{i}",
        )


class TestBuildPhase1Indexes:
    def test_empty_store_returns_empty_indexes(self, tmp_path):
        store = BeliefStore(persistence_path=tmp_path / "beliefs.jsonl")
        prop_store, bm25, id_map = build_phase1_indexes(
            store, embedder=StubEmbedder(),
        )
        assert id_map == {}

    def test_populated_store_builds_id_map(self, tmp_path):
        store = BeliefStore(persistence_path=tmp_path / "beliefs.jsonl")
        _populate(store)
        prop_store, bm25, id_map = build_phase1_indexes(
            store, embedder=StubEmbedder(),
        )
        # Every chunk maps to a source_proposition_id
        assert len(id_map) == 4
        assert set(id_map.values()) == {"src-0", "src-1", "src-2", "src-3"}


class TestBuildPhase1Retriever:
    def test_empty_store_returns_empty_list(self, tmp_path):
        store = BeliefStore(persistence_path=tmp_path / "beliefs.jsonl")
        retrieve = build_phase1_retriever(store, embedder=StubEmbedder())
        assert retrieve("anything", 5) == []

    def test_retrieves_proposition_ids(self, tmp_path):
        store = BeliefStore(persistence_path=tmp_path / "beliefs.jsonl")
        _populate(store)
        retrieve = build_phase1_retriever(store, embedder=StubEmbedder())
        results = retrieve("tea coffee mornings", top_k=2)
        # Every returned id must be a source_proposition_id (starts with 'src-')
        assert all(r.startswith("src-") for r in results)
        # Should return at most top_k items
        assert len(results) <= 2
