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
    LazyPhase1Retriever,
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


class _SimilarityEmbedder:
    """Deterministic embedder that can EXPRESS similarity (StubEmbedder's
    hash-vectors are pairwise ~orthogonal, so topic clustering correctly
    yields all-None on them). Texts sharing a keyword get near-identical
    vectors; everything else lands on its own axis."""

    dim = 16

    _KEYWORD_AXIS = {"sushi": 0, "fish": 0, "tea": 3, "rent": 6}

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for i, t in enumerate(texts):
            v = [0.0] * self.dim
            axis = None
            for kw, ax in self._KEYWORD_AXIS.items():
                if kw in t.lower():
                    axis = ax
                    break
            if axis is None:
                axis = 8 + (i % 7)
            v[axis] = 1.0
            v[(axis + 1) % self.dim] = 0.05 * (i % 3)  # slight wobble
            norm = sum(x * x for x in v) ** 0.5
            out.append([x / norm for x in v])
        return out


class TestBuildPhase1Indexes:
    def test_empty_store_returns_empty_indexes(self, tmp_path):
        store = BeliefStore(persistence_path=tmp_path / "beliefs.jsonl")
        prop_store, bm25, id_map, songline = build_phase1_indexes(
            store, embedder=StubEmbedder(), enable_entities=False,
            enable_songline=False,
        )
        assert id_map == {}
        assert songline is None

    def test_populated_store_builds_id_map(self, tmp_path):
        store = BeliefStore(persistence_path=tmp_path / "beliefs.jsonl")
        _populate(store)
        prop_store, bm25, id_map, songline = build_phase1_indexes(
            store, embedder=StubEmbedder(), enable_entities=False,
            enable_songline=False,
        )
        assert len(id_map) == 4
        assert set(id_map.values()) == {"src-0", "src-1", "src-2", "src-3"}
        assert songline is None

    def test_populated_store_builds_songline(self, tmp_path):
        store = BeliefStore(persistence_path=tmp_path / "beliefs.jsonl")
        _populate(store)
        prop_store, bm25, id_map, songline = build_phase1_indexes(
            store, embedder=StubEmbedder(), enable_entities=False,
        )
        # With enable_entities=False we skip spaCy (which may not be
        # installed in CI); songline graph still builds from session/
        # speaker/temporal channels.
        assert songline is not None

    def test_bridge_populates_topic_channel(self, tmp_path):
        # The two fish/sushi beliefs share a keyword axis in the
        # similarity embedder → same cluster → topic channel + edge.
        store = BeliefStore(persistence_path=tmp_path / "beliefs.jsonl")
        _populate(store)
        _, _, _, songline = build_phase1_indexes(
            store, embedder=_SimilarityEmbedder(), enable_entities=False,
        )
        assert songline is not None
        topics = songline._channel_index.get("topic", {})
        assert topics, "topic channel should be populated"
        # sushi + fish chunks share a topic
        cids = {cid for members in topics.values() for cid in members}
        assert len(cids) >= 2
        # topic_of agrees with the channel index
        some = next(iter(cids))
        assert songline.topic_of(some) is not None

    def test_topic_labels_deterministic_across_rebuilds(self, tmp_path):
        store = BeliefStore(persistence_path=tmp_path / "beliefs.jsonl")
        _populate(store)

        def build_topic_map():
            _, _, _, sl = build_phase1_indexes(
                store, embedder=_SimilarityEmbedder(), enable_entities=False,
            )
            return {
                cid: sl.topic_of(cid)
                for members in sl._channel_index.get("topic", {}).values()
                for cid in members
            }

        assert build_topic_map() == build_topic_map()

    def test_patha_topics_off_disables_channel(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PATHA_TOPICS", "off")
        store = BeliefStore(persistence_path=tmp_path / "beliefs.jsonl")
        _populate(store)
        _, _, _, songline = build_phase1_indexes(
            store, embedder=_SimilarityEmbedder(), enable_entities=False,
        )
        assert songline is not None
        assert not songline._channel_index.get("topic", {})


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


class TestLazyPhase1Retriever:
    def test_builds_on_first_call_not_construction(self, tmp_path):
        """Constructing the retriever must NOT build indexes yet —
        that's the whole point of laziness. Claude Desktop startup
        stays fast."""
        store = BeliefStore(persistence_path=tmp_path / "beliefs.jsonl")
        _populate(store)
        retriever = LazyPhase1Retriever(store, embedder=StubEmbedder())
        # Before any call: not built
        assert retriever.is_built is False
        # First call builds
        retriever("tea", 2)
        assert retriever.is_built is True

    def test_invalidate_triggers_rebuild(self, tmp_path):
        store = BeliefStore(persistence_path=tmp_path / "beliefs.jsonl")
        _populate(store)
        retriever = LazyPhase1Retriever(store, embedder=StubEmbedder())
        retriever("initial", 2)
        assert retriever.is_built is True

        retriever.invalidate()
        # After invalidate: not built
        assert retriever.is_built is False

        # Next call rebuilds
        retriever("after-invalidate", 2)
        assert retriever.is_built is True

    def test_newly_added_belief_findable_after_invalidate(self, tmp_path):
        """Realistic MCP flow: ingest → query. The new belief must be
        retrievable after invalidate() runs."""
        store = BeliefStore(persistence_path=tmp_path / "beliefs.jsonl")
        _populate(store)
        retriever = LazyPhase1Retriever(store, embedder=StubEmbedder())
        retriever("tea", 2)

        # Add a new belief. Without invalidate, it wouldn't be indexed.
        store.add(
            proposition="I just got a new laptop",
            asserted_at=datetime(2024, 2, 1),
            asserted_in_session="s-new",
            source_proposition_id="src-new",
        )
        retriever.invalidate()

        # Rebuild happens on next call. The new belief should be
        # retrievable (StubEmbedder is deterministic so relative
        # scoring is predictable).
        results = retriever("laptop", 5)
        assert "src-new" in results

    def test_empty_store_returns_empty(self, tmp_path):
        store = BeliefStore(persistence_path=tmp_path / "beliefs.jsonl")
        retriever = LazyPhase1Retriever(store, embedder=StubEmbedder())
        assert retriever("anything", 5) == []
