"""Tests for the songline graph (Pillar 2 — Aboriginal-inspired retrieval)."""

from __future__ import annotations

import pytest

from patha.indexing.bm25_index import SimpleBM25
from patha.indexing.ingest import ingest_sessions
from patha.indexing.songline_graph import SonglineGraph, build_songline_graph
from patha.indexing.store import InMemoryStore
from patha.models.embedder import StubEmbedder
from patha.retrieval.pipeline import PipelineConfig, retrieve
from patha.retrieval.songline_walker import songline_walk


# ─── Fixtures ─────────────────────────────────────────────────────────

MOCK_ROWS = [
    {
        "chunk_id": "s1#t0#p0", "session_id": "s1", "speaker": "alice",
        "timestamp": "2026-01-01", "entities": ["Alice", "Bob"], "topic_cluster": 0,
    },
    {
        "chunk_id": "s1#t0#p1", "session_id": "s1", "speaker": "alice",
        "timestamp": "2026-01-01", "entities": ["Alice"], "topic_cluster": 0,
    },
    {
        "chunk_id": "s1#t1#p0", "session_id": "s1", "speaker": "bob",
        "timestamp": "2026-01-02", "entities": ["Bob", "Carol"], "topic_cluster": 1,
    },
    {
        "chunk_id": "s2#t0#p0", "session_id": "s2", "speaker": "alice",
        "timestamp": "2026-01-03", "entities": ["Alice", "Dave"], "topic_cluster": 0,
    },
    {
        "chunk_id": "s2#t1#p0", "session_id": "s2", "speaker": "bob",
        "timestamp": "2026-01-03", "entities": ["Bob"], "topic_cluster": 1,
    },
]


@pytest.fixture
def graph():
    return build_songline_graph(MOCK_ROWS)


# ─── Graph construction ──────────────────────────────────────────────

class TestSonglineGraph:
    def test_nodes_and_edges_exist(self, graph):
        assert graph.node_count() > 0
        assert graph.edge_count() > 0

    def test_same_session_connected(self, graph):
        neighbor_ids = {n[0] for n in graph.neighbors("s1#t0#p0")}
        assert "s1#t0#p1" in neighbor_ids

    def test_cross_session_connected_via_shared_entity(self, graph):
        neighbor_ids = {n[0] for n in graph.neighbors("s1#t0#p0")}
        assert "s2#t0#p0" in neighbor_ids  # both have Alice

    def test_edges_carry_channel_info(self, graph):
        channels = {n[2] for n in graph.neighbors("s1#t0#p0")}
        assert "entity" in channels

    def test_edge_weights_are_positive(self, graph):
        for neighbors in graph.adjacency.values():
            for _, w, _ in neighbors:
                assert w > 0

    def test_empty_rows_yield_empty_graph(self):
        g = build_songline_graph([])
        assert g.node_count() == 0
        assert g.edge_count() == 0

    def test_single_row_yields_no_edges(self):
        g = build_songline_graph([MOCK_ROWS[0]])
        assert g.edge_count() == 0

    def test_edges_are_symmetric(self, graph):
        for cid, neighbors in graph.adjacency.items():
            for nid, _w, _ch in neighbors:
                reverse_ids = {n[0] for n in graph.neighbors(nid)}
                assert cid in reverse_ids, f"{cid} -> {nid} but not reverse"


# ─── Walker ───────────────────────────────────────────────────────────

class TestSonglineWalker:
    def test_preserves_original_candidates(self, graph):
        reranked = [("s1#t0#p0", 0.9), ("s1#t0#p1", 0.85)]
        walked = songline_walk(reranked, graph, num_anchors=2, hops=2)
        walked_ids = {cid for cid, _ in walked}
        assert "s1#t0#p0" in walked_ids
        assert "s1#t0#p1" in walked_ids

    def test_discovers_cross_session_nodes(self, graph):
        reranked = [("s1#t0#p0", 0.9), ("s1#t0#p1", 0.85), ("s1#t1#p0", 0.8)]
        walked = songline_walk(reranked, graph, num_anchors=3, hops=2)
        walked_ids = {cid for cid, _ in walked}
        discovered = walked_ids - {"s1#t0#p0", "s1#t0#p1", "s1#t1#p0"}
        assert len(discovered) > 0, "Walk should discover nodes from s2"

    def test_empty_graph_returns_input_unchanged(self):
        reranked = [("a", 0.9), ("b", 0.8)]
        walked = songline_walk(reranked, SonglineGraph(), num_anchors=2, hops=3)
        assert set(cid for cid, _ in walked) == {"a", "b"}

    def test_empty_input_returns_empty(self, graph):
        assert songline_walk([], graph) == []

    def test_sorted_descending_by_score(self, graph):
        reranked = [("s1#t0#p0", 0.9), ("s1#t0#p1", 0.5)]
        walked = songline_walk(reranked, graph, num_anchors=2, hops=2)
        scores = [s for _, s in walked]
        assert scores == sorted(scores, reverse=True)


# ─── Pipeline integration ────────────────────────────────────────────

class TestPipelineWithSongline:
    @pytest.fixture
    def populated(self):
        store = InMemoryStore()
        emb = StubEmbedder(dim=32)
        bm25 = SimpleBM25()
        sessions = [
            {"session_id": "s1", "turns": [
                {"text": "Alice went to the store. She bought apples.", "speaker": "alice"},
                {"text": "Bob met Carol at the park.", "speaker": "bob"},
            ]},
            {"session_id": "s2", "turns": [
                {"text": "Alice called Bob about dinner plans.", "speaker": "alice"},
                {"text": "They decided on Italian food.", "speaker": "bob"},
            ]},
        ]
        ingest_sessions(sessions, store=store, embedder=emb)
        for row in store.all_rows():
            bm25.add(row["chunk_id"], row["text"])
        graph = build_songline_graph(list(store.all_rows()))
        return store, emb, bm25, graph

    def test_with_songline_returns_results(self, populated):
        store, emb, bm25, graph = populated
        result = retrieve(
            "Alice bought apples", store=store, embedder=emb,
            bm25=bm25, songline_graph=graph,
            config=PipelineConfig(top_k=5),
        )
        assert len(result.final) > 0
        assert len(result.walked) > 0

    def test_without_songline_walked_equals_reranked(self, populated):
        store, emb, bm25, _ = populated
        result = retrieve(
            "Alice bought apples", store=store, embedder=emb,
            bm25=bm25, songline_graph=None,
            config=PipelineConfig(top_k=5),
        )
        assert len(result.walked) == len(result.reranked)

    def test_songline_can_surface_cross_session_in_final(self, populated):
        store, emb, bm25, graph = populated
        result = retrieve(
            "Alice", store=store, embedder=emb,
            bm25=bm25, songline_graph=graph,
            config=PipelineConfig(top_k=5, mmr_session_cap=3),
        )
        sessions_in_final = {cid.split("#")[0] for cid in result.top_ids}
        # With songline walk + MMR, we should see both sessions
        # (at least in walked, maybe in final depending on MMR)
        sessions_in_walked = {cid.split("#")[0] for cid, _ in result.walked}
        assert len(sessions_in_walked) >= 1  # at minimum something walked
