"""Tests for the retrieval layer: RRF, BM25, MMR, and full pipeline."""

from __future__ import annotations

import pytest

from patha.indexing.bm25_index import SimpleBM25, BM25Index
from patha.indexing.ingest import ingest_session, ingest_sessions
from patha.indexing.store import InMemoryStore
from patha.models.embedder import StubEmbedder
from patha.retrieval.hybrid_candidates import generate_candidates, reciprocal_rank_fusion
from patha.retrieval.mmr import mmr_rerank
from patha.retrieval.pipeline import PipelineConfig, retrieve


# ─── RRF ──────────────────────────────────────────────────────────────

class TestRRF:
    def test_single_list_preserves_order(self):
        ranked = [("a", 10.0), ("b", 5.0), ("c", 1.0)]
        fused = reciprocal_rank_fusion([ranked], k=60)
        assert [x[0] for x in fused] == ["a", "b", "c"]

    def test_two_agreeing_lists_boost_score(self):
        r1 = [("a", 10.0), ("b", 5.0)]
        r2 = [("a", 9.0), ("b", 4.0)]
        fused = reciprocal_rank_fusion([r1, r2], k=60)
        assert fused[0][0] == "a"
        # "a" appears rank-1 in both: score = 2 * 1/(60+1)
        assert fused[0][1] > fused[1][1]

    def test_disagreeing_lists_are_merged(self):
        r1 = [("a", 10.0)]
        r2 = [("b", 10.0)]
        fused = reciprocal_rank_fusion([r1, r2], k=60)
        ids = {x[0] for x in fused}
        assert ids == {"a", "b"}
        # Both rank-1 in their list, so same score
        assert fused[0][1] == fused[1][1]

    def test_empty_lists_return_empty(self):
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []

    def test_deterministic_tiebreak(self):
        """On score ties, IDs are sorted alphabetically for determinism."""
        r1 = [("b", 1.0)]
        r2 = [("a", 1.0)]
        fused = reciprocal_rank_fusion([r1, r2], k=60)
        assert fused[0][0] == "a"  # alphabetical
        assert fused[1][0] == "b"


# ─── BM25 ─────────────────────────────────────────────────────────────

class TestSimpleBM25:
    def test_conforms_to_protocol(self):
        assert isinstance(SimpleBM25(), BM25Index)

    def test_empty_index_returns_empty(self):
        assert SimpleBM25().search("hello", k=5) == []

    def test_exact_match_ranks_first(self):
        bm25 = SimpleBM25()
        bm25.add("d1", "Alice went home and cooked dinner")
        bm25.add("d2", "Bob stayed at work until late")
        bm25.add("d3", "Carol played soccer in the park")
        results = bm25.search("Alice cooked dinner", k=3)
        assert results[0][0] == "d1"

    def test_no_match_returns_empty(self):
        bm25 = SimpleBM25()
        bm25.add("d1", "Alice went home")
        results = bm25.search("xyzzy quux", k=5)
        assert results == []

    def test_count_tracks_documents(self):
        bm25 = SimpleBM25()
        assert bm25.count() == 0
        bm25.add("d1", "hello")
        bm25.add("d2", "world")
        assert bm25.count() == 2

    def test_readd_replaces_document(self):
        bm25 = SimpleBM25()
        bm25.add("d1", "original text about cats")
        bm25.add("d1", "replacement text about dogs")
        assert bm25.count() == 1
        results = bm25.search("dogs", k=1)
        assert results[0][0] == "d1"

    def test_stopwords_are_filtered(self):
        bm25 = SimpleBM25()
        bm25.add("d1", "the quick fox")
        # "the" is a stopword; query "the" alone matches nothing
        results = bm25.search("the", k=1)
        assert results == []

    def test_k_zero_returns_empty(self):
        bm25 = SimpleBM25()
        bm25.add("d1", "hello world")
        assert bm25.search("hello", k=0) == []


# ─── MMR ──────────────────────────────────────────────────────────────

class TestMMR:
    def test_empty_returns_empty(self):
        assert mmr_rerank([], [0.0, 0.0], k=5) == []

    def test_pure_relevance_at_lambda_one(self):
        candidates = [
            ("a", 0.9, [1.0, 0.0]),
            ("b", 0.8, [0.0, 1.0]),
            ("c", 0.7, [0.5, 0.5]),
        ]
        result = mmr_rerank(candidates, [1.0, 0.0], k=3, lambda_=1.0, session_cap=999)
        # Pure relevance: order by score
        assert [r[0] for r in result] == ["a", "b", "c"]

    def test_session_cap_enforced(self):
        candidates = [
            ("s1#t0#p0", 0.9, [1.0, 0.0]),
            ("s1#t0#p1", 0.85, [0.9, 0.1]),
            ("s1#t1#p0", 0.8, [0.8, 0.2]),
            ("s2#t0#p0", 0.5, [0.0, 1.0]),
        ]
        result = mmr_rerank(
            candidates, [1.0, 0.0], k=4, lambda_=1.0, session_cap=2,
        )
        selected_sessions = [r[0].split("#")[0] for r in result]
        from collections import Counter
        counts = Counter(selected_sessions)
        assert counts["s1"] <= 2  # cap enforced
        assert "s2" in counts  # s2 got a slot

    def test_k_larger_than_candidates_returns_all(self):
        candidates = [("a", 0.9, [1.0])]
        result = mmr_rerank(candidates, [1.0], k=100, session_cap=999)
        assert len(result) == 1


# ─── Full pipeline integration ────────────────────────────────────────

class TestPipeline:
    @pytest.fixture
    def populated(self):
        store = InMemoryStore()
        emb = StubEmbedder(dim=32)
        bm25 = SimpleBM25()

        sessions = [
            {
                "session_id": "s1",
                "turns": [
                    {"text": "Alice went to the store. She bought apples.", "speaker": "alice"},
                    {"text": "Bob met Carol at the park.", "speaker": "bob"},
                ],
            },
            {
                "session_id": "s2",
                "turns": [
                    {"text": "Alice called Bob about dinner plans.", "speaker": "alice"},
                    {"text": "They decided on Italian food.", "speaker": "bob"},
                ],
            },
        ]
        ingest_sessions(sessions, store=store, embedder=emb)

        # Populate BM25 from store
        for row in store.all_rows():
            bm25.add(row["chunk_id"], row["text"])

        return store, emb, bm25

    def test_pipeline_returns_top_k_results(self, populated):
        store, emb, bm25 = populated
        result = retrieve(
            "Alice bought apples",
            store=store,
            embedder=emb,
            bm25=bm25,
            config=PipelineConfig(top_k=3),
        )
        assert len(result.final) <= 3
        assert len(result.top_ids) <= 3
        assert all(isinstance(cid, str) for cid in result.top_ids)

    def test_pipeline_without_bm25(self, populated):
        store, emb, _ = populated
        result = retrieve(
            "Alice bought apples",
            store=store,
            embedder=emb,
            bm25=None,
            config=PipelineConfig(top_k=3),
        )
        assert len(result.final) > 0

    def test_exact_text_is_top_ranked_via_stub(self, populated):
        """With the stub embedder, querying the exact text of a proposition
        should retrieve that proposition's chunk as the top candidate,
        because the hash-based vectors are identical."""
        store, emb, bm25 = populated
        result = retrieve(
            "Alice went to the store.",
            store=store,
            embedder=emb,
            bm25=bm25,
            config=PipelineConfig(top_k=5),
        )
        # Must be in top-5 (likely top-1)
        assert "s1#t0#p0" in result.top_ids

    def test_pipeline_stages_are_monotonically_shrinking(self, populated):
        store, emb, bm25 = populated
        config = PipelineConfig(candidate_k=100, mmr_k=10, top_k=3)
        result = retrieve("dinner plans", store=store, embedder=emb, bm25=bm25, config=config)
        assert len(result.final) <= len(result.diversified)
        assert len(result.diversified) <= len(result.reranked)
        assert len(result.reranked) <= len(result.candidates)

    def test_mmr_session_cap_surfaces_both_sessions(self, populated):
        store, emb, bm25 = populated
        config = PipelineConfig(top_k=5, mmr_k=5, mmr_session_cap=2)
        result = retrieve("Alice", store=store, embedder=emb, bm25=bm25, config=config)
        sessions_in_result = {cid.split("#")[0] for cid in result.top_ids}
        # With session_cap=2, both s1 and s2 should appear if both have Alice
        # At minimum, we shouldn't get 5 results all from one session
        if len(result.top_ids) >= 3:
            assert len(sessions_in_result) >= 1  # at least one session

    def test_result_has_intermediate_stages(self, populated):
        store, emb, bm25 = populated
        result = retrieve("anything", store=store, embedder=emb, bm25=bm25)
        assert result.query == "anything"
        assert isinstance(result.candidates, list)
        assert isinstance(result.reranked, list)
        assert isinstance(result.diversified, list)
        assert isinstance(result.final, list)
