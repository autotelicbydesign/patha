"""Tests for the cross-encoder reranker (unit tests with mock, no real model)."""

from __future__ import annotations

from patha.chunking.views import VIEW_NAMES
from patha.indexing.store import InMemoryStore


def _make_row(chunk_id: str, text: str, vec: list[float]) -> dict:
    return {
        "chunk_id": chunk_id,
        "session_id": chunk_id.split("#")[0],
        "turn_idx": 0,
        "prop_idx": 0,
        "text": text,
        "speaker": None,
        "timestamp": None,
        "entities": [],
        "views": {
            name: {"text": text, "embedding": vec}
            for name in VIEW_NAMES
        },
    }


class TestRerankerInterface:
    """Test the reranker callable interface without loading a real model."""

    def test_reranker_module_imports(self):
        """Verify the reranker module is importable."""
        from patha.retrieval import reranker
        assert hasattr(reranker, "CrossEncoderReranker")

    def test_identity_reranker_passthrough(self):
        """The pipeline's identity reranker returns candidates unchanged."""
        from patha.retrieval.pipeline import _identity_reranker

        candidates = [("a", 0.9), ("b", 0.5), ("c", 0.3)]
        store = InMemoryStore()
        result = _identity_reranker("query", candidates, store)
        assert result == candidates

    def test_reranker_callable_signature(self):
        """A mock reranker with the correct signature works in the pipeline."""
        from patha.models.embedder import StubEmbedder
        from patha.indexing.store import InMemoryStore
        from patha.retrieval.pipeline import retrieve, PipelineConfig

        store = InMemoryStore()
        embedder = StubEmbedder(dim=8)
        vec = embedder.embed(["test text"])[0]
        store.upsert([
            _make_row("s1#t0#p0", "test text", vec),
            _make_row("s2#t0#p0", "other text", embedder.embed(["other text"])[0]),
        ])

        # Mock reranker that reverses the order
        def mock_reranker(query, candidates, store):
            return list(reversed(candidates))

        config = PipelineConfig(per_view_k=10, bm25_k=10, candidate_k=10,
                                mmr_k=5, top_k=2)
        result = retrieve(
            "test text",
            store=store,
            embedder=embedder,
            config=config,
            reranker=mock_reranker,
        )
        # Should run without error and return results
        assert len(result.top_ids) <= 2
        assert len(result.reranked) > 0
