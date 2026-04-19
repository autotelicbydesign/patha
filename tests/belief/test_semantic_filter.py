"""Tests for SemanticBeliefFilter — the semantic pre-filter that narrows
the belief store before supersession reasoning in the MCP server.

Uses a stub embedder so these tests are fast and don't download models.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from patha.belief.semantic_filter import SemanticBeliefFilter
from patha.belief.types import (
    Belief,
    Pramana,
    ResolutionStatus,
    Validity,
)


class _KeywordEmbedder:
    """Bag-of-words unit-norm embedder for deterministic testing.
    Encodes each text into a fixed vocabulary one-hot vector, L2-normalised.
    """
    VOCAB = [
        "sushi", "fish", "eat", "raw", "weather", "nice", "today",
        "laptop", "macbook", "pro", "m3", "rent", "monthly",
        "vegetarian", "diet", "meat",
    ]

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for text in texts:
            t = text.lower()
            v = np.array(
                [1.0 if w in t else 0.0 for w in self.VOCAB],
                dtype=np.float32,
            )
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n
            vecs.append(v)
        return np.vstack(vecs) if vecs else np.zeros((0, len(self.VOCAB)))


def _belief(text: str, id_: str) -> Belief:
    return Belief(
        id=id_,
        proposition=text,
        asserted_at=datetime(2024, 1, 1),
        asserted_in_session="s1",
        source_proposition_id=id_,
        confidence=1.0,
        pramana=Pramana.PRATYAKSA,
        validity=Validity(mode="permanent"),
        status=ResolutionStatus.CURRENT,
    )


class TestSemanticFilter:
    def test_top_k_basic(self):
        f = SemanticBeliefFilter(embedder=_KeywordEmbedder(), min_similarity=0.1)
        beliefs = [
            _belief("I love sushi and raw fish", "b1"),
            _belief("the weather is nice today", "b2"),
            _belief("I eat sushi every week", "b3"),
            _belief("my laptop is a macbook pro", "b4"),
        ]
        ids = f.top_k(query="do I eat sushi?", beliefs=beliefs, k=2)
        assert "b3" in ids  # most similar
        assert "b1" in ids
        assert "b4" not in ids
        assert "b2" not in ids

    def test_min_similarity_floor(self):
        f = SemanticBeliefFilter(embedder=_KeywordEmbedder(), min_similarity=0.9)
        beliefs = [
            _belief("completely unrelated topic", "b1"),
            _belief("another unrelated one", "b2"),
        ]
        ids = f.top_k(query="do I eat sushi?", beliefs=beliefs, k=5)
        assert ids == []

    def test_empty_beliefs(self):
        f = SemanticBeliefFilter(embedder=_KeywordEmbedder())
        assert f.top_k(query="anything", beliefs=[], k=5) == []

    def test_k_cap(self):
        f = SemanticBeliefFilter(embedder=_KeywordEmbedder(), min_similarity=0.0)
        beliefs = [
            _belief(f"sushi text {i}", f"b{i}") for i in range(10)
        ]
        ids = f.top_k(query="sushi", beliefs=beliefs, k=3)
        assert len(ids) == 3

    def test_scores_returned(self):
        f = SemanticBeliefFilter(embedder=_KeywordEmbedder(), min_similarity=0.1)
        beliefs = [
            _belief("I eat sushi every week", "b1"),
            _belief("the weather is nice", "b2"),
        ]
        scored = f.top_k_with_scores(query="do I eat sushi?", beliefs=beliefs, k=5)
        assert len(scored) >= 1
        assert scored[0][0] == "b1"  # sushi belief ranks first
        assert scored[0][1] > 0     # non-zero similarity
        # Scores are descending
        if len(scored) > 1:
            assert scored[0][1] >= scored[1][1]

    def test_ranking_order(self):
        """Higher overlap → higher rank."""
        f = SemanticBeliefFilter(embedder=_KeywordEmbedder(), min_similarity=0.0)
        beliefs = [
            _belief("sushi fish raw", "high"),     # 3 overlap terms
            _belief("sushi", "mid"),                # 1 overlap term
            _belief("random", "low"),               # 0 overlap terms
        ]
        ids = f.top_k(query="sushi fish raw", beliefs=beliefs, k=3)
        assert ids[0] == "high"
        assert ids[1] == "mid"
