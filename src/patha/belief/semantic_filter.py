"""Semantic pre-filter for the belief store.

Problem:
    When the MCP server is asked "what do I currently eat?", it runs
    supersession/summary over EVERY belief in the store. On a store
    with thousands of beliefs from unrelated topics, the false-positive
    rate of the contradiction detector compounds, and the summary is
    polluted with irrelevant material.

Solution:
    Before handing the store to supersession, filter to the top-k
    beliefs whose proposition text is semantically similar to the
    query. This is a lightweight analogue of Phase 1 retrieval —
    it doesn't require building a full LanceDB index; it just embeds
    the belief texts and the query using the embedder that's already
    a project dependency (sentence-transformers MiniLM).

The semantic filter is stateless and cheap: it re-embeds the (usually
small-per-user) belief store on each query. For stores with thousands
of beliefs we'd want a persistent index — deferred until someone
actually hits that scale in practice.

Why not wire full Phase 1:
    Phase 1 has its own proposition store, BM25 index, songline graph,
    and multi-view Vedic encoder. Wiring all that into the MCP server
    is a larger project. This module gets 80% of the benefit (narrow
    context → fewer false-positive supersessions) in ~100 lines.

Usage:
    filter = SemanticBeliefFilter()
    relevant_ids = filter.top_k(
        query="what do I eat?",
        beliefs=store.all(),
        k=20,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from patha.belief.types import Belief, BeliefId


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class _LazyMiniLM:
    """Lazy-loaded MiniLM embedder. Avoids import-time model download."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return np.asarray(
            self._model.encode(texts, normalize_embeddings=True),
            dtype=np.float32,
        )


@dataclass
class SemanticBeliefFilter:
    """Cosine-similarity filter over a set of beliefs.

    Parameters
    ----------
    embedder
        Embedder with `.embed(list[str]) -> np.ndarray` returning
        unit-norm row vectors. Defaults to all-MiniLM-L6-v2.
    min_similarity
        Minimum cosine similarity for a belief to be considered
        relevant. 0.0 disables the floor; 0.35 is a sensible default
        for MiniLM.
    """

    embedder: Embedder | None = None
    min_similarity: float = 0.25

    def __post_init__(self):
        if self.embedder is None:
            self.embedder = _LazyMiniLM()

    def top_k(
        self,
        query: str,
        beliefs: list[Belief],
        k: int = 20,
    ) -> list[BeliefId]:
        """Return the top-k belief ids whose proposition is most similar
        to the query. Filters by min_similarity first, then ranks by cosine.
        Returns fewer than k if not enough beliefs clear the floor.
        """
        if not beliefs:
            return []
        texts = [b.proposition for b in beliefs]
        # Embed query + all propositions in one batch
        vecs = self.embedder.embed([query, *texts])
        query_vec = vecs[0]
        prop_vecs = vecs[1:]
        sims = prop_vecs @ query_vec  # unit-norm rows → dot == cosine

        ranked = sorted(
            zip(beliefs, sims),
            key=lambda bs: float(bs[1]),
            reverse=True,
        )
        keep: list[BeliefId] = []
        for belief, sim in ranked:
            if float(sim) < self.min_similarity:
                break
            keep.append(belief.id)
            if len(keep) >= k:
                break
        return keep

    def top_k_with_scores(
        self,
        query: str,
        beliefs: list[Belief],
        k: int = 20,
    ) -> list[tuple[BeliefId, float]]:
        """Same as top_k but also returns the similarity scores."""
        if not beliefs:
            return []
        texts = [b.proposition for b in beliefs]
        vecs = self.embedder.embed([query, *texts])
        query_vec = vecs[0]
        prop_vecs = vecs[1:]
        sims = prop_vecs @ query_vec
        ranked = sorted(
            zip(beliefs, sims),
            key=lambda bs: float(bs[1]),
            reverse=True,
        )
        out = []
        for belief, sim in ranked:
            s = float(sim)
            if s < self.min_similarity:
                break
            out.append((belief.id, s))
            if len(out) >= k:
                break
        return out
