"""Hybrid candidate generation: 7 dense Patha views + BM25 fused via RRF.

This is the first stage of the retrieval pipeline. It generates a broad
candidate pool by running the query against all 7 embedding views in the
store plus the BM25 sparse index, then fusing the 8 ranked lists via
Reciprocal Rank Fusion (RRF). The output is a single list of
``(chunk_id, rrf_score)`` pairs sorted by fused score, ready for the
cross-encoder reranker.

RRF is the fusion method because it is:
- rank-based (no score normalization needed across heterogeneous retrievers),
- proven to match or beat score-level fusion on BEIR-style tasks,
- parameter-free except for ``k`` (the smoothing constant, default 60).

The Vedic insight: each view is an independent "fingerprint" of the same
proposition from a different contextual angle. RRF treats them as
independent voters. If *any one* view recalls a chunk, it enters the
candidate pool — the same redundancy principle as *jata/ghana* recitation,
where any positional context can verify a word.
"""

from __future__ import annotations

from typing import Sequence

from patha.chunking.views import VIEW_NAMES
from patha.indexing.bm25_index import BM25Index
from patha.indexing.store import Store
from patha.models.embedder import Embedder


def reciprocal_rank_fusion(
    ranked_lists: Sequence[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists into one via RRF.

    For each item appearing in any list, its RRF score is the sum of
    ``1 / (k + rank)`` across all lists where it appears. ``rank`` is
    1-indexed. Items not in a list receive no contribution from that list.

    Parameters
    ----------
    ranked_lists
        Each element is a list of ``(id, score)`` pairs sorted descending
        by score. The raw scores are ignored — only rank order matters.
    k
        Smoothing constant. Default 60 (standard in the literature).

    Returns
    -------
    list[tuple[str, float]]
        Fused ``(id, rrf_score)`` pairs sorted descending by RRF score.
        Ties are broken by alphabetical id for determinism.
    """
    scores: dict[str, float] = {}
    for rlist in ranked_lists:
        for rank_0, (item_id, _raw_score) in enumerate(rlist):
            rank = rank_0 + 1  # 1-indexed
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return fused


def generate_candidates(
    query: str,
    *,
    store: Store,
    embedder: Embedder,
    bm25: BM25Index | None = None,
    per_view_k: int = 300,
    bm25_k: int = 300,
    total_k: int = 2000,
    rrf_k: int = 60,
) -> list[tuple[str, float]]:
    """Generate the hybrid candidate pool for a single query.

    1. Embed the query once.
    2. Search each of the 7 views in the store (dense retrieval).
    3. Search the BM25 index (sparse retrieval), if provided.
    4. Fuse all 7 (or 8) ranked lists via RRF.
    5. Return the top ``total_k`` candidates.

    Parameters
    ----------
    query
        The raw query string.
    store
        Proposition store with multi-view embeddings.
    embedder
        Same embedder used at ingest time — critical for vector alignment.
    bm25
        Optional BM25 index. When ``None``, only dense views are used.
    per_view_k
        How many candidates to pull from each dense view. Default 300.
        With 7 views, this gives up to 2100 unique candidates before fusion.
    bm25_k
        How many candidates to pull from BM25. Default 300.
    total_k
        Maximum number of candidates to return after fusion. Default 2000.
    rrf_k
        RRF smoothing constant. Default 60.

    Returns
    -------
    list[tuple[str, float]]
        ``(chunk_id, rrf_score)`` pairs sorted descending by score.
    """
    query_vec = embedder.embed([query])[0]

    ranked_lists: list[list[tuple[str, float]]] = []

    # Dense: one search per Patha view
    for view_name in VIEW_NAMES:
        hits = store.search_view(view_name, query_vec, k=per_view_k)
        ranked_lists.append(hits)

    # Sparse: BM25
    if bm25 is not None:
        bm25_hits = bm25.search(query, k=bm25_k)
        ranked_lists.append(bm25_hits)

    fused = reciprocal_rank_fusion(ranked_lists, k=rrf_k)
    return fused[:total_k]
