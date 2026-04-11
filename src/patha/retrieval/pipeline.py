"""Full retrieval pipeline orchestrator.

Wires the complete query path from the plan:

    query
      -> hybrid candidates (7 dense views + BM25 via RRF)    [2000]
      -> pointwise rerank (stub/Qwen3)                       [100]
      -> ColBERT verification (stub/PLAID)                    [100]
      -> songline walks from top-3 anchors                    [~140]
      -> MMR diversity pass with session cap                  [30]
      -> listwise rerank                                      [5]
      -> Hafiz ensemble check                                 [5]

Each stage is a pluggable callable so we can ablate by replacing any
stage with an identity function. The pipeline itself is a pure function
with no side effects.

Currently implemented stages:
  1. hybrid_candidates (RRF over 7 views + BM25)
  2. pointwise reranker (pluggable, identity stub by default)
  3. songline walks from top-3 anchors (Pillar 2 — Aboriginal)
  4. MMR diversity with session cap
Still pass-through stubs: ColBERT verification, listwise rerank, Hafiz ensemble.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from patha.chunking.views import VIEW_NAMES
from patha.indexing.bm25_index import BM25Index
from patha.indexing.store import Store
from patha.models.embedder import Embedder
from patha.indexing.songline_graph import SonglineGraph
from patha.retrieval.hybrid_candidates import generate_candidates
from patha.retrieval.mmr import mmr_rerank
from patha.retrieval.songline_walker import songline_walk


@dataclass
class PipelineConfig:
    """Knobs for each retrieval stage."""

    # Candidate generation
    per_view_k: int = 300
    bm25_k: int = 300
    candidate_k: int = 2000
    rrf_k: int = 60

    # Songline walks
    songline_anchors: int = 3
    songline_hops: int = 3
    songline_branch: int = 5
    songline_bonus: float = 0.05

    # MMR diversity
    mmr_k: int = 30
    mmr_lambda: float = 0.7
    mmr_session_cap: int = 2

    # Final output
    top_k: int = 5


@dataclass
class RetrievalResult:
    """Output of a single query through the pipeline."""

    query: str
    candidates: list[tuple[str, float]] = field(default_factory=list)
    reranked: list[tuple[str, float]] = field(default_factory=list)
    walked: list[tuple[str, float]] = field(default_factory=list)
    diversified: list[tuple[str, float]] = field(default_factory=list)
    final: list[tuple[str, float]] = field(default_factory=list)

    @property
    def top_ids(self) -> list[str]:
        """Chunk IDs in the final top-K, in rank order."""
        return [cid for cid, _ in self.final]


# Type alias for a reranker function. Takes (query, candidates as
# [(chunk_id, score)], store) and returns re-scored [(chunk_id, score)].
Reranker = Callable[
    [str, list[tuple[str, float]], Store],
    list[tuple[str, float]],
]


def _identity_reranker(
    query: str,
    candidates: list[tuple[str, float]],
    store: Store,
) -> list[tuple[str, float]]:
    """Pass-through: return candidates unchanged. Used as a stub."""
    return candidates


def retrieve(
    query: str,
    *,
    store: Store,
    embedder: Embedder,
    bm25: BM25Index | None = None,
    songline_graph: SonglineGraph | None = None,
    config: PipelineConfig | None = None,
    reranker: Reranker | None = None,
) -> RetrievalResult:
    """Run the full retrieval pipeline for a single query.

    Parameters
    ----------
    query
        Raw query string.
    store
        Proposition store with multi-view embeddings.
    embedder
        Same embedder used at ingest time.
    bm25
        Optional BM25 index.
    songline_graph
        Optional songline multi-modal graph. When provided, the walker
        augments the reranked list with graph-discovered neighbors before
        MMR diversity. When ``None``, the walk stage is skipped.
    config
        Pipeline configuration. Defaults to ``PipelineConfig()``.
    reranker
        Optional reranker function. When ``None``, the pointwise rerank
        stage is an identity pass-through (candidates flow straight to MMR).

    Returns
    -------
    RetrievalResult
        Contains intermediate results at each stage for ablation analysis.
    """
    if config is None:
        config = PipelineConfig()
    if reranker is None:
        reranker = _identity_reranker

    result = RetrievalResult(query=query)

    # Stage 1: hybrid candidate generation (7 views + BM25, RRF)
    result.candidates = generate_candidates(
        query,
        store=store,
        embedder=embedder,
        bm25=bm25,
        per_view_k=config.per_view_k,
        bm25_k=config.bm25_k,
        total_k=config.candidate_k,
        rrf_k=config.rrf_k,
    )

    # Stage 2: pointwise rerank (stub or real cross-encoder)
    result.reranked = reranker(query, result.candidates, store)

    # Stage 3: songline walks from top-N anchors (Pillar 2 — Aboriginal)
    if songline_graph is not None:
        result.walked = songline_walk(
            result.reranked,
            songline_graph,
            num_anchors=config.songline_anchors,
            hops=config.songline_hops,
            max_branch=config.songline_branch,
            walk_bonus=config.songline_bonus,
        )
    else:
        result.walked = list(result.reranked)

    # Stage 4: MMR diversity with session cap
    # We need embeddings for MMR. Pull v1 embedding from the store.
    mmr_input: list[tuple[str, float, list[float]]] = []
    for cid, score in result.walked:
        row = store.get(cid)
        if row is not None:
            vec = row["views"]["v1"]["embedding"]
            mmr_input.append((cid, score, vec))

    query_vec = embedder.embed([query])[0]
    result.diversified = mmr_rerank(
        mmr_input,
        query_vec,
        k=config.mmr_k,
        lambda_=config.mmr_lambda,
        session_cap=config.mmr_session_cap,
    )

    # Stage 5+6: listwise rerank + Hafiz ensemble — not yet implemented.
    # For now, just take the top-K from MMR output.
    result.final = result.diversified[: config.top_k]

    return result
