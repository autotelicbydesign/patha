"""Phase 1 ↔ Phase 2 bridge for the MCP server.

Phase 1 (Vedic 7-view + BM25 + RRF retrieval) is Patha's core
differentiator for paraphrase-robust recall. This bridge runs Phase 1
over the Phase 2 BeliefStore so every MCP query goes through the full
retrieval pipeline instead of a simple cosine-over-all-beliefs filter.

## Design

`LazyPhase1Retriever` is a callable (query, top_k) -> list[belief_id]
that:

  - **Builds lazily.** The 7-view index is constructed on the first
    actual query, not at MCP server startup. Claude Desktop starts
    instantly; the one-time cost (~5-30 s depending on store size)
    is paid on the first retrieval.
  - **Marks dirty on ingest.** After each `patha_ingest`, the bridge's
    `.invalidate()` is called. The next query triggers a rebuild so
    newly-added beliefs are findable.
  - **Returns source_proposition_ids**, not Phase-1 chunk_ids. That's
    the contract `IntegratedPatha.query()` expects for its
    `phase1_retrieve` hook.

## Trade-offs

  - **First-query latency.** A 100-belief store takes ~3 s to index,
    a 1000-belief store ~15 s, on CPU. All subsequent queries within
    the same session are <100 ms.
  - **After-ingest latency.** An ingest followed by a query pays the
    rebuild cost again. Fine in practice — users don't usually ingest
    and query back-to-back.
  - **Memory.** 7x the raw embedding RAM. For a 1000-belief store
    with MiniLM (384-dim, float32), that's ~11 MB. Negligible.

## Why this is the default (as of v0.9)

Earlier versions kept Phase 1 opt-in on the theory that "personal
memory is small, 7-view is overkill." That was wrong:

  1. Phase 1 is the *point*. Vedic multi-view retrieval is Patha's
     differentiator; shipping it off-by-default means shipping a
     watered-down product.
  2. "Overkill" was a guess. We never measured MiniLM cosine vs.
     7-view retrieval on real personal memory. Paraphrase coverage
     is Phase 1's whole thing — the scale at which MiniLM alone
     loses robustness is not clearly large.
  3. Startup cost is manageable via lazy build. Normal users don't
     notice a one-time cost on first query.

If you specifically want the old cosine-only behavior (e.g.,
benchmarking, tiny stores, offline profiling), set
`PATHA_PHASE1=off` in the MCP config.
"""

from __future__ import annotations

import threading
from typing import Callable

from patha.belief.store import BeliefStore
from patha.chunking.propositionizer import Proposition
from patha.chunking.views import build_views, VIEW_NAMES
from patha.indexing.bm25_index import SimpleBM25
from patha.indexing.store import InMemoryStore
from patha.models.embedder_st import SentenceTransformerEmbedder
from patha.retrieval.pipeline import PipelineConfig, retrieve


Phase1Retriever = Callable[[str, int], list[str]]


def _belief_to_proposition(b, turn_idx: int) -> Proposition:
    """Convert a Belief to a Phase-1 Proposition."""
    return Proposition(
        text=b.proposition,
        session_id=b.asserted_in_session,
        turn_idx=turn_idx,
        prop_idx=0,
        timestamp=b.asserted_at.isoformat() if b.asserted_at else None,
    )


def build_phase1_indexes(
    belief_store: BeliefStore,
    *,
    embedder=None,
) -> tuple[InMemoryStore, SimpleBM25, dict[str, str]]:
    """Embed every belief across the 7 Vedic views + build a BM25 index.

    Returns:
        store           — InMemoryStore populated with all beliefs' views
        bm25            — SimpleBM25 index over proposition text
        id_map          — dict mapping Phase-1 chunk_id → BeliefStore source_proposition_id
    """
    if embedder is None:
        embedder = SentenceTransformerEmbedder()

    store = InMemoryStore()
    bm25 = SimpleBM25()
    id_map: dict[str, str] = {}

    beliefs = list(belief_store.all())
    if not beliefs:
        return store, bm25, id_map

    propositions = [_belief_to_proposition(b, i) for i, b in enumerate(beliefs)]
    views_per_prop = build_views(propositions)

    embeddings_per_view: dict[str, list[list[float]]] = {}
    for view_name in VIEW_NAMES:
        view_texts = [views_per_prop[i][view_name] for i in range(len(propositions))]
        embeddings_per_view[view_name] = embedder.embed(view_texts)

    rows: list[dict] = []
    for i, (prop, belief) in enumerate(zip(propositions, beliefs)):
        view_payload = {
            name: {
                "text": views_per_prop[i][name],
                "embedding": embeddings_per_view[name][i],
            }
            for name in VIEW_NAMES
        }
        row = {
            "chunk_id": prop.chunk_id,
            "session_id": prop.session_id,
            "turn_idx": prop.turn_idx,
            "prop_idx": prop.prop_idx,
            "text": prop.text,
            "speaker": prop.speaker,
            "timestamp": prop.timestamp,
            "entities": [],
            "views": view_payload,
        }
        rows.append(row)
        id_map[prop.chunk_id] = belief.source_proposition_id
        bm25.add(prop.chunk_id, prop.text)

    store.upsert(rows)
    return store, bm25, id_map


class LazyPhase1Retriever:
    """Phase 1 retriever that builds indexes on first use and rebuilds
    after invalidation.

    Thread-safe: concurrent builds are serialized via an internal lock
    so only one index-build happens at a time per retriever instance.
    """

    def __init__(
        self,
        belief_store: BeliefStore,
        *,
        embedder=None,
        reranker=None,
        config: PipelineConfig | None = None,
        use_reranker: bool = True,
    ) -> None:
        """
        reranker
            Optional pre-built reranker callable. If None and
            `use_reranker=True` (default), a CrossEncoderReranker is
            lazy-built on first query. Set `use_reranker=False` for
            dense-only retrieval (faster, ~3-5pp less accurate).
        """
        self._belief_store = belief_store
        self._provided_embedder = embedder
        self._embedder = embedder
        self._provided_reranker = reranker
        self._reranker = reranker
        self._use_reranker = use_reranker
        self._config = config or PipelineConfig(top_k=20)
        self._lock = threading.Lock()
        self._indexes: tuple | None = None
        self._dirty = True

    def _ensure_embedder(self):
        if self._embedder is None:
            self._embedder = SentenceTransformerEmbedder()
        return self._embedder

    def _ensure_reranker(self):
        if not self._use_reranker:
            return None
        if self._reranker is None:
            from patha.retrieval.reranker import CrossEncoderReranker
            self._reranker = CrossEncoderReranker(top_n=0)
        return self._reranker

    def invalidate(self) -> None:
        """Mark the indexes stale. Next call rebuilds."""
        with self._lock:
            self._dirty = True

    def _ensure_built(self) -> None:
        if not self._dirty and self._indexes is not None:
            return
        with self._lock:
            if not self._dirty and self._indexes is not None:
                return
            embedder = self._ensure_embedder()
            self._indexes = build_phase1_indexes(
                self._belief_store, embedder=embedder,
            )
            self._dirty = False

    def __call__(self, query: str, top_k: int) -> list[str]:
        self._ensure_built()
        assert self._indexes is not None
        prop_store, bm25, id_map = self._indexes
        if not id_map:
            return []
        embedder = self._ensure_embedder()
        reranker = self._ensure_reranker()
        result = retrieve(
            query, store=prop_store, embedder=embedder, bm25=bm25,
            config=self._config, reranker=reranker,
        )
        chunk_ids = [cid for cid, _ in result.final[:top_k]]
        return [id_map[cid] for cid in chunk_ids if cid in id_map]

    @property
    def is_built(self) -> bool:
        return self._indexes is not None and not self._dirty


def build_phase1_retriever(
    belief_store: BeliefStore,
    *,
    embedder=None,
    config: PipelineConfig | None = None,
) -> Phase1Retriever:
    """Return a lazy Phase 1 retriever (callable + invalidate()).

    Back-compat name. New code can construct `LazyPhase1Retriever`
    directly if it needs the `.invalidate()` handle — the MCP server
    does. For simple callers, this returns a bare callable.
    """
    return LazyPhase1Retriever(
        belief_store, embedder=embedder, config=config,
    )
