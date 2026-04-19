"""Phase 1 ↔ Phase 2 bridge for the MCP server.

Phase 1 (Vedic 7-view + BM25 + songline + RRF) is designed for corpus-
scale retrieval (LongMemEval, thousands of turns). Phase 2's BeliefStore
is designed for personal-memory scale (hundreds to low thousands of
beliefs). At personal scale, the lightweight `semantic_filter` is
usually sufficient; full Phase 1 is overkill.

This module exists for the minority of users whose belief store grows
beyond the scale where cosine-over-all-beliefs stays cheap. It rebuilds
Phase 1's in-memory indexes from the existing BeliefStore so no
separate corpus ingestion is needed.

Usage:
    from patha.belief.store import BeliefStore
    from patha.phase1_bridge import build_phase1_retriever

    belief_store = BeliefStore(persistence_path=Path("~/.patha/beliefs.jsonl"))
    retrieve = build_phase1_retriever(belief_store)
    patha = IntegratedPatha(belief_layer=..., phase1_retrieve=retrieve)
    # IntegratedPatha.query() now uses Phase 1 retrieval under the hood.

Trade-offs:
  - Startup cost: embeds every belief across 7 views on first call
    (~5-60 seconds depending on store size and device).
  - Memory: ~7x the raw text size in embedding RAM.
  - Correctness: the bridge is read-only from BeliefStore's POV, so
    concurrent ingest is safe (but won't update the index until you
    rebuild).
"""

from __future__ import annotations

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
        id_map          — dict mapping Phase-1 chunk_id → BeliefStore belief id
    """
    if embedder is None:
        embedder = SentenceTransformerEmbedder()

    store = InMemoryStore()
    bm25 = SimpleBM25()
    id_map: dict[str, str] = {}

    beliefs = list(belief_store.all())
    if not beliefs:
        return store, bm25, id_map

    # Build propositions + views for every belief
    propositions = [_belief_to_proposition(b, i) for i, b in enumerate(beliefs)]
    views_per_prop = build_views(propositions)

    # Embed once per view, batched across all propositions
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


def build_phase1_retriever(
    belief_store: BeliefStore,
    *,
    embedder=None,
    config: PipelineConfig | None = None,
) -> Phase1Retriever:
    """Return a callable (query, top_k) -> list[source_proposition_id].

    The returned callable matches IntegratedPatha's `phase1_retrieve`
    contract. It runs the full Phase 1 pipeline (7-view dense + BM25
    → RRF → MMR) and maps Phase-1 chunk_ids back to the
    source_proposition_ids the BeliefStore tracks.
    """
    if embedder is None:
        embedder = SentenceTransformerEmbedder()

    prop_store, bm25, id_map = build_phase1_indexes(
        belief_store, embedder=embedder,
    )
    cfg = config or PipelineConfig(top_k=20)

    def _retrieve(query: str, top_k: int) -> list[str]:
        if not id_map:
            return []
        result = retrieve(
            query, store=prop_store, embedder=embedder, bm25=bm25,
            config=cfg,
        )
        chunk_ids = [cid for cid, _ in result.final[:top_k]]
        # Map Phase-1 chunk_ids back to Phase-2 source_proposition_ids
        return [id_map[cid] for cid in chunk_ids if cid in id_map]

    return _retrieve
