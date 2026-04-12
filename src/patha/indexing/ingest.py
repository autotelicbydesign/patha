"""Ingestion orchestrator.

Turns raw conversation data into indexed rows in the store. The pipeline is:

    raw turn text
        -> propositionize (deterministic, rule-based)
        -> build 7 Vedic patha views per proposition
        -> embed each view across all propositions in the turn (7 batched calls)
        -> assemble row dicts
        -> store.upsert

No LLM is invoked anywhere. Everything is deterministic given a fixed
embedder and input, so re-ingestion of the same corpus produces a
bit-identical index.

Entity and timestamp enrichment are optional. When missing, the views
module degrades gracefully (v5→v4, v6→v1, v7→v1), so ingestion is still
complete and the schema is constant. Real NER / temporal parsing is added
later in ``query/entities.py`` and ``query/temporal.py``.
"""

from __future__ import annotations

from typing import Iterable

from patha.chunking.propositionizer import Proposition, propositionize
from patha.chunking.views import VIEW_NAMES, build_views
from patha.indexing.store import Store
from patha.models.embedder import Embedder


def ingest_turn(
    text: str,
    *,
    session_id: str,
    turn_idx: int,
    store: Store,
    embedder: Embedder,
    speaker: str | None = None,
    timestamp: str | None = None,
    entities_per_prop: list[list[str]] | None = None,
    entity_enricher: object | None = None,
) -> list[str]:
    """Ingest a single turn into the store.

    Returns the list of ``chunk_id``s created for this turn. Empty list if
    the turn produced no propositions (e.g. empty or whitespace-only input).

    Parameters
    ----------
    entity_enricher
        Optional ``EntityEnricher`` instance. When provided and
        ``entities_per_prop`` is ``None``, entities are extracted
        automatically via spaCy NER.
    """
    props = propositionize(
        text,
        session_id=session_id,
        turn_idx=turn_idx,
        speaker=speaker,
        timestamp=timestamp,
    )
    if not props:
        return []

    if entities_per_prop is not None and len(entities_per_prop) != len(props):
        raise ValueError(
            f"entities_per_prop length {len(entities_per_prop)} "
            f"does not match proposition count {len(props)}"
        )

    # Auto-extract entities if enricher provided and no manual entities
    if entities_per_prop is None and entity_enricher is not None:
        prop_texts = [p.text for p in props]
        entities_per_prop = entity_enricher.extract_batch(prop_texts)

    timestamps = [timestamp] * len(props) if timestamp else None
    views_per_prop = build_views(
        props,
        entities=entities_per_prop,
        timestamps=timestamps,
    )

    # Batch one embedder call per view (7 total), each over all propositions
    # in the turn. For the stub embedder this is equivalent to per-text calls;
    # for real GPU embedders this maximizes batch utilization.
    embeddings_per_view: dict[str, list[list[float]]] = {}
    for view_name in VIEW_NAMES:
        view_texts = [views_per_prop[i][view_name] for i in range(len(props))]
        embeddings_per_view[view_name] = embedder.embed(view_texts)

    rows: list[dict] = []
    chunk_ids: list[str] = []
    for i, prop in enumerate(props):
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
            "entities": entities_per_prop[i] if entities_per_prop else [],
            "views": view_payload,
        }
        rows.append(row)
        chunk_ids.append(prop.chunk_id)

    store.upsert(rows)
    return chunk_ids


def ingest_session(
    turns: list[dict],
    *,
    session_id: str,
    store: Store,
    embedder: Embedder,
    entity_enricher: object | None = None,
) -> list[str]:
    """Ingest a whole session.

    ``turns`` is a list of dicts with at least a ``text`` key and optional
    ``speaker``, ``timestamp``, ``entities_per_prop`` keys. ``turn_idx`` is
    assigned by position in the list. Returns the flat list of chunk_ids
    created across all turns.

    When ``entity_enricher`` is provided, entities are automatically
    extracted for each proposition via spaCy NER.
    """
    all_ids: list[str] = []
    for turn_idx, turn in enumerate(turns):
        ids = ingest_turn(
            turn["text"],
            session_id=session_id,
            turn_idx=turn_idx,
            store=store,
            embedder=embedder,
            speaker=turn.get("speaker"),
            timestamp=turn.get("timestamp"),
            entities_per_prop=turn.get("entities_per_prop"),
            entity_enricher=entity_enricher,
        )
        all_ids.extend(ids)
    return all_ids


def ingest_sessions(
    sessions: Iterable[dict],
    *,
    store: Store,
    embedder: Embedder,
    entity_enricher: object | None = None,
) -> dict[str, list[str]]:
    """Ingest multiple sessions.

    ``sessions`` is an iterable of dicts with ``session_id`` and ``turns`` keys.
    Returns a map from session_id to the list of chunk_ids created for that
    session.
    """
    result: dict[str, list[str]] = {}
    for session in sessions:
        sid = session["session_id"]
        result[sid] = ingest_session(
            session["turns"],
            session_id=sid,
            store=store,
            embedder=embedder,
            entity_enricher=entity_enricher,
        )
    return result
