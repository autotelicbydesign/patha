"""End-to-end ingestion tests.

These are the first real integration tests: propositionize -> views ->
embed -> store, all wired together with the stub embedder and in-memory
store. They verify row shape, the multi-view schema, determinism of
re-ingestion, and that exact-text search round-trips through the stub
embedder's hash determinism.
"""

from __future__ import annotations

import pytest

from patha.chunking.views import VIEW_NAMES
from patha.indexing.ingest import ingest_session, ingest_sessions, ingest_turn
from patha.indexing.store import InMemoryStore
from patha.models.embedder import StubEmbedder


@pytest.fixture
def fresh():
    return InMemoryStore(), StubEmbedder(dim=48)


def test_ingest_empty_turn_creates_no_rows(fresh):
    store, emb = fresh
    ids = ingest_turn("", session_id="s1", turn_idx=0, store=store, embedder=emb)
    assert ids == []
    assert store.count() == 0


def test_ingest_single_sentence_creates_one_row(fresh):
    store, emb = fresh
    ids = ingest_turn("Hello world.", session_id="s1", turn_idx=0, store=store, embedder=emb)
    assert len(ids) == 1
    assert ids[0] == "s1#t0#p0"
    assert store.count() == 1


def test_ingest_multi_sentence_creates_one_row_per_proposition(fresh):
    store, emb = fresh
    ids = ingest_turn(
        "Alice went home. Bob stayed behind. Carol left.",
        session_id="s1",
        turn_idx=2,
        store=store,
        embedder=emb,
    )
    assert len(ids) == 3
    assert ids == ["s1#t2#p0", "s1#t2#p1", "s1#t2#p2"]


def test_ingested_row_has_full_seven_view_schema(fresh):
    store, emb = fresh
    ingest_turn("Hello.", session_id="s1", turn_idx=0, store=store, embedder=emb)
    row = store.get("s1#t0#p0")
    assert row is not None
    assert set(row["views"].keys()) == set(VIEW_NAMES)
    for name in VIEW_NAMES:
        assert "text" in row["views"][name]
        assert "embedding" in row["views"][name]
        assert len(row["views"][name]["embedding"]) == 48


def test_ingest_preserves_metadata(fresh):
    store, emb = fresh
    ingest_turn(
        "Hi there.",
        session_id="s5",
        turn_idx=3,
        store=store,
        embedder=emb,
        speaker="alice",
        timestamp="2026-04-10T14:00:00Z",
    )
    row = store.get("s5#t3#p0")
    assert row["speaker"] == "alice"
    assert row["timestamp"] == "2026-04-10T14:00:00Z"
    assert row["session_id"] == "s5"
    assert row["turn_idx"] == 3


def test_ingest_with_entities_populates_v5_and_v6(fresh):
    store, emb = fresh
    ingest_turn(
        "Alice went home.",
        session_id="s1",
        turn_idx=0,
        store=store,
        embedder=emb,
        entities_per_prop=[["Alice"]],
    )
    row = store.get("s1#t0#p0")
    assert row["entities"] == ["Alice"]
    assert "Alice" in row["views"]["v5"]["text"]
    assert row["views"]["v6"]["text"] == "fact about Alice: Alice went home."


def test_entities_length_mismatch_raises(fresh):
    store, emb = fresh
    with pytest.raises(ValueError, match="entities_per_prop length"):
        ingest_turn(
            "A. B.",
            session_id="s1",
            turn_idx=0,
            store=store,
            embedder=emb,
            entities_per_prop=[["X"]],  # only 1, but "A. B." yields 2 props
        )


def test_reingestion_is_idempotent(fresh):
    """Same input + same embedder -> same rows. This is the Vedic determinism guarantee."""
    store, emb = fresh
    ids_first = ingest_turn(
        "Alpha. Beta. Gamma.",
        session_id="s1",
        turn_idx=0,
        store=store,
        embedder=emb,
    )
    snapshot_first = {cid: store.get(cid) for cid in ids_first}

    ids_second = ingest_turn(
        "Alpha. Beta. Gamma.",
        session_id="s1",
        turn_idx=0,
        store=store,
        embedder=emb,
    )
    assert ids_first == ids_second
    for cid in ids_second:
        assert store.get(cid) == snapshot_first[cid]
    assert store.count() == 3


def test_exact_text_search_roundtrips_via_stub_determinism(fresh):
    """Critical property: embedding the exact same text as a view should
    retrieve that chunk as top-1 with cosine ~1.0, because StubEmbedder is
    a pure hash and identical strings map to identical vectors."""
    store, emb = fresh
    ingest_turn(
        "Alice went home. Bob stayed.",
        session_id="s1",
        turn_idx=0,
        store=store,
        embedder=emb,
    )
    # v1 of the first prop is literally "Alice went home."
    query_vec = emb.embed(["Alice went home."])[0]
    results = store.search_view("v1", query_vec, k=2)
    assert len(results) == 2
    assert results[0][0] == "s1#t0#p0"
    assert results[0][1] > 0.999  # essentially 1.0


def test_ingest_session_multiple_turns(fresh):
    store, emb = fresh
    turns = [
        {"text": "Hi there.", "speaker": "alice"},
        {"text": "How are you? I am well.", "speaker": "bob"},
        {"text": "Great to hear.", "speaker": "alice"},
    ]
    ids = ingest_session(turns, session_id="conv1", store=store, embedder=emb)
    # turn 0: 1 prop, turn 1: 2 props, turn 2: 1 prop -> 4 total
    assert len(ids) == 4
    assert store.count() == 4
    # verify turn indices were assigned correctly
    assert store.get("conv1#t0#p0")["speaker"] == "alice"
    assert store.get("conv1#t1#p1")["speaker"] == "bob"
    assert store.get("conv1#t2#p0")["speaker"] == "alice"


def test_ingest_sessions_multiple_sessions(fresh):
    store, emb = fresh
    sessions = [
        {
            "session_id": "s1",
            "turns": [{"text": "First session, first turn."}],
        },
        {
            "session_id": "s2",
            "turns": [
                {"text": "Second session begins."},
                {"text": "Second session continues."},
            ],
        },
    ]
    result = ingest_sessions(sessions, store=store, embedder=emb)
    assert set(result.keys()) == {"s1", "s2"}
    assert len(result["s1"]) == 1
    assert len(result["s2"]) == 2
    assert store.count() == 3
    assert store.get("s1#t0#p0") is not None
    assert store.get("s2#t1#p0") is not None
