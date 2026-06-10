"""End-to-end: a narrative question routed through Memory.recall().

Proves the full Phase 4 path is reachable through the public API: a
narrative query produces strategy="narrative" with a populated
Recall.narrative, ordered beats, and current/history split — while a
plain retrieval question on the same store does NOT route narrative.

Uses a stub Phase 1 retriever (synthetic songline graph + id_map) so
the test is fast and deterministic — no embedder/spaCy/reranker load.
The graph + store are wired exactly as the real LazyPhase1Retriever
would expose them (.songline_graph, .id_map, callable).
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path

import patha
from patha.indexing.songline_graph import SonglineGraph


class _FakeRetriever:
    """Stand-in for LazyPhase1Retriever exposing the three things the
    narrative path needs: .songline_graph, .id_map, and __call__."""

    def __init__(self, graph, id_map):
        self._graph = graph
        self._id_map = id_map

    @property
    def songline_graph(self):
        return self._graph

    @property
    def id_map(self):
        return self._id_map

    def __call__(self, question: str, top_k: int):
        return []  # no semantic anchors; the walk uses the entity channel

    def invalidate(self):
        pass


def _wire_agency_store(mem) -> None:
    """Add an agency timeline directly to the store + a matching graph,
    then swap in the fake retriever. b4 supersedes b2 (a revision)."""
    store = mem._patha.belief_layer.store

    def add(bid, prop, month, session):
        store.add(
            proposition=prop,
            asserted_at=datetime(2025, month, 1, 12, 0),
            asserted_in_session=session,
            source_proposition_id=f"prop-{bid}",
            belief_id=bid,
        )

    add("b1", "agency is about removing constraints", 1, "s1")
    add("b2", "agency means having more options", 3, "s2")
    add("b3", "agency is fundamentally about leverage", 6, "s3")
    add("b4", "agency is really about ownership, not options", 9, "s4")
    store.supersede("b2", "b4")

    graph = SonglineGraph(
        adjacency=defaultdict(list),
        _channel_index=defaultdict(lambda: defaultdict(set)),
    )
    chunks = ["c1", "c2", "c3", "c4"]
    for c in chunks:
        graph._channel_index["entity"]["agency"].add(c)
    for a in chunks:
        for b in chunks:
            if a != b:
                graph.adjacency[a].append((b, 0.5, "entity"))
    id_map = {f"c{i}": f"prop-b{i}" for i in [1, 2, 3, 4]}

    mem._phase1_retriever = _FakeRetriever(graph, id_map)
    mem._narrative_enabled = True


class TestNarrativeRecall:
    def test_narrative_question_routes_narrative(self, tmp_path: Path):
        mem = patha.Memory(
            path=tmp_path / "store.jsonl", detector="stub",
            enable_phase1=True, enable_ganita=True,
        )
        _wire_agency_store(mem)

        rec = mem.recall(
            "how has my thinking on agency evolved?",
            include_history=True,
        )
        assert rec.strategy == "narrative"
        assert rec.narrative is not None
        assert rec.narrative.theme == "agency"
        # ordered beats, at least the 4 agency beliefs
        assert rec.narrative.beat_count >= 4
        dates = [b.asserted_at for b in rec.narrative.beats]
        assert dates == sorted(dates)
        # the through-line and timeline summary are present + prompt-ready
        assert "agency" in rec.summary
        assert rec.tokens >= 0

    def test_supersession_split_into_current_and_history(self, tmp_path: Path):
        mem = patha.Memory(
            path=tmp_path / "store.jsonl", detector="stub",
            enable_phase1=True,
        )
        _wire_agency_store(mem)
        rec = mem.recall(
            "trace my thinking on agency", include_history=True,
        )
        assert rec.strategy == "narrative"
        # b2 was superseded by b4 → b2 lands in history, b4 in current
        history_ids = {c["id"] for c in rec.history}
        current_ids = {c["id"] for c in rec.current}
        assert "b2" in history_ids
        assert "b4" in current_ids

    def test_plain_question_does_not_route_narrative(self, tmp_path: Path):
        mem = patha.Memory(
            path=tmp_path / "store.jsonl", detector="stub",
            enable_phase1=True,
        )
        _wire_agency_store(mem)
        # A non-narrative question must fall through to retrieval.
        rec = mem.recall("what did I say about agency?")
        assert rec.strategy != "narrative"
        assert rec.narrative is None

    def test_narrative_disabled_when_flag_off(self, tmp_path: Path):
        mem = patha.Memory(
            path=tmp_path / "store.jsonl", detector="stub",
            enable_phase1=True, enable_narrative=False,
        )
        _wire_agency_store(mem)
        mem._narrative_enabled = False  # explicit: flag honored
        rec = mem.recall("how has my thinking on agency evolved?")
        assert rec.strategy != "narrative"
        assert rec.narrative is None
