"""Tests for the narrative songline-walk recall strategy
(retrieval/narrative_walk.py).

Builds a small real BeliefStore + a hand-constructed SonglineGraph (so
the test exercises the actual graph/store interfaces, not mocks) and
asserts the walker:
  - orders beats by time
  - folds in supersession ancestors (the evolution signal)
  - degrades to None when the theme has < 2 beats
  - stays on-theme (doesn't drag in off-theme session-mates)
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from patha.belief.store import BeliefStore
from patha.indexing.songline_graph import SonglineGraph
from patha.retrieval.narrative_walk import narrative_walk


def _store_with_agency_timeline() -> tuple[BeliefStore, dict[str, str]]:
    """4 agency beliefs across 4 dates; b4 supersedes b2 (a revision).
    Plus one off-theme belief b9 sharing a session with b1.

    Returns (store, id_map) where id_map maps chunk_id -> proposition_id.
    """
    store = BeliefStore()

    def add(bid, prop, month, session="s1"):
        return store.add(
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
    add("b9", "I had pasta for dinner", 1, "s1")  # off-theme, shares s1

    # b4 supersedes b2 (revised view on agency)
    store.supersede("b2", "b4")

    # id_map: chunk_id -> proposition_id (1:1 in the Memory store world)
    id_map = {f"c{i}": f"prop-b{i}" for i in [1, 2, 3, 4, 9]}
    return store, id_map


def _agency_graph() -> SonglineGraph:
    """Hand-built graph: agency chunks linked by entity + temporal edges;
    b9 linked to b1 only by session (off-theme)."""
    g = SonglineGraph(
        adjacency=defaultdict(list),
        _channel_index=defaultdict(lambda: defaultdict(set)),
    )
    agency_chunks = ["c1", "c2", "c3", "c4"]
    # entity channel members for "agency"
    for c in agency_chunks:
        g._channel_index["entity"]["agency"].add(c)
    # entity edges among agency chunks (undirected, both directions)
    for a in agency_chunks:
        for b in agency_chunks:
            if a != b:
                g.adjacency[a].append((b, 0.5, "entity"))
    # session edge c1<->c9 (off-theme link that must NOT pull c9 in)
    g.adjacency["c1"].append(("c9", 0.4, "session"))
    g.adjacency["c9"].append(("c1", 0.4, "session"))
    return g


class TestNarrativeWalk:
    def test_orders_beats_by_time(self):
        store, id_map = _store_with_agency_timeline()
        graph = _agency_graph()
        res = narrative_walk(
            "how has my thinking on agency evolved?",
            "evolution", "agency",
            graph=graph, id_map=id_map, store=store,
        )
        assert res is not None
        dates = [b.asserted_at for b in res.beats]
        assert dates == sorted(dates), "beats must be temporally ordered"
        assert res.theme == "agency"

    def test_folds_in_superseded_ancestor(self):
        # b2 is superseded by b4 and is NOT in the entity channel walk
        # start set directly... actually it is; but the key assertion is
        # that the superseded b2 appears tagged as a revision.
        store, id_map = _store_with_agency_timeline()
        graph = _agency_graph()
        res = narrative_walk(
            "how has my thinking on agency evolved?",
            "evolution", "agency",
            graph=graph, id_map=id_map, store=store,
        )
        assert res is not None
        ids = {b.belief_id for b in res.beats}
        assert "b2" in ids and "b4" in ids
        b2 = next(b for b in res.beats if b.belief_id == "b2")
        assert b2.supersession_status in ("revised-from", "superseded")

    def test_stays_on_theme(self):
        # b9 (pasta) shares session s1 with b1 but is off-theme; the walk
        # must NOT include it (session edges only followed to on-theme nodes).
        store, id_map = _store_with_agency_timeline()
        graph = _agency_graph()
        res = narrative_walk(
            "trace my thinking on agency",
            "throughline", "agency",
            graph=graph, id_map=id_map, store=store,
        )
        assert res is not None
        ids = {b.belief_id for b in res.beats}
        assert "b9" not in ids, "off-theme session-mate must be excluded"

    def test_through_line_present(self):
        store, id_map = _store_with_agency_timeline()
        graph = _agency_graph()
        res = narrative_walk(
            "how has my thinking on agency evolved?",
            "evolution", "agency",
            graph=graph, id_map=id_map, store=store,
        )
        assert res is not None
        assert "agency" in res.through_line
        # has at least the 4 agency beats
        assert res.beat_count >= 4

    def test_unknown_theme_returns_none(self):
        store, id_map = _store_with_agency_timeline()
        graph = _agency_graph()
        res = narrative_walk(
            "how has my thinking on quantum gravity evolved?",
            "evolution", "quantumgravity",
            graph=graph, id_map=id_map, store=store,
        )
        assert res is None, "no on-theme beats → degrade to retrieval"

    def test_single_beat_returns_none(self):
        # A theme with only one belief is not a narrative.
        store = BeliefStore()
        store.add(
            proposition="solitude is underrated",
            asserted_at=datetime(2025, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="prop-x1",
            belief_id="x1",
        )
        g = SonglineGraph(
            adjacency=defaultdict(list),
            _channel_index=defaultdict(lambda: defaultdict(set)),
        )
        g._channel_index["entity"]["solitude"].add("cx1")
        id_map = {"cx1": "prop-x1"}
        res = narrative_walk(
            "trace my thinking on solitude", "throughline", "solitude",
            graph=g, id_map=id_map, store=store,
        )
        assert res is None
