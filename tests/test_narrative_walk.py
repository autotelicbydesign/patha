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


# ─── Topic-cluster gate (dogfood F4/F5 fixes) ───────────────────────


def _agency_fixture_with_topics():
    """Agency store + graph where beliefs b5/b6 are PARAPHRASES (no
    'agency' substring) linked to anchors only via topic clusters."""
    store = BeliefStore()

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
    # b5: paraphrase — same topic, no 'agency' substring, reachable ONLY
    # via a topic edge from b1's cluster.
    add("b5", "true autonomy comes from owning your decisions", 5, "s5")
    # b6: paraphrase reachable ONLY via a temporal edge from b1 — the
    # old substring gate excluded this; the cluster-share gate admits it.
    add("b6", "self-direction matters more than raw options", 1, "s6")
    # b9: off-theme, in its own cluster, session-linked to b1.
    add("b9", "I had pasta for dinner", 1, "s1")

    g = SonglineGraph(
        adjacency=defaultdict(list),
        _channel_index=defaultdict(lambda: defaultdict(set)),
    )
    # entity channel: only b1/b2 mention 'agency' literally
    for c in ("c1", "c2"):
        g._channel_index["entity"]["agency"].add(c)
    g.adjacency["c1"].append(("c2", 0.5, "entity"))
    g.adjacency["c2"].append(("c1", 0.5, "entity"))
    # topic channel: c1, c2, c5, c6 share topic "t0"; c9 in "t9"
    for c in ("c1", "c2", "c5", "c6"):
        g._channel_index["topic"]["t0"].add(c)
    g._channel_index["topic"]["t9"].add("c9")
    # topic edge c2 -> c5 (paraphrase reachable via topic edge)
    g.adjacency["c2"].append(("c5", 0.5, "topic"))
    g.adjacency["c5"].append(("c2", 0.5, "topic"))
    # temporal edge c1 -> c6 (same day) — the headline gate case
    g.adjacency["c1"].append(("c6", 0.4, "temporal"))
    g.adjacency["c6"].append(("c1", 0.4, "temporal"))
    # session edge c1 -> c9 (off-theme; different topic cluster)
    g.adjacency["c1"].append(("c9", 0.4, "session"))
    g.adjacency["c9"].append(("c1", 0.4, "session"))

    # Padding: two disconnected off-theme pairs so the anchor cluster
    # t0 (4 members) stays under the walker's >50%-of-nodes mega-cluster
    # guard (4/9 nodes). Unreachable from the anchors, so they never
    # surface in beats and need no store entries.
    for a, b in (("c20", "c21"), ("c22", "c23")):
        g.adjacency[a].append((b, 0.3, "session"))
        g.adjacency[b].append((a, 0.3, "session"))

    id_map = {f"c{i}": f"prop-b{i}" for i in [1, 2, 5, 6, 9]}
    return store, g, id_map


class TestTopicClusterGate:
    def test_walker_traverses_topic_edge_to_paraphrase(self):
        store, g, id_map = _agency_fixture_with_topics()
        res = narrative_walk(
            "trace my thinking on agency", "throughline", "agency",
            graph=g, id_map=id_map, store=store,
        )
        assert res is not None
        ids = {b.belief_id for b in res.beats}
        assert "b5" in ids, "paraphrase via topic edge must be included"

    def test_temporal_edge_to_cluster_mate_passes_gate(self):
        # THE headline regression: b6 has no 'agency' substring and is
        # reachable only via a temporal edge — under the old substring
        # gate it was excluded; sharing anchor cluster t0 admits it.
        store, g, id_map = _agency_fixture_with_topics()
        res = narrative_walk(
            "trace my thinking on agency", "throughline", "agency",
            graph=g, id_map=id_map, store=store,
        )
        assert res is not None
        ids = {b.belief_id for b in res.beats}
        assert "b6" in ids, (
            "temporal edge to a cluster-mate paraphrase must pass the gate"
        )

    def test_off_theme_session_mate_still_excluded_with_topics(self):
        store, g, id_map = _agency_fixture_with_topics()
        res = narrative_walk(
            "trace my thinking on agency", "throughline", "agency",
            graph=g, id_map=id_map, store=store,
        )
        assert res is not None
        ids = {b.belief_id for b in res.beats}
        assert "b9" not in ids, (
            "off-theme session-mate in a different cluster stays excluded"
        )

    def test_mega_cluster_topic_ignored(self):
        # If an anchor's cluster spans >50% of nodes, it must not admit
        # off-theme temporal neighbors.
        store, g, id_map = _agency_fixture_with_topics()
        # blow t0 up to cover ALL nodes including c9
        g._channel_index["topic"]["t0"] |= {"c9"}
        g._topic_by_chunk = None  # reset lazy cache
        res = narrative_walk(
            "trace my thinking on agency", "throughline", "agency",
            graph=g, id_map=id_map, store=store,
        )
        assert res is not None
        ids = {b.belief_id for b in res.beats}
        assert "b9" not in ids, "mega-cluster must not blow the gate open"
