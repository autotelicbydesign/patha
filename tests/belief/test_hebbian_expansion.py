"""Tests for Innovation #1 — Hebbian-cluster-aware retrieval.

Patha's belief layer accumulates Hebbian co-retrieval edges between
beliefs that surface in the same query. Innovation #1 turns that
*recorded* signal into a *runtime* signal: when the user queries again,
the Phase-1 candidate set is expanded with each seed belief's strongest
co-retrieval neighbors. This is the cluster-aware retrieval step that
neither cosine, BM25, nor songlines provide on their own — it's a
learned routing prior, accumulated from real usage.

What we test:

  1. With expansion off, the candidate set is exactly Phase 1's output.
  2. With expansion on but an empty Hebbian graph, we still get the
     Phase 1 candidates verbatim (graceful no-op).
  3. After a query has fired, subsequent queries pull in the co-retrieved
     cluster via the expansion.
  4. Session seeding bootstraps the graph at first use, so a fresh
     store without query history still expands.
  5. Expansion respects max_added and skips superseded neighbors.
"""

from __future__ import annotations

from datetime import datetime

from patha.belief.contradiction import StubContradictionDetector
from patha.belief.layer import BeliefLayer
from patha.belief.store import BeliefStore
from patha.integrated import IntegratedPatha


def _build(hebbian_expansion: bool, **kwargs) -> IntegratedPatha:
    layer = BeliefLayer(
        store=BeliefStore(),
        detector=StubContradictionDetector(),
    )
    patha = IntegratedPatha(
        belief_layer=layer,
        hebbian_expansion=hebbian_expansion,
        **kwargs,
    )
    return patha


def _ingest_three_beliefs(p: IntegratedPatha) -> tuple[str, str, str]:
    """Ingest three unrelated beliefs in three sessions; return ids."""
    ev1 = p.ingest(
        proposition="I bought a road bike in Lisbon",
        asserted_at=datetime(2024, 1, 1),
        asserted_in_session="s1",
        source_proposition_id="p1",
    )
    ev2 = p.ingest(
        proposition="I am training for a triathlon",
        asserted_at=datetime(2024, 2, 1),
        asserted_in_session="s2",
        source_proposition_id="p2",
    )
    ev3 = p.ingest(
        proposition="My favourite color is blue",
        asserted_at=datetime(2024, 3, 1),
        asserted_in_session="s3",
        source_proposition_id="p3",
    )
    return ev1.new_belief.id, ev2.new_belief.id, ev3.new_belief.id


# ─── Behaviour ───────────────────────────────────────────────────────


class TestHebbianExpansionDisabled:
    def test_off_by_default_in_integrated(self) -> None:
        """`IntegratedPatha`'s default is no expansion (back-compat)."""
        p = _build(hebbian_expansion=False)
        b1, b2, b3 = _ingest_three_beliefs(p)
        # Manually pre-populate Hebbian — even with edges present,
        # expansion off means nothing extra surfaces.
        p.belief_layer.hebbian.record_coretrieval([b1, b2])

        # Phase 1 returns only b1; expansion off → only b1 in result.
        p._phase1_retrieve = lambda q, k: ["p1"]
        resp = p.query("anything", at_time=datetime(2024, 6, 1))
        assert {b.id for b in resp.retrieval_result.current} == {b1}


class TestHebbianExpansionEnabled:
    def test_empty_graph_is_noop(self) -> None:
        """No Hebbian edges → expansion changes nothing."""
        p = _build(hebbian_expansion=True, hebbian_session_seed_weight=0.0)
        b1, b2, b3 = _ingest_three_beliefs(p)
        p._phase1_retrieve = lambda q, k: ["p1"]
        resp = p.query("anything", at_time=datetime(2024, 6, 1))
        assert {b.id for b in resp.retrieval_result.current} == {b1}

    def test_co_retrieval_seeds_then_expands_on_next_query(self) -> None:
        """First query records co-retrieval; second query expands cluster."""
        p = _build(hebbian_expansion=True, hebbian_session_seed_weight=0.0)
        b1, b2, b3 = _ingest_three_beliefs(p)

        # First query: Phase 1 returns p1 AND p2 → co-retrieval recorded.
        p._phase1_retrieve = lambda q, k: ["p1", "p2"]
        p.query("query 1", at_time=datetime(2024, 6, 1))

        # Verify the edge actually got recorded by the belief layer's
        # plasticity hook.
        assert p.belief_layer.hebbian.weight(b1, b2) > 0

        # Second query: Phase 1 returns ONLY p1, but expansion should
        # bring in b2 since the edge exists.
        p._phase1_retrieve = lambda q, k: ["p1"]
        resp = p.query("query 2", at_time=datetime(2024, 6, 2))
        ids_returned = {b.id for b in resp.retrieval_result.current}
        assert b1 in ids_returned
        assert b2 in ids_returned

    def test_session_seeding_bootstraps_fresh_store(self) -> None:
        """With session seeding on, beliefs in the same session expand
        together even before any query has fired."""
        p = _build(
            hebbian_expansion=True, hebbian_session_seed_weight=0.05
        )
        # Ingest two beliefs in the same session — they should get
        # a weak Hebbian edge from the seeding pass on first query.
        ev1 = p.ingest(
            proposition="I rented an apartment in Lisbon",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="apt-search",
            source_proposition_id="apt-1",
        )
        ev2 = p.ingest(
            proposition="The rent is 1500 EUR per month",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="apt-search",
            source_proposition_id="apt-2",
        )
        # Phase 1 returns only apt-1, but session-seeding should pull
        # apt-2 in as a neighbor.
        p._phase1_retrieve = lambda q, k: ["apt-1"]
        resp = p.query(
            "where do I live", at_time=datetime(2024, 6, 1)
        )
        ids_returned = {b.id for b in resp.retrieval_result.current}
        assert ev1.new_belief.id in ids_returned
        assert ev2.new_belief.id in ids_returned

    def test_max_added_caps_expansion(self) -> None:
        """Expansion respects hebbian_max_added."""
        p = _build(
            hebbian_expansion=True,
            hebbian_session_seed_weight=0.0,
            hebbian_max_added=1,
            hebbian_top_k_per_seed=10,
        )
        # Ingest five beliefs and pre-record co-retrieval among them
        # so b1 has many Hebbian neighbors.
        evs = []
        for i in range(5):
            ev = p.ingest(
                proposition=f"Fact number {i}",
                asserted_at=datetime(2024, 1, i + 1),
                asserted_in_session=f"s{i}",
                source_proposition_id=f"p{i}",
            )
            evs.append(ev.new_belief.id)
        # Pre-populate Hebbian so b0 has 4 neighbors.
        p.belief_layer.hebbian.record_coretrieval(evs)
        # Phase 1 returns just p0; with max_added=1, expansion should
        # pull in exactly one neighbor.
        p._phase1_retrieve = lambda q, k: ["p0"]
        resp = p.query("q", at_time=datetime(2024, 6, 1))
        ids = {b.id for b in resp.retrieval_result.current}
        # Original seed + 1 added = 2 ids total.
        assert len(ids) == 2
        assert evs[0] in ids

    def test_expansion_skips_superseded_neighbors(self) -> None:
        """A Hebbian edge to a superseded belief is dead weight; skip."""
        p = _build(hebbian_expansion=True, hebbian_session_seed_weight=0.0)
        # Use a scripted detector so the second ingest supersedes b1.
        from patha.belief.types import (
            ContradictionLabel,
            ContradictionResult,
        )

        class _Det:
            def detect_batch(self, pairs):
                return [
                    ContradictionResult(
                        label=ContradictionLabel.CONTRADICTS,
                        confidence=0.95,
                    )
                    if "now eat fish" in pair[1]
                    else ContradictionResult(
                        label=ContradictionLabel.NEUTRAL, confidence=0.5
                    )
                    for pair in pairs
                ]

        p.belief_layer.detector = _Det()

        ev1 = p.ingest(
            proposition="I am vegetarian",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
        )
        ev2 = p.ingest(
            proposition="I now eat fish again",
            asserted_at=datetime(2024, 6, 1),
            asserted_in_session="s2",
            source_proposition_id="p2",
        )
        ev3 = p.ingest(
            proposition="My favourite color is blue",
            asserted_at=datetime(2024, 7, 1),
            asserted_in_session="s3",
            source_proposition_id="p3",
        )

        # Pre-populate a Hebbian edge from b3 → (now-superseded) b1.
        p.belief_layer.hebbian.record_coretrieval(
            [ev3.new_belief.id, ev1.new_belief.id]
        )
        # Phase 1 returns p3; expansion should skip b1 (superseded).
        p._phase1_retrieve = lambda q, k: ["p3"]
        resp = p.query("q", at_time=datetime(2024, 8, 1))
        ids = {b.id for b in resp.retrieval_result.current}
        assert ev3.new_belief.id in ids
        assert ev1.new_belief.id not in ids
