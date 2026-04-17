"""Tests that plasticity mechanisms actually fire during normal
BeliefLayer operation (v0.3 requirement).

v0.2 had the classes defined and unit-tested, but they didn't
affect runtime behaviour. v0.3 wires them into the ingest/query paths.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from patha.belief.contradiction import StubContradictionDetector
from patha.belief.layer import BeliefLayer, PlasticityConfig
from patha.belief.store import BeliefStore


# ─── Helpers ────────────────────────────────────────────────────────

def _ingest(layer: BeliefLayer, prop: str, *, at: datetime, session: str = "s1") -> str:
    ev = layer.ingest(
        proposition=prop,
        asserted_at=at,
        asserted_in_session=session,
        source_proposition_id=f"prop-{prop[:20]}",
    )
    return ev.new_belief.id


# ─── LTD (query-time decay) ──────────────────────────────────────────

class TestLTDOnQuery:
    def test_decay_fires_on_stale_beliefs(self) -> None:
        layer = BeliefLayer(
            store=BeliefStore(),
            detector=StubContradictionDetector(),
            plasticity=PlasticityConfig(
                ltd_on_query=True,
                ltd_half_life_days=100,
                ltd_floor=0.0,
                hebbian_on_query=False,  # isolate LTD
                homeostasis_on_ingest=False,
                pruning_on_ingest=False,
            ),
        )
        bid = _ingest(layer, "old fact", at=datetime(2023, 1, 1))
        assert layer.store.get(bid).confidence == 1.0  # type: ignore[union-attr]

        # Query ~2 years later — should decay by ~4 half-lives → ~0.0625
        result = layer.query([bid], at_time=datetime(2025, 1, 1))
        assert len(result.current) == 1
        conf = layer.store.get(bid).confidence  # type: ignore[union-attr]
        assert conf < 0.25

    def test_ltd_off_leaves_confidence_alone(self) -> None:
        layer = BeliefLayer(
            store=BeliefStore(),
            detector=StubContradictionDetector(),
            plasticity=PlasticityConfig(ltd_on_query=False),
        )
        bid = _ingest(layer, "x", at=datetime(2020, 1, 1))
        layer.query([bid], at_time=datetime(2030, 1, 1))
        assert layer.store.get(bid).confidence == 1.0  # type: ignore[union-attr]

    def test_plasticity_master_switch(self) -> None:
        layer = BeliefLayer(
            store=BeliefStore(),
            detector=StubContradictionDetector(),
            plasticity=PlasticityConfig(
                enabled=False,
                # All sub-flags on, but master switch off
                ltd_on_query=True,
                hebbian_on_query=True,
                homeostasis_on_ingest=True,
                pruning_on_ingest=True,
            ),
        )
        bid = _ingest(layer, "x", at=datetime(2020, 1, 1))
        layer.query([bid], at_time=datetime(2030, 1, 1))
        assert layer.store.get(bid).confidence == 1.0  # type: ignore[union-attr]


# ─── Hebbian (query-time co-retrieval) ──────────────────────────────

class TestHebbianOnQuery:
    def test_co_retrieved_pair_gets_edge(self) -> None:
        layer = BeliefLayer(
            store=BeliefStore(),
            detector=StubContradictionDetector(),
            plasticity=PlasticityConfig(
                hebbian_on_query=True,
                hebbian_learning_rate=0.1,
                ltd_on_query=False,  # isolate Hebbian
                homeostasis_on_ingest=False,
                pruning_on_ingest=False,
            ),
        )
        a = _ingest(layer, "a", at=datetime(2024, 1, 1))
        b = _ingest(layer, "b", at=datetime(2024, 1, 2))

        # Pre-query: no edge
        assert layer.hebbian.weight(a, b) == 0.0

        layer.query([a, b], at_time=datetime(2024, 6, 1))
        assert layer.hebbian.weight(a, b) == pytest.approx(0.1)

        layer.query([a, b], at_time=datetime(2024, 6, 2))
        assert layer.hebbian.weight(a, b) == pytest.approx(0.2)

    def test_single_belief_query_no_edge(self) -> None:
        layer = BeliefLayer(
            store=BeliefStore(),
            detector=StubContradictionDetector(),
            plasticity=PlasticityConfig(
                hebbian_on_query=True,
                ltd_on_query=False,
                homeostasis_on_ingest=False,
                pruning_on_ingest=False,
            ),
        )
        a = _ingest(layer, "alone", at=datetime(2024, 1, 1))
        layer.query([a], at_time=datetime(2024, 6, 1))
        assert len(layer.hebbian) == 0

    def test_related_surface_after_repeated_coretrieval(self) -> None:
        layer = BeliefLayer(
            store=BeliefStore(),
            detector=StubContradictionDetector(),
            plasticity=PlasticityConfig(
                hebbian_on_query=True,
                hebbian_learning_rate=0.1,
                ltd_on_query=False,
                homeostasis_on_ingest=False,
                pruning_on_ingest=False,
            ),
        )
        a = _ingest(layer, "a", at=datetime(2024, 1, 1))
        b = _ingest(layer, "b", at=datetime(2024, 1, 2))
        c = _ingest(layer, "c", at=datetime(2024, 1, 3))

        # a+b together 3 times, a+c once
        for _ in range(3):
            layer.query([a, b], at_time=datetime(2024, 6, 1))
        layer.query([a, c], at_time=datetime(2024, 6, 1))

        related = layer.hebbian.related(a, top_k=2)
        assert related[0][0] == b  # most-related first
        assert related[1][0] == c


# ─── Homeostasis (ingest-scheduled) ─────────────────────────────────

class TestHomeostasisOnIngest:
    def test_fires_on_interval(self) -> None:
        layer = BeliefLayer(
            store=BeliefStore(),
            detector=StubContradictionDetector(),
            plasticity=PlasticityConfig(
                homeostasis_on_ingest=True,
                homeostasis_interval_ingests=3,
                homeostasis_target_mean=0.5,
                ltd_on_query=False,
                hebbian_on_query=False,
                pruning_on_ingest=False,
            ),
        )
        # Ingest 3 beliefs at confidence 1.0 (the default from .add).
        for i in range(3):
            _ingest(layer, f"b{i}", at=datetime(2024, 1, i + 1), session=f"s{i}")

        # On the 3rd ingest, homeostasis fires and rescales to mean 0.5.
        mean = sum(b.confidence for b in layer.store.current()) / 3
        assert mean == pytest.approx(0.5, rel=1e-3)

    def test_does_not_fire_between_intervals(self) -> None:
        layer = BeliefLayer(
            store=BeliefStore(),
            detector=StubContradictionDetector(),
            plasticity=PlasticityConfig(
                homeostasis_on_ingest=True,
                homeostasis_interval_ingests=100,
                homeostasis_target_mean=0.5,
                ltd_on_query=False,
                hebbian_on_query=False,
                pruning_on_ingest=False,
            ),
        )
        for i in range(5):
            _ingest(layer, f"b{i}", at=datetime(2024, 1, i + 1), session=f"s{i}")
        # Interval=100 not yet reached → confidences still 1.0
        assert all(b.confidence == 1.0 for b in layer.store.current())


# ─── Pruning (ingest-scheduled) ─────────────────────────────────────

class TestPruningOnIngest:
    def test_deep_supersession_chain_gets_pruned(self) -> None:
        from patha.belief.types import ContradictionLabel, ContradictionResult

        # Scripted detector that always contradicts → every new belief
        # supersedes the previous one, building a deep chain.
        class AlwaysContradicts:
            def detect(self, p1, p2):
                return ContradictionResult(
                    label=ContradictionLabel.CONTRADICTS, confidence=0.99
                )

            def detect_batch(self, pairs):
                return [self.detect(p1, p2) for p1, p2 in pairs]

        layer = BeliefLayer(
            store=BeliefStore(),
            detector=AlwaysContradicts(),
            contradiction_threshold=0.5,
            plasticity=PlasticityConfig(
                pruning_on_ingest=True,
                pruning_interval_ingests=5,
                pruning_max_depth=2,
                ltd_on_query=False,
                hebbian_on_query=False,
                homeostasis_on_ingest=False,
            ),
        )
        # Ingest 5 beliefs in a chain. By ingest 5, pruning fires and
        # beliefs more than 2 hops from current get archived.
        for i in range(5):
            _ingest(layer, f"claim {i}", at=datetime(2024, 1, i + 1))

        archived = layer.store.archived()
        # The earliest beliefs (depth 3, 4) should be archived.
        assert len(archived) >= 2


# ─── Default behaviour (plasticity on by default) ───────────────────

class TestDefaults:
    def test_layer_plasticity_on_by_default(self) -> None:
        layer = BeliefLayer(
            store=BeliefStore(), detector=StubContradictionDetector()
        )
        assert layer.plasticity.enabled is True
        assert layer.plasticity.ltd_on_query is True
        assert layer.plasticity.hebbian_on_query is True
        assert layer.plasticity.homeostasis_on_ingest is True
        assert layer.plasticity.pruning_on_ingest is True
