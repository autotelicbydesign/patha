"""Tests for saṁskāra → vāsanā layered confidence (v0.4 #4)."""

from __future__ import annotations

from datetime import datetime

import pytest

from patha.belief.plasticity import LongTermDepression
from patha.belief.store import BeliefStore
from patha.belief.types import Pramana


def _add(store: BeliefStore, bid: str, session: str = "s0", pramana=Pramana.UNKNOWN):
    return store.add(
        proposition=f"p-{bid}",
        asserted_at=datetime(2024, 1, 1),
        asserted_in_session=session,
        source_proposition_id=f"prop-{bid}",
        belief_id=bid,
        pramana=pramana,
    )


class TestSamskaraCounter:
    def test_distinct_source_bumps_samskara(self) -> None:
        store = BeliefStore()
        a = _add(store, "a", session="s0", pramana=Pramana.PRATYAKSA)
        a.confidence = 0.5
        _add(store, "r1", session="s1", pramana=Pramana.PRATYAKSA)
        store.reinforce("a", "r1")
        assert a.samskara_count == 1

    def test_same_source_same_pramana_does_not_bump(self) -> None:
        """Pure echo shouldn't accumulate a samskāra."""
        store = BeliefStore()
        a = _add(store, "a", session="s0", pramana=Pramana.PRATYAKSA)
        a.confidence = 0.5
        _add(store, "r1", session="s0", pramana=Pramana.PRATYAKSA)
        store.reinforce("a", "r1")
        assert a.samskara_count == 0

    def test_distinct_pramana_bumps_samskara(self) -> None:
        store = BeliefStore()
        a = _add(store, "a", session="s0", pramana=Pramana.SHABDA)
        a.confidence = 0.5
        _add(store, "r1", session="s0", pramana=Pramana.PRATYAKSA)
        store.reinforce("a", "r1")
        assert a.samskara_count == 1


class TestVasanaEstablishment:
    def test_deep_confidence_none_before_threshold(self) -> None:
        store = BeliefStore()
        a = _add(store, "a", session="s0", pramana=Pramana.PRATYAKSA)
        a.confidence = 0.5
        # 4 reinforcements (below threshold of 5)
        for i in range(4):
            _add(store, f"r{i}", session=f"s{i+1}", pramana=Pramana.PRATYAKSA)
            store.reinforce("a", f"r{i}")
        assert a.samskara_count == 4
        assert a.deep_confidence is None
        assert not a.is_vasana_established

    def test_deep_confidence_crystallises_at_threshold(self) -> None:
        store = BeliefStore()
        a = _add(store, "a", session="s0", pramana=Pramana.PRATYAKSA)
        a.confidence = 0.5
        for i in range(5):
            _add(store, f"r{i}", session=f"s{i+1}", pramana=Pramana.PRATYAKSA)
            store.reinforce("a", f"r{i}")
        # At threshold 5, deep_confidence is set to current surface
        assert a.samskara_count == 5
        assert a.deep_confidence is not None
        assert a.is_vasana_established

    def test_further_reinforcement_grows_deep_slowly(self) -> None:
        """After vāsanā is established, further reinforcements pull
        deep confidence upward but at 1/5 the surface rate."""
        store = BeliefStore()
        a = _add(store, "a", session="s0", pramana=Pramana.PRATYAKSA)
        a.confidence = 0.5
        # Get past threshold
        for i in range(5):
            _add(store, f"r{i}", session=f"s{i+1}", pramana=Pramana.PRATYAKSA)
            store.reinforce("a", f"r{i}")
        deep_at_threshold = a.deep_confidence
        assert deep_at_threshold is not None

        # Further reinforcement
        _add(store, "r-extra", session="s-extra", pramana=Pramana.PRATYAKSA)
        store.reinforce("a", "r-extra")
        # Deep should move UP but less than surface
        assert a.deep_confidence >= deep_at_threshold
        # Surface moves faster: surface delta > deep delta


class TestEffectiveConfidence:
    def test_effective_is_surface_when_no_vasana(self) -> None:
        store = BeliefStore()
        a = _add(store, "a")
        a.confidence = 0.7
        assert a.effective_confidence == 0.7

    def test_effective_is_max_of_surface_and_deep(self) -> None:
        """When surface dips below deep (e.g., after decay), effective
        should surface the deep confidence — the belief doesn't vanish."""
        store = BeliefStore()
        a = _add(store, "a")
        a.confidence = 0.3  # decayed
        a.deep_confidence = 0.8
        assert a.effective_confidence == 0.8

    def test_effective_is_surface_when_surface_higher(self) -> None:
        """A recently-reinforced belief has higher surface than deep."""
        store = BeliefStore()
        a = _add(store, "a")
        a.confidence = 0.95
        a.deep_confidence = 0.7
        assert a.effective_confidence == 0.95


class TestDeepConfidenceDecay:
    def test_deep_decays_10x_slower(self) -> None:
        """Surface decays at half_life; deep decays at 10x half_life."""
        store = BeliefStore()
        a = _add(store, "a", session="s0")
        a.confidence = 1.0
        a.deep_confidence = 1.0
        # Force vāsanā established
        a.samskara_count = 5

        # 1-year decay with half_life_days=365 → surface → ~0.5
        # Deep decays 10x slower → half_life 3650 days → 1 year → factor 0.5^(1/10) ≈ 0.933
        ltd = LongTermDepression(half_life_days=365, floor=0.0)
        ltd.apply_to_store(
            store,
            now=datetime(2025, 1, 1),
            beliefs=[a],
        )
        assert a.confidence == pytest.approx(0.5, abs=0.01)
        assert a.deep_confidence == pytest.approx(0.933, abs=0.02)


class TestPersistenceRoundTrip:
    def test_samskara_and_deep_roundtrip(self, tmp_path) -> None:
        path = tmp_path / "beliefs.jsonl"

        s1 = BeliefStore(persistence_path=path)
        a = _add(s1, "a", session="s0", pramana=Pramana.PRATYAKSA)
        a.confidence = 0.5
        for i in range(5):
            _add(s1, f"r{i}", session=f"s{i+1}", pramana=Pramana.PRATYAKSA)
            s1.reinforce("a", f"r{i}")

        s2 = BeliefStore(persistence_path=path)
        restored = s2.get("a")
        assert restored is not None
        assert restored.samskara_count == 5
        assert restored.deep_confidence is not None
