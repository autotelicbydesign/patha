"""Tests for neuroplasticity-inspired belief maintenance mechanisms."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from patha.belief.plasticity import (
    HebbianAssociation,
    HomeostaticRegulation,
    LongTermDepression,
    LongTermPotentiation,
    SynapticPruning,
)
from patha.belief.store import BeliefStore


# ─── Fixtures ────────────────────────────────────────────────────────

def _add(store: BeliefStore, bid: str, asserted_at: datetime | None = None):
    return store.add(
        proposition=f"p-{bid}",
        asserted_at=asserted_at or datetime(2024, 1, 1, 12, 0),
        asserted_in_session="s1",
        source_proposition_id=f"prop-{bid}",
        belief_id=bid,
    )


# ─── Long-term potentiation ──────────────────────────────────────────

class TestLongTermPotentiation:
    def test_new_confidence_closes_gap(self) -> None:
        ltp = LongTermPotentiation(gap_closure=0.3)
        assert ltp.new_confidence(0.5) == pytest.approx(0.65)
        assert ltp.new_confidence(0.0) == pytest.approx(0.3)

    def test_caps_at_one(self) -> None:
        ltp = LongTermPotentiation(gap_closure=0.5)
        assert ltp.new_confidence(0.99) <= 1.0

    def test_apply_mutates_belief(self) -> None:
        store = BeliefStore()
        b = _add(store, "b1")
        b.confidence = 0.5
        LongTermPotentiation(0.4).apply(b)
        assert b.confidence == pytest.approx(0.7)

    def test_rejects_invalid_rate(self) -> None:
        with pytest.raises(ValueError):
            LongTermPotentiation(gap_closure=0)
        with pytest.raises(ValueError):
            LongTermPotentiation(gap_closure=1.5)


# ─── Long-term depression (decay) ────────────────────────────────────

class TestLongTermDepression:
    def test_decay_halves_at_half_life(self) -> None:
        ltd = LongTermDepression(half_life_days=100)
        result = ltd.decayed(1.0, 100)
        assert result == pytest.approx(0.5, rel=1e-3)

    def test_no_decay_at_zero_age(self) -> None:
        ltd = LongTermDepression(half_life_days=100)
        assert ltd.decayed(0.8, 0) == 0.8

    def test_floor_is_respected(self) -> None:
        ltd = LongTermDepression(half_life_days=100, floor=0.2)
        # At very large age, value should land at floor, not 0
        result = ltd.decayed(1.0, 10_000)
        assert result == pytest.approx(0.2, abs=1e-3)

    def test_apply_to_store_mutates_based_on_age(self) -> None:
        store = BeliefStore()
        _add(store, "old", asserted_at=datetime(2023, 1, 1))
        _add(store, "recent", asserted_at=datetime(2024, 12, 1))

        ltd = LongTermDepression(half_life_days=365)
        updated = ltd.apply_to_store(store, now=datetime(2025, 1, 1))
        # Both should be updated: old is aged ~2 years, recent ~1 month
        assert updated == 2
        assert store.get("old").confidence < 0.3  # type: ignore[union-attr]
        assert store.get("recent").confidence > 0.9  # type: ignore[union-attr]

    def test_reinforcement_resets_decay_clock(self) -> None:
        """A reinforcing assertion is treated as a fresh access timestamp."""
        store = BeliefStore()
        _add(store, "old", asserted_at=datetime(2023, 1, 1))
        # A reinforcing belief from 2024-12
        _add(store, "reinforcer", asserted_at=datetime(2024, 12, 1))
        store.reinforce("old", "reinforcer")

        ltd = LongTermDepression(half_life_days=365)
        ltd.apply_to_store(store, now=datetime(2025, 1, 1))
        # 'old' should have decayed only from its reinforcement timestamp,
        # i.e., ~1 month, so stay high.
        assert store.get("old").confidence > 0.8  # type: ignore[union-attr]

    def test_rejects_nonpositive_half_life(self) -> None:
        with pytest.raises(ValueError):
            LongTermDepression(half_life_days=0)


# ─── Synaptic pruning ────────────────────────────────────────────────

class TestSynapticPruning:
    def test_prunes_beyond_max_depth(self) -> None:
        store = BeliefStore()
        # Build a chain of 6 beliefs: b0 <- b1 <- b2 <- b3 <- b4 <- b5
        # b5 is current; b0 is 5 hops deep.
        for i in range(6):
            _add(store, f"b{i}")
        for i in range(5):
            store.supersede(f"b{i}", f"b{i+1}")

        pruned = SynapticPruning(max_depth=3).prune(store)
        # b0 (depth 5) and b1 (depth 4) should be pruned
        assert "b0" in pruned
        assert "b1" in pruned
        # b2 (depth 3) and beyond stay
        assert "b2" not in pruned
        assert store.get("b0").confidence == 0.0  # type: ignore[union-attr]
        # Non-destructive: belief still exists
        assert store.get("b0") is not None

    def test_current_beliefs_not_pruned(self) -> None:
        store = BeliefStore()
        _add(store, "b1")
        pruned = SynapticPruning(max_depth=10).prune(store)
        assert pruned == []

    def test_rejects_invalid_depth(self) -> None:
        with pytest.raises(ValueError):
            SynapticPruning(max_depth=0)


# ─── Homeostatic regulation ──────────────────────────────────────────

class TestHomeostaticRegulation:
    def test_normalises_to_target_mean(self) -> None:
        store = BeliefStore()
        _add(store, "b1").confidence = 0.9
        _add(store, "b2").confidence = 0.9
        _add(store, "b3").confidence = 0.9

        updated = HomeostaticRegulation(target_mean=0.6).apply(store)
        assert updated == 3
        mean = sum(b.confidence for b in store.current()) / 3
        assert mean == pytest.approx(0.6, rel=1e-3)

    def test_preserves_relative_ordering(self) -> None:
        store = BeliefStore()
        _add(store, "b1").confidence = 0.9
        _add(store, "b2").confidence = 0.6
        _add(store, "b3").confidence = 0.3

        HomeostaticRegulation(target_mean=0.7).apply(store)
        # b1 > b2 > b3 must still hold
        assert store.get("b1").confidence > store.get("b2").confidence  # type: ignore[union-attr]
        assert store.get("b2").confidence > store.get("b3").confidence  # type: ignore[union-attr]

    def test_handles_empty_store(self) -> None:
        store = BeliefStore()
        assert HomeostaticRegulation().apply(store) == 0

    def test_rejects_invalid_target(self) -> None:
        with pytest.raises(ValueError):
            HomeostaticRegulation(target_mean=0)
        with pytest.raises(ValueError):
            HomeostaticRegulation(target_mean=1.5)


# ─── Hebbian association ─────────────────────────────────────────────

class TestHebbianAssociation:
    def test_pair_weight_grows_with_coretrieval(self) -> None:
        heb = HebbianAssociation(learning_rate=0.1)
        heb.record_coretrieval(["a", "b"])
        heb.record_coretrieval(["a", "b"])
        heb.record_coretrieval(["a", "b"])
        assert heb.weight("a", "b") == pytest.approx(0.3)

    def test_symmetric(self) -> None:
        heb = HebbianAssociation(learning_rate=0.1)
        heb.record_coretrieval(["a", "b"])
        assert heb.weight("a", "b") == heb.weight("b", "a")

    def test_self_weight_zero(self) -> None:
        heb = HebbianAssociation()
        heb.record_coretrieval(["a", "b"])
        assert heb.weight("a", "a") == 0.0

    def test_triples_create_three_edges(self) -> None:
        heb = HebbianAssociation(learning_rate=0.1)
        heb.record_coretrieval(["a", "b", "c"])
        assert heb.weight("a", "b") == pytest.approx(0.1)
        assert heb.weight("a", "c") == pytest.approx(0.1)
        assert heb.weight("b", "c") == pytest.approx(0.1)

    def test_related_returns_top_k(self) -> None:
        heb = HebbianAssociation(learning_rate=0.1)
        # a-b: 3 co-occurrences; a-c: 2; a-d: 1
        for _ in range(3):
            heb.record_coretrieval(["a", "b"])
        for _ in range(2):
            heb.record_coretrieval(["a", "c"])
        heb.record_coretrieval(["a", "d"])

        top2 = heb.related("a", top_k=2)
        assert len(top2) == 2
        assert top2[0][0] == "b"
        assert top2[1][0] == "c"

    def test_decay_shrinks_weights(self) -> None:
        heb = HebbianAssociation(learning_rate=0.1, decay=0.5)
        heb.record_coretrieval(["a", "b"])  # weight = 0.1
        # First tick of decay happens on the next record_coretrieval
        heb.record_coretrieval(["c", "d"])
        # a-b weight is now 0.05 (decayed by 0.5)
        assert heb.weight("a", "b") == pytest.approx(0.05)
        assert heb.weight("c", "d") == pytest.approx(0.1)

    def test_serialise_deserialise_round_trip(self) -> None:
        heb = HebbianAssociation(learning_rate=0.1)
        heb.record_coretrieval(["a", "b"])
        heb.record_coretrieval(["a", "c"])
        data = heb.serialize()

        restored = HebbianAssociation.deserialize(data)
        assert restored.weight("a", "b") == heb.weight("a", "b")
        assert restored.weight("a", "c") == heb.weight("a", "c")

    def test_rejects_invalid_rates(self) -> None:
        with pytest.raises(ValueError):
            HebbianAssociation(learning_rate=0)
        with pytest.raises(ValueError):
            HebbianAssociation(decay=1.5)
