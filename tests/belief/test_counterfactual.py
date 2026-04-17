"""Tests for order-sensitive / counterfactual belief operations (v0.4 #5)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from patha.belief.counterfactual import (
    order_sensitivity,
    replay_in_order,
)
from patha.belief.store import BeliefStore
from patha.belief.types import ResolutionStatus


def _build_chain_store(tmp_path: Path) -> Path:
    """Build a persistent store with a known supersession chain.

    A -> B (supersede A), B -> C (supersede B). Final state:
      C is CURRENT
      B is SUPERSEDED
      A is SUPERSEDED
    """
    path = tmp_path / "chain.jsonl"
    store = BeliefStore(persistence_path=path)
    store.add(
        proposition="I live in Sydney",
        asserted_at=datetime(2022, 1, 1),
        asserted_in_session="s1",
        source_proposition_id="p1",
        belief_id="A",
    )
    store.add(
        proposition="I moved to Sofia",
        asserted_at=datetime(2023, 1, 1),
        asserted_in_session="s2",
        source_proposition_id="p2",
        belief_id="B",
    )
    store.supersede("A", "B")
    store.add(
        proposition="I moved to Berlin",
        asserted_at=datetime(2024, 1, 1),
        asserted_in_session="s3",
        source_proposition_id="p3",
        belief_id="C",
    )
    store.supersede("B", "C")
    return path


# ─── replay_in_order ───────────────────────────────────────────────

class TestReplay:
    def test_replay_in_original_order_matches_original(
        self, tmp_path: Path
    ) -> None:
        path = _build_chain_store(tmp_path)
        replayed = replay_in_order(path, ["A", "B", "C"])

        assert len(replayed) == 3
        assert replayed.get("A").status == ResolutionStatus.SUPERSEDED  # type: ignore[union-attr]
        assert replayed.get("B").status == ResolutionStatus.SUPERSEDED  # type: ignore[union-attr]
        assert replayed.get("C").status == ResolutionStatus.CURRENT  # type: ignore[union-attr]

    def test_replay_in_different_order_preserves_relations(
        self, tmp_path: Path
    ) -> None:
        """Supersede events reference ids, so reordering the add events
        doesn't break the supersede relations — they still point at
        the right pairs, but the 'temporal perception' of the chain
        changes (A is now the 'newest' added)."""
        path = _build_chain_store(tmp_path)
        replayed = replay_in_order(path, ["C", "B", "A"])

        assert len(replayed) == 3
        # Supersede relations still exist; C is still CURRENT
        # (because the superseded-by edges are attached to A and B).
        assert replayed.get("C").status == ResolutionStatus.CURRENT  # type: ignore[union-attr]
        assert replayed.get("A").is_superseded  # type: ignore[union-attr]
        assert replayed.get("B").is_superseded  # type: ignore[union-attr]

    def test_replay_does_not_mutate_original(self, tmp_path: Path) -> None:
        path = _build_chain_store(tmp_path)
        size_before = path.stat().st_size
        replay_in_order(path, ["B", "A", "C"])
        size_after = path.stat().st_size
        assert size_before == size_after

    def test_unlisted_ids_appended(self, tmp_path: Path) -> None:
        """IDs missing from the explicit ordering get appended in
        original order."""
        path = _build_chain_store(tmp_path)
        # Only specify 2 of 3 ids
        replayed = replay_in_order(path, ["C", "A"])
        # B should be appended
        assert len(replayed) == 3
        assert replayed.get("B") is not None


# ─── order_sensitivity ─────────────────────────────────────────────

class TestOrderSensitivity:
    def test_identical_ordering_has_zero_divergence(
        self, tmp_path: Path
    ) -> None:
        path = _build_chain_store(tmp_path)
        result = order_sensitivity(
            path,
            orderings=[
                ["A", "B", "C"],
                ["A", "B", "C"],
            ],
        )
        assert result["divergence"] == 0.0

    def test_reversed_ordering_still_stable_for_linear_chain(
        self, tmp_path: Path
    ) -> None:
        """A linear supersession chain is stable under reordering —
        supersede edges are content-addressed, so final state is the
        same. Divergence = 0 for this particular chain."""
        path = _build_chain_store(tmp_path)
        result = order_sensitivity(
            path,
            orderings=[
                ["A", "B", "C"],
                ["C", "B", "A"],
            ],
        )
        # Edges point from superseded_by to superseder regardless of
        # add-order, so the final current/superseded sets are stable
        assert result["divergence"] == 0.0

    def test_per_ordering_captures_state(self, tmp_path: Path) -> None:
        path = _build_chain_store(tmp_path)
        result = order_sensitivity(
            path,
            orderings=[["A", "B", "C"], ["C", "B", "A"]],
        )
        assert len(result["per_ordering"]) == 2
        # Both orderings end with C current, A and B superseded
        for per in result["per_ordering"]:
            assert "C" in per["current_ids"]
            assert "A" in per["superseded_ids"]
            assert "B" in per["superseded_ids"]

    def test_rejects_single_ordering(self, tmp_path: Path) -> None:
        path = _build_chain_store(tmp_path)
        with pytest.raises(ValueError, match="at least two"):
            order_sensitivity(path, orderings=[["A", "B"]])
