"""Tests for confidence-weighted supersession (D3 advanced)."""

from __future__ import annotations

from datetime import datetime

import pytest

from patha.belief.store import BeliefStore
from patha.belief.types import (
    ContradictionLabel,
    ContradictionResult,
    Pramana,
    ResolutionStatus,
)


def _add(store: BeliefStore, bid: str, conf: float, when: datetime, pramana=Pramana.PRATYAKSA):
    return store.add(
        proposition=f"p-{bid}",
        asserted_at=when,
        asserted_in_session=f"sess-{bid}",
        source_proposition_id=f"prop-{bid}",
        belief_id=bid,
        confidence=conf,
        pramana=pramana,
    )


class TestConfidenceWeightedDisabled:
    """Default (confidence_weighted=False): pure temporal supersession when
    pramāṇas tie, regardless of confidence."""

    def test_temporal_wins_when_pramanas_equal(self) -> None:
        store = BeliefStore()
        _add(store, "old", conf=0.95, when=datetime(2024, 1, 1))
        _add(store, "new", conf=0.30, when=datetime(2024, 2, 1))

        status = store.resolve_contradiction("old", "new")
        # Default: temporal supersession — new (even though less confident) wins.
        assert status == ResolutionStatus.SUPERSEDED
        assert store.get("old").status == ResolutionStatus.SUPERSEDED  # type: ignore[union-attr]


class TestConfidenceWeightedEnabled:
    def test_higher_confidence_sublates_lower(self) -> None:
        """Pramāṇas equal, new confidence >> old → new sublates old."""
        store = BeliefStore()
        _add(store, "old", conf=0.30, when=datetime(2024, 1, 1))
        _add(store, "new", conf=0.95, when=datetime(2024, 2, 1))

        status = store.resolve_contradiction(
            "old", "new", confidence_weighted=True
        )
        assert status == ResolutionStatus.BADHITA
        assert store.get("old").status == ResolutionStatus.BADHITA  # type: ignore[union-attr]

    def test_higher_confidence_old_sublates_new(self) -> None:
        """Even older belief wins if its confidence is meaningfully higher."""
        store = BeliefStore()
        _add(store, "old", conf=0.95, when=datetime(2024, 1, 1))
        _add(store, "new", conf=0.30, when=datetime(2024, 2, 1))

        status = store.resolve_contradiction(
            "old", "new", confidence_weighted=True
        )
        # Old is much more confident → new is sublated
        assert status == ResolutionStatus.BADHITA
        assert store.get("new").status == ResolutionStatus.BADHITA  # type: ignore[union-attr]
        assert store.get("old").is_current  # type: ignore[union-attr]

    def test_within_margin_falls_back_to_temporal(self) -> None:
        """Confidence within margin → temporal supersession wins."""
        store = BeliefStore()
        _add(store, "old", conf=0.70, when=datetime(2024, 1, 1))
        _add(store, "new", conf=0.80, when=datetime(2024, 2, 1))

        status = store.resolve_contradiction(
            "old", "new",
            confidence_weighted=True,
            confidence_margin=0.2,  # delta=0.10 < margin
        )
        assert status == ResolutionStatus.SUPERSEDED

    def test_margin_is_configurable(self) -> None:
        """Tighter margin makes more pairs confidence-weighted."""
        store = BeliefStore()
        _add(store, "old", conf=0.60, when=datetime(2024, 1, 1))
        _add(store, "new", conf=0.75, when=datetime(2024, 2, 1))

        status = store.resolve_contradiction(
            "old", "new",
            confidence_weighted=True,
            confidence_margin=0.10,  # delta=0.15 > margin
        )
        assert status == ResolutionStatus.BADHITA

    def test_pramana_hierarchy_still_wins_over_confidence(self) -> None:
        """Pramāṇa hierarchy is tier 1; confidence-weighting is tier 2.

        Even if old's confidence is higher, a stronger-pramāṇa new
        belief still sublates old.
        """
        store = BeliefStore()
        _add(
            store, "old", conf=0.95, when=datetime(2024, 1, 1),
            pramana=Pramana.SHABDA,
        )
        _add(
            store, "new", conf=0.50, when=datetime(2024, 2, 1),
            pramana=Pramana.PRATYAKSA,
        )

        status = store.resolve_contradiction(
            "old", "new", confidence_weighted=True
        )
        # Pramāṇa hierarchy: PRATYAKṢA > SHABDA → new sublates old
        # (even though old's confidence was higher)
        assert status == ResolutionStatus.BADHITA
        assert store.get("old").status == ResolutionStatus.BADHITA  # type: ignore[union-attr]


class TestLayerIntegration:
    def test_layer_confidence_weighted_flag_routes_through(self) -> None:
        """BeliefLayer's confidence_weighted_supersession flag actually
        reaches the store's resolve_contradiction call."""
        from patha.belief.layer import BeliefLayer, PlasticityConfig

        class AlwaysContradicts:
            def detect(self, p1, p2):
                return ContradictionResult(
                    label=ContradictionLabel.CONTRADICTS, confidence=0.95
                )

            def detect_batch(self, pairs):
                return [self.detect(p1, p2) for p1, p2 in pairs]

        layer = BeliefLayer(
            store=BeliefStore(),
            detector=AlwaysContradicts(),
            plasticity=PlasticityConfig(enabled=False),
            contradiction_threshold=0.5,
            confidence_weighted_supersession=True,
            confidence_margin=0.2,
        )

        e1 = layer.ingest(
            proposition="high conf old claim",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
            pramana=Pramana.PRATYAKSA,
            confidence=0.95,
        )
        e2 = layer.ingest(
            proposition="low conf new claim",
            asserted_at=datetime(2024, 2, 1),
            asserted_in_session="s2",
            source_proposition_id="p2",
            pramana=Pramana.PRATYAKSA,
            confidence=0.30,
        )

        # Confidence-weighted: old (higher confidence) wins, new is BADHITA
        assert layer.store.get(e2.new_belief.id).status == ResolutionStatus.BADHITA  # type: ignore[union-attr]
        assert layer.store.get(e1.new_belief.id).is_current  # type: ignore[union-attr]
