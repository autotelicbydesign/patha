"""Tests for the ingest-time sliding window (D2 hybrid completion)."""

from __future__ import annotations

from datetime import datetime

from patha.belief.contradiction import StubContradictionDetector
from patha.belief.layer import BeliefLayer, PlasticityConfig
from patha.belief.store import BeliefStore
from patha.belief.types import (
    ContradictionLabel,
    ContradictionResult,
)


class _ContradictAllDetector:
    def detect(self, p1, p2):
        return ContradictionResult(
            label=ContradictionLabel.CONTRADICTS, confidence=0.95
        )

    def detect_batch(self, pairs):
        return [self.detect(p1, p2) for p1, p2 in pairs]


class TestIngestWindow:
    def test_without_window_checks_all_beliefs(self) -> None:
        """Default behaviour: contradiction checks run against every
        current belief, regardless of age."""
        layer = BeliefLayer(
            store=BeliefStore(),
            detector=_ContradictAllDetector(),
            plasticity=PlasticityConfig(enabled=False),
            contradiction_threshold=0.5,
            ingest_window_days=None,  # no window
        )
        # Old belief
        layer.ingest(
            proposition="x",
            asserted_at=datetime(2020, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
        )
        # New belief — old should still be a candidate
        ev = layer.ingest(
            proposition="y",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s2",
            source_proposition_id="p2",
        )
        # Detector says CONTRADICTS for every pair, so ev should
        # supersede the old belief.
        assert ev.action == "superseded"

    def test_window_filters_out_old_beliefs(self) -> None:
        """With a 30-day window, a belief from 4 years ago isn't a
        candidate for contradiction."""
        layer = BeliefLayer(
            store=BeliefStore(),
            detector=_ContradictAllDetector(),
            plasticity=PlasticityConfig(enabled=False),
            contradiction_threshold=0.5,
            ingest_window_days=30,
        )
        layer.ingest(
            proposition="x",
            asserted_at=datetime(2020, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
        )
        ev = layer.ingest(
            proposition="y",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s2",
            source_proposition_id="p2",
        )
        # Old belief filtered out by window → no candidates → ADDED
        assert ev.action == "added"

    def test_window_includes_beliefs_inside_window(self) -> None:
        """A belief from last week is inside a 30-day window."""
        layer = BeliefLayer(
            store=BeliefStore(),
            detector=_ContradictAllDetector(),
            plasticity=PlasticityConfig(enabled=False),
            contradiction_threshold=0.5,
            ingest_window_days=30,
        )
        layer.ingest(
            proposition="x",
            asserted_at=datetime(2024, 1, 25),
            asserted_in_session="s1",
            source_proposition_id="p1",
        )
        ev = layer.ingest(
            proposition="y",
            asserted_at=datetime(2024, 2, 1),
            asserted_in_session="s2",
            source_proposition_id="p2",
        )
        # 7 days apart → inside the 30-day window → supersession fires
        assert ev.action == "superseded"

    def test_window_exactly_at_boundary(self) -> None:
        """Boundary case: belief from exactly 30 days ago."""
        layer = BeliefLayer(
            store=BeliefStore(),
            detector=_ContradictAllDetector(),
            plasticity=PlasticityConfig(enabled=False),
            contradiction_threshold=0.5,
            ingest_window_days=30,
        )
        layer.ingest(
            proposition="x",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
        )
        ev = layer.ingest(
            proposition="y",
            asserted_at=datetime(2024, 1, 31),  # exactly 30 days later
            asserted_in_session="s2",
            source_proposition_id="p2",
        )
        # 30 days → inclusive boundary → candidate
        assert ev.action == "superseded"

    def test_window_plus_explicit_candidates(self) -> None:
        """Explicit candidate_belief_ids intersect with the window."""
        layer = BeliefLayer(
            store=BeliefStore(),
            detector=_ContradictAllDetector(),
            plasticity=PlasticityConfig(enabled=False),
            contradiction_threshold=0.5,
            ingest_window_days=30,
        )
        e1 = layer.ingest(
            proposition="old",
            asserted_at=datetime(2020, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
        )
        e2 = layer.ingest(
            proposition="recent",
            asserted_at=datetime(2024, 1, 25),
            asserted_in_session="s2",
            source_proposition_id="p2",
        )
        ev = layer.ingest(
            proposition="new",
            asserted_at=datetime(2024, 2, 1),
            asserted_in_session="s3",
            source_proposition_id="p3",
            candidate_belief_ids=[e1.new_belief.id, e2.new_belief.id],
        )
        # Only 'recent' is inside the window; 'old' filtered.
        # Detector says all contradict; so the recent one gets superseded.
        assert ev.action == "superseded"
        assert e2.new_belief.id in ev.affected_belief_ids
        assert e1.new_belief.id not in ev.affected_belief_ids
