"""Tests for vṛtti-aware direct-answer policy (v0.5 #6)."""

from __future__ import annotations

from datetime import datetime

import pytest

from patha.belief.direct_answer import DirectAnswerer
from patha.belief.store import BeliefStore
from patha.belief.types import (
    ContradictionLabel,
    ContradictionResult,
    Pramana,
    ResolutionStatus,
)


def _populate_store() -> tuple[BeliefStore, dict[str, str]]:
    """Make a store with mixed-confidence current beliefs."""
    store = BeliefStore()
    high = store.add(
        proposition="I live in Sofia",
        asserted_at=datetime(2024, 1, 1),
        asserted_in_session="s1",
        source_proposition_id="p1",
        belief_id="high",
        confidence=0.9,
    )
    low = store.add(
        proposition="I think the meeting is on Thursday",
        asserted_at=datetime(2024, 1, 2),
        asserted_in_session="s2",
        source_proposition_id="p2",
        belief_id="low",
        confidence=0.3,
    )
    return store, {"high": high.id, "low": low.id}


class TestVikalpaFiltering:
    def test_low_confidence_filtered_by_default(self) -> None:
        """Default: vikalpa beliefs omitted from direct answers."""
        store, ids = _populate_store()
        answerer = DirectAnswerer(store)
        result = answerer.try_answer(
            "What do I currently believe about my location?",
            list(ids.values()),
            at_time=datetime(2024, 6, 1),
        )
        assert result is not None
        # High-confidence surfaces
        assert "Sofia" in result.text
        # Low-confidence filtered
        assert "Thursday" not in result.text
        assert "high" in result.belief_ids
        assert "low" not in result.belief_ids

    def test_surface_vikalpa_opt_in(self) -> None:
        """Callers can opt in to surfacing low-confidence beliefs."""
        store, ids = _populate_store()
        answerer = DirectAnswerer(store, surface_vikalpa=True)
        result = answerer.try_answer(
            "What do I currently believe?",
            list(ids.values()),
            at_time=datetime(2024, 6, 1),
        )
        assert result is not None
        assert "Sofia" in result.text
        assert "Thursday" in result.text

    def test_custom_vikalpa_threshold(self) -> None:
        """Callers can tune where the vikalpa line is."""
        store, ids = _populate_store()
        # A low threshold means even 0.3 passes
        answerer = DirectAnswerer(store, vikalpa_threshold=0.1)
        result = answerer.try_answer(
            "What do I currently believe?",
            list(ids.values()),
            at_time=datetime(2024, 6, 1),
        )
        assert result is not None
        assert "Thursday" in result.text  # 0.3 > 0.1 → passes


class TestViparyayaFlagging:
    def test_disputed_belief_flagged(self) -> None:
        """Disputed beliefs are still surfaced but with a caveat."""
        store = BeliefStore()
        store.add(
            proposition="Alice is the lead",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
            belief_id="a",
        )
        store.add(
            proposition="Bob is the lead",
            asserted_at=datetime(2024, 2, 1),
            asserted_in_session="s2",
            source_proposition_id="p2",
            belief_id="b",
        )
        store.dispute("a", "b")
        answerer = DirectAnswerer(store)
        result = answerer.try_answer(
            "Who is currently the lead?",
            ["a", "b"],
            at_time=datetime(2024, 6, 1),
        )
        assert result is not None
        assert "disputed" in result.text.lower()

    def test_flag_viparyaya_disable(self) -> None:
        """Callers can disable the disputed caveat."""
        store = BeliefStore()
        store.add(
            proposition="Alice is the lead",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
            belief_id="a",
        )
        store.add(
            proposition="Bob is the lead",
            asserted_at=datetime(2024, 2, 1),
            asserted_in_session="s2",
            source_proposition_id="p2",
            belief_id="b",
        )
        store.dispute("a", "b")
        answerer = DirectAnswerer(store, flag_viparyaya=False)
        result = answerer.try_answer(
            "Who is currently the lead?",
            ["a", "b"],
            at_time=datetime(2024, 6, 1),
        )
        assert result is not None
        assert "disputed" not in result.text.lower()


class TestVasanaSurvival:
    def test_decayed_surface_with_deep_still_surfaces(self) -> None:
        """A belief whose surface confidence has decayed below
        vikalpa_threshold but whose deep (vāsanā) confidence is high
        should still be surfaced — effective_confidence is the gate."""
        store = BeliefStore()
        b = store.add(
            proposition="I was born in Sofia",
            asserted_at=datetime(2020, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
            belief_id="b",
            confidence=0.3,  # decayed surface
        )
        b.deep_confidence = 0.9  # strong vāsanā
        answerer = DirectAnswerer(store)
        result = answerer.try_answer(
            "Where do I currently live?",
            ["b"],
            at_time=datetime(2024, 1, 1),
        )
        # effective_confidence = max(0.3, 0.9) = 0.9 → surfaces
        assert result is not None
        assert "Sofia" in result.text


class TestEmptyAfterFiltering:
    def test_all_vikalpa_returns_none(self) -> None:
        """If every candidate is vikalpa and surface_vikalpa=False,
        nothing to surface → return None → caller falls back to LLM."""
        store = BeliefStore()
        store.add(
            proposition="I think X is true",
            asserted_at=datetime(2024, 1, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
            belief_id="b",
            confidence=0.2,
        )
        answerer = DirectAnswerer(store)
        result = answerer.try_answer(
            "What do I currently believe about X?",
            ["b"],
            at_time=datetime(2024, 6, 1),
        )
        assert result is None  # fell through
