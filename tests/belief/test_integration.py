"""End-to-end integration tests for the belief layer.

Exercises the full path: BeliefLayer + BeliefStore + detector +
validity extraction + query filtering + summary rendering. Uses only
the stub detector so these stay fast and deterministic.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from patha.belief import (
    BeliefLayer,
    BeliefStore,
    ContradictionLabel,
    ContradictionResult,
    StubContradictionDetector,
)


class _ScriptedDetector:
    def __init__(self, scripts):
        self.scripts = scripts

    def detect(self, p1, p2):
        return self.scripts.get(
            (p1, p2),
            ContradictionResult(label=ContradictionLabel.NEUTRAL, confidence=0.5),
        )

    def detect_batch(self, pairs):
        return [self.detect(p1, p2) for p1, p2 in pairs]


class TestEndToEnd:
    def test_typical_preference_shift(self) -> None:
        """Full story: user changes preference, asks what's current."""
        scripts = {
            (
                "I love sushi and eat it every week",
                "I am avoiding raw fish for medical reasons",
            ): ContradictionResult(
                label=ContradictionLabel.CONTRADICTS, confidence=0.92
            ),
        }
        layer = BeliefLayer(
            store=BeliefStore(),
            detector=_ScriptedDetector(scripts),
        )

        e1 = layer.ingest(
            proposition="I love sushi and eat it every week",
            asserted_at=datetime(2023, 6, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
        )
        e2 = layer.ingest(
            proposition="I am avoiding raw fish for medical reasons",
            asserted_at=datetime(2024, 2, 1),
            asserted_in_session="s2",
            source_proposition_id="p2",
        )

        # Old belief should be superseded, new current
        assert e2.action == "superseded"
        assert layer.store.get(e1.new_belief.id).is_superseded  # type: ignore[union-attr]

        # Query current-only: only the new belief surfaces
        result = layer.query(
            [e1.new_belief.id, e2.new_belief.id], at_time=datetime(2024, 6, 1)
        )
        assert len(result.current) == 1
        assert "avoiding raw fish" in result.current[0].proposition

        # Query with history: both surface
        result_hist = layer.query(
            [e1.new_belief.id, e2.new_belief.id],
            at_time=datetime(2024, 6, 1),
            include_history=True,
        )
        assert len(result_hist.current) == 1
        assert len(result_hist.history) == 1

    def test_temporally_bounded_expires(self) -> None:
        layer = BeliefLayer(
            store=BeliefStore(), detector=StubContradictionDetector()
        )
        ev = layer.ingest(
            proposition=(
                "I am on paternity leave for three weeks starting March 1"
            ),
            asserted_at=datetime(2024, 3, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
        )
        # Inside the window -> current
        r_inside = layer.query([ev.new_belief.id], at_time=datetime(2024, 3, 15))
        assert len(r_inside.current) == 1
        # Outside the window -> filtered out
        r_outside = layer.query([ev.new_belief.id], at_time=datetime(2024, 4, 30))
        assert len(r_outside.current) == 0

    def test_persistence_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "beliefs.jsonl"
        scripts = {
            (
                "I live in Sydney",
                "I moved to Sofia in January",
            ): ContradictionResult(
                label=ContradictionLabel.CONTRADICTS, confidence=0.95
            ),
        }

        # Session 1: ingest through the layer
        layer1 = BeliefLayer(
            store=BeliefStore(persistence_path=path),
            detector=_ScriptedDetector(scripts),
        )
        layer1.ingest(
            proposition="I live in Sydney",
            asserted_at=datetime(2022, 6, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
        )
        e2 = layer1.ingest(
            proposition="I moved to Sofia in January",
            asserted_at=datetime(2024, 1, 15),
            asserted_in_session="s2",
            source_proposition_id="p2",
        )

        # Session 2: reopen from disk, state must be consistent
        layer2 = BeliefLayer(
            store=BeliefStore(persistence_path=path),
            detector=_ScriptedDetector(scripts),
        )
        result = layer2.query(
            [b.id for b in layer2.store.all()],
            at_time=datetime(2024, 6, 1),
        )
        assert len(result.current) == 1
        assert "Sofia" in result.current[0].proposition

    def test_compression_claim_holds_for_long_chain(self) -> None:
        """A chain of five supersessions compresses to one current belief."""
        scripts = {
            (f"City {i}", f"City {i+1}"): ContradictionResult(
                label=ContradictionLabel.CONTRADICTS, confidence=0.95
            )
            for i in range(5)
        }
        layer = BeliefLayer(
            store=BeliefStore(), detector=_ScriptedDetector(scripts)
        )
        ids = []
        for i in range(6):
            ev = layer.ingest(
                proposition=f"City {i}",
                asserted_at=datetime(2020 + i, 1, 1),
                asserted_in_session=f"s{i}",
                source_proposition_id=f"p{i}",
            )
            ids.append(ev.new_belief.id)

        # Current-only summary compresses 6 historical assertions to 1
        current_only = layer.query(ids, at_time=datetime(2030, 1, 1))
        with_history = layer.query(
            ids, at_time=datetime(2030, 1, 1), include_history=True
        )

        assert len(current_only.current) == 1
        assert len(with_history.history) == 5
        # Summary token count must grow when history is included
        assert (
            with_history.tokens_in_summary
            > current_only.tokens_in_summary
        )

        # The current-only rendering mentions only the latest city
        rendered = layer.render_summary(current_only)
        assert "City 5" in rendered
        for i in range(5):
            assert f"City {i}\n" not in rendered and f"City {i}]" not in rendered
