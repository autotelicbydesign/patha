"""Tests for BeliefLayer (the public API)."""

from __future__ import annotations

from datetime import datetime

import pytest

from patha.belief.contradiction import StubContradictionDetector
from patha.belief.layer import (
    BeliefLayer,
    BeliefQueryResult,
    IngestEvent,
)
from patha.belief.store import BeliefStore
from patha.belief.types import (
    ContradictionLabel,
    ContradictionResult,
    Validity,
)


# ─── Scripted detector for deterministic tests ───────────────────────

class ScriptedDetector:
    """A detector that returns pre-programmed verdicts per pair."""

    def __init__(self, scripts: dict[tuple[str, str], ContradictionResult]):
        self.scripts = scripts
        self.default = ContradictionResult(
            label=ContradictionLabel.NEUTRAL, confidence=0.5
        )

    def detect(self, p1: str, p2: str) -> ContradictionResult:
        return self.scripts.get((p1, p2), self.default)

    def detect_batch(self, pairs):
        return [self.detect(p1, p2) for p1, p2 in pairs]


# ─── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def stub_layer() -> BeliefLayer:
    return BeliefLayer(
        store=BeliefStore(), detector=StubContradictionDetector()
    )


def _ingest(
    layer: BeliefLayer,
    prop: str,
    *,
    at: datetime | None = None,
    session: str = "s1",
    pid: str | None = None,
    validity: Validity | None = None,
) -> IngestEvent:
    return layer.ingest(
        proposition=prop,
        asserted_at=at if at is not None else datetime(2024, 1, 1, 12, 0),
        asserted_in_session=session,
        source_proposition_id=pid if pid is not None else f"prop-{prop[:10]}",
        validity=validity,
    )


# ─── ingest ──────────────────────────────────────────────────────────

class TestIngestAdd:
    def test_first_proposition_is_added(self, stub_layer: BeliefLayer) -> None:
        ev = _ingest(stub_layer, "I love sushi")
        assert ev.action == "added"
        assert ev.new_belief.proposition == "I love sushi"
        assert ev.affected_belief_ids == ()
        assert ev.contradictions_detected == 0

    def test_two_unrelated_propositions_both_added(
        self, stub_layer: BeliefLayer
    ) -> None:
        ev1 = _ingest(stub_layer, "I love sushi", pid="p1")
        ev2 = _ingest(
            stub_layer, "The weather is lovely today", pid="p2"
        )
        assert ev1.action == "added"
        assert ev2.action == "added"


class TestIngestSupersede:
    def test_stub_detector_supersedes_on_negation(
        self, stub_layer: BeliefLayer
    ) -> None:
        # Stub detector fires when there's content overlap + asymmetric negation.
        # But stub confidence is 0.6 which is below default threshold 0.75.
        # So we need a layer with lower threshold.
        layer = BeliefLayer(
            store=BeliefStore(),
            detector=StubContradictionDetector(),
            contradiction_threshold=0.5,
        )
        _ingest(layer, "I love sushi regularly", pid="p1")
        ev = _ingest(
            layer, "I never eat sushi anymore", pid="p2"
        )
        assert ev.action == "superseded"
        assert len(ev.affected_belief_ids) == 1
        assert ev.contradictions_detected == 1

    def test_scripted_contradiction_supersedes(self) -> None:
        scripts = {
            ("I love sushi", "I'm avoiding raw fish"): ContradictionResult(
                label=ContradictionLabel.CONTRADICTS, confidence=0.95
            ),
        }
        layer = BeliefLayer(
            store=BeliefStore(), detector=ScriptedDetector(scripts)
        )
        _ingest(layer, "I love sushi", pid="p1")
        ev = _ingest(layer, "I'm avoiding raw fish", pid="p2")
        assert ev.action == "superseded"
        old = layer.store.get(ev.affected_belief_ids[0])
        assert old is not None and old.is_superseded
        assert ev.new_belief.is_current

    def test_below_threshold_does_not_supersede(self) -> None:
        # Confidence below threshold is ignored
        scripts = {
            ("I love sushi", "I'm avoiding raw fish"): ContradictionResult(
                label=ContradictionLabel.CONTRADICTS, confidence=0.6
            ),
        }
        layer = BeliefLayer(
            store=BeliefStore(),
            detector=ScriptedDetector(scripts),
            contradiction_threshold=0.75,
        )
        _ingest(layer, "I love sushi", pid="p1")
        ev = _ingest(layer, "I'm avoiding raw fish", pid="p2")
        assert ev.action == "added"  # not superseded, because conf too low

    def test_one_new_supersedes_multiple_olds(self) -> None:
        scripts = {
            ("I love sushi", "I'm vegetarian now"): ContradictionResult(
                label=ContradictionLabel.CONTRADICTS, confidence=0.95
            ),
            ("I love steak", "I'm vegetarian now"): ContradictionResult(
                label=ContradictionLabel.CONTRADICTS, confidence=0.95
            ),
        }
        layer = BeliefLayer(
            store=BeliefStore(), detector=ScriptedDetector(scripts)
        )
        _ingest(layer, "I love sushi", pid="p1")
        _ingest(layer, "I love steak", pid="p2")
        ev = _ingest(layer, "I'm vegetarian now", pid="p3")
        assert ev.action == "superseded"
        assert len(ev.affected_belief_ids) == 2
        assert ev.contradictions_detected == 2


class TestIngestReinforce:
    def test_scripted_entailment_reinforces(self) -> None:
        scripts = {
            ("I have a golden retriever", "I have a dog"): ContradictionResult(
                label=ContradictionLabel.ENTAILS, confidence=0.9
            ),
        }
        layer = BeliefLayer(
            store=BeliefStore(), detector=ScriptedDetector(scripts)
        )
        first = _ingest(layer, "I have a golden retriever", pid="p1")
        first.new_belief.confidence = 0.5  # simulate some prior decay
        ev = _ingest(layer, "I have a dog", pid="p2")
        assert ev.action == "reinforced"
        # Existing belief should have reinforced_by edge
        existing = layer.store.get(ev.affected_belief_ids[0])
        assert existing is not None
        assert ev.new_belief.id in existing.reinforced_by
        assert existing.confidence > 0.5  # bumped

    def test_only_first_entailment_reinforces(self) -> None:
        # If the new assertion entails multiple, reinforce only one
        scripts = {
            ("I have a golden retriever", "I have a pet"): ContradictionResult(
                label=ContradictionLabel.ENTAILS, confidence=0.9
            ),
            ("I have a cat", "I have a pet"): ContradictionResult(
                label=ContradictionLabel.ENTAILS, confidence=0.9
            ),
        }
        layer = BeliefLayer(
            store=BeliefStore(), detector=ScriptedDetector(scripts)
        )
        _ingest(layer, "I have a golden retriever", pid="p1")
        _ingest(layer, "I have a cat", pid="p2")
        ev = _ingest(layer, "I have a pet", pid="p3")
        assert ev.action == "reinforced"
        assert len(ev.affected_belief_ids) == 1  # only one, not both


# ─── query ───────────────────────────────────────────────────────────

class TestQuery:
    def test_current_only_by_default(self) -> None:
        scripts = {
            ("I live in Sydney", "I moved to Sofia"): ContradictionResult(
                label=ContradictionLabel.CONTRADICTS, confidence=0.95
            ),
        }
        layer = BeliefLayer(
            store=BeliefStore(), detector=ScriptedDetector(scripts)
        )
        e1 = _ingest(layer, "I live in Sydney", pid="p1")
        e2 = _ingest(layer, "I moved to Sofia", pid="p2")

        result = layer.query([e1.new_belief.id, e2.new_belief.id])
        assert len(result.current) == 1
        assert result.current[0].proposition == "I moved to Sofia"
        assert result.history == []

    def test_include_history_returns_ancestors(self) -> None:
        scripts = {
            ("I live in Sydney", "I moved to Sofia"): ContradictionResult(
                label=ContradictionLabel.CONTRADICTS, confidence=0.95
            ),
            ("I moved to Sofia", "I moved to Berlin"): ContradictionResult(
                label=ContradictionLabel.CONTRADICTS, confidence=0.95
            ),
        }
        layer = BeliefLayer(
            store=BeliefStore(), detector=ScriptedDetector(scripts)
        )
        _ingest(layer, "I live in Sydney", pid="p1")
        _ingest(layer, "I moved to Sofia", pid="p2")
        e3 = _ingest(layer, "I moved to Berlin", pid="p3")

        result = layer.query([e3.new_belief.id], include_history=True)
        assert len(result.current) == 1
        assert len(result.history) == 2  # Sydney, Sofia
        props = {b.proposition for b in result.history}
        assert "I live in Sydney" in props
        assert "I moved to Sofia" in props

    def test_validity_filter_applies(self) -> None:
        layer = BeliefLayer(
            store=BeliefStore(), detector=StubContradictionDetector()
        )
        validity = Validity(
            mode="dated_range",
            start=datetime(2024, 1, 1),
            end=datetime(2024, 6, 30),
            source="explicit",
        )
        ev = _ingest(
            layer,
            "I am training for a marathon",
            at=datetime(2024, 1, 1),
            pid="p1",
            validity=validity,
        )

        # Inside the validity window → returned
        result_in = layer.query([ev.new_belief.id], at_time=datetime(2024, 3, 15))
        assert len(result_in.current) == 1

        # Outside the validity window → filtered out
        result_out = layer.query(
            [ev.new_belief.id], at_time=datetime(2024, 8, 1)
        )
        assert len(result_out.current) == 0

    def test_missing_ids_ignored(self, stub_layer: BeliefLayer) -> None:
        result = stub_layer.query(["nonexistent1", "nonexistent2"])
        assert result.current == []

    def test_duplicate_ids_deduped(self, stub_layer: BeliefLayer) -> None:
        ev = _ingest(stub_layer, "I love sushi")
        result = stub_layer.query(
            [ev.new_belief.id, ev.new_belief.id, ev.new_belief.id]
        )
        assert len(result.current) == 1

    def test_tokens_in_summary_tracks_content(
        self, stub_layer: BeliefLayer
    ) -> None:
        short = _ingest(stub_layer, "Hi", pid="p1")
        long = _ingest(
            stub_layer,
            "I have a very long and elaborate belief about many topics",
            pid="p2",
        )
        r_short = stub_layer.query([short.new_belief.id])
        r_long = stub_layer.query([long.new_belief.id])
        assert r_short.tokens_in_summary < r_long.tokens_in_summary


# ─── rendering ───────────────────────────────────────────────────────

class TestAutoValidityExtraction:
    def test_validity_extracted_from_proposition(
        self, stub_layer: BeliefLayer
    ) -> None:
        ev = _ingest(
            stub_layer,
            "I'm avoiding raw fish for three weeks",
            at=datetime(2024, 3, 1),
        )
        assert ev.new_belief.validity.mode == "dated_range"
        assert ev.new_belief.validity.source == "explicit"

    def test_explicit_validity_overrides_extraction(
        self, stub_layer: BeliefLayer
    ) -> None:
        explicit = Validity(
            mode="dated_range",
            start=datetime(2024, 1, 1),
            end=datetime(2024, 12, 31),
            source="explicit",
        )
        ev = _ingest(
            stub_layer,
            "I'm avoiding raw fish for three weeks",  # would extract
            at=datetime(2024, 3, 1),
            validity=explicit,
        )
        # Explicit kwarg wins over extracted
        assert ev.new_belief.validity.end == datetime(2024, 12, 31)

    def test_no_marker_no_validity_means_permanent(
        self, stub_layer: BeliefLayer
    ) -> None:
        ev = _ingest(stub_layer, "I love sushi")
        assert ev.new_belief.validity.mode == "permanent"

    def test_auto_extraction_can_be_disabled(self) -> None:
        from patha.belief.layer import BeliefLayer
        layer = BeliefLayer(
            detector=StubContradictionDetector(),
            auto_extract_validity=False,
        )
        ev = layer.ingest(
            proposition="I'm out for three weeks",
            asserted_at=datetime(2024, 3, 1),
            asserted_in_session="s1",
            source_proposition_id="p1",
        )
        assert ev.new_belief.validity.mode == "permanent"


class TestRenderSummary:
    def test_empty_result(self, stub_layer: BeliefLayer) -> None:
        result = BeliefQueryResult(current=[], history=[], tokens_in_summary=1)
        rendered = stub_layer.render_summary(result)
        assert "no current" in rendered.lower()

    def test_renders_current_only(self, stub_layer: BeliefLayer) -> None:
        ev = _ingest(
            stub_layer,
            "I moved to Sofia",
            at=datetime(2024, 3, 1),
        )
        result = stub_layer.query([ev.new_belief.id])
        rendered = stub_layer.render_summary(result)
        assert "I moved to Sofia" in rendered
        assert "2024-03-01" in rendered
        assert "superseded" not in rendered.lower()

    def test_renders_history_when_requested(self) -> None:
        scripts = {
            ("I live in Sydney", "I moved to Sofia"): ContradictionResult(
                label=ContradictionLabel.CONTRADICTS, confidence=0.95
            ),
        }
        layer = BeliefLayer(
            store=BeliefStore(), detector=ScriptedDetector(scripts)
        )
        _ingest(layer, "I live in Sydney", at=datetime(2023, 1, 1))
        e2 = _ingest(
            layer, "I moved to Sofia", at=datetime(2024, 3, 1)
        )
        result = layer.query([e2.new_belief.id], include_history=True)
        rendered = layer.render_summary(result, include_history=True)
        assert "I moved to Sofia" in rendered
        assert "I live in Sydney" in rendered
        assert "superseded" in rendered.lower()
