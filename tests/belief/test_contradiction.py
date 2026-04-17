"""Tests for contradiction detection.

The stub detector gets thorough testing (fast, deterministic, no model
download). The NLI detector has one integration test marked `slow` that
actually loads a small NLI model — skipped unless explicitly selected.
"""

from __future__ import annotations

import pytest

from patha.belief.contradiction import (
    ContradictionDetector,
    NLIContradictionDetector,
    StubContradictionDetector,
)
from patha.belief.types import ContradictionLabel, ContradictionResult


# ─── Stub behaviour ──────────────────────────────────────────────────

class TestStubContradictionDetector:
    def setup_method(self) -> None:
        self.detector = StubContradictionDetector()

    def test_satisfies_protocol(self) -> None:
        assert isinstance(self.detector, ContradictionDetector)

    def test_asymmetric_negation_with_overlap_fires(self) -> None:
        r = self.detector.detect(
            "I love sushi",
            "I'm avoiding sushi for health reasons",
        )
        assert r.label == ContradictionLabel.CONTRADICTS
        assert 0.0 <= r.confidence <= 1.0
        assert r.rationale is not None and "negation" in r.rationale

    def test_same_polarity_is_neutral(self) -> None:
        r = self.detector.detect(
            "I love sushi",
            "I love sashimi and ramen",
        )
        assert r.label == ContradictionLabel.NEUTRAL

    def test_no_content_overlap_is_neutral(self) -> None:
        r = self.detector.detect(
            "I love sushi",
            "The weather is lovely today",
        )
        assert r.label == ContradictionLabel.NEUTRAL

    def test_returns_contradiction_result_type(self) -> None:
        r = self.detector.detect("a", "b")
        assert isinstance(r, ContradictionResult)

    def test_batch_matches_individual(self) -> None:
        pairs = [
            ("I love sushi", "I'm avoiding sushi"),
            ("I love dogs", "The weather is nice"),
        ]
        batch_results = self.detector.detect_batch(pairs)
        individual = [self.detector.detect(p1, p2) for p1, p2 in pairs]
        assert len(batch_results) == 2
        for b, i in zip(batch_results, individual):
            assert b.label == i.label
            assert b.confidence == i.confidence

    def test_empty_batch(self) -> None:
        assert self.detector.detect_batch([]) == []

    def test_confidence_in_range(self) -> None:
        r = self.detector.detect("anything", "something")
        assert 0.0 <= r.confidence <= 1.0

    @pytest.mark.parametrize("neg_cue", [
        "not", "never", "avoid", "avoiding", "stopped", "used to",
    ])
    def test_various_negation_cues_trigger(self, neg_cue: str) -> None:
        r = self.detector.detect(
            "I eat sushi",
            f"I {neg_cue} sushi regularly",
        )
        # Should detect contradiction when one has a negation cue and
        # the other doesn't, and there's content overlap ("sushi").
        assert r.label == ContradictionLabel.CONTRADICTS


# ─── Protocol conformance ────────────────────────────────────────────

class TestProtocol:
    def test_stub_conforms(self) -> None:
        assert isinstance(StubContradictionDetector(), ContradictionDetector)

    def test_nli_conforms(self) -> None:
        # Conformance check without loading the model.
        detector = NLIContradictionDetector()
        assert isinstance(detector, ContradictionDetector)


# ─── NLI integration (slow; skipped by default) ──────────────────────

@pytest.mark.slow
class TestNLIContradictionDetector:
    """Integration tests that actually load and run the NLI model.

    Skipped by default — run with `pytest -m slow` to exercise.
    Requires network access on first run for model download.
    """

    @pytest.fixture(scope="class")
    def detector(self) -> NLIContradictionDetector:
        # Use the smaller base model for tests to cut download time.
        from patha.belief.contradiction import SMALL_NLI_MODEL
        return NLIContradictionDetector(model_name=SMALL_NLI_MODEL)

    def test_clear_contradiction(self, detector: NLIContradictionDetector) -> None:
        r = detector.detect(
            "I love sushi and eat it every week.",
            "I hate sushi and have never eaten it.",
        )
        assert r.label == ContradictionLabel.CONTRADICTS
        assert r.confidence > 0.5

    def test_clear_entailment(self, detector: NLIContradictionDetector) -> None:
        r = detector.detect(
            "I own a golden retriever named Lily.",
            "I have a dog.",
        )
        assert r.label == ContradictionLabel.ENTAILS

    def test_unrelated_is_neutral(
        self, detector: NLIContradictionDetector
    ) -> None:
        r = detector.detect(
            "I love sushi.",
            "The weather in Sydney is warm today.",
        )
        assert r.label == ContradictionLabel.NEUTRAL

    def test_batch_inference_preserves_order(
        self, detector: NLIContradictionDetector
    ) -> None:
        pairs = [
            ("I love sushi.", "I hate sushi."),
            ("I have a dog.", "The weather is warm."),
            ("I own a Ford.", "I own a car."),
        ]
        results = detector.detect_batch(pairs)
        assert len(results) == 3
        assert results[0].label == ContradictionLabel.CONTRADICTS
        assert results[1].label == ContradictionLabel.NEUTRAL
        assert results[2].label == ContradictionLabel.ENTAILS
