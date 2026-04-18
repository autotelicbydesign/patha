"""Tests for AdhyasaAwareDetector (v0.5)."""

from __future__ import annotations

import pytest

from patha.belief.adhyasa_detector import AdhyasaAwareDetector
from patha.belief.contradiction import (
    ContradictionDetector,
    StubContradictionDetector,
)
from patha.belief.types import ContradictionLabel, ContradictionResult


class _SpyDetector:
    """Captures the pairs sent to it."""

    def __init__(self, verdicts: dict | None = None):
        self.received: list[tuple[str, str]] = []
        self.verdicts = verdicts or {}

    def detect(self, p1: str, p2: str) -> ContradictionResult:
        return self.detect_batch([(p1, p2)])[0]

    def detect_batch(self, pairs):
        self.received.extend(pairs)
        return [
            self.verdicts.get(
                (p1, p2),
                ContradictionResult(label=ContradictionLabel.NEUTRAL, confidence=0.5),
            )
            for p1, p2 in pairs
        ]


class TestAdhyasaAwareDetector:
    def test_satisfies_protocol(self) -> None:
        d = AdhyasaAwareDetector(inner=StubContradictionDetector())
        assert isinstance(d, ContradictionDetector)

    def test_rewrites_pair_when_superimposition_detected(self) -> None:
        """Sushi vs raw-fish pair gets rewritten before inner detector sees it."""
        spy = _SpyDetector()
        d = AdhyasaAwareDetector(inner=spy)
        d.detect(
            "I love sushi and eat it every week",
            "I am avoiding raw fish on my doctor's advice",
        )
        assert len(spy.received) == 1
        sent_p1, sent_p2 = spy.received[0]
        # p2 was rewritten: 'raw fish' → 'sushi'
        assert "sushi" in sent_p2.lower()
        assert "raw fish" not in sent_p2.lower()
        # Metrics
        assert d.adhyasa_rewrites == 1
        assert d.inner_calls == 1

    def test_passes_through_when_no_superimposition(self) -> None:
        spy = _SpyDetector()
        d = AdhyasaAwareDetector(inner=spy)
        d.detect("I love sushi", "the weather is nice today")
        sent_p1, sent_p2 = spy.received[0]
        # Unchanged
        assert sent_p2 == "the weather is nice today"
        assert d.adhyasa_rewrites == 0
        assert d.inner_calls == 1

    def test_returns_inner_verdict_on_rewritten_pair(self) -> None:
        """When rewrite fires, the returned verdict is what the inner
        detector said about the rewritten pair."""
        p1 = "I love sushi every week"
        p2_original = "I am avoiding raw fish"
        # The rewritten p2 will be "I am avoiding sushi"
        expected_rewritten = "I am avoiding sushi"
        spy = _SpyDetector(
            verdicts={
                (p1, expected_rewritten): ContradictionResult(
                    label=ContradictionLabel.CONTRADICTS, confidence=0.95
                ),
            }
        )
        d = AdhyasaAwareDetector(inner=spy)
        r = d.detect(p1, p2_original)
        assert r.label == ContradictionLabel.CONTRADICTS
        assert r.confidence == 0.95

    def test_batch_partial_rewrite(self) -> None:
        """Batch with one rewritable pair and one not — inner sees
        rewrite for one, original for the other."""
        spy = _SpyDetector()
        d = AdhyasaAwareDetector(inner=spy)
        pairs = [
            ("I love sushi", "I avoid raw fish"),          # rewrite
            ("I live in Sofia", "The weather is nice"),    # no rewrite
        ]
        d.detect_batch(pairs)
        assert len(spy.received) == 2
        # First pair was rewritten
        _, rewritten = spy.received[0]
        assert "sushi" in rewritten.lower()
        # Second pair passed through
        assert spy.received[1] == ("I live in Sofia", "The weather is nice")
        assert d.adhyasa_rewrites == 1
        assert d.inner_calls == 2

    def test_empty_batch(self) -> None:
        d = AdhyasaAwareDetector(inner=_SpyDetector())
        assert d.detect_batch([]) == []

    def test_metrics_accumulate(self) -> None:
        d = AdhyasaAwareDetector(inner=_SpyDetector())
        d.detect("I love sushi", "I avoid raw fish")
        d.detect("I love sushi", "I avoid raw fish")
        d.detect("unrelated", "also unrelated")
        assert d.adhyasa_rewrites == 2
        assert d.inner_calls == 3

    def test_e2e_stub_gains_contradiction_on_sushi_pair(self) -> None:
        """Integration demo: the stub detector's asymmetric-negation
        heuristic DOES fire on 'I love sushi' vs 'I avoid sushi' — so
        wrapping in adhyāsa lifts a real failure case."""
        # Raw stub on original pair — stub NEUTRAL (no word overlap)
        stub = StubContradictionDetector()
        raw = stub.detect(
            "I love sushi every week",
            "I am avoiding raw fish on my doctor's advice",
        )
        assert raw.label == ContradictionLabel.NEUTRAL

        # Wrapped: same pair, stub now sees the rewrite and catches it
        wrapped = AdhyasaAwareDetector(inner=StubContradictionDetector())
        lifted = wrapped.detect(
            "I love sushi every week",
            "I am avoiding raw fish on my doctor's advice",
        )
        assert lifted.label == ContradictionLabel.CONTRADICTS
