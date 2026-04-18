"""Tests for AdhyasaAwareDetector (v0.5 + v0.6 both-paths refactor).

v0.6 changed behaviour: when superimposition is detected, adhyāsa
runs the inner detector on BOTH the original and the rewritten pair,
then takes the stronger CONTRADICTS verdict. Rationale: rewriting
doesn't always preserve contradiction (sometimes NLI returns NEUTRAL
on a rewrite that loses the surrounding context). Running both and
picking the stronger signal is more robust.
"""

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

    def __init__(self, verdicts: dict | None = None, default_conf: float = 0.5):
        self.received: list[tuple[str, str]] = []
        self.verdicts = verdicts or {}
        self.default_conf = default_conf

    def detect(self, p1: str, p2: str) -> ContradictionResult:
        return self.detect_batch([(p1, p2)])[0]

    def detect_batch(self, pairs):
        self.received.extend(pairs)
        return [
            self.verdicts.get(
                (p1, p2),
                ContradictionResult(
                    label=ContradictionLabel.NEUTRAL, confidence=self.default_conf
                ),
            )
            for p1, p2 in pairs
        ]


class TestAdhyasaAwareDetector:
    def test_satisfies_protocol(self) -> None:
        d = AdhyasaAwareDetector(inner=StubContradictionDetector())
        assert isinstance(d, ContradictionDetector)

    def test_runs_inner_on_both_original_and_rewrite(self) -> None:
        """When superimposition fires, inner sees both the original
        pair AND the rewritten pair."""
        spy = _SpyDetector()
        d = AdhyasaAwareDetector(inner=spy)
        d.detect(
            "I love sushi and eat it every week",
            "I am avoiding raw fish on my doctor's advice",
        )
        # Both original and rewritten pair submitted
        assert len(spy.received) == 2
        originals = [p for p in spy.received if "raw fish" in p[1]]
        rewrites = [p for p in spy.received if "sushi" in p[1]]
        assert len(originals) >= 1
        assert len(rewrites) >= 1
        assert d.adhyasa_rewrites == 1
        assert d.inner_calls == 2

    def test_passes_through_when_no_superimposition(self) -> None:
        """No rewrite → inner sees the pair once."""
        spy = _SpyDetector()
        d = AdhyasaAwareDetector(inner=spy)
        d.detect("I love sushi", "the weather is nice today")
        assert len(spy.received) == 1
        assert d.adhyasa_rewrites == 0
        assert d.inner_calls == 1

    def test_picks_stronger_contradiction_signal(self) -> None:
        """If inner says CONTRADICTS on the original but NEUTRAL on the
        rewrite (the real bug we saw in practice), we still return
        CONTRADICTS."""
        p1 = "I love sushi and eat it every week"
        p2_original = "I am avoiding raw fish on my doctor's advice"
        rewritten = "I am avoiding sushi on my doctor's advice"
        spy = _SpyDetector(verdicts={
            (p1, p2_original): ContradictionResult(
                label=ContradictionLabel.CONTRADICTS, confidence=0.95
            ),
            (p1, rewritten): ContradictionResult(
                label=ContradictionLabel.NEUTRAL, confidence=0.92
            ),
        })
        d = AdhyasaAwareDetector(inner=spy)
        r = d.detect(p1, p2_original)
        assert r.label == ContradictionLabel.CONTRADICTS
        assert r.confidence == 0.95

    def test_picks_stronger_contradiction_from_rewrite(self) -> None:
        """Conversely, if inner is weak on original but strong on
        rewrite, we return the strong rewrite verdict."""
        p1 = "I love sushi"
        p2_original = "I am avoiding raw fish"
        rewritten = "I am avoiding sushi"
        spy = _SpyDetector(verdicts={
            (p1, p2_original): ContradictionResult(
                label=ContradictionLabel.NEUTRAL, confidence=0.6
            ),
            (p1, rewritten): ContradictionResult(
                label=ContradictionLabel.CONTRADICTS, confidence=0.98
            ),
        })
        d = AdhyasaAwareDetector(inner=spy)
        r = d.detect(p1, p2_original)
        assert r.label == ContradictionLabel.CONTRADICTS
        assert r.confidence == 0.98

    def test_batch_partial_rewrite(self) -> None:
        """Batch with one rewritable pair and one not — metrics count
        rewrites and inner_calls correctly (original + rewrite = 2
        for rewritable; 1 for non-rewritable)."""
        spy = _SpyDetector()
        d = AdhyasaAwareDetector(inner=spy)
        pairs = [
            ("I love sushi", "I avoid raw fish"),          # rewrite → 2 calls
            ("I live in Sofia", "The weather is nice"),    # no rewrite → 1 call
        ]
        d.detect_batch(pairs)
        assert d.adhyasa_rewrites == 1
        assert d.inner_calls == 3  # 2 + 1

    def test_empty_batch(self) -> None:
        d = AdhyasaAwareDetector(inner=_SpyDetector())
        assert d.detect_batch([]) == []

    def test_metrics_accumulate(self) -> None:
        d = AdhyasaAwareDetector(inner=_SpyDetector())
        d.detect("I love sushi", "I avoid raw fish")       # 1 rewrite, 2 calls
        d.detect("I love sushi", "I avoid raw fish")       # 1 rewrite, 2 calls
        d.detect("unrelated", "also unrelated")            # 0 rewrite, 1 call
        assert d.adhyasa_rewrites == 2
        assert d.inner_calls == 5

    def test_e2e_stub_gains_contradiction_on_sushi_pair(self) -> None:
        """Integration: the stub detector's asymmetric-negation
        heuristic fires on 'I love sushi' vs 'I avoid sushi' — so
        wrapping in adhyāsa lifts the sushi/raw-fish failure case."""
        stub = StubContradictionDetector()
        raw = stub.detect(
            "I love sushi every week",
            "I am avoiding raw fish on my doctor's advice",
        )
        assert raw.label == ContradictionLabel.NEUTRAL  # stub can't catch it

        wrapped = AdhyasaAwareDetector(inner=StubContradictionDetector())
        lifted = wrapped.detect(
            "I love sushi every week",
            "I am avoiding raw fish on my doctor's advice",
        )
        # Rewritten pair 'I love sushi' vs 'I am avoiding sushi' DOES
        # fire the stub's heuristic → CONTRADICTS
        assert lifted.label == ContradictionLabel.CONTRADICTS
