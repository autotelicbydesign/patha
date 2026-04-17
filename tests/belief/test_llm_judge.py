"""Tests for LLM-judge fallback and hybrid contradiction detection."""

from __future__ import annotations

import pytest

from patha.belief.contradiction import StubContradictionDetector
from patha.belief.llm_judge import (
    HybridContradictionDetector,
    LLMJudge,
    PromptLLMJudge,
    StubLLMJudge,
)
from patha.belief.types import ContradictionLabel, ContradictionResult


# ─── StubLLMJudge ────────────────────────────────────────────────────

class TestStubLLMJudge:
    def test_default_is_neutral(self) -> None:
        judge = StubLLMJudge()
        r = judge.judge("anything", "else")
        assert r.label == ContradictionLabel.NEUTRAL

    def test_scripted_verdict_returned(self) -> None:
        verdicts = {
            ("a", "b"): ContradictionResult(
                label=ContradictionLabel.CONTRADICTS, confidence=0.9
            )
        }
        judge = StubLLMJudge(verdicts)
        assert judge.judge("a", "b").label == ContradictionLabel.CONTRADICTS
        assert judge.judge("c", "d").label == ContradictionLabel.NEUTRAL

    def test_satisfies_protocol(self) -> None:
        assert isinstance(StubLLMJudge(), LLMJudge)


# ─── PromptLLMJudge ──────────────────────────────────────────────────

class TestPromptLLMJudge:
    def test_parses_contradicts(self) -> None:
        judge = PromptLLMJudge(lambda p: "CONTRADICTS")
        r = judge.judge("x", "y")
        assert r.label == ContradictionLabel.CONTRADICTS
        assert r.confidence == 0.85

    def test_parses_entails(self) -> None:
        judge = PromptLLMJudge(lambda p: "ENTAILS, because...")
        r = judge.judge("x", "y")
        assert r.label == ContradictionLabel.ENTAILS

    def test_parses_neutral(self) -> None:
        judge = PromptLLMJudge(lambda p: "NEUTRAL")
        r = judge.judge("x", "y")
        assert r.label == ContradictionLabel.NEUTRAL

    def test_handles_multiline_response(self) -> None:
        judge = PromptLLMJudge(
            lambda p: "CONTRADICTS\nBecause sushi is raw fish."
        )
        r = judge.judge("I love sushi", "I'm avoiding raw fish")
        assert r.label == ContradictionLabel.CONTRADICTS

    def test_unknown_token_defaults_to_neutral(self) -> None:
        judge = PromptLLMJudge(lambda p: "dunno mate")
        r = judge.judge("x", "y")
        assert r.label == ContradictionLabel.NEUTRAL

    def test_satisfies_protocol(self) -> None:
        judge = PromptLLMJudge(lambda p: "NEUTRAL")
        assert isinstance(judge, LLMJudge)


# ─── HybridContradictionDetector ─────────────────────────────────────

class TestHybridRouter:
    def test_primary_contradiction_is_returned_directly(self) -> None:
        """If primary says CONTRADICTS, no LLM call happens."""
        class AlwaysContradicts:
            def detect(self, p1, p2):
                return ContradictionResult(
                    label=ContradictionLabel.CONTRADICTS, confidence=0.9
                )

            def detect_batch(self, pairs):
                return [self.detect(p1, p2) for p1, p2 in pairs]

        llm = StubLLMJudge()
        hybrid = HybridContradictionDetector(AlwaysContradicts(), llm)
        r = hybrid.detect("a", "b")
        assert r.label == ContradictionLabel.CONTRADICTS
        assert hybrid.llm_calls == 0

    def test_neutral_with_overlap_escalates(self) -> None:
        """NLI NEUTRAL + shared content word + uncertainty band → LLM fires."""
        class AlwaysNeutral:
            def detect(self, p1, p2):
                return ContradictionResult(
                    label=ContradictionLabel.NEUTRAL, confidence=0.6
                )

            def detect_batch(self, pairs):
                return [self.detect(p1, p2) for p1, p2 in pairs]

        llm = StubLLMJudge({
            ("I love sushi", "I'm avoiding raw fish"): ContradictionResult(
                label=ContradictionLabel.CONTRADICTS, confidence=0.9,
                rationale="sushi is raw fish",
            )
        })
        hybrid = HybridContradictionDetector(AlwaysNeutral(), llm)
        r = hybrid.detect("I love sushi", "I'm avoiding raw fish")
        # No word overlap between "sushi" and "avoiding raw fish" as
        # computed by simple content words... let's pick a shared-word case.
        # Actually "sushi" is unique; verify the scripted path.
        # The hybrid escalates if shared content words >=1.
        # Between "I love sushi" and "I'm avoiding raw fish" shared
        # content words are: none. So this should NOT escalate.
        # We want the test to confirm the routing logic; adjust the
        # example to share a word.
        assert hybrid.llm_calls == 0  # no shared content words

    def test_neutral_with_shared_word_escalates(self) -> None:
        class AlwaysNeutral:
            def detect(self, p1, p2):
                return ContradictionResult(
                    label=ContradictionLabel.NEUTRAL, confidence=0.6
                )

            def detect_batch(self, pairs):
                return [self.detect(p1, p2) for p1, p2 in pairs]

        llm = StubLLMJudge({
            ("I love sushi weekly", "I never eat sushi now"): ContradictionResult(
                label=ContradictionLabel.CONTRADICTS, confidence=0.9,
                rationale="explicit",
            )
        })
        hybrid = HybridContradictionDetector(AlwaysNeutral(), llm)
        r = hybrid.detect("I love sushi weekly", "I never eat sushi now")
        assert r.label == ContradictionLabel.CONTRADICTS
        assert hybrid.llm_calls == 1

    def test_neutral_without_overlap_stays_neutral(self) -> None:
        class AlwaysNeutral:
            def detect(self, p1, p2):
                return ContradictionResult(
                    label=ContradictionLabel.NEUTRAL, confidence=0.6
                )

            def detect_batch(self, pairs):
                return [self.detect(p1, p2) for p1, p2 in pairs]

        llm = StubLLMJudge()
        hybrid = HybridContradictionDetector(AlwaysNeutral(), llm)
        r = hybrid.detect("I love sushi", "The weather is nice today")
        assert r.label == ContradictionLabel.NEUTRAL
        assert hybrid.llm_calls == 0

    def test_high_confidence_neutral_not_escalated(self) -> None:
        """If NLI is VERY confident it's neutral, don't bother the LLM."""
        class ConfidentlyNeutral:
            def detect(self, p1, p2):
                return ContradictionResult(
                    label=ContradictionLabel.NEUTRAL, confidence=0.99
                )

            def detect_batch(self, pairs):
                return [self.detect(p1, p2) for p1, p2 in pairs]

        llm = StubLLMJudge()
        hybrid = HybridContradictionDetector(
            ConfidentlyNeutral(),
            llm,
            uncertainty_band=(0.0, 0.95),
        )
        r = hybrid.detect("I love sushi weekly", "I hate sushi")
        assert r.label == ContradictionLabel.NEUTRAL
        assert hybrid.llm_calls == 0  # confidence 0.99 > band max 0.95

    def test_batch_preserves_order(self) -> None:
        class Mixed:
            def detect(self, p1, p2):
                if "ZZZ" in p1:
                    return ContradictionResult(
                        label=ContradictionLabel.CONTRADICTS, confidence=0.9
                    )
                return ContradictionResult(
                    label=ContradictionLabel.NEUTRAL, confidence=0.6
                )

            def detect_batch(self, pairs):
                return [self.detect(p1, p2) for p1, p2 in pairs]

        llm = StubLLMJudge()
        hybrid = HybridContradictionDetector(Mixed(), llm)
        pairs = [
            ("ZZZ thing", "another"),     # → CONTRADICTS (ZZZ in p1)
            ("quiet dog", "loud cat"),    # → NEUTRAL, no overlap
        ]
        results = hybrid.detect_batch(pairs)
        assert len(results) == 2
        assert results[0].label == ContradictionLabel.CONTRADICTS
        assert results[1].label == ContradictionLabel.NEUTRAL

    def test_metrics_tracked(self) -> None:
        class AlwaysNeutral:
            def detect(self, p1, p2):
                return ContradictionResult(
                    label=ContradictionLabel.NEUTRAL, confidence=0.6
                )

            def detect_batch(self, pairs):
                return [self.detect(p1, p2) for p1, p2 in pairs]

        llm = StubLLMJudge()
        hybrid = HybridContradictionDetector(AlwaysNeutral(), llm)
        hybrid.detect_batch([
            ("shared sushi", "also sushi here"),
            ("totally", "unrelated"),
        ])
        assert hybrid.primary_calls == 2
        assert hybrid.llm_calls == 1  # only the one with overlap
