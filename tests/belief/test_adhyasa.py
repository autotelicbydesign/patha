"""Tests for adhyāsa superimposition detection (v0.4 #6)."""

from __future__ import annotations

import pytest

from patha.belief.adhyasa import (
    AdhyasaResult,
    HandCuratedOntology,
    check_superimposition,
)


# ─── HandCuratedOntology ────────────────────────────────────────────

class TestHandCuratedOntology:
    def test_sushi_includes_raw_fish(self) -> None:
        ont = HandCuratedOntology()
        eq = ont.equivalent_terms("sushi")
        assert "raw fish" in eq
        assert "sushi" in eq

    def test_raw_fish_includes_sushi(self) -> None:
        """Equivalence is symmetric."""
        ont = HandCuratedOntology()
        eq = ont.equivalent_terms("raw fish")
        assert "sushi" in eq

    def test_term_not_in_any_class_returns_singleton(self) -> None:
        ont = HandCuratedOntology()
        eq = ont.equivalent_terms("zeppelin")
        assert eq == {"zeppelin"}

    def test_case_insensitive(self) -> None:
        ont = HandCuratedOntology()
        assert "raw fish" in ont.equivalent_terms("SUSHI")


# ─── check_superimposition ─────────────────────────────────────────

class TestCheckSuperimposition:
    def test_sushi_vs_raw_fish_detected(self) -> None:
        result = check_superimposition(
            "I love sushi and eat it every week",
            "I am avoiding raw fish on my doctor's advice",
        )
        assert result.superimposition_detected
        # Specifically identified the terms
        assert result.p1_term == "sushi"
        assert result.p2_term == "raw fish"
        # Rewritten p2 substitutes 'raw fish' with 'sushi' so NLI can
        # see the contradiction
        assert "sushi" in result.rewritten_p2.lower()  # type: ignore[union-attr]
        assert "raw fish" not in result.rewritten_p2.lower()  # type: ignore[union-attr]

    def test_vegetarian_vs_meat_free_detected(self) -> None:
        result = check_superimposition(
            "I am vegetarian",
            "I have been meat-free for two years",
        )
        # 'meat-free' is in the vegetarian equivalence class
        assert result.superimposition_detected

    def test_unrelated_propositions_not_detected(self) -> None:
        result = check_superimposition(
            "I love sushi",
            "The weather is nice today",
        )
        assert not result.superimposition_detected
        assert result.p1_term is None

    def test_identical_terms_not_flagged(self) -> None:
        """If both propositions use the SAME lexeme, there's no
        superimposition (NLI would catch it already)."""
        result = check_superimposition(
            "I love sushi every week",
            "I am avoiding sushi now",
        )
        # 'sushi' in both — no substitution needed
        assert not result.superimposition_detected

    def test_no_equivalence_class_for_unknown_term(self) -> None:
        """Terms not in the ontology don't trigger false-positive
        superimposition."""
        result = check_superimposition(
            "I love xylophones",
            "I am avoiding bassoons",
        )
        assert not result.superimposition_detected


# ─── Rewriting correctness ─────────────────────────────────────────

class TestRewriting:
    def test_rewritten_preserves_sentence_structure(self) -> None:
        result = check_superimposition(
            "I love sushi",
            "I am avoiding raw fish on my doctor's advice",
        )
        assert result.rewritten_p2 is not None
        # Structure should be preserved, only the target term swapped
        assert "avoiding" in result.rewritten_p2
        assert "doctor" in result.rewritten_p2
        assert "sushi" in result.rewritten_p2

    def test_rewrite_is_case_preserving_in_structure(self) -> None:
        result = check_superimposition(
            "I love Sushi",
            "I am avoiding Raw Fish on my diet",
        )
        assert result.superimposition_detected
        # Should still substitute (case-insensitive match)
        assert "sushi" in result.rewritten_p2.lower()  # type: ignore[union-attr]


# ─── Downstream pipeline hint ──────────────────────────────────────

class TestPipelineUsage:
    def test_adhyasa_then_nli_pipeline(self) -> None:
        """Demonstrates intended usage: check adhyāsa, if detected
        run NLI (or any contradiction detector) on the rewritten pair.

        Does not require a real NLI — just shows the wiring.
        """
        from patha.belief.contradiction import StubContradictionDetector
        from patha.belief.types import ContradictionLabel

        stub = StubContradictionDetector()
        p1 = "I love sushi every week"
        p2 = "I am avoiding raw fish on my doctor's advice"

        # Raw NLI on the original pair (the failure case)
        raw = stub.detect(p1, p2)
        # Stub fires only on asymmetric negation with word overlap.
        # 'sushi' and 'raw fish' share no word → raw result is NEUTRAL.
        assert raw.label == ContradictionLabel.NEUTRAL

        # Adhyāsa rewrite
        adh = check_superimposition(p1, p2)
        assert adh.superimposition_detected
        rewritten = adh.rewritten_p2

        # Stub on rewritten pair: now both mention 'sushi', with
        # asymmetric negation — stub DOES fire CONTRADICTS
        rewritten_check = stub.detect(p1, rewritten)
        assert rewritten_check.label == ContradictionLabel.CONTRADICTS
