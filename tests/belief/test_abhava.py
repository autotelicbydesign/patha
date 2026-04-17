"""Tests for abhāva classification (Nyāya four-fold negation)."""

from __future__ import annotations

import pytest

from patha.belief.abhava import AbhavaKind, classify_abhava


class TestAtyantabhava:
    @pytest.mark.parametrize("text", [
        "I have never lived in Paris",
        "I've never smoked",
        "We have never been to Bhutan",
    ])
    def test_never_constructions(self, text: str) -> None:
        r = classify_abhava(text)
        assert r.kind == AbhavaKind.ATYANTABHAVA


class TestPradhvamsabhava:
    @pytest.mark.parametrize("text", [
        "I stopped smoking last year",
        "I quit my job at Canva",
        "I gave up coffee three months ago",
        "I no longer eat meat",
        "I don't drink coffee anymore",
    ])
    def test_destructive_constructions(self, text: str) -> None:
        r = classify_abhava(text)
        assert r.kind == AbhavaKind.PRADHVAMSABHAVA

    def test_referenced_state_extracted(self) -> None:
        r = classify_abhava("I stopped smoking cigarettes")
        assert r.kind == AbhavaKind.PRADHVAMSABHAVA
        assert r.referenced_state is not None
        assert "smoking" in r.referenced_state


class TestPragabhava:
    @pytest.mark.parametrize("text", [
        "I haven't started the project yet",
        "I have not finished the report yet",
        "I haven't begun training",
    ])
    def test_prior_absence_constructions(self, text: str) -> None:
        r = classify_abhava(text)
        assert r.kind == AbhavaKind.PRAGABHAVA


class TestAnyonyabhava:
    @pytest.mark.parametrize("text", [
        "I am not a nurse",
        "She is not an engineer",
        "They are not a client",
    ])
    def test_identity_negation(self, text: str) -> None:
        r = classify_abhava(text)
        assert r.kind == AbhavaKind.ANYONYABHAVA

    def test_no_referenced_state(self) -> None:
        r = classify_abhava("I am not a doctor")
        # Identity negations don't reference a positive state
        assert r.referenced_state is None


class TestNonNegation:
    @pytest.mark.parametrize("text", [
        "I live in Sofia",
        "The meeting is at 3pm",
        "I love sushi",
    ])
    def test_no_negation_returns_none(self, text: str) -> None:
        r = classify_abhava(text)
        assert r.kind == AbhavaKind.NONE


class TestAmbiguousNegation:
    def test_plain_negation_without_specific_pattern_is_unknown(self) -> None:
        """Negation present but not matching any specific kind."""
        r = classify_abhava("not exactly right")
        # Has 'not' but no specific pattern
        assert r.kind in (AbhavaKind.UNKNOWN, AbhavaKind.ANYONYABHAVA)


class TestCueReturned:
    def test_cue_populated_on_hit(self) -> None:
        r = classify_abhava("I no longer eat meat")
        assert r.cue  # non-empty
        assert r.cue.lower().startswith("no longer") or "no longer" in r.cue.lower()
