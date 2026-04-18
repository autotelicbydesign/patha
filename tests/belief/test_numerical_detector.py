"""Tests for NumericalAwareDetector.

Covers both arms of the numerical/value-replacement detector:
  - check_numerical_contradiction: shared subject, differing numbers
  - check_value_replacement:       shared 'my X is Y', differing values
"""

from __future__ import annotations

from patha.belief.numerical_detector import (
    NumericalAwareDetector,
    _canonical_num,
    _extract_subject_number_pairs,
    _extract_subject_value_pairs,
    _normalise_subject,
    check_numerical_contradiction,
    check_value_replacement,
)
from patha.belief.types import ContradictionLabel, ContradictionResult


class _SpyDetector:
    def __init__(self):
        self.received: list[tuple[str, str]] = []

    def detect(self, p1: str, p2: str) -> ContradictionResult:
        return self.detect_batch([(p1, p2)])[0]

    def detect_batch(self, pairs):
        self.received.extend(pairs)
        return [
            ContradictionResult(
                label=ContradictionLabel.NEUTRAL, confidence=0.5,
            )
            for _ in pairs
        ]


class TestCanonicalNum:
    def test_digit_string(self):
        assert _canonical_num("1500") == "1500"
        assert _canonical_num("1500.0") == "1500"
        assert _canonical_num("3.5") == "3.5"

    def test_word_number(self):
        assert _canonical_num("three") == "3"
        assert _canonical_num("SEVEN") == "7"
        assert _canonical_num("fifteen") == "15"


class TestNormaliseSubject:
    def test_strips_trailing_s(self):
        assert _normalise_subject("closes") == "close"
        assert _normalise_subject("cars") == "car"
        assert _normalise_subject("minutes") == "minute"

    def test_preserves_ss(self):
        assert _normalise_subject("class") == "class"
        assert _normalise_subject("glass") == "glass"

    def test_too_short(self):
        # <=3 chars: don't strip
        assert _normalise_subject("cat") == "cat"
        assert _normalise_subject("bus") == "bus"


class TestExtractSubjectNumberPairs:
    def test_rent(self):
        pairs = _extract_subject_number_pairs("My rent is 1500 a month")
        assert ("rent", "1500") in pairs

    def test_commute_with_minutes(self):
        pairs = _extract_subject_number_pairs(
            "My commute is 45 minutes by train"
        )
        assert ("commute", "45") in pairs

    def test_store_closes_at_time(self):
        pairs = _extract_subject_number_pairs(
            "The store closes at 9pm on weekdays"
        )
        # 'closes' normalised to 'close'
        assert ("close", "9") in pairs

    def test_speak_languages(self):
        pairs = _extract_subject_number_pairs("I speak three languages")
        # number word → digit, subject normalised
        assert ("language", "3") in pairs


class TestCheckNumericalContradiction:
    def test_rent_change(self):
        assert check_numerical_contradiction(
            "My rent is 1500 a month",
            "My rent went up to 1800",
        )

    def test_commute_change(self):
        assert check_numerical_contradiction(
            "My commute is 45 minutes",
            "My commute is now 15 minutes",
        )

    def test_closing_time_change(self):
        assert check_numerical_contradiction(
            "The store closes at 8pm",
            "The store closes at 9pm now",
        )

    def test_same_number_no_contradiction(self):
        assert not check_numerical_contradiction(
            "My rent is 1500",
            "My rent has been 1500 for years",
        )

    def test_different_subjects_no_contradiction(self):
        assert not check_numerical_contradiction(
            "My rent is 1500",
            "My commute is 45 minutes",
        )

    def test_word_digit_equivalence(self):
        """'three' and '3' compare equal after canonicalisation."""
        assert not check_numerical_contradiction(
            "I speak three languages",
            "I speak 3 languages",
        )


class TestValueReplacement:
    def test_email_change(self):
        assert check_value_replacement(
            "My email is alice@old.com",
            "My new email is alice@new.com",
        )

    def test_laptop_change(self):
        assert check_value_replacement(
            "My laptop is a 2019 MacBook Pro",
            "My laptop is an M3 MacBook Pro",
        )

    def test_same_value_no_contradiction(self):
        assert not check_value_replacement(
            "My email is alice@example.com",
            "My email is alice@example.com",
        )

    def test_property_not_in_canonical_list(self):
        """'dog' is not in _VALUE_PROPERTIES — don't fire on
        'my dog is X' vs 'my dog is Y'."""
        assert not check_value_replacement(
            "My dog is friendly",
            "My dog is shy",
        )

    def test_extract_pairs_single_property(self):
        pairs = _extract_subject_value_pairs(
            "My landlord is Alice Smith"
        )
        assert len(pairs) == 1
        assert pairs[0][0] == "landlord"
        assert "alice smith" in pairs[0][1]


class TestNumericalAwareDetector:
    def test_fires_on_rent(self):
        spy = _SpyDetector()
        det = NumericalAwareDetector(inner=spy)
        r = det.detect("My rent is 1500", "My rent went up to 1800")
        assert r.label == ContradictionLabel.CONTRADICTS
        assert r.confidence == 0.9
        assert det.numerical_overrides == 1
        assert det.inner_calls == 0
        assert spy.received == []

    def test_fires_on_email(self):
        spy = _SpyDetector()
        det = NumericalAwareDetector(inner=spy)
        r = det.detect(
            "My email is alice@old.com",
            "My new email is alice@new.com",
        )
        assert r.label == ContradictionLabel.CONTRADICTS

    def test_delegates_when_no_match(self):
        spy = _SpyDetector()
        det = NumericalAwareDetector(inner=spy)
        det.detect("I love sushi", "The weather is nice")
        assert det.numerical_overrides == 0
        assert det.inner_calls == 1
        assert len(spy.received) == 1

    def test_batch_mixed(self):
        spy = _SpyDetector()
        det = NumericalAwareDetector(inner=spy)
        pairs = [
            ("My rent is 1500", "My rent went up to 1800"),    # numerical
            ("My email is x@old.com", "My email is x@new.com"), # value-replace
            ("I love jazz", "The weather is nice"),             # inner
        ]
        results = det.detect_batch(pairs)
        assert results[0].label == ContradictionLabel.CONTRADICTS
        assert results[1].label == ContradictionLabel.CONTRADICTS
        assert results[2].label == ContradictionLabel.NEUTRAL
        assert det.numerical_overrides == 2
        assert det.inner_calls == 1

    def test_empty_batch(self):
        det = NumericalAwareDetector(inner=_SpyDetector())
        assert det.detect_batch([]) == []
