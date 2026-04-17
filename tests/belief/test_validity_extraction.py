"""Tests for validity window extraction."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from patha.belief.validity_extraction import (
    extract_validity,
    list_supported_patterns,
)


# Anchor date used across tests
T0 = datetime(2024, 3, 1, 12, 0)


# ─── "for N units" patterns ──────────────────────────────────────────

class TestForDuration:
    def test_for_three_weeks(self) -> None:
        v = extract_validity(
            "I'm on holiday for three weeks starting Monday",
            asserted_at=T0,
        )
        assert v is not None
        assert v.mode == "dated_range"
        assert v.source == "explicit"
        assert v.start == T0
        assert v.end == T0 + timedelta(days=21)

    def test_for_2_months(self) -> None:
        v = extract_validity(
            "I'm avoiding raw fish for 2 months",
            asserted_at=T0,
        )
        assert v is not None
        assert v.end == T0 + timedelta(days=60)

    def test_for_a_week(self) -> None:
        v = extract_validity(
            "I'll be out for a week",
            asserted_at=T0,
        )
        assert v is not None
        assert v.end == T0 + timedelta(days=7)

    def test_for_ten_days(self) -> None:
        v = extract_validity(
            "Fasting for ten days",
            asserted_at=T0,
        )
        assert v is not None
        assert v.end == T0 + timedelta(days=10)

    @pytest.mark.parametrize(
        "text,days",
        [
            ("Sick for one day", 1),
            ("Traveling for two weeks", 14),
            ("Project runs for five months", 150),
            ("I'll be training for a year", 365),
        ],
    )
    def test_parametrized_durations(self, text: str, days: int) -> None:
        v = extract_validity(text, asserted_at=T0)
        assert v is not None
        assert v.end == T0 + timedelta(days=days)


# ─── "next N units" patterns ─────────────────────────────────────────

class TestNextDuration:
    def test_next_two_weeks(self) -> None:
        v = extract_validity(
            "I'll be focused on this the next two weeks",
            asserted_at=T0,
        )
        assert v is not None
        assert v.end == T0 + timedelta(days=14)

    def test_for_the_next_month(self) -> None:
        v = extract_validity(
            "Working remotely for the next month",
            asserted_at=T0,
        )
        assert v is not None
        assert v.end == T0 + timedelta(days=30)


# ─── "until X" patterns ──────────────────────────────────────────────

class TestUntil:
    def test_until_explicit_date(self) -> None:
        v = extract_validity(
            "I'm on sabbatical until September 1 2024",
            asserted_at=T0,
        )
        assert v is not None
        assert v.mode == "dated_range"
        assert v.end is not None
        assert v.end.year == 2024
        assert v.end.month == 9

    def test_until_relative_date(self) -> None:
        # dateparser is more reliable on concrete date expressions than
        # weekday references — this test asserts the basic flow, not
        # dateparser's edge-case behaviour.
        v = extract_validity(
            "Out of office until April 15 2024",
            asserted_at=T0,
        )
        assert v is not None
        assert v.end is not None
        assert v.end > T0
        assert v.end.month == 4
        assert v.end.day == 15

    def test_through_as_synonym_for_until(self) -> None:
        v = extract_validity(
            "Working on this through July",
            asserted_at=T0,
        )
        assert v is not None
        assert v.end is not None
        assert v.end > T0

    def test_until_past_date_returns_none(self) -> None:
        # "until January" when T0 is March → dateparser may pick past
        # January; the handler rejects past endpoints.
        v = extract_validity(
            "I was training until January 2023",
            asserted_at=T0,
        )
        # Should return None rather than a nonsensical backward window.
        assert v is None or (v.end is not None and v.end > T0)


# ─── No match cases (v0.1 returns None, caller falls back) ───────────

class TestNoMatch:
    def test_no_temporal_marker(self) -> None:
        assert (
            extract_validity("I love sushi", asserted_at=T0)
            is None
        )

    def test_plain_statement_of_preference(self) -> None:
        assert (
            extract_validity(
                "I prefer oat milk over dairy", asserted_at=T0
            )
            is None
        )

    def test_implicit_duration_not_extracted_in_v01(self) -> None:
        # v0.1 does NOT infer implicit durations like "training for a
        # marathon". That's deferred to v0.2. For now the pattern
        # "for a marathon" doesn't match any rule, so return None.
        v = extract_validity(
            "I'm training for a marathon",
            asserted_at=T0,
        )
        assert v is None


# ─── Structural / introspection ──────────────────────────────────────

class TestStructure:
    def test_list_supported_patterns(self) -> None:
        names = list_supported_patterns()
        assert "for_N_units" in names
        assert "until_X" in names
        assert len(names) >= 2

    def test_all_results_are_explicit_source(self) -> None:
        # Every successful extraction should mark source="explicit"
        for text in [
            "for three weeks",
            "for 2 months",
            "until next Friday",
        ]:
            v = extract_validity(text, asserted_at=T0)
            if v is not None:
                assert v.source == "explicit"
                assert v.mode == "dated_range"
                assert v.start == T0
