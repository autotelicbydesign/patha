"""Tests for belief layer data types."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from patha.belief.types import (
    Belief,
    ContradictionLabel,
    ContradictionResult,
    Validity,
)


# ─── ContradictionResult ─────────────────────────────────────────────

class TestContradictionResult:
    def test_basic_construction(self) -> None:
        r = ContradictionResult(
            label=ContradictionLabel.CONTRADICTS, confidence=0.9
        )
        assert r.label == ContradictionLabel.CONTRADICTS
        assert r.confidence == 0.9
        assert r.rationale is None

    def test_rationale_optional(self) -> None:
        r = ContradictionResult(
            label=ContradictionLabel.NEUTRAL,
            confidence=0.5,
            rationale="insufficient evidence",
        )
        assert r.rationale == "insufficient evidence"

    def test_rejects_out_of_range_confidence(self) -> None:
        with pytest.raises(ValueError, match="confidence must be in"):
            ContradictionResult(
                label=ContradictionLabel.CONTRADICTS, confidence=1.5
            )
        with pytest.raises(ValueError):
            ContradictionResult(
                label=ContradictionLabel.CONTRADICTS, confidence=-0.1
            )

    def test_immutable(self) -> None:
        r = ContradictionResult(
            label=ContradictionLabel.CONTRADICTS, confidence=0.9
        )
        with pytest.raises(Exception):
            r.confidence = 0.1  # type: ignore[misc]


# ─── Validity ────────────────────────────────────────────────────────

class TestValidity:
    def test_default_is_permanent(self) -> None:
        v = Validity()
        assert v.mode == "permanent"
        assert v.source == "default"

    def test_permanent_always_valid(self) -> None:
        v = Validity(mode="permanent")
        t_past = datetime(1900, 1, 1)
        t_future = datetime(3000, 1, 1)
        assert v.is_valid_at(t_past)
        assert v.is_valid_at(t_future)

    def test_permanent_respects_start_boundary(self) -> None:
        v = Validity(mode="permanent", start=datetime(2020, 1, 1))
        assert not v.is_valid_at(datetime(2019, 12, 31))
        assert v.is_valid_at(datetime(2020, 1, 1))
        assert v.is_valid_at(datetime(2030, 1, 1))

    def test_dated_range_requires_end(self) -> None:
        with pytest.raises(ValueError, match="dated_range.*end timestamp"):
            Validity(mode="dated_range", start=datetime(2020, 1, 1))

    def test_dated_range_bounds(self) -> None:
        v = Validity(
            mode="dated_range",
            start=datetime(2024, 1, 1),
            end=datetime(2024, 6, 30),
        )
        assert not v.is_valid_at(datetime(2023, 12, 31))
        assert v.is_valid_at(datetime(2024, 3, 15))
        assert v.is_valid_at(datetime(2024, 6, 30))
        assert not v.is_valid_at(datetime(2024, 7, 1))

    def test_duration_requires_half_life(self) -> None:
        with pytest.raises(ValueError, match="duration.*half_life_days"):
            Validity(mode="duration", start=datetime(2024, 1, 1))

    def test_duration_hard_end_is_three_half_lives(self) -> None:
        v = Validity(
            mode="duration",
            start=datetime(2024, 1, 1),
            half_life_days=30,
        )
        # 3 * 30 = 90 days, hard end at 2024-04-01 (leap year)
        assert v.is_valid_at(datetime(2024, 3, 31))
        assert not v.is_valid_at(datetime(2024, 4, 2))

    def test_decay_always_valid_after_start(self) -> None:
        v = Validity(
            mode="decay",
            start=datetime(2024, 1, 1),
            half_life_days=365,
        )
        # Decay affects confidence, not hard validity
        assert v.is_valid_at(datetime(2024, 1, 1))
        assert v.is_valid_at(datetime(2030, 1, 1))
        assert not v.is_valid_at(datetime(2023, 1, 1))

    def test_rejects_negative_half_life(self) -> None:
        with pytest.raises(ValueError, match="half_life_days must be positive"):
            Validity(mode="duration", start=datetime(2024, 1, 1), half_life_days=-5)


# ─── Belief ──────────────────────────────────────────────────────────

def _make_belief(
    bid: str = "b1",
    proposition: str = "I love sushi",
    **overrides,
) -> Belief:
    defaults = dict(
        id=bid,
        proposition=proposition,
        asserted_at=datetime(2024, 1, 1, 12, 0),
        asserted_in_session="sess-001",
        source_proposition_id=f"prop-{bid}",
    )
    defaults.update(overrides)
    return Belief(**defaults)


class TestBelief:
    def test_basic_construction(self) -> None:
        b = _make_belief()
        assert b.id == "b1"
        assert b.proposition == "I love sushi"
        assert b.confidence == 1.0
        assert b.validity.mode == "permanent"
        assert b.supersedes == []
        assert b.superseded_by == []
        assert b.reinforced_by == []

    def test_is_current_when_nothing_supersedes(self) -> None:
        b = _make_belief()
        assert b.is_current
        assert not b.is_superseded

    def test_is_superseded_when_superseded_by_nonempty(self) -> None:
        b = _make_belief(superseded_by=["b2"])
        assert not b.is_current
        assert b.is_superseded

    def test_rejects_out_of_range_confidence(self) -> None:
        with pytest.raises(ValueError, match="confidence must be in"):
            _make_belief(confidence=1.5)
        with pytest.raises(ValueError):
            _make_belief(confidence=-0.1)

    def test_mutable_fields_independent_per_instance(self) -> None:
        # Regression guard: field(default_factory=list) — not mutable default.
        b1 = _make_belief(bid="b1")
        b2 = _make_belief(bid="b2")
        b1.supersedes.append("x")
        assert b2.supersedes == []

    def test_can_carry_validity(self) -> None:
        v = Validity(
            mode="dated_range",
            start=datetime(2024, 1, 1),
            end=datetime(2024, 6, 30),
            source="explicit",
        )
        b = _make_belief(validity=v)
        assert b.validity.mode == "dated_range"
        assert b.validity.source == "explicit"
