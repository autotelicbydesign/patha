"""Tests for composition (time-series-of-sums — the chained pramāṇa).

Covers: intent detection (incl. the theft guards that keep narrative
themes and scalar frames off the route), the production/eval trend-rule
cross-test (they must never drift), bucketing rules (gap, currency,
count-by-beliefs), degradation to scalar, and recall() integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from patha.belief.composition import (
    CompositionIntent,
    classify_trend,
    compose_scalar,
    compose_series,
    detect_composition,
)


# ─── Detection ──────────────────────────────────────────────────────


class TestDetection:
    def test_composition_phrasings(self):
        for q, op, gran in (
            ("how has my spending on the bike evolved?", "sum", "month"),
            ("how much did I spend on coffee gear each month?", "sum", "month"),
            ("how has my travel spending changed year over year?", "sum", "year"),
            ("how many yoga classes did I go to each month?", "count", "month"),
            ("how has my average spend on dinners out changed month to month?",
             "avg", "month"),
        ):
            got = detect_composition(q)
            assert got is not None, q
            assert (got.op, got.granularity) == (op, gran), q

    def test_theft_guards(self):
        # narrative themes stay narrative (EvolutionEval protection):
        assert detect_composition(
            "how has my thinking about budgeting evolved?") is None
        assert detect_composition(
            "how has my thinking about meditation evolved?") is None
        # scalar frames stay plain gaṇita:
        assert detect_composition(
            "how much have I spent on the aquarium in total?") is None
        assert detect_composition(
            "how much did I spend on board games altogether?") is None
        # plain retrieval:
        assert detect_composition("what did I say about the saddle?") is None


class TestTrendRuleCrossPinned:
    def test_production_mirrors_eval_frozen_rule(self):
        # the eval owns the frozen v1 copy; production must never drift
        from eval.composition_eval import classify_trend as eval_trend
        cases = [
            [], [50], [10, 20, 30], [30, 20, 10], [100, 95, 102],
            [40, 45, 300, 42], [40, 45, 90, 42], [10, 30, 20],
            [40, 44, 200], [15.99, 15.99, 15.99], [0, 0, 0], [1, 1, 2],
        ]
        for vals in cases:
            assert classify_trend(vals) == eval_trend(vals), vals


# ─── Bucketing rules ────────────────────────────────────────────────


@dataclass
class _T:
    entity: str
    attribute: str
    value: float
    unit: str
    belief_id: str
    time: str | None
    entity_aliases: list = field(default_factory=list)

    def matches_entity(self, q):
        return q in (self.entity, *self.entity_aliases)


class _Index:
    def __init__(self, tuples):
        self._tuples = tuples

    def all_for(self, hint):
        return [t for t in self._tuples if t.matches_entity(hint)]


@dataclass
class _B:
    id: str
    proposition: str
    asserted_at: datetime
    source_proposition_id: str = "p"


class _Store:
    def __init__(self, beliefs):
        self._b = beliefs

    def current(self):
        return list(self._b)


_INTENT = CompositionIntent(op="sum", granularity="month")


class TestComposeSeries:
    def _index(self):
        return _Index([
            _T("bike", "expense", 40.0, "USD", "b1", "2026-01-05T09:00:00"),
            _T("bike", "expense", 65.0, "USD", "b2", "2026-02-10T09:00:00"),
            _T("bike", "expense", 90.0, "USD", "b3", "2026-04-01T09:00:00"),
        ])

    def test_gap_rule_no_fabricated_zero(self):
        r = compose_series("how has my spending on the bike evolved?",
                           _INTENT, self._index())
        assert [b.period for b in r.buckets] == ["2026-01", "2026-02", "2026-04"]
        assert "2026-03" not in {b.period for b in r.buckets}
        assert r.trend == "rising"
        assert r.buckets[0].contributing_belief_ids == ["b1"]

    def test_currency_rule_excludes_non_usd(self):
        idx = _Index([
            _T("hotel", "expense", 300.0, "USD", "b1", "2026-01-05T00:00:00"),
            _T("hotel", "expense", 220.0, "EUR", "b2", "2026-02-05T00:00:00"),
            _T("hotel", "expense", 180.0, "USD", "b3", "2026-03-05T00:00:00"),
        ])
        r = compose_series("how has my spending on the hotel evolved?",
                           _INTENT, idx)
        assert [b.period for b in r.buckets] == ["2026-01", "2026-03"]

    def test_degradation_under_two_buckets(self):
        idx = _Index([
            _T("camera", "expense", 600.0, "USD", "b1", "2026-01-05T00:00:00"),
            _T("camera", "expense", 150.0, "USD", "b2", "2026-01-20T00:00:00"),
        ])
        assert compose_series("how has my camera spending evolved?",
                              _INTENT, idx) is None
        scalar = compose_scalar("how has my camera spending evolved?",
                                _INTENT, idx)
        assert scalar is not None and scalar.value == 750.0
        assert scalar.operator == "sum"

    def test_count_by_beliefs_when_no_tuples(self):
        store = _Store([
            _B("b1", "went to a hot yoga class", datetime(2026, 1, 5)),
            _B("b2", "took an evening yoga class", datetime(2026, 1, 20)),
            _B("b3", "went to a morning yoga class", datetime(2026, 2, 3)),
            _B("b4", "repotted the monstera", datetime(2026, 2, 4)),
        ])
        intent = CompositionIntent(op="count", granularity="month")
        r = compose_series("how many yoga classes did I go to each month?",
                           intent, _Index([]), store=store)
        assert [(b.period, b.value) for b in r.buckets] == [
            ("2026-01", 2.0), ("2026-02", 1.0),
        ]
        assert r.buckets[0].unit == "item"


# ─── recall() integration ───────────────────────────────────────────


class TestRouting:
    def test_composition_routes_and_neighbours_hold(self, tmp_path):
        import patha
        mem = patha.Memory(path=tmp_path / "b.jsonl", detector="stub",
                           enable_phase1=False)
        for text, when in (
            ("spent $40 on a chain for the bike", datetime(2026, 1, 5)),
            ("spent $65 on tires for the bike", datetime(2026, 2, 10)),
            ("spent $90 on a bike tune-up", datetime(2026, 3, 1)),
        ):
            mem.remember(text, asserted_at=when)
        r = mem.recall("how has my spending on the bike evolved?")
        assert r.strategy == "composition"
        assert r.composition is not None
        assert [b.period for b in r.composition.buckets] == [
            "2026-01", "2026-02", "2026-03",
        ]
        assert r.composition.trend == "rising"
        assert r.tokens == 0
        # plain scalar stays plain gaṇita:
        r2 = mem.recall("how much did I spend on the bike in total?")
        assert r2.strategy == "ganita" and r2.composition is None