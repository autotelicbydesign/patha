"""Tests for the gaṇita (procedural arithmetic) layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from patha.belief.ganita import (
    GanitaIndex,
    GanitaTuple,
    answer_aggregation_question,
    detect_aggregation,
    extract_entity_hints,
    extract_tuples,
)


class TestCurrencyExtraction:
    def test_dollar_sign(self):
        ts = extract_tuples("I spent $50 on a saddle for the bike", belief_id="b1")
        assert any(t.value == 50.0 and t.unit == "USD" for t in ts)

    def test_with_commas(self):
        ts = extract_tuples("I raised $3,750 for charity", belief_id="b1")
        assert any(t.value == 3750.0 for t in ts)

    def test_decimal(self):
        ts = extract_tuples("It cost $99.99", belief_id="b1")
        assert any(abs(t.value - 99.99) < 0.01 for t in ts)

    def test_dollars_word(self):
        ts = extract_tuples("They paid me 200 dollars", belief_id="b1")
        assert any(t.value == 200.0 and t.unit == "USD" for t in ts)

    def test_attribute_inference(self):
        spent = extract_tuples("I spent $50 on bike helmet", belief_id="b1")
        earned = extract_tuples("I earned $100 from selling old books", belief_id="b1")
        assert spent[0].attribute == "expense"
        assert earned[0].attribute == "income"


class TestFalsePositiveFilters:
    """Three filter classes that prevent the most common spurious
    extractions on real conversational text. All real purchases pass;
    these specific shapes get dropped."""

    # ─── Code fix 1: range filtering ─────────────────────────────

    def test_range_with_to(self):
        """'racks range from $100 to $500' → no purchases extracted."""
        ts = extract_tuples(
            "Bike racks range from $100 to $500 depending on size.",
            belief_id="b1",
        )
        usd = [t for t in ts if t.unit == "USD"]
        assert usd == [], (
            f"expected no currency tuples from a range expression; "
            f"got {[t.value for t in usd]}"
        )

    def test_range_with_dash(self):
        ts = extract_tuples(
            "Helmets typically run $50-$200.", belief_id="b1",
        )
        usd = [t for t in ts if t.unit == "USD"]
        assert usd == []

    def test_range_with_en_dash(self):
        ts = extract_tuples(
            "Around $80–$120 for decent brushes.", belief_id="b1",
        )
        usd = [t for t in ts if t.unit == "USD"]
        assert usd == []

    def test_range_does_not_swallow_unrelated_purchase(self):
        """A range in one sentence shouldn't suppress a real purchase
        in another."""
        ts = extract_tuples(
            "Bike racks run $100 to $500. I bought a $50 saddle.",
            belief_id="b1",
        )
        values = sorted(t.value for t in ts if t.unit == "USD")
        assert 50.0 in values  # saddle survives
        assert 100.0 not in values  # range dropped
        assert 500.0 not in values  # range dropped

    # ─── Code fix 2: hypothetical / aspirational filter ──────────

    def test_thinking_about(self):
        ts = extract_tuples(
            "I'm thinking about a $300 helmet but haven't decided.",
            belief_id="b1",
        )
        usd = [t for t in ts if t.unit == "USD"]
        assert usd == []

    def test_would_cost(self):
        ts = extract_tuples(
            "A new wheelset would cost $450 — way out of budget.",
            belief_id="b1",
        )
        usd = [t for t in ts if t.unit == "USD"]
        assert usd == []

    def test_considering(self):
        ts = extract_tuples(
            "Considering the $200 model from the bike shop.",
            belief_id="b1",
        )
        usd = [t for t in ts if t.unit == "USD"]
        assert usd == []

    def test_if_i_bought(self):
        ts = extract_tuples(
            "If I bought the carbon frame, $1200 minimum.",
            belief_id="b1",
        )
        usd = [t for t in ts if t.unit == "USD"]
        assert usd == []

    # ─── Code fix 3: negated-purchase filter ─────────────────────

    def test_didnt_buy(self):
        ts = extract_tuples(
            "I didn't buy the $400 frame in the end.", belief_id="b1",
        )
        usd = [t for t in ts if t.unit == "USD"]
        assert usd == []

    def test_couldnt_afford(self):
        ts = extract_tuples(
            "Couldn't afford the $800 wheelset, sadly.", belief_id="b1",
        )
        usd = [t for t in ts if t.unit == "USD"]
        assert usd == []

    def test_returned(self):
        ts = extract_tuples(
            "Returned the $120 helmet — wrong size.", belief_id="b1",
        )
        usd = [t for t in ts if t.unit == "USD"]
        assert usd == []

    def test_decided_against(self):
        ts = extract_tuples(
            "Decided against the $250 lock.", belief_id="b1",
        )
        usd = [t for t in ts if t.unit == "USD"]
        assert usd == []

    # ─── Real purchases still pass ───────────────────────────────

    def test_real_purchase_with_distant_range_in_other_sentence(self):
        """A 'range' phrase >50 chars from a real purchase doesn't
        pollute the extraction."""
        long_text = (
            "Bike rack prices vary; you can find them anywhere from "
            "$100 to $500. " + ("Filler. " * 20) +
            "Yesterday I actually bought a $50 saddle."
        )
        ts = extract_tuples(long_text, belief_id="b1")
        values = sorted(t.value for t in ts if t.unit == "USD")
        assert 50.0 in values
    def test_hours(self):
        ts = extract_tuples("I spent 3.5 hours on yoga today", belief_id="b1")
        durations = [t for t in ts if t.attribute == "duration"]
        assert durations
        assert durations[0].value == 3.5
        assert durations[0].unit == "hour"

    def test_weeks(self):
        ts = extract_tuples("It took me 5.5 weeks to read the book", belief_id="b1")
        durations = [t for t in ts if t.attribute == "duration"]
        assert durations
        assert durations[0].value == 5.5
        assert durations[0].unit == "week"


class TestPercent:
    def test_percent(self):
        ts = extract_tuples("My portfolio is 30% bonds", belief_id="b1")
        pct = [t for t in ts if t.attribute == "percentage"]
        assert pct
        assert pct[0].value == 30
        assert pct[0].unit == "%"


class TestCountExtraction:
    def test_digit_count(self):
        ts = extract_tuples("I own 4 bikes", belief_id="b1")
        counts = [t for t in ts if t.attribute == "count"]
        assert counts
        assert counts[0].value == 4
        assert counts[0].entity == "bike"

    def test_word_count(self):
        ts = extract_tuples("I have twenty short stories", belief_id="b1")
        counts = [t for t in ts if t.attribute == "count"]
        assert any(c.value == 20 for c in counts)


class TestEntityCanon:
    def test_plural_singular(self):
        a = extract_tuples("I have 4 bikes", belief_id="b1")
        b = extract_tuples("I have 4 bicycles", belief_id="b1")
        # Both should canonicalize to "bike"
        ents_a = {t.entity for t in a if t.attribute == "count"}
        ents_b = {t.entity for t in b if t.attribute == "count"}
        assert "bike" in ents_a
        assert "bike" in ents_b


# ─── Index ───────────────────────────────────────────────────────────


class TestIndex:
    def test_in_memory_add_lookup(self):
        idx = GanitaIndex()
        t1 = GanitaTuple(entity="bike", attribute="expense", value=50,
                         unit="USD", time=None, belief_id="b1", raw_text="$50")
        t2 = GanitaTuple(entity="bike", attribute="expense", value=75,
                         unit="USD", time=None, belief_id="b2", raw_text="$75")
        idx.add(t1)
        idx.add(t2)
        rows = idx.all_for("bike", "expense")
        assert len(rows) == 2
        assert sum(r.value for r in rows) == 125

    def test_alias_lookup(self):
        idx = GanitaIndex()
        idx.add(GanitaTuple(entity="bike", attribute="expense", value=50,
                            unit="USD", time=None, belief_id="b1", raw_text="$50"))
        # Query with "bicycles" (plural alias)
        rows = idx.all_for("bicycles", "expense")
        assert len(rows) == 1

    def test_persistence(self, tmp_path):
        path = tmp_path / "ganita.jsonl"
        idx1 = GanitaIndex(persistence_path=path)
        idx1.add(GanitaTuple(entity="bike", attribute="expense", value=50,
                             unit="USD", time=None, belief_id="b1", raw_text="$50"))
        idx1.add(GanitaTuple(entity="bike", attribute="expense", value=75,
                             unit="USD", time=None, belief_id="b2", raw_text="$75"))
        # Reload — open a new instance over the same path
        idx2 = GanitaIndex(persistence_path=path)
        rows = idx2.all_for("bike", "expense")
        assert len(rows) == 2
        assert sum(r.value for r in rows) == 125


# ─── Aggregation detection ─────────────────────────────────────────


class TestAggregationDetection:
    def test_sum_keywords(self):
        for q in [
            "How much did I spend in total on bikes?",
            "What's the total of my workshop expenses?",
            "How much altogether?",
            "Combined cost of meals?",
        ]:
            assert detect_aggregation(q) == "sum"

    def test_count_keywords(self):
        for q in [
            "How many bikes do I own?",
            "Count of stories I've written",
            "Number of yoga classes?",
        ]:
            assert detect_aggregation(q) == "count"

    def test_average(self):
        assert detect_aggregation("What is the average age?") == "average"
        assert detect_aggregation("On average, how much per month?") == "average"

    def test_difference_beats_sum(self):
        # "how much more" must match difference, not sum
        assert detect_aggregation("How much more did I spend in Hawaii vs Tokyo?") == "difference"

    def test_no_match(self):
        assert detect_aggregation("What's the weather today?") is None

    def test_extract_hints(self):
        hints = extract_entity_hints("How much did I spend on bikes this year?")
        assert "bike" in hints  # canonicalized


# ─── End-to-end: aggregation answer ────────────────────────────────


class TestAggregationAnswer:
    def test_bike_total_recovers_185(self):
        idx = GanitaIndex()
        # Mirrors the actual LongMemEval question $185 = 50+75+30+30
        for amount, label in [(50, "saddle"), (75, "helmet"), (30, "lights"), (30, "gloves")]:
            ts = extract_tuples(
                f"I spent ${amount} on a {label} for my bike",
                belief_id=f"b-{amount}",
            )
            for t in ts:
                idx.add(t)
        result = answer_aggregation_question(
            "How much total money have I spent on bike-related expenses?", idx,
        )
        assert result is not None
        assert result.operator == "sum"
        assert result.value == 185
        assert result.unit == "USD"
        assert len(result.contributing_belief_ids) == 4

    def test_count_bikes_owned(self):
        idx = GanitaIndex()
        for line in [
            "I bought 1 mountain bike",
            "I own 1 road bike",
            "I have 2 commuter bikes",
        ]:
            for t in extract_tuples(line, belief_id=line):
                if t.attribute == "count":
                    idx.add(t)
        # All should be canonicalized to entity="bike", attribute="count"
        ts = idx.all_for("bike", "count")
        # We expect at least one entry per line, all under "bike"
        assert len(ts) >= 3

    def test_no_op_when_no_aggregation_keyword(self):
        idx = GanitaIndex()
        for t in extract_tuples("I spent $50 on a saddle", belief_id="b1"):
            idx.add(t)
        result = answer_aggregation_question(
            "What did I buy for my bike?", idx,
        )
        # No aggregation operator → returns None
        assert result is None

    def test_returns_none_if_no_matching_entity(self):
        idx = GanitaIndex()
        for t in extract_tuples("I spent $50 on a saddle for the bike", belief_id="b1"):
            idx.add(t)
        result = answer_aggregation_question(
            "How much have I spent on jewelry?", idx,
        )
        # No "jewelry" tuples → None
        assert result is None
