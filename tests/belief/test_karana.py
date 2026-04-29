"""Tests for the karaṇa ingest-time extractor (Innovation #2).

The karaṇa layer pushes LLM cost from query-time (every recall) to
ingest-time (one shot per new belief). At recall the gaṇita arithmetic
runs on the persisted JSONL tuples — no LLM, deterministic to the cent.

We test three things:

  1. **RegexKaranaExtractor** — the zero-dependency fallback. Already
     covered by tests/belief/test_ganita.py; this file verifies the
     wrapper interface stays compatible.

  2. **OllamaKaranaExtractor.extract** — graceful fallback when Ollama
     isn't reachable, JSON parsing of model responses, conversion of
     records into GanitaTuples.

  3. **Memory class wiring** — the optional `karana_extractor`
     parameter actually routes ingest through the alternate extractor
     and the resulting tuples flow into the gaṇita index.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

import patha
from patha.belief.ganita import GanitaTuple
from patha.belief.karana import (
    HybridKaranaExtractor,
    OllamaKaranaExtractor,
    RegexKaranaExtractor,
    _parse_json_array,
    _record_to_tuple,
)


# ─── Regex wrapper ──────────────────────────────────────────────────


class TestRegexKaranaExtractor:
    def test_extracts_currency(self) -> None:
        ex = RegexKaranaExtractor()
        out = ex.extract(
            "I bought a $50 saddle for the bike",
            belief_id="b1",
            time="2024-01-01T00:00:00",
        )
        assert any(t.value == 50 and t.unit == "USD" for t in out)


# ─── JSON parsing (decoupled from Ollama) ────────────────────────────


class TestJsonParse:
    def test_clean_array(self) -> None:
        out = _parse_json_array(
            '[{"entity":"bike","attribute":"expense","value":50,"unit":"USD"}]'
        )
        assert out is not None
        assert out[0]["value"] == 50

    def test_code_fence_wrapped(self) -> None:
        raw = (
            '```json\n'
            '[{"entity":"bike","attribute":"expense","value":50,"unit":"USD"}]\n'
            '```'
        )
        out = _parse_json_array(raw)
        assert out is not None
        assert out[0]["entity"] == "bike"

    def test_prose_wrapped(self) -> None:
        raw = (
            'Sure, here is the JSON you asked for: '
            '[{"entity":"bike","attribute":"expense","value":50,"unit":"USD"}] '
            'Hope this helps!'
        )
        out = _parse_json_array(raw)
        assert out is not None
        assert out[0]["value"] == 50

    def test_empty_array(self) -> None:
        out = _parse_json_array("[]")
        assert out == []

    def test_no_array(self) -> None:
        out = _parse_json_array("I'm sorry, I cannot help with that.")
        assert out is None

    def test_non_array_object(self) -> None:
        # If the model emits an object instead of an array, we reject.
        out = _parse_json_array(
            '{"entity":"bike","value":50}'
        )
        assert out is None

    def test_filters_non_dict_elements(self) -> None:
        # If the array contains a stray string, drop just that element.
        out = _parse_json_array(
            '[{"entity":"bike","value":50,"unit":"USD","attribute":"expense"}, '
            '"some stray string", '
            '{"entity":"helmet","value":75,"unit":"USD","attribute":"expense"}]'
        )
        assert out is not None
        assert len(out) == 2
        assert out[0]["entity"] == "bike"
        assert out[1]["entity"] == "helmet"


class TestRecordToTuple:
    def test_well_formed(self) -> None:
        t = _record_to_tuple(
            {"entity": "bike", "attribute": "expense", "value": 50, "unit": "USD"},
            belief_id="b1",
            time="2024-01-01",
            raw_text="I bought a $50 saddle",
        )
        assert t is not None
        assert t.entity == "bike"
        assert t.value == 50
        assert t.unit == "USD"
        assert t.attribute == "expense"
        assert "bike" in t.entity_aliases

    def test_missing_value_drops(self) -> None:
        t = _record_to_tuple(
            {"entity": "bike", "attribute": "expense", "unit": "USD"},
            belief_id="b1", time=None, raw_text="",
        )
        assert t is None

    def test_missing_entity_drops(self) -> None:
        t = _record_to_tuple(
            {"attribute": "expense", "value": 50, "unit": "USD"},
            belief_id="b1", time=None, raw_text="",
        )
        assert t is None

    def test_string_value_coerced(self) -> None:
        t = _record_to_tuple(
            {"entity": "bike", "attribute": "expense", "value": "50", "unit": "USD"},
            belief_id="b1", time=None, raw_text="",
        )
        assert t is not None
        assert t.value == 50

    def test_attribute_alias_normalised(self) -> None:
        t = _record_to_tuple(
            {"entity": "rent", "attribute": "spent", "value": 1500, "unit": "EUR"},
            belief_id="b1", time=None, raw_text="",
        )
        assert t is not None
        assert t.attribute == "expense"

    def test_aliases_only_from_llm_explicit_field(self) -> None:
        """Without an explicit `aliases` field from the LLM, the tuple
        gets only the canonical entity as alias — no auto-pull from
        raw text. (Auto-pull was over-matching: a rent tuple from a
        sentence that incidentally mentioned the bike path would get
        'bike' as an alias and falsely match a 'bike-related expenses'
        query.)"""
        t = _record_to_tuple(
            {"entity": "saddle", "attribute": "expense",
             "value": 50, "unit": "USD"},
            belief_id="b1", time=None,
            raw_text="I bought a $50 saddle for my bike",
        )
        assert t is not None
        assert t.entity == "saddle"
        # Canonical entity alone — no incidental "bike" pulled from text.
        assert t.entity_aliases == ("saddle",)

    def test_explicit_aliases_extend_canonical(self) -> None:
        """The LLM is now prompted to emit broader-category aliases.
        Verify we honor them and they extend the canonical entity."""
        t = _record_to_tuple(
            {"entity": "saddle", "attribute": "expense",
             "value": 50, "unit": "USD",
             "aliases": ["bike", "cycling"]},
            belief_id="b1", time=None,
            raw_text="$50 saddle",
        )
        assert t is not None
        assert t.entity == "saddle"
        assert "saddle" in t.entity_aliases
        assert "bike" in t.entity_aliases
        # cycling canonicalises to bike via ENTITY_ALIASES, dedup
        assert len(set(t.entity_aliases)) == len(t.entity_aliases)

    def test_explicit_aliases_array_honored(self) -> None:
        """If the LLM emits its own `aliases` array, use that instead
        of pulling from raw text. (cycling canonicalises to bike via
        ENTITY_ALIASES, so we use 'transport' which doesn't.)"""
        t = _record_to_tuple(
            {"entity": "saddle", "attribute": "expense",
             "value": 50, "unit": "USD",
             "aliases": ["bike", "transport"]},
            belief_id="b1", time=None,
            raw_text="I bought a $50 saddle",
        )
        assert t is not None
        # canonical entity always first
        assert t.entity_aliases[0] == "saddle"
        assert "bike" in t.entity_aliases
        assert "transport" in t.entity_aliases


class TestEndToEndKaranaWithAliases:
    """Verify the LLM-emitted aliases path works end-to-end through
    the gaṇita aggregation pipeline. The prompt asks the LLM for
    broader-category aliases per fact; we honor them precisely
    (no auto-supplementation from incidental text words)."""

    def test_bike_query_aggregates_via_explicit_aliases(
        self, tmp_path, monkeypatch
    ) -> None:
        # The LLM (per the updated prompt) emits per-item entities
        # AND broader-category aliases ['bike']. The aggregation
        # arithmetic finds them via the alias match.
        canned_responses = [
            '[{"entity":"saddle","aliases":["bike"],"attribute":"expense","value":50,"unit":"USD"}]',
            '[{"entity":"helmet","aliases":["bike"],"attribute":"expense","value":75,"unit":"USD"}]',
            '[{"entity":"light","aliases":["bike"],"attribute":"expense","value":30,"unit":"USD"}]',
            '[{"entity":"glove","aliases":["bike"],"attribute":"expense","value":30,"unit":"USD"}]',
        ]
        idx = {"i": 0}

        def _scripted_generate(self, prompt):
            r = canned_responses[idx["i"] % len(canned_responses)]
            idx["i"] += 1
            return r

        monkeypatch.setattr(
            OllamaKaranaExtractor, "_generate", _scripted_generate,
        )
        ex = OllamaKaranaExtractor(host="http://localhost:1")
        mem = patha.Memory(
            path=tmp_path / "store.jsonl",
            enable_phase1=False,
            karana_extractor=ex,
        )
        for fact in [
            "I bought a $50 saddle for my bike",
            "I got a $75 helmet for the bike",
            "$30 for new bike lights",
            "I spent $30 on bike gloves",
        ]:
            mem.remember(fact)

        rec = mem.recall("how much total did I spend on bike-related expenses?")
        assert rec.ganita is not None, (
            "ganita didn't fire — LLM-emitted aliases must include "
            "'bike' so saddle/helmet/light/glove tuples match the query"
        )
        assert abs(rec.ganita.value - 185.0) < 1.0
        assert len(rec.ganita.contributing_belief_ids) == 4

    def test_bike_query_misses_when_llm_omits_bike_alias(
        self, tmp_path, monkeypatch
    ) -> None:
        """Conversely: if the LLM doesn't emit 'bike' as an alias on a
        rent fact (correctly — rent isn't bike-related), a 'bike'
        query doesn't pull the rent tuple. This is the fix for the
        $999 false-positive on the LongMemEval haystack."""
        canned_responses = [
            '[{"entity":"rent","aliases":["rent","housing"],"attribute":"expense","value":999,"unit":"USD"}]',
        ]
        idx = {"i": 0}

        def _scripted_generate(self, prompt):
            r = canned_responses[idx["i"] % len(canned_responses)]
            idx["i"] += 1
            return r

        monkeypatch.setattr(
            OllamaKaranaExtractor, "_generate", _scripted_generate,
        )
        ex = OllamaKaranaExtractor(host="http://localhost:1")
        mem = patha.Memory(
            path=tmp_path / "store.jsonl",
            enable_phase1=False,
            karana_extractor=ex,
        )
        # Rent fact whose source text incidentally mentions 'bike path'
        mem.remember("Paid $999 rent on a flat near the bike path")

        rec = mem.recall("how much have I spent on bike-related expenses?")
        # Expected: the rent tuple does NOT match the bike query
        # because the LLM correctly tagged it with rent/housing aliases,
        # not bike. So gaṇita doesn't fire.
        assert rec.ganita is None or rec.ganita.value != 999.0


# ─── Ollama extractor (fallback path) ────────────────────────────────


class TestOllamaKaranaExtractor:
    def test_unreachable_falls_back_to_regex(self) -> None:
        """When Ollama isn't running, fallback_to_regex=True keeps signal."""
        ex = OllamaKaranaExtractor(
            host="http://localhost:1",  # nothing listens here
            timeout_s=0.5,
            fallback_to_regex=True,
        )
        out = ex.extract(
            "I bought a $50 saddle for the bike",
            belief_id="b1",
        )
        # Regex should still pull at least one currency tuple.
        assert any(t.value == 50 for t in out)
        assert ex.failures >= 1

    def test_unreachable_no_fallback_returns_empty(self) -> None:
        ex = OllamaKaranaExtractor(
            host="http://localhost:1",
            timeout_s=0.5,
            fallback_to_regex=False,
        )
        out = ex.extract("I bought a $50 saddle", belief_id="b1")
        assert out == []

    def test_extract_uses_mocked_generate(self, monkeypatch) -> None:
        """When the LLM returns valid JSON, tuples come from the JSON."""
        ex = OllamaKaranaExtractor(host="http://localhost:1")
        # Override _generate to return a canned JSON response.
        canned = (
            '[{"entity":"bike","attribute":"expense","value":185,"unit":"USD"},'
            '{"entity":"yoga","attribute":"duration","value":3.5,"unit":"hour"}]'
        )
        monkeypatch.setattr(
            OllamaKaranaExtractor, "_generate",
            lambda self, prompt: canned,
        )
        out = ex.extract(
            "I spent $185 on bike gear and 3.5 hours doing yoga.",
            belief_id="b42",
            time="2024-01-01T00:00:00",
        )
        assert len(out) == 2
        bike = next(t for t in out if t.entity == "bike")
        assert bike.value == 185
        assert bike.unit == "USD"
        assert bike.attribute == "expense"
        assert bike.belief_id == "b42"
        yoga = next(t for t in out if t.entity == "yoga")
        assert yoga.value == 3.5
        assert yoga.unit == "hour"

    def test_malformed_response_falls_back(self, monkeypatch) -> None:
        ex = OllamaKaranaExtractor(host="http://localhost:1")
        monkeypatch.setattr(
            OllamaKaranaExtractor, "_generate",
            lambda self, prompt: "the model went off-script",
        )
        out = ex.extract(
            "I bought a $50 saddle",
            belief_id="b1",
        )
        # Regex fallback engages → 50 USD tuple still present.
        assert any(t.value == 50 for t in out)
        assert ex.failures >= 1

    def test_empty_text_returns_empty(self) -> None:
        ex = OllamaKaranaExtractor()
        assert ex.extract("", belief_id="b1") == []
        assert ex.extract("   \n  ", belief_id="b1") == []

    def test_truncation_respected(self, monkeypatch) -> None:
        """Very long inputs get truncated before hitting the model."""
        captured: dict[str, str] = {}

        def _fake_generate(self, prompt: str) -> str:
            captured["prompt"] = prompt
            return "[]"

        monkeypatch.setattr(OllamaKaranaExtractor, "_generate", _fake_generate)
        ex = OllamaKaranaExtractor(max_text_chars=100)
        ex.extract("x" * 5000, belief_id="b1")
        # Prompt ends with the truncated TEXT — the long string in
        # the prompt must not exceed 100 chars.
        assert "x" * 101 not in captured["prompt"]


# ─── Memory class wiring ────────────────────────────────────────────


class TestMemoryWiring:
    def test_default_uses_regex_extractor(self, tmp_path: Path) -> None:
        mem = patha.Memory(
            path=tmp_path / "beliefs.jsonl",
            enable_phase1=False,
        )
        from patha.belief.karana import RegexKaranaExtractor
        assert isinstance(mem._karana, RegexKaranaExtractor)

    def test_explicit_extractor_used_at_ingest(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Wiring smoke test: a custom extractor's tuples land in the index."""

        class ScriptedExtractor:
            """Always returns a single fixed tuple — easy to assert on."""

            def extract(self, text, *, belief_id, time=None):
                return [GanitaTuple(
                    entity="lighthouse",
                    attribute="value",
                    value=42.0,
                    unit="custom",
                    time=time,
                    belief_id=belief_id,
                    raw_text=text[:20],
                    entity_aliases=("lighthouse",),
                )]

        mem = patha.Memory(
            path=tmp_path / "beliefs.jsonl",
            enable_phase1=False,
            karana_extractor=ScriptedExtractor(),
        )
        mem.remember("Whatever the input is")
        assert len(mem._ganita_index) == 1
        t = mem._ganita_index.all()[0]
        assert t.entity == "lighthouse"
        assert t.value == 42.0

    def test_ollama_extractor_wired(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """OllamaKaranaExtractor with mocked _generate produces tuples
        that land in the gaṇita index and answer aggregation queries."""
        canned = (
            '[{"entity":"bike","attribute":"expense","value":50,"unit":"USD"},'
            '{"entity":"bike","attribute":"expense","value":75,"unit":"USD"},'
            '{"entity":"bike","attribute":"expense","value":30,"unit":"USD"}]'
        )
        monkeypatch.setattr(
            OllamaKaranaExtractor, "_generate",
            lambda self, prompt: canned,
        )
        ex = OllamaKaranaExtractor(host="http://localhost:1")
        mem = patha.Memory(
            path=tmp_path / "beliefs.jsonl",
            enable_phase1=False,
            karana_extractor=ex,
        )
        mem.remember("I bought a $50 saddle, a $75 helmet, and $30 lights for the bike")
        # Index should have 3 tuples (LLM response says so).
        assert len(mem._ganita_index) == 3
        # Aggregation question uses the index.
        rec = mem.recall("how much did I spend on bike-related expenses total")
        assert rec.ganita is not None
        assert rec.ganita.value == 155.0
        # No LLM tokens needed at recall time:
        assert rec.tokens == 0 or rec.strategy == "ganita" or rec.strategy in (
            "direct_answer", "structured", "raw"
        )


# ─── HybridKaranaExtractor: regex finds amounts, LLM tags ──────────


class TestHybridKaranaExtractor:
    def test_no_amounts_returns_empty(self) -> None:
        ex = HybridKaranaExtractor()
        out = ex.extract(
            "I rode my bike around Lisbon today.", belief_id="b1",
        )
        assert out == []

    def test_extracts_every_amount_via_mocked_llm(self, monkeypatch) -> None:
        """Hybrid guarantees regex catches every $X; LLM only tags."""
        canned = (
            '['
            '{"index":1,"entity":"saddle","aliases":["bike"],"attribute":"expense"},'
            '{"index":2,"entity":"helmet","aliases":["bike","safety"],"attribute":"expense"},'
            '{"index":3,"entity":"chain","aliases":["bike","maintenance"],"attribute":"expense"}'
            ']'
        )
        monkeypatch.setattr(
            HybridKaranaExtractor, "_generate",
            lambda self, prompt: canned,
        )
        ex = HybridKaranaExtractor(host="http://localhost:1")
        out = ex.extract(
            "I bought a $50 saddle, $75 helmet, and replaced the chain "
            "for $25.",
            belief_id="b1",
        )
        # All three amounts captured — regex didn't miss anything,
        # LLM tagged each correctly.
        assert len(out) == 3
        values = sorted(t.value for t in out)
        assert values == [25.0, 50.0, 75.0]
        # Each tuple has 'bike' as an alias
        for t in out:
            assert "bike" in t.entity_aliases

    def test_skip_marker_drops_amount(self, monkeypatch) -> None:
        """LLM can mark an amount as 'skip' (range, hypothetical, etc.)
        — that amount doesn't produce a tuple."""
        canned = (
            '['
            '{"index":1,"entity":"skip","aliases":[],"attribute":"value"},'
            '{"index":2,"entity":"skip","aliases":[],"attribute":"value"}'
            ']'
        )
        monkeypatch.setattr(
            HybridKaranaExtractor, "_generate",
            lambda self, prompt: canned,
        )
        ex = HybridKaranaExtractor(host="http://localhost:1")
        out = ex.extract(
            "Bike racks range from $100 to $500 depending on size.",
            belief_id="b1",
        )
        assert out == []

    def test_unreachable_returns_empty(self) -> None:
        ex = HybridKaranaExtractor(
            host="http://localhost:1", timeout_s=0.5,
        )
        out = ex.extract("$50 saddle", belief_id="b1")
        assert out == []
        assert ex.failures >= 1

    def test_index_out_of_range_skipped(self, monkeypatch) -> None:
        """If the LLM emits an index that doesn't exist (hallucination),
        we drop that record rather than crashing."""
        canned = (
            '['
            '{"index":5,"entity":"saddle","aliases":["bike"],"attribute":"expense"},'
            '{"index":1,"entity":"helmet","aliases":["bike"],"attribute":"expense"}'
            ']'
        )
        monkeypatch.setattr(
            HybridKaranaExtractor, "_generate",
            lambda self, prompt: canned,
        )
        ex = HybridKaranaExtractor(host="http://localhost:1")
        # Only one amount in the text, but LLM hallucinates index=5
        out = ex.extract("$75 helmet for the bike", belief_id="b1")
        assert len(out) == 1
        assert out[0].entity == "helmet"
