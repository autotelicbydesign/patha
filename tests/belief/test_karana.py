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

    def test_aliases_supplemented_from_raw_text(self) -> None:
        """When the LLM emits entity='saddle' but the source text mentions
        'bike', the aliases should include 'bike' so a question that
        asks about bike-related expenses still matches."""
        t = _record_to_tuple(
            {"entity": "saddle", "attribute": "expense",
             "value": 50, "unit": "USD"},
            belief_id="b1", time=None,
            raw_text="I bought a $50 saddle for my bike",
        )
        assert t is not None
        assert t.entity == "saddle"
        assert "bike" in t.entity_aliases

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
    """Verify the alias path works end-to-end through the gaṇita
    aggregation pipeline — the LLM extracts saddle/helmet/light/glove
    expenses, but a question about 'bike' still finds and sums them."""

    def test_bike_query_aggregates_saddle_helmet_etc(
        self, tmp_path, monkeypatch
    ) -> None:
        # The LLM emits per-item entities (saddle, helmet, light, glove)
        # but each tuple's raw_text mentions 'bike', so context aliases
        # should pull them in for a 'bike' query.
        canned_responses = [
            '[{"entity":"saddle","attribute":"expense","value":50,"unit":"USD"}]',
            '[{"entity":"helmet","attribute":"expense","value":75,"unit":"USD"}]',
            '[{"entity":"light","attribute":"expense","value":30,"unit":"USD"}]',
            '[{"entity":"glove","attribute":"expense","value":30,"unit":"USD"}]',
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

        # Aggregation question with 'bike' in it — should hit all 4
        # expense tuples via the text-context aliases.
        rec = mem.recall("how much total did I spend on bike-related expenses?")
        assert rec.ganita is not None, (
            "ganita didn't fire — text-context aliases must pull "
            "saddle/helmet/light/glove tuples in for a 'bike' query"
        )
        assert abs(rec.ganita.value - 185.0) < 1.0
        assert len(rec.ganita.contributing_belief_ids) == 4


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
