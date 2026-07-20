"""Tests for DepParseKaranaExtractor (karaṇa v2 — the no-LLM bet).

The eval (KaranaEval) owns quality measurement; this suite pins the
mechanics: veto families, entity-attachment shapes, unit typing, and
the regression that mattered — verbless list fragments must terminate
(spaCy Tokens are ephemeral views, so `tok.head is tok` is False even
at the root; the head-walk must compare indices).
"""

from __future__ import annotations

import pytest

spacy = pytest.importorskip("spacy")
try:
    spacy.load("en_core_web_sm")
except OSError:
    pytest.skip("en_core_web_sm not installed", allow_module_level=True)

from patha.belief.karana import DepParseKaranaExtractor


@pytest.fixture(scope="module")
def ex():
    return DepParseKaranaExtractor()


def _tuples(ex, text):
    return ex.extract(text, belief_id="t")


class TestEntityAttachment:
    def test_multiple_amounts_claim_nearest_objects(self, ex):
        ts = _tuples(ex, "spent $40 on the pump and $85 on the saddle bag")
        got = {(t.entity, t.value) for t in ts}
        assert got == {("pump", 40.0), ("saddle bag", 85.0)}

    def test_copular_subject(self, ex):
        ts = _tuples(ex, "the vet visit was $120")
        assert ts and ts[0].entity == "vet visit" and ts[0].value == 120.0

    def test_charge_verb_prefers_subject(self, ex):
        ts = _tuples(ex, "the cat sitter charged $150 total")
        assert [(t.entity, t.value) for t in ts] == [("cat sitter", 150.0)]

    def test_part_whole_alias(self, ex):
        ts = _tuples(ex, "replaced the chain for $28 — the bike deserves it")
        assert ts[0].entity == "chain"
        assert "bike" in ts[0].entity_aliases

    def test_verbless_list_fragment_terminates_and_attaches(self, ex):
        # THE regression: this shape hung forever before the index
        # comparison fix; it must both terminate and attach correctly
        ts = _tuples(
            ex, "big month: gym renewal $89, one physio session $70, "
                "and new running shoes $130",
        )
        got = {(t.entity, t.value) for t in ts if t.unit == "USD"}
        assert ("gym renewal", 89.0) in got
        assert ("physio session", 70.0) in got
        # the count of 'one physio session' is context, not a fact
        assert not any(t.unit == "item" for t in ts)


class TestUnits:
    def test_currency_words_and_symbols(self, ex):
        ts = _tuples(ex, "paid 220 euros for the hotel in Berlin")
        assert ts[0].unit == "EUR" and ts[0].value == 220.0
        ts = _tuples(ex, "the workshop fee was £150")
        assert ts[0].unit == "GBP"

    def test_decimal_boundary_regression(self, ex):
        # "$15.99 again" must never become (again, 99, item)
        ts = _tuples(ex, "the streaming subscription renewed at $15.99 again")
        assert [(t.value, t.unit) for t in ts] == [(15.99, "USD")]

    def test_counts_and_distances(self, ex):
        ts = _tuples(ex, "picked up three more succulents for the office")
        assert ts and ts[0].unit == "item" and ts[0].value == 3.0
        ts = _tuples(ex, "it was 34 degrees out; ran 8k anyway")
        assert [(t.entity, t.value, t.unit) for t in ts] == [("run", 8.0, "km")]

    def test_ages_win_over_their_count(self, ex):
        ts = _tuples(ex, "my three nephews are 4, 7, and 12")
        assert {t.value for t in ts if t.unit == "years"} == {4.0, 7.0, 12.0}
        assert not any(t.unit == "item" for t in ts)


class TestVetoes:
    @pytest.mark.parametrize("text", [
        "the new laptop will probably run me somewhere between $1,200 and $1,500",
        "if I took the Berlin job I'd be looking at about $95,000",
        "imagine dropping $300 on a single dinner — not happening",
        "the airline finally refunded the $240 for the cancelled flight",
        "turns out I was overcharged; they knocked $60 off the bill",
        "the brakes are going to cost a couple hundred to sort out",
        "meeting moved to 3pm; the build takes 45 minutes now and we're on version 2.7",
    ])
    def test_no_fabricated_tuples(self, ex, text):
        assert _tuples(ex, text) == []


class TestFallback:
    def test_missing_model_falls_back_loudly(self, monkeypatch):
        ex = DepParseKaranaExtractor()
        monkeypatch.setattr(
            "spacy.load",
            lambda *a, **k: (_ for _ in ()).throw(OSError("no model")),
        )
        ts = ex.extract("spent $40 on the pump", belief_id="t")
        assert ex.parser_available is False   # loud, inspectable
        assert ts and ts[0].value == 40.0     # regex fallback still works
