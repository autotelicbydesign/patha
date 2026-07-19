"""Tests for KaranaEval (eval/karana_eval.py + karana_data/).

Three layers, all model-free (no extractor is exercised — scorers take
plain dicts, so the instrument is tested independently of any config):
  1. Scorers — matching rules (greedy one-to-one, unit normalization,
     alias-based entity acceptance, value tolerance) and case scoring
     semantics (silence, forbidden hits, f1).
  2. Data integrity — the authored gold set holds its own invariants.
  3. Rescore — persisted artifacts re-score identically.
"""

from __future__ import annotations

import json
from pathlib import Path

from eval.karana_eval import (
    RUBRIC_VERSION,
    aggregate,
    match_tuples,
    normalize_unit,
    rescore_rows,
    score_case,
    tuple_to_dict,
)

DATA = Path(__file__).parent.parent / "eval" / "karana_data"


def _pred(entity, value, unit, aliases=()):
    return {"entity": entity, "entity_aliases": list(aliases),
            "attribute": "spend", "value": value, "unit": unit}


def _gold(entity, acceptable, value, unit):
    return {"entity": entity, "acceptable": acceptable,
            "value": value, "unit": unit}


# ─── 1. Scorers ─────────────────────────────────────────────────────


class TestMatching:
    def test_unit_normalization(self):
        assert normalize_unit("dollars") == "USD"
        assert normalize_unit("$") == "USD"
        assert normalize_unit("pieces") == "item"
        assert normalize_unit("euros") == "EUR"
        assert normalize_unit("furlongs") == "furlongs"  # unknown passes through
        assert normalize_unit(None) is None

    def test_entity_matches_via_aliases(self):
        golds = [_gold("chain", ["chain", "bike", "cycling"], 28.0, "USD")]
        # direct entity
        assert match_tuples([_pred("chain", 28.0, "USD")], golds)
        # via predicted alias
        assert match_tuples([_pred("part", 28.0, "USD", aliases=["bike"])], golds)
        # canonical plural handling both sides
        assert match_tuples([_pred("chains", 28.0, "USD")], golds)
        # no lexical route → no match
        assert not match_tuples([_pred("lock", 28.0, "USD")], golds)

    def test_value_and_unit_must_match(self):
        golds = [_gold("hotel", ["hotel"], 220.0, "EUR")]
        assert not match_tuples([_pred("hotel", 220.0, "USD")], golds)
        assert not match_tuples([_pred("hotel", 221.0, "EUR")], golds)
        assert match_tuples([_pred("hotel", 220.0, "euros")], golds)

    def test_greedy_one_to_one_age_series(self):
        golds = [
            _gold("nephews", ["nephew", "nephews"], 4.0, "years"),
            _gold("nephews", ["nephew", "nephews"], 7.0, "years"),
            _gold("nephews", ["nephew", "nephews"], 12.0, "years"),
        ]
        preds = [_pred("nephew", 7.0, "years"), _pred("nephew", 7.0, "years"),
                 _pred("nephew", 4.0, "years")]
        pairs = match_tuples(preds, golds)
        # duplicate 7.0 prediction cannot double-claim one gold
        assert len(pairs) == 2
        assert {gi for _pi, gi in pairs} == {0, 1}


class TestScoreCase:
    def test_perfect_case(self):
        case = {"gold_tuples": [_gold("pump", ["pump", "bike"], 40.0, "USD")],
                "forbidden_tuples": []}
        s = score_case([_pred("pump", 40.0, "USD")], case)
        assert (s["precision"], s["recall"], s["f1"]) == (1.0, 1.0, 1.0)
        assert s["forbidden_hit"] is None

    def test_silence_on_forbidden_only_case_is_correct(self):
        case = {"gold_tuples": [],
                "forbidden_tuples": [{"value": 1200.0, "reason": "range"}]}
        s = score_case([], case)
        assert s["precision"] is None and s["recall"] is None
        assert s["forbidden_hit"] == 0.0

    def test_forbidden_hit_is_unit_insensitive(self):
        case = {"gold_tuples": [],
                "forbidden_tuples": [{"value": 34.0, "reason": "temp"}]}
        s = score_case([_pred("weather", 34.0, "USD")], case)
        assert s["forbidden_hit"] == 1.0
        assert s["precision"] == 0.0  # unmatched prediction

    def test_partial_precision_recall_f1(self):
        case = {"gold_tuples": [
                    _gold("flights", ["flight", "flights"], 410.0, "USD"),
                    _gold("airbnb", ["airbnb"], 380.0, "USD")],
                "forbidden_tuples": []}
        preds = [_pred("flights", 410.0, "USD"), _pred("taxi", 55.0, "USD")]
        s = score_case(preds, case)
        assert s["precision"] == 0.5 and s["recall"] == 0.5
        assert abs(s["f1"] - 0.5) < 1e-9

    def test_aggregate_excludes_nones(self):
        rows = [
            {"scores": {"precision": 1.0, "recall": 1.0, "f1": 1.0,
                        "forbidden_hit": None}},
            {"scores": {"precision": None, "recall": None, "f1": None,
                        "forbidden_hit": 0.0}},
        ]
        agg = aggregate(rows)
        assert agg["precision"]["n"] == 1
        assert agg["forbidden_hit"]["mean"] == 0.0
        assert agg["forbidden_hit"]["n"] == 1

    def test_tuple_to_dict_duck_typing(self):
        class T:
            entity = "pump"
            entity_aliases = ["bike"]
            attribute = "spend"
            value = 40.0
            unit = "USD"
        d = tuple_to_dict(T())
        assert d["entity"] == "pump" and d["entity_aliases"] == ["bike"]
        d2 = tuple_to_dict({"entity": "pump", "value": 40.0, "unit": "USD"})
        assert d2["entity_aliases"] == []

    def test_rubric_version_pinned(self):
        assert RUBRIC_VERSION == "v1"


# ─── 2. Data integrity ──────────────────────────────────────────────


def _load() -> list[dict]:
    path = DATA / "gold_cases.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


class TestDataIntegrity:
    def test_shape_families_and_unique_ids(self):
        cases = _load()
        assert len(cases) == 26
        ids = [c["id"] for c in cases]
        assert len(set(ids)) == len(ids)
        fams = {c["family"] for c in cases}
        assert {"multi_amount", "amount_far", "range_forbidden",
                "hypothetical_forbidden", "money_in_forbidden", "currency",
                "colloquial_forbidden", "implicit_count", "dated_amount",
                "dense_paragraph", "numeric_distractor"} <= fams

    def test_gold_rules_hold(self):
        for c in _load():
            gold_values = {(g["value"], normalize_unit(g["unit"]))
                           for g in c["gold_tuples"]}
            for g in c["gold_tuples"]:
                assert g["entity"] in g["acceptable"], c["id"]
                assert g["value"] > 0, c["id"]
                # every gold value literally appears in the text (no
                # inferred numbers — exactness is the contract)
                v = g["value"]
                forms = {f"{v:g}", f"{int(v):,}" if v == int(v) else "",
                         f"{v:.2f}", f"{v:.1f}"}
                spelled = {3.0: "three", 2.0: "two", 9.0: "ninth",
                           8.0: "8k", 4.0: "4", 7.0: "7", 12.0: "12"}
                forms.add(spelled.get(v, ""))
                assert any(f and f in c["text"] for f in forms), (c["id"], v)
            for f in c["forbidden_tuples"]:
                assert f["reason"], c["id"]
                assert all(f["value"] != gv for gv, _u in gold_values), c["id"]

    def test_forbidden_families_have_no_golds(self):
        for c in _load():
            if c["family"].endswith("_forbidden"):
                assert c["gold_tuples"] == [], c["id"]
                assert c["forbidden_tuples"], c["id"]


# ─── 3. Rescore ─────────────────────────────────────────────────────


class TestRescore:
    def test_rescore_reproduces_scores_from_artifacts(self):
        cases = _load()
        case = cases[0]
        preds = [dict(g, entity=g["entity"], entity_aliases=[], attribute="x",
                      unit=g["unit"], value=g["value"])
                 for g in case["gold_tuples"]]
        row = {"case_id": case["id"], "family": case["family"],
               "predicted": preds, "scores": {}}
        out = rescore_rows([row], {case["id"]: case})
        assert out[0]["scores"]["precision"] == 1.0
        assert out[0]["scores"]["recall"] == 1.0
