"""Tests for CompositionEval (eval/composition_eval.py + composition_data/).

Three layers, all model-free (the test_evolution_eval.py pattern):
  1. Scorers — pure functions, hand-verified cases (the frozen rubric,
     including the frozen classify_trend rule).
  2. Data integrity — dev scenarios load, schema holds, and the gold
     rules the runner's docstring declares are actually true of the
     data: trend labels equal classify_trend(expected values), bucket
     indices are in range, multi-currency exclusions never contribute.
  3. Rescore — persisted artifacts re-score identically.
"""

from __future__ import annotations

import json
from pathlib import Path

from eval.composition_eval import (
    RUBRIC_VERSION,
    aggregate,
    classify_trend,
    rescore_rows,
    score_bucket_periods,
    score_bucket_values,
    score_question,
    score_receipts,
    score_routed,
    score_scalar,
    score_trend,
)

DATA = Path(__file__).parent.parent / "eval" / "composition_data"


# ─── 1. Scorers (frozen rubric v1) ──────────────────────────────────


class TestClassifyTrend:
    def test_rising_falling_flat(self):
        assert classify_trend([10, 20, 30]) == "rising"
        assert classify_trend([30, 20, 10]) == "falling"
        assert classify_trend([100, 95, 102]) == "flat"  # spread ≤ 10%

    def test_spike_dominates(self):
        # one bucket > 2.5× the rest, rest roughly level
        assert classify_trend([40, 45, 300, 42]) == "spike"
        # peak not dominant enough → not spike; ambiguous → None
        assert classify_trend([40, 45, 90, 42]) is None

    def test_none_cases(self):
        assert classify_trend([]) is None
        assert classify_trend([50]) is None          # < 2 buckets
        assert classify_trend([10, 30, 20]) is None  # ambiguous zigzag

    def test_order_spike_beats_rising(self):
        # monotone series ending in a dominant peak: spike wins (checked
        # first), not rising — the frozen precedence.
        assert classify_trend([40, 44, 200]) == "spike"


class TestScorers:
    def test_routed_bidirectional(self):
        assert score_routed("composition", "composition") == 1.0
        assert score_routed("ganita", "composition") == 0.0
        # theft direction: composition stealing a plain-sum question
        assert score_routed("composition", "ganita") == 0.0

    def test_bucket_periods_jaccard_penalizes_fabricated_zero(self):
        e = [{"period": "2026-01"}, {"period": "2026-03"}]  # gap in 02
        exact = [{"period": "2026-01"}, {"period": "2026-03"}]
        fabricated = exact + [{"period": "2026-02"}]  # zero-filled gap
        assert score_bucket_periods(exact, e) == 1.0
        assert score_bucket_periods(fabricated, e) == 2 / 3
        assert score_bucket_periods([], []) is None

    def test_bucket_values_cent_exact_common_periods_only(self):
        e = [{"period": "2026-01", "value": 40.0},
             {"period": "2026-02", "value": 65.0}]
        r = [{"period": "2026-01", "value": 40.004},   # within half-cent
             {"period": "2026-02", "value": 65.02}]    # off by 2 cents
        assert score_bucket_values(r, e) == 0.5
        # no common periods → None (bucket_periods owns that miss)
        assert score_bucket_values([{"period": "2026-09", "value": 1}], e) is None

    def test_receipts_exact_set(self):
        e = [{"period": "2026-01", "contributing": [0, 2]}]
        assert score_receipts([{"period": "2026-01", "contributing": [2, 0]}], e) == 1.0
        assert score_receipts([{"period": "2026-01", "contributing": [0]}], e) == 0.0
        assert score_receipts([{"period": "2026-01"}], e) == 0.0  # missing receipts

    def test_trend_and_scalar(self):
        assert score_trend("rising", "rising") == 1.0
        assert score_trend(None, "rising") == 0.0
        assert score_trend("rising", None) is None
        assert score_scalar(830.0, 830.0) == 1.0
        assert score_scalar(None, 830.0) == 0.0
        assert score_scalar(830.0, None) is None

    def test_score_question_gates_on_expected_path(self):
        q = {
            "expected_route": "composition",
            "expected_buckets": [{"period": "2026-01", "value": 40.0,
                                  "contributing": [0]}],
            "expected_trend": None,
        }
        # wrong route: routed owns the failure, content scorers None
        s = score_question("ganita", None, None, 40.0, q)
        assert s["routed"] == 0.0
        assert all(s[k] is None for k in
                   ("bucket_periods", "bucket_values", "receipts", "trend",
                    "scalar"))
        # right route: content scorers engage
        s = score_question(
            "composition",
            [{"period": "2026-01", "value": 40.0, "contributing": [0]}],
            None, None, q,
        )
        assert s["routed"] == 1.0
        assert s["bucket_periods"] == 1.0 and s["receipts"] == 1.0

    def test_aggregate_excludes_nones(self):
        rows = [
            {"scores": {"routed": 1.0, "bucket_periods": 1.0,
                        "bucket_values": 1.0, "receipts": None,
                        "trend": None, "scalar": None}},
            {"scores": {"routed": 0.0, "bucket_periods": None,
                        "bucket_values": None, "receipts": None,
                        "trend": None, "scalar": 1.0}},
        ]
        agg = aggregate(rows)
        assert agg["routed"]["mean"] == 0.5 and agg["routed"]["n"] == 2
        assert agg["bucket_periods"]["n"] == 1
        assert agg["scalar"]["n"] == 1

    def test_rubric_version_pinned(self):
        assert RUBRIC_VERSION == "v1"


# ─── 2. Data integrity (the gold rules, enforced) ───────────────────


def _load() -> list[dict]:
    path = DATA / "dev_scenarios.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


class TestDataIntegrity:
    def test_shape_and_families(self):
        scenarios = _load()
        assert len(scenarios) >= 20
        assert not any(s.get("heldout") for s in scenarios)  # dev-only
        fams = {s["family"] for s in scenarios}
        # the design's required coverage
        assert {"monthly_sum", "multi_currency", "single_bucket_degradation",
                "routing_negative_ganita",
                "routing_negative_narrative"} <= fams

    def test_gold_rules_hold(self):
        for s in _load():
            n = len(s["propositions"])
            for q in s["questions"]:
                eb = q.get("expected_buckets") or []
                contributing_all: set[int] = set()
                periods = [b["period"] for b in eb]
                # periods sorted + unique (a series, not a bag)
                assert periods == sorted(set(periods)), s["id"]
                for b in eb:
                    assert b["contributing"], (s["id"], "empty receipts")
                    for i in b["contributing"]:
                        assert 0 <= i < n, s["id"]
                    contributing_all |= set(b["contributing"])
                # currency rule: excluded indices never contribute
                for i in q.get("excluded_currency_indices", []):
                    assert i not in contributing_all, (s["id"], "currency leak")
                # distractors never contribute
                for i in q.get("distractor_indices", []):
                    assert i not in contributing_all, (s["id"], "distractor leak")
                # trend rule: the frozen function is the referee
                if q.get("expected_trend") is not None:
                    vals = [b["value"] for b in eb]
                    assert classify_trend(vals) == q["expected_trend"], s["id"]
                # composition golds need ≥2 buckets; degradation cases
                # route to ganita with a scalar instead
                if q["expected_route"] == "composition":
                    assert len(eb) >= 2, s["id"]
                if q["expected_route"] == "ganita" and q.get("expected_value") is not None:
                    assert q["expected_value"] > 0, s["id"]

    def test_scalar_golds_match_contributing_sums(self):
        # For sum-op ganita golds, the scalar must equal the sum of the
        # dollar amounts in its contributing propositions (receipts are
        # the contract on the scalar path too).
        import re
        for s in _load():
            for q in s["questions"]:
                if (q["expected_route"] == "ganita"
                        and q.get("op") == "sum"
                        and q.get("expected_value") is not None
                        and q.get("expected_value_contributing")):
                    total = 0.0
                    for i in q["expected_value_contributing"]:
                        m = re.search(r"\$(\d+(?:\.\d+)?)",
                                      s["propositions"][i]["text"])
                        assert m, (s["id"], i)
                        total += float(m.group(1))
                    assert abs(total - q["expected_value"]) < 0.005, s["id"]


# ─── 3. Rescore path ────────────────────────────────────────────────


class TestRescore:
    def test_rescore_reproduces_scores_from_artifacts(self):
        scenario = {
            "id": "s1",
            "questions": [{
                "q": "how has my spending on x evolved?",
                "expected_route": "composition",
                "expected_buckets": [
                    {"period": "2026-01", "value": 40.0, "contributing": [0]},
                    {"period": "2026-02", "value": 60.0, "contributing": [1]},
                ],
                "expected_trend": "rising",
            }],
        }
        row = {
            "scenario_id": "s1",
            "question": "how has my spending on x evolved?",
            "route": "composition",
            "buckets": [
                {"period": "2026-01", "value": 40.0, "contributing": [0]},
                {"period": "2026-02", "value": 60.0, "contributing": [1]},
            ],
            "trend": "rising",
            "scalar_value": None,
            "scores": {},  # stale/absent — rescore must rebuild
        }
        out = rescore_rows([row], {"s1": scenario})
        assert out[0]["scores"]["routed"] == 1.0
        assert out[0]["scores"]["bucket_periods"] == 1.0
        assert out[0]["scores"]["bucket_values"] == 1.0
        assert out[0]["scores"]["receipts"] == 1.0
        assert out[0]["scores"]["trend"] == 1.0
