"""Tests for RouterEval (eval/router_eval.py + router_data/).

Three layers, all model-free:
  1. Scorers — pure functions, hand-verified cases (the frozen rubric).
  2. Data integrity — the dev JSONL loads, schema fields present, gold
     labels valid, class/family counts pinned, boundary discipline held.
  3. Adapter + determinism — the intent router is deterministic and its
     documented current-behaviour mappings hold (these pins describe
     the pre-composition/absence/analogy router; update them, with the
     confusion-matrix re-report, when those gates ship).
"""

from __future__ import annotations

import json
from pathlib import Path

from eval.router_eval import (
    INTENT_ROUTER_COVERAGE,
    ROUTES,
    RUBRIC_VERSION,
    aggregate,
    boundary_table,
    boundary_verdict,
    confusion_matrix,
    intent_router,
    per_class_metrics,
    rescore_rows,
    route_from_strategy,
    run_questions,
    score_question,
)

DATA_PATH = (
    Path(__file__).parent.parent / "eval" / "router_data" / "dev_questions.jsonl"
)


def _load() -> list[dict]:
    return [
        json.loads(line)
        for line in DATA_PATH.read_text().splitlines() if line.strip()
    ]


# ─── 1. Scorers (frozen rubric v1) ──────────────────────────────────


class TestScorers:
    def test_score_question_exact(self):
        assert score_question("narrative", "narrative") == {
            "exact": 1.0, "acceptable": 1.0,
        }
        assert score_question("retrieval", "narrative") == {
            "exact": 0.0, "acceptable": 0.0,
        }

    def test_score_question_secondary(self):
        # secondary hit: exact misses, acceptable credits
        s = score_question("synthesis", "narrative", "synthesis")
        assert s == {"exact": 0.0, "acceptable": 1.0}
        # gold hit with a secondary present: both credit
        s = score_question("narrative", "narrative", "synthesis")
        assert s == {"exact": 1.0, "acceptable": 1.0}
        # off-route with a secondary present: neither credits
        s = score_question("retrieval", "narrative", "synthesis")
        assert s == {"exact": 0.0, "acceptable": 0.0}
        # secondary=None behaves exactly like no secondary
        assert score_question("synthesis", "narrative", None) == {
            "exact": 0.0, "acceptable": 0.0,
        }

    def test_confusion_matrix_counts_and_zero_fill(self):
        rows = [
            {"gold_route": "absence", "predicted_route": "retrieval"},
            {"gold_route": "absence", "predicted_route": "retrieval"},
            {"gold_route": "narrative", "predicted_route": "narrative"},
            {"gold_route": "composition", "predicted_route": "narrative"},
        ]
        m = confusion_matrix(rows)
        assert m["absence"]["retrieval"] == 2
        assert m["narrative"]["narrative"] == 1
        assert m["composition"]["narrative"] == 1
        # zero-filled everywhere else, all six classes present both ways
        assert set(m) == set(ROUTES)
        assert all(set(m[g]) == set(ROUTES) for g in ROUTES)
        assert m["analogy"]["analogy"] == 0

    def test_per_class_metrics_hand_verified(self):
        rows = [
            # narrative: 2 gold, 1 hit; 1 stolen composition question
            {"gold_route": "narrative", "predicted_route": "narrative"},
            {"gold_route": "narrative", "predicted_route": "retrieval"},
            {"gold_route": "composition", "predicted_route": "narrative"},
            # retrieval: 1 gold, 1 hit
            {"gold_route": "retrieval", "predicted_route": "retrieval"},
        ]
        pc = per_class_metrics(confusion_matrix(rows))
        assert pc["narrative"]["support"] == 2
        assert pc["narrative"]["predicted"] == 2
        assert pc["narrative"]["recall"] == 0.5
        assert pc["narrative"]["precision"] == 0.5
        assert pc["retrieval"]["recall"] == 1.0
        assert pc["retrieval"]["precision"] == 0.5  # narrative miss landed here
        # never predicted → precision undefined, not zero
        assert pc["absence"]["precision"] is None
        # no gold questions → recall undefined, not zero
        assert pc["absence"]["support"] == 0
        assert pc["absence"]["recall"] is None
        # composition: gold exists but never predicted
        assert pc["composition"]["recall"] == 0.0
        assert pc["composition"]["precision"] is None

    def test_boundary_verdict(self):
        assert boundary_verdict("absence", "absence", "synthesis") == "gold"
        assert boundary_verdict("synthesis", "absence", "synthesis") == "secondary"
        assert boundary_verdict("retrieval", "absence", "synthesis") == "off"
        assert boundary_verdict("retrieval", "absence", None) == "off"

    def test_boundary_table_projects_only_boundary_rows(self):
        rows = [
            {"id": "a", "question": "q1", "gold_route": "absence",
             "acceptable_secondary": "synthesis", "boundary": True,
             "predicted_route": "synthesis"},
            {"id": "b", "question": "q2", "gold_route": "retrieval",
             "acceptable_secondary": None, "boundary": False,
             "predicted_route": "retrieval"},
        ]
        table = boundary_table(rows)
        assert len(table) == 1
        assert table[0]["id"] == "a"
        assert table[0]["verdict"] == "secondary"

    def test_aggregate_means_and_counts(self):
        rows = [
            {"scores": {"exact": 1.0, "acceptable": 1.0}},
            {"scores": {"exact": 0.0, "acceptable": 1.0}},
            {"scores": {"exact": 0.0, "acceptable": 0.0}},
        ]
        agg = aggregate(rows)
        assert agg["exact"]["mean"] == 1 / 3 and agg["exact"]["n"] == 3
        assert agg["acceptable"]["mean"] == 2 / 3 and agg["acceptable"]["n"] == 3
        empty = aggregate([])
        assert empty["exact"]["mean"] is None and empty["exact"]["n"] == 0

    def test_route_from_strategy_mappings(self):
        # what recall() emits today
        assert route_from_strategy("ganita") == "synthesis"
        assert route_from_strategy("narrative") == "narrative"
        assert route_from_strategy("direct_answer") == "retrieval"
        assert route_from_strategy("structured") == "retrieval"
        assert route_from_strategy("raw") == "retrieval"
        # forward mappings for roadmap items 3-5
        assert route_from_strategy("composition") == "composition"
        assert route_from_strategy("abhava") == "absence"
        assert route_from_strategy("upamana") == "analogy"
        # unknown degrades like the system does: to retrieval
        assert route_from_strategy("something-new") == "retrieval"

    def test_run_questions_harness_with_stub_router(self):
        # The harness is testable with any callable — no patha needed.
        questions = [
            {"id": "x1", "family": "f", "question": "alpha?",
             "gold_route": "retrieval", "acceptable_secondary": None,
             "boundary": False},
            {"id": "x2", "family": "f", "question": "beta?",
             "gold_route": "absence", "acceptable_secondary": "retrieval",
             "boundary": True},
        ]
        rows = run_questions(questions, lambda q: "retrieval")
        assert [r["predicted_route"] for r in rows] == ["retrieval", "retrieval"]
        assert rows[0]["scores"] == {"exact": 1.0, "acceptable": 1.0}
        assert rows[1]["scores"] == {"exact": 0.0, "acceptable": 1.0}
        # artifacts persisted for re-scoring
        assert rows[1]["gold_route"] == "absence"
        assert rows[1]["acceptable_secondary"] == "retrieval"
        assert rows[1]["boundary"] is True

    def test_rescore_reapplies_current_golds_to_artifacts(self):
        # A persisted row re-scores against a corrected gold label
        # without re-running the router (the artifact is the prediction).
        row = {
            "id": "x1", "family": "f", "question": "alpha?",
            "gold_route": "retrieval", "acceptable_secondary": None,
            "boundary": False, "predicted_route": "absence",
            "scores": {"exact": 0.0, "acceptable": 0.0},
        }
        corrected = {
            "id": "x1", "family": "f", "question": "alpha?",
            "gold_route": "absence", "acceptable_secondary": "retrieval",
            "boundary": True,
        }
        out = rescore_rows([row], {"x1": corrected})
        assert out[0]["predicted_route"] == "absence"  # artifact untouched
        assert out[0]["gold_route"] == "absence"
        assert out[0]["scores"] == {"exact": 1.0, "acceptable": 1.0}

    def test_rubric_version_pinned(self):
        # Rubric changes require a version bump — this test forces the
        # conversation if anyone edits scorers without bumping.
        assert RUBRIC_VERSION == "v1"


# ─── 2. Data integrity ──────────────────────────────────────────────


class TestDataIntegrity:
    def test_dev_set_shape(self):
        qs = _load()
        assert len(qs) == 90
        # dev-only: nothing sealed in this file
        assert not any(q.get("heldout") for q in qs)
        ids = [q["id"] for q in qs]
        assert len(ids) == len(set(ids))

    def test_schema_fields(self):
        for q in _load():
            assert q["question"].strip(), q["id"]
            assert q["notes"].strip(), q["id"]  # every gold is argued
            assert q["source"].strip(), q["id"]
            assert isinstance(q["boundary"], bool), q["id"]
            assert q["gold_route"] in ROUTES, q["id"]
            sec = q["acceptable_secondary"]
            if sec is not None:
                assert sec in ROUTES, q["id"]
                assert sec != q["gold_route"], q["id"]

    def test_family_counts_pinned(self):
        fams = [q["family"] for q in _load()]
        # rt-retr-13 re-familied to boundary 2026-07-08: "am/do I still
        # X" is AbsenceEval's still-family shape — gold corrected to
        # absence (secondary retrieval) for inter-instrument consistency.
        assert {f: fams.count(f) for f in set(fams)} == {
            "retrieval_plain": 13,
            "synthesis_plain": 14,
            "narrative_plain": 14,
            "composition_plain": 12,
            "absence_plain": 12,
            "analogy_plain": 9,
            "boundary": 16,
        }

    def test_gold_route_distribution_pinned(self):
        golds = [q["gold_route"] for q in _load()]
        assert {r: golds.count(r) for r in ROUTES} == {
            "retrieval": 15,
            "synthesis": 17,
            "narrative": 17,
            "composition": 13,
            "absence": 17,
            "analogy": 11,
        }
        # all six classes represented — the matrix has no empty gold row
        assert all(golds.count(r) > 0 for r in ROUTES)

    def test_boundary_discipline(self):
        qs = _load()
        boundary = [q for q in qs if q["boundary"]]
        plain = [q for q in qs if not q["boundary"]]
        # the boundary flag and the family agree
        assert all(q["family"] == "boundary" for q in boundary)
        assert all(q["family"] != "boundary" for q in plain)
        # only boundary questions may carry a secondary; not all must
        # (genuinely-unambiguous boundary cases keep secondary null)
        assert all(q["acceptable_secondary"] is None for q in plain)
        with_secondary = [
            q for q in boundary if q["acceptable_secondary"] is not None
        ]
        assert len(with_secondary) >= 10
        assert len(with_secondary) < len(boundary)

    def test_plain_family_gold_matches_family_route(self):
        route_of_family = {
            "retrieval_plain": "retrieval",
            "synthesis_plain": "synthesis",
            "narrative_plain": "narrative",
            "composition_plain": "composition",
            "absence_plain": "absence",
            "analogy_plain": "analogy",
        }
        for q in _load():
            if q["family"] in route_of_family:
                assert q["gold_route"] == route_of_family[q["family"]], q["id"]

    def test_questions_are_unique(self):
        texts = [q["question"] for q in _load()]
        assert len(texts) == len(set(texts))


# ─── 3. Adapter + determinism ───────────────────────────────────────


class TestIntentRouterAdapter:
    def test_deterministic_over_the_full_dev_set(self):
        qs = _load()
        first = [intent_router(q["question"]) for q in qs]
        second = [intent_router(q["question"]) for q in qs]
        assert first == second
        assert all(r in ROUTES for r in first)

    def test_coverage_declaration_matches_emissions(self):
        emitted = {intent_router(q["question"]) for q in _load()}
        assert emitted <= set(INTENT_ROUTER_COVERAGE)

    def test_gate_order_hand_verified(self):
        # The four-class routing recall() performs today (absence gate
        # shipped 2026-07-08, between synthesis and narrative).
        assert intent_router(
            "how much did I spend on the bike in total?") == "synthesis"
        assert intent_router(
            "how has my thinking about bouldering evolved?") == "narrative"
        assert intent_router(
            "what did I say about the saddle?") == "retrieval"
        assert intent_router("have I ever lived abroad?") == "absence"
        # gaṇita gate is consulted BEFORE narrative (recall()'s order):
        # a count marker wins even when a change marker is present.
        assert intent_router(
            "how many times have I changed my mind about remote work?"
        ) == "synthesis"
        # …and the absence gate's aggregation guard keeps threshold
        # phrasings OFF the absence route ("have I ever spent more than
        # 100…" is arithmetic territory). gaṇita's own detector does
        # not claim thresholds today, so this lands in retrieval — the
        # pin is that it must never land in absence.
        assert intent_router(
            "have I ever spent more than 100 in one go on food?"
        ) in ("synthesis", "retrieval")

    def test_documented_current_confusions(self):
        # These pins DESCRIBE the pre-composition/analogy router — the
        # baseline the remaining roadmap gates must move. Update them
        # (with a re-reported confusion matrix) when the corresponding
        # gate ships. (The absence line moved 2026-07-08 when the
        # anupalabdhi gate shipped — see test_gate_order_hand_verified.)
        #
        # Composition-gold splits into narrative (evolution marker)…
        assert intent_router(
            "how has my spending on the bike evolved?") == "narrative"
        # …or retrieval (no marker at all).
        assert intent_router(
            "show me my coffee spending over time") == "retrieval"
        # Analogy-gold: the 'most similar' superlative trips the
        # gaṇita 'most' (max) marker.
        assert intent_router(
            "what past project was most similar to this one?") == "synthesis"
