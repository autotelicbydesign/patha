"""Tests for AnalogyEval (eval/analogy_eval.py + analogy_data/).

Three layers, all model-free:
  1. Scorers — pure functions, hand-verified cases (frozen rubric v1),
     plus the content_tokens normalizer both scorer and data share.
  2. Data integrity — the authored invariants re-checked against the
     committed JSONL (not just at authoring time): content-word
     DISJOINTNESS between question and gold sessions, trap dominance,
     filler blandness, session references, negative-control hygiene.
  3. Determinism + rescore — the stub floor is seeded (byte-identical
     runs); persisted artifacts re-score identically.
"""

from __future__ import annotations

import json
from pathlib import Path

from eval.analogy_eval import (
    RUBRIC_VERSION,
    RandomSessionAnswerer,
    aggregate,
    candidate_sessions,
    content_tokens,
    rescore_rows,
    run_scenario,
    score_analogue_hit,
    score_question,
    score_routed,
    score_structure_overlap,
    score_trap_resistance,
)

DATA = Path(__file__).parent.parent / "eval" / "analogy_data"


def _load() -> list[dict]:
    path = DATA / "dev_scenarios.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# ─── 1. Scorers (frozen rubric v1) ──────────────────────────────────


class TestContentTokens:
    def test_stopwords_frame_vocabulary_and_plurals(self):
        toks = content_tokens(
            "does this situation remind me of anything like the runs?"
        )
        # frame words gone, 'runs' plural-stripped
        assert toks == {"run"}

    def test_short_tokens_dropped(self):
        assert "km" not in content_tokens("an 18 km run")


class TestScorers:
    def test_routed_claim_vs_gold(self):
        assert score_routed(True, "analogy") == 1.0
        assert score_routed(False, "analogy") == 0.0
        # negative controls: claiming analogy on a ganita question fails
        assert score_routed(True, "ganita") == 0.0
        assert score_routed(False, "narrative") == 1.0

    def test_analogue_hit_at_k(self):
        gold = ["s-a", "s-b"]
        assert score_analogue_hit(["s-a"], gold, 1) == 1.0
        assert score_analogue_hit(["s-x", "s-b"], gold, 1) == 0.0
        assert score_analogue_hit(["s-x", "s-b"], gold, 2) == 1.0
        assert score_analogue_hit([], gold, 1) == 0.0   # claimed route owns it
        assert score_analogue_hit(["s-a"], [], 1) is None  # no gold → None

    def test_trap_resistance_only_on_trap_cases(self):
        assert score_trap_resistance(["s-gold"], ["s-gold"], None) is None
        assert score_trap_resistance(["s-gold"], ["s-gold"], "s-trap") == 1.0
        assert score_trap_resistance(["s-trap", "s-gold"], ["s-gold"], "s-trap") == 0.0

    def test_structure_overlap_half_token_rule(self):
        # "external deadline" → tokens {external, deadline}; naming one
        # of two (0.5) counts, per the ≥half rule
        assert score_structure_overlap(
            "there was a deadline involved", ["external deadline"],
        ) == 1.0
        assert score_structure_overlap(
            "something about advice", ["external deadline"],
        ) == 0.0
        assert score_structure_overlap("anything", []) is None

    def test_score_question_gates_on_routing_claim(self):
        q = {
            "expected_route": "analogy",
            "gold_analogue_sessions": ["s-a"],
            "shared_structure": ["sunk cost"],
        }
        s = score_question(False, [], "", q)
        assert s["routed"] == 0.0
        assert all(
            s[k] is None
            for k in ("analogue_hit_1", "analogue_hit_2", "trap_resistance",
                      "structure_overlap")
        )
        s = score_question(True, ["s-a"], "the sunk cost pattern", q)
        assert s["routed"] == 1.0 and s["analogue_hit_1"] == 1.0
        assert s["structure_overlap"] == 1.0

    def test_aggregate_excludes_nones(self):
        rows = [
            {"scores": {"routed": 1.0, "analogue_hit_1": 1.0,
                        "analogue_hit_2": 1.0, "trap_resistance": None,
                        "structure_overlap": 0.5}},
            {"scores": {"routed": 0.0, "analogue_hit_1": None,
                        "analogue_hit_2": None, "trap_resistance": None,
                        "structure_overlap": None}},
        ]
        agg = aggregate(rows)
        assert agg["routed"]["mean"] == 0.5 and agg["routed"]["n"] == 2
        assert agg["analogue_hit_1"]["n"] == 1
        assert agg["trap_resistance"]["n"] == 0

    def test_rubric_version_pinned(self):
        assert RUBRIC_VERSION == "v1"


# ─── 2. Data integrity (authored invariants, re-checked) ────────────


class TestDataIntegrity:
    def test_shape_and_families(self):
        scenarios = _load()
        assert len(scenarios) == 16
        fams = [s["family"] for s in scenarios]
        assert fams.count("surface_trap") == 4
        assert fams.count("routing_negative") == 4
        assert fams.count("core_analogy") == 6
        assert fams.count("multi_candidate") == 2

    def test_session_references_valid(self):
        for s in _load():
            sessions = {p["session"] for p in s["propositions"]}
            for q in s["questions"]:
                for g in q["gold_analogue_sessions"]:
                    assert g in sessions, (s["id"], g)
                trap = q.get("surface_trap_session")
                if trap:
                    assert trap in sessions, (s["id"], trap)
                    assert trap not in q["gold_analogue_sessions"], s["id"]

    def test_disjointness_invariant(self):
        # THE defining property: an analogy question shares ≤ 2 content
        # tokens with each gold session's full text.
        for s in _load():
            text_by_session: dict[str, list[str]] = {}
            for p in s["propositions"]:
                text_by_session.setdefault(p["session"], []).append(p["text"])
            for q in s["questions"]:
                qt = content_tokens(q["q"])
                for g in q["gold_analogue_sessions"]:
                    overlap = qt & content_tokens(" ".join(text_by_session[g]))
                    assert len(overlap) <= 2, (s["id"], g, sorted(overlap))

    def test_trap_dominance(self):
        # the lexically-seductive session must out-overlap the gold
        for s in _load():
            text_by_session: dict[str, list[str]] = {}
            for p in s["propositions"]:
                text_by_session.setdefault(p["session"], []).append(p["text"])
            for q in s["questions"]:
                trap = q.get("surface_trap_session")
                if not trap:
                    continue
                qt = content_tokens(q["q"])
                t_overlap = qt & content_tokens(" ".join(text_by_session[trap]))
                g_overlap = qt & content_tokens(
                    " ".join(text_by_session[q["gold_analogue_sessions"][0]])
                )
                assert len(t_overlap) >= 3, (s["id"], sorted(t_overlap))
                assert len(t_overlap) > len(g_overlap), s["id"]

    def test_negative_controls_carry_no_gold(self):
        for s in _load():
            if s["family"] != "routing_negative":
                continue
            for q in s["questions"]:
                assert q["expected_route"] != "analogy", s["id"]
                assert q["gold_analogue_sessions"] == [], s["id"]
                assert q["shared_structure"] == [], s["id"]

    def test_analogy_scenarios_have_filler_sessions(self):
        # the chance-floor control: ≥ 4 candidate sessions per analogy
        # scenario (gold + foil + 2 fillers)
        for s in _load():
            if s["family"] == "routing_negative":
                continue
            assert len(candidate_sessions(s)) >= 4, s["id"]


# ─── 3. Determinism + rescore ───────────────────────────────────────


class TestDeterminismAndRescore:
    def test_stub_floor_is_byte_identical(self):
        scenario = _load()[0]
        a = run_scenario(scenario, RandomSessionAnswerer())
        b = run_scenario(scenario, RandomSessionAnswerer())
        strip = lambda rows: [
            {k: v for k, v in r.items() if k != "seconds"} for r in rows
        ]
        assert strip(a) == strip(b)

    def test_rescore_reproduces_scores_from_artifacts(self):
        scenario = _load()[0]
        rows = run_scenario(scenario, RandomSessionAnswerer())
        wiped = [{**r, "scores": {}} for r in rows]
        rescored = rescore_rows(wiped, {scenario["id"]: scenario})
        assert [r["scores"] for r in rescored] == [r["scores"] for r in rows]
