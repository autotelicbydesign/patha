"""Tests for AbsenceEval (eval/absence_eval.py + absence_data/).

Three layers, all model-free:
  1. Scorers — pure functions, hand-verified cases (the frozen rubric).
  2. Data integrity — the dev JSONL loads, schema holds, the composition
     quotas hold (8 traps, >=2 per taxonomy kind, 4 routing controls),
     gold labels are internally consistent with the abhāva taxonomy.
  3. Determinism — the stub harness is run-to-run identical and the
     authoring script regenerates the committed file byte-identically;
     the stub floor (the red bar) is pinned.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from eval.absence_eval import (
    RUBRIC_VERSION,
    TAXONOMY,
    aggregate,
    canonicalize_locus,
    rescore_rows,
    run_scenario,
    score_contrast,
    score_false_absence,
    score_kind,
    score_locus,
    score_question,
    score_routed,
    score_verdict,
    stub_answerer,
)

DATA_DIR = Path(__file__).parent.parent / "eval" / "absence_data"

_SCOPE_FOR_KIND = {
    "atyantabhava": "ever",
    "pragabhava": "yet",
    "pradhvamsabhava": "still",
    "anyonyabhava": "identity",
}


# ─── 1. Scorers (frozen rubric v1) ──────────────────────────────────


class TestCanonicalizeLocus:
    def test_lower_strip_deplural(self):
        assert canonicalize_locus(" Venues ") == "venue"
        assert canonicalize_locus("Japan") == "japan"
        assert canonicalize_locus("violin lessons") == "violin lesson"

    def test_non_plural_s_endings_guarded(self):
        # the ganita _canonicalize_entity discipline: chess/tennis/status
        assert canonicalize_locus("chess") == "chess"
        assert canonicalize_locus("tennis") == "tennis"
        assert canonicalize_locus("status") == "status"
        # 'class' ends in -ss → guarded
        assert canonicalize_locus("italian class") == "italian class"


class TestScorers:
    def test_routed_absence_gold(self):
        assert score_routed("absence", "absence") == 1.0
        assert score_routed("retrieval", "absence") == 0.0
        assert score_routed(None, "absence") == 0.0

    def test_routed_controls_pass_on_any_non_absence_route(self):
        assert score_routed("retrieval", "retrieval") == 1.0
        # the gate only judges absence-theft, not the exact destination
        assert score_routed("synthesis", "retrieval") == 1.0
        assert score_routed("absence", "retrieval") == 0.0
        assert score_routed("absence", "synthesis") == 0.0

    def test_verdict(self):
        assert score_verdict("absent", "absent") == 1.0
        assert score_verdict("present", "absent") == 0.0
        assert score_verdict("absent", "present") == 0.0
        assert score_verdict("absent", None) is None  # controls

    def test_false_absence_is_the_catastrophe(self):
        # trap answered absent through the absence route → 1.0
        assert score_false_absence("absence", "absent", "present") == 1.0
        # trap answered present → 0.0
        assert score_false_absence("absence", "present", "present") == 0.0
        # trap routed elsewhere made NO absence claim → 0.0, not None
        assert score_false_absence("retrieval", "absent", "present") == 0.0
        # not a trap → None (verdict owns gold-absent correctness)
        assert score_false_absence("absence", "absent", "absent") is None
        assert score_false_absence("absence", "absent", None) is None

    def test_kind(self):
        assert score_kind("pragabhava", "pragabhava") == 1.0
        assert score_kind("atyantabhava", "pragabhava") == 0.0
        assert score_kind("unknown", "pragabhava") == 0.0  # the stub
        assert score_kind(None, "pragabhava") == 0.0
        assert score_kind("pragabhava", None) is None  # traps carry no kind

    def test_locus(self):
        assert score_locus("japan", "japan") == 1.0
        # canonicalization absorbs case + plural
        assert score_locus("Venues", "venue") == 1.0
        assert score_locus("wedding venue", "venue") == 0.0  # no fuzz
        assert score_locus(None, "venue") == 0.0
        assert score_locus("", "venue") == 0.0
        assert score_locus("venue", None) is None  # controls

    def test_contrast_f1(self):
        assert score_contrast([0, 2], [0, 2]) == 1.0
        # subset: p=1, r=0.5 → F1 = 2/3
        assert score_contrast([0], [0, 2]) == 2 / 3
        # padding: p=2/3, r=1 → F1 = 0.8 (citing everything is not free)
        assert abs(score_contrast([0, 1, 2], [0, 2]) - 0.8) < 1e-12
        assert score_contrast([1, 3], [0, 2]) == 0.0
        assert score_contrast([], [0, 2]) == 0.0
        assert score_contrast([0], None) is None  # controls
        assert score_contrast([0], []) is None

    def test_score_question_control_gates_everything_but_routed(self):
        gold = {
            "expected_route": "retrieval", "expected_kind": None,
            "expected_verdict": None, "expected_locus": None,
            "expected_contrast_ids": None,
        }
        s = score_question({"route": "absence", "verdict": "absent"}, gold)
        assert s["routed"] == 0.0
        assert all(
            s[k] is None
            for k in ("verdict", "false_absence", "kind", "locus", "contrast")
        )

    def test_score_question_not_routed_gates_answer_scorers(self):
        gold = {
            "expected_route": "absence", "expected_kind": "pragabhava",
            "expected_verdict": "absent", "expected_locus": "venue",
            "expected_contrast_ids": [1, 2],
        }
        s = score_question({"route": "retrieval"}, gold)
        assert s["routed"] == 0.0
        assert s["false_absence"] is None  # gold-absent, not a trap
        assert all(s[k] is None for k in ("verdict", "kind", "locus", "contrast"))

    def test_score_question_trap_not_routed_still_scores_false_absence(self):
        gold = {
            "expected_route": "absence", "expected_kind": None,
            "expected_verdict": "present", "expected_locus": "triathlon",
            "expected_contrast_ids": [0],
        }
        s = score_question({"route": "retrieval", "verdict": "absent"}, gold)
        assert s["routed"] == 0.0
        assert s["false_absence"] == 0.0  # no absence claim was made
        assert s["verdict"] is None

    def test_score_question_full_absent_case(self):
        gold = {
            "expected_route": "absence", "expected_kind": "pradhvamsabhava",
            "expected_verdict": "absent", "expected_locus": "veg box",
            "expected_contrast_ids": [0, 1],
        }
        answer = {
            "route": "absence", "verdict": "absent",
            "kind": "pradhvamsabhava", "locus": "veg box",
            "cited_indices": [0, 1],
        }
        s = score_question(answer, gold)
        assert s == {
            "routed": 1.0, "verdict": 1.0, "false_absence": None,
            "kind": 1.0, "locus": 1.0, "contrast": 1.0,
        }

    def test_score_question_trap_answered_absent_is_the_red_case(self):
        gold = {
            "expected_route": "absence", "expected_kind": None,
            "expected_verdict": "present", "expected_locus": "puppy name",
            "expected_contrast_ids": [1],
        }
        answer = {
            "route": "absence", "verdict": "absent", "kind": "pragabhava",
            "locus": "puppy name", "cited_indices": [],
        }
        s = score_question(answer, gold)
        assert s["routed"] == 1.0
        assert s["verdict"] == 0.0
        assert s["false_absence"] == 1.0
        assert s["kind"] is None       # no kind gold on traps
        assert s["locus"] == 1.0       # locus can still be right
        assert s["contrast"] == 0.0    # evidence not cited

    def test_aggregate_excludes_nones(self):
        rows = [
            {"scores": {"routed": 1.0, "verdict": 1.0, "false_absence": None,
                        "kind": 1.0, "locus": 1.0, "contrast": 0.5}},
            {"scores": {"routed": 1.0, "verdict": 0.0, "false_absence": 1.0,
                        "kind": None, "locus": 0.0, "contrast": None}},
        ]
        agg = aggregate(rows)
        assert agg["verdict"]["mean"] == 0.5 and agg["verdict"]["n"] == 2
        assert agg["false_absence"]["mean"] == 1.0 and agg["false_absence"]["n"] == 1
        assert agg["kind"]["mean"] == 1.0 and agg["kind"]["n"] == 1
        assert agg["contrast"]["mean"] == 0.5 and agg["contrast"]["n"] == 1

    def test_rescore_reproduces_scores_from_artifacts(self):
        scenario = {
            "id": "s1",
            "questions": [{
                "q": "have I ever been to Japan?",
                "gold": {
                    "expected_route": "absence",
                    "expected_kind": "atyantabhava",
                    "expected_verdict": "absent",
                    "expected_locus": "japan",
                    "expected_contrast_ids": [0, 2],
                },
            }],
        }
        row = {
            "scenario_id": "s1",
            "question": "have I ever been to Japan?",
            "answer": {"route": "absence", "verdict": "absent",
                       "kind": "atyantabhava", "locus": "Japan",
                       "cited_indices": [0]},
            "scores": {},  # rescore ignores stale scores
        }
        out = rescore_rows([row], {"s1": scenario})
        assert out[0]["scores"]["verdict"] == 1.0
        assert out[0]["scores"]["kind"] == 1.0
        assert out[0]["scores"]["locus"] == 1.0
        assert out[0]["scores"]["contrast"] == 2 / 3

    def test_rubric_version_pinned(self):
        # Rubric changes require a version bump — this test forces the
        # conversation if anyone edits scorers without bumping.
        assert RUBRIC_VERSION == "v1"


# ─── 2. Data integrity ──────────────────────────────────────────────


def _load(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines() if line.strip()
    ]


class TestDataIntegrity:
    def test_dev_set_shape_and_split_marking(self):
        scenarios = _load(DATA_DIR / "dev_scenarios.jsonl")
        assert len(scenarios) == 24
        assert all(s["split"] == "dev" for s in scenarios)
        assert not any(s.get("heldout") for s in scenarios)
        ids = [s["id"] for s in scenarios]
        assert len(ids) == len(set(ids))

    def test_composition_quotas(self):
        scenarios = _load(DATA_DIR / "dev_scenarios.jsonl")
        fams = [s["family"] for s in scenarios]
        counts = {f: fams.count(f) for f in set(fams)}
        assert counts == {
            "absent_atyantabhava": 3, "absent_pragabhava": 3,
            "absent_pradhvamsabhava": 3, "absent_anyonyabhava": 3,
            "trap_present": 8, "routing_control": 4,
        }
        # >=2 absent cases per taxonomy kind (the prompt's floor; we ship 3)
        kind_counts: dict[str, int] = {}
        for s in scenarios:
            for q in s["questions"]:
                k = q["gold"]["expected_kind"]
                if k is not None:
                    kind_counts[k] = kind_counts.get(k, 0) + 1
        assert set(kind_counts) == set(TAXONOMY)
        assert all(v >= 2 for v in kind_counts.values())
        # traps cover all four temporal scopes, 2 each
        trap_scopes = [
            q["scope"]
            for s in scenarios if s["family"] == "trap_present"
            for q in s["questions"]
        ]
        assert {sc: trap_scopes.count(sc) for sc in set(trap_scopes)} == {
            "ever": 2, "yet": 2, "still": 2, "identity": 2,
        }
        # controls: 2 retrieval + 2 synthesis
        ctrl_routes = [
            q["gold"]["expected_route"]
            for s in scenarios if s["family"] == "routing_control"
            for q in s["questions"]
        ]
        assert sorted(ctrl_routes) == [
            "retrieval", "retrieval", "synthesis", "synthesis",
        ]

    def test_taxonomy_strings_match_production_abhava_kinds(self):
        # gold kinds must stay in lock-step with the four-fold taxonomy
        # in src/patha/belief/abhava.py (read-only import, no models)
        from patha.belief.abhava import AbhavaKind

        production = {k.value for k in AbhavaKind}
        assert set(TAXONOMY) <= production
        for s in _load(DATA_DIR / "dev_scenarios.jsonl"):
            for q in s["questions"]:
                k = q["gold"]["expected_kind"]
                assert k is None or k in TAXONOMY, s["id"]

    def test_scenario_internal_consistency(self):
        from datetime import datetime

        for s in _load(DATA_DIR / "dev_scenarios.jsonl"):
            n = len(s["propositions"])
            assert n >= 3, s["id"]
            dates = [
                datetime.fromisoformat(p["asserted_at"])
                for p in s["propositions"]
            ]
            assert dates == sorted(dates), s["id"]
            for prop in s["propositions"]:
                assert prop["text"].strip(), s["id"]
                assert prop["session"], s["id"]
            for q in s["questions"]:
                g = q["gold"]
                if g["expected_route"] != "absence":
                    # controls carry no absence gold at all
                    assert q["type"] == "control", s["id"]
                    assert g["expected_verdict"] is None, s["id"]
                    assert g["expected_kind"] is None, s["id"]
                    assert g["expected_locus"] is None, s["id"]
                    assert g["expected_contrast_ids"] is None, s["id"]
                    continue
                assert q["type"] == "absence", s["id"]
                assert g["expected_verdict"] in ("absent", "present"), s["id"]
                if g["expected_verdict"] == "absent":
                    assert g["expected_kind"] in TAXONOMY, s["id"]
                    # scope and kind must agree (ever/yet/still/identity)
                    assert q["scope"] == _SCOPE_FOR_KIND[g["expected_kind"]], s["id"]
                else:
                    # traps: no kind gold, but evidence must exist
                    assert g["expected_kind"] is None, s["id"]
                # locus is a canonical fixed point (scorer-comparable)
                locus = g["expected_locus"]
                assert locus and canonicalize_locus(locus) == locus, s["id"]
                # contrast ids: non-empty, valid, unique
                contrast = g["expected_contrast_ids"]
                assert contrast, s["id"]
                assert all(0 <= i < n for i in contrast), s["id"]
                assert len(set(contrast)) == len(contrast), s["id"]

    def test_trap_families_align_with_present_verdicts(self):
        for s in _load(DATA_DIR / "dev_scenarios.jsonl"):
            for q in s["questions"]:
                verdict = q["gold"]["expected_verdict"]
                if s["family"] == "trap_present":
                    assert verdict == "present", s["id"]
                elif s["family"].startswith("absent_"):
                    assert verdict == "absent", s["id"]
                    # family name encodes the gold kind
                    assert s["family"] == f"absent_{q['gold']['expected_kind']}", s["id"]


# ─── 3. Determinism + the pinned floor ──────────────────────────────


class TestDeterminismAndFloor:
    def _run_all(self) -> list[dict]:
        rows: list[dict] = []
        for s in _load(DATA_DIR / "dev_scenarios.jsonl"):
            rows.extend(run_scenario(s, answerer=stub_answerer, detector="stub"))
        return rows

    def test_stub_harness_is_deterministic(self):
        def strip_time(rows: list[dict]) -> list[dict]:
            return [{k: v for k, v in r.items() if k != "seconds"} for r in rows]

        assert strip_time(self._run_all()) == strip_time(self._run_all())

    def test_stub_floor_is_pinned(self):
        # THE red bar the implementation phase must turn green. If this
        # test starts failing because the numbers improved, the floor
        # documentation in absence_data/README.md moves with it.
        agg = aggregate(self._run_all())
        assert agg["routed"]["mean"] == 20 / 24 and agg["routed"]["n"] == 24
        assert agg["false_absence"]["mean"] == 1.0 and agg["false_absence"]["n"] == 8
        # always-absent gets gold-absent verdicts right for free —
        # exactly why verdict is not the headline metric
        assert agg["verdict"]["mean"] == 0.6 and agg["verdict"]["n"] == 20
        assert agg["kind"]["mean"] == 0.0 and agg["kind"]["n"] == 12
        assert agg["locus"]["mean"] == 0.0 and agg["locus"]["n"] == 20
        assert agg["contrast"]["mean"] == 0.0 and agg["contrast"]["n"] == 20

    def test_rescore_roundtrip_on_real_rows(self):
        rows = self._run_all()
        by_id = {s["id"]: s for s in _load(DATA_DIR / "dev_scenarios.jsonl")}
        rescored = rescore_rows(rows, by_id)
        assert [r["scores"] for r in rescored] == [r["scores"] for r in rows]

    def test_authoring_is_byte_identical(self, tmp_path: Path):
        out = tmp_path / "regen.jsonl"
        subprocess.run(
            [sys.executable, "-m", "eval.absence_data.author_dev",
             "--output", str(out)],
            check=True, capture_output=True,
            cwd=str(DATA_DIR.parent.parent),
        )
        committed = (DATA_DIR / "dev_scenarios.jsonl").read_bytes()
        assert out.read_bytes() == committed, (
            "dev_scenarios.jsonl is not author_dev.py's output — "
            "regenerate or fix the authoring script"
        )
