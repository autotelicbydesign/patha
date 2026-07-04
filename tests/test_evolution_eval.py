"""Tests for EvolutionEval (eval/evolution_eval.py + evolution_data/).

Three layers, all model-free:
  1. Scorers — pure functions, hand-verified cases (the frozen rubric).
  2. Data integrity — both JSONL files load, schema fields present, the
     dev/held-out split holds, gold indices are consistent.
  3. Generator determinism — regeneration is byte-identical.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from eval.evolution_eval import (
    RUBRIC_VERSION,
    aggregate,
    score_coverage,
    score_ordering,
    score_origin,
    score_precision,
    score_question,
    score_supersession,
)

DATA_DIR = Path(__file__).parent.parent / "eval" / "evolution_data"


# ─── 1. Scorers (frozen rubric v1) ──────────────────────────────────


class TestScorers:
    def test_coverage(self):
        assert score_coverage([0, 2, 4], [0, 2, 4]) == 1.0
        assert score_coverage([0, 2], [0, 2, 4]) == 2 / 3
        assert score_coverage([], [0, 2]) == 0.0
        assert score_coverage([1, 3], [0, 2]) == 0.0
        # distractors present don't hurt coverage
        assert score_coverage([0, 1, 2, 3, 4], [0, 2, 4]) == 1.0

    def test_precision(self):
        assert score_precision([0, 2, 4], [0, 2, 4]) == 1.0
        assert score_precision([0, 1, 2], [0, 2]) == 2 / 3
        assert score_precision([1, 3], [0, 2]) == 0.0
        assert score_precision([], [0, 2]) is None  # undefined, not zero

    def test_ordering_perfect_and_reversed(self):
        assert score_ordering([0, 2, 4], [0, 2, 4]) == 1.0
        assert score_ordering([4, 2, 0], [0, 2, 4]) == 0.0
        # one swap among three: 2 of 3 pairs concordant
        assert score_ordering([2, 0, 4], [0, 2, 4]) == 2 / 3

    def test_ordering_ignores_distractors_and_needs_two(self):
        # distractor 1 interleaved — ordering judged on gold beats only
        assert score_ordering([0, 1, 2, 4], [0, 2, 4]) == 1.0
        assert score_ordering([0], [0, 2, 4]) is None
        assert score_ordering([1, 3], [0, 2, 4]) is None  # no gold present

    def test_origin(self):
        assert score_origin([0, 2, 4], 0) == 1.0
        assert score_origin([2, 0, 4], 0) == 0.0
        # a distractor first also fails origin (strict rubric)
        assert score_origin([1, 0, 2], 0) == 0.0
        assert score_origin([], 0) is None

    def test_supersession(self):
        statuses = {0: "revised-from", 2: "current", 4: "current"}
        # pair applicable + tagged
        assert score_supersession([0, 2, 4], statuses, [[0, 4]]) == 1.0
        # pair applicable, old NOT tagged
        assert score_supersession(
            [0, 2, 4], {0: "current", 4: "current"}, [[0, 4]],
        ) == 0.0
        # pair not applicable (new end missing) → None
        assert score_supersession([0, 2], statuses, [[0, 4]]) is None
        # no expectations → None
        assert score_supersession([0, 2, 4], statuses, []) is None
        # mixed: one tagged, one not → 0.5
        st = {0: "superseded", 2: "current", 4: "current", 6: "current"}
        assert score_supersession(
            [0, 2, 4, 6], st, [[0, 4], [2, 6]],
        ) == 0.5

    def test_score_question_not_routed_gates_everything(self):
        q = {
            "expected_beat_order": [0, 2],
            "expected_origin": 0,
            "expected_supersessions": [],
        }
        s = score_question(False, [], {}, q)
        assert s["routed"] == 0.0
        assert all(
            s[k] is None
            for k in ("coverage", "precision", "ordering", "origin",
                      "supersession")
        )

    def test_aggregate_excludes_nones(self):
        rows = [
            {"scores": {"routed": 1.0, "coverage": 1.0, "precision": 1.0,
                        "ordering": None, "origin": 1.0,
                        "supersession": None}},
            {"scores": {"routed": 1.0, "coverage": 0.5, "precision": 1.0,
                        "ordering": 1.0, "origin": 0.0,
                        "supersession": 0.0}},
        ]
        agg = aggregate(rows)
        assert agg["coverage"]["mean"] == 0.75 and agg["coverage"]["n"] == 2
        assert agg["ordering"]["mean"] == 1.0 and agg["ordering"]["n"] == 1
        assert agg["supersession"]["n"] == 1

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
    def test_dev_set_shape(self):
        scenarios = _load(DATA_DIR / "dev_scenarios.jsonl")
        assert len(scenarios) == 36
        assert not any(s.get("heldout") for s in scenarios)
        fams = {s["family"] for s in scenarios}
        assert fams == {
            "progressive_revelation", "multi_factor_change",
            "perspective_shift", "reversed_belief_chain",
        }

    def test_heldout_set_shape_and_seal(self):
        scenarios = _load(DATA_DIR / "heldout_scenarios.jsonl")
        assert len(scenarios) == 16
        assert all(s.get("heldout") is True for s in scenarios)
        fams = {s["family"] for s in scenarios}
        assert len(fams) == 4

    def test_split_ratio(self):
        dev = _load(DATA_DIR / "dev_scenarios.jsonl")
        held = _load(DATA_DIR / "heldout_scenarios.jsonl")
        ratio = len(dev) / (len(dev) + len(held))
        assert 0.65 <= ratio <= 0.75  # the committed 70/30 split

    def test_scenario_internal_consistency(self):
        for path in (
            DATA_DIR / "dev_scenarios.jsonl",
            DATA_DIR / "heldout_scenarios.jsonl",
        ):
            for s in _load(path):
                n = len(s["propositions"])
                assert n >= 3, s["id"]
                for prop in s["propositions"]:
                    assert prop["text"].strip()
                    assert "asserted_at" in prop
                for q in s["questions"]:
                    gold = q["expected_beat_order"]
                    dist = q["distractor_indices"]
                    assert gold, s["id"]
                    # indices valid + disjoint + jointly exhaustive
                    assert all(0 <= i < n for i in gold + dist), s["id"]
                    assert not set(gold) & set(dist), s["id"]
                    assert set(gold) | set(dist) == set(range(n)), s["id"]
                    # gold order must be chronological (the arc IS time)
                    dates = [
                        s["propositions"][i]["asserted_at"] for i in gold
                    ]
                    assert dates == sorted(dates), s["id"]
                    assert q["expected_origin"] == gold[0], s["id"]
                    for old, new in q.get("expected_supersessions", []):
                        assert old in gold and new in gold, s["id"]
                        assert gold.index(old) < gold.index(new), s["id"]

    def test_heldout_domains_disjoint_from_dev(self):
        dev_themes = {
            q["expected_theme"]
            for s in _load(DATA_DIR / "dev_scenarios.jsonl")
            for q in s["questions"]
        }
        held_themes = {
            q["expected_theme"]
            for s in _load(DATA_DIR / "heldout_scenarios.jsonl")
            for q in s["questions"]
        }
        assert not dev_themes & held_themes, (
            f"held-out themes must be disjoint: {dev_themes & held_themes}"
        )


# ─── 3. Generator determinism ───────────────────────────────────────


class TestGeneratorDeterminism:
    def test_regeneration_is_byte_identical(self, tmp_path: Path):
        out = tmp_path / "regen.jsonl"
        subprocess.run(
            [sys.executable, "-m", "eval.evolution_data.generate_scenarios",
             "--output", str(out)],
            check=True, capture_output=True,
            cwd=str(DATA_DIR.parent.parent),
        )
        committed = (DATA_DIR / "dev_scenarios.jsonl").read_bytes()
        assert out.read_bytes() == committed, (
            "dev_scenarios.jsonl is not the generator's output — "
            "regenerate or fix the generator"
        )
