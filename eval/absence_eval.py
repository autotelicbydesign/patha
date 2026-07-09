"""AbsenceEval — scorers + harness + CLI runner (anupalabdhi).

Measures whether Patha can answer absence questions ("have I ever…",
"have I decided yet…", "do I still…", "am I a…") with the epistemics
the roadmap demands: assert absence only after qualified search, name
the absence's kind (four-fold Nyāya taxonomy — see
src/patha/belief/abhava.py), and cite the nearest *present* beliefs as
contrast. Scenario format + the frozen scoring rubric are documented in
eval/absence_data/README.md. Rubric changes require a version bump and
a re-report — never silent edits.

Rubric history:
- v1 (frozen 2026-07-08): routed / verdict / false_absence / kind /
  locus / contrast. Frozen before the first reported run; the recall
  path does not exist yet, so the first run is the stub floor
  (false_absence_rate 1.000 — the red bar the implementation must
  turn green).

Design notes:
- The absence recall path is NOT built yet. The harness scores a
  provided *answerer* callable `answerer(scenario, question, *,
  detector) -> dict` and defaults to `stub_answerer` (always routes
  absence, always claims absent, cites nothing — the floor). A future
  real answerer wraps Memory: ingest the scenario's propositions,
  recall, and map cited belief ids back to proposition indices (the
  EvolutionEval belief_id→index pattern) before returning.
- Scoring is by exact proposition index; loci compare under
  `canonicalize_locus` (mirrors ganita's `_canonicalize_entity`
  lower/strip/de-plural discipline, without the alias table). No
  fuzzy text matching anywhere.
- Scorers are pure functions over (answer artifacts, gold) —
  unit-testable without any model.
- `false_absence` is the headline catastrophe metric: over gold-PRESENT
  trap questions, 1.0 iff the system routed absence AND claimed
  "absent" (it told the user "you never…" when a positive belief
  exists). It is deliberately NOT gated on routing — a trap that
  routes elsewhere made no absence claim and scores 0.0. Lower is
  better; every other scorer is higher-better.
- Artifacts (the full normalized answer per question) are persisted in
  the results JSON so runs can be re-scored under future rubric
  versions without re-running the system (eval/rescore.py pattern).

Usage:
    uv run python -m eval.absence_eval \\
        --data eval/absence_data/dev_scenarios.jsonl \\
        --output runs/absence/dev-stub-floor.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

RUBRIC_VERSION = "v1"

ROUTES = ("absence", "retrieval", "synthesis", "narrative")
TAXONOMY = ("pragabhava", "pradhvamsabhava", "anyonyabhava", "atyantabhava")
VERDICTS = ("absent", "present")

# guarded non-plural endings — same discipline as ganita's
# _canonicalize_entity (chess/tennis/status class)
_NON_PLURAL_S_ENDINGS = ("ss", "is", "us")


def canonicalize_locus(text: str) -> str:
    """Lowercase, strip punctuation tails, naive de-pluralization.

    Mirrors `patha.belief.ganita._canonicalize_entity` minus the alias
    table, so locus comparison is stable without importing production
    code into the scorers. Gold loci are authored as fixed points of
    this function (asserted by the data integrity tests)."""
    t = text.lower().strip().rstrip(".,;")
    if len(t) > 3 and t.endswith("s") and not t.endswith(_NON_PLURAL_S_ENDINGS):
        return t[:-1]
    return t


# ─── Scorers (frozen rubric — see absence_data/README.md) ───────────


def score_routed(predicted_route: str | None, expected_route: str) -> float:
    """The gate. For absence-gold questions: did the system route
    absence? For routing-negative controls: did it correctly NOT route
    absence? (Whether a control lands on retrieval vs synthesis is a
    different instrument's business — this eval judges the absence gate
    only, so any non-absence route passes a control.)"""
    if expected_route == "absence":
        return 1.0 if predicted_route == "absence" else 0.0
    return 1.0 if predicted_route != "absence" else 0.0


def score_verdict(
    predicted_verdict: str | None, expected_verdict: str | None
) -> float | None:
    """Exact verdict match ("absent"/"present"). None when gold has no
    verdict (routing controls)."""
    if expected_verdict is None:
        return None
    return 1.0 if predicted_verdict == expected_verdict else 0.0


def score_false_absence(
    predicted_route: str | None,
    predicted_verdict: str | None,
    expected_verdict: str | None,
) -> float | None:
    """The catastrophic failure, scored separately: on a gold-PRESENT
    trap, did the system claim absence? 1.0 iff it routed absence AND
    answered "absent". A trap routed elsewhere made no absence claim →
    0.0. None on non-trap questions. Aggregate mean = false_absence_rate
    (LOWER is better — the only scorer with that polarity)."""
    if expected_verdict != "present":
        return None
    claimed = predicted_route == "absence" and predicted_verdict == "absent"
    return 1.0 if claimed else 0.0


def score_kind(
    predicted_kind: str | None, expected_kind: str | None
) -> float | None:
    """Exact four-fold taxonomy match. None when gold has no kind
    (trap/present questions and controls — we do not assert what kind a
    "present" answer has)."""
    if expected_kind is None:
        return None
    return 1.0 if predicted_kind == expected_kind else 0.0


def score_locus(
    predicted_locus: str | None, expected_locus: str | None
) -> float | None:
    """Canonicalized exact match of the absence's locus entity. None
    when gold has no locus (controls); 0.0 when the answer names no
    locus."""
    if expected_locus is None:
        return None
    if not predicted_locus:
        return 0.0
    return (
        1.0
        if canonicalize_locus(predicted_locus) == canonicalize_locus(expected_locus)
        else 0.0
    )


def score_contrast(
    cited: list[int], expected: list[int] | None
) -> float | None:
    """F1 of cited proposition indices vs gold contrast set (for absent
    verdicts: the present beliefs constitutive of the absence claim;
    for present verdicts: the minimal evidence set). F1 rather than
    bare recall so citing everything is not a free lunch. None when
    gold defines no contrast set (controls)."""
    if not expected:
        return None
    if not cited:
        return 0.0
    c, g = set(cited), set(expected)
    inter = len(c & g)
    if inter == 0:
        return 0.0
    p = inter / len(c)
    r = inter / len(g)
    return 2 * p * r / (p + r)


def score_question(answer: dict, gold: dict) -> dict:
    """Apply the full frozen rubric to one question's answer artifacts.

    Gating: verdict/kind/locus/contrast are outputs of the absence
    path, so they are None unless the question is absence-gold AND the
    system routed absence. false_absence is never gated (see its
    docstring) but is None on non-trap questions."""
    route = answer.get("route")
    expected_route = gold["expected_route"]
    scores: dict[str, float | None] = {
        "routed": score_routed(route, expected_route),
        "verdict": None,
        "false_absence": None,
        "kind": None,
        "locus": None,
        "contrast": None,
    }
    if expected_route != "absence":
        return scores
    scores["false_absence"] = score_false_absence(
        route, answer.get("verdict"), gold["expected_verdict"]
    )
    if route != "absence":
        return scores
    scores["verdict"] = score_verdict(answer.get("verdict"), gold["expected_verdict"])
    scores["kind"] = score_kind(answer.get("kind"), gold.get("expected_kind"))
    scores["locus"] = score_locus(answer.get("locus"), gold.get("expected_locus"))
    scores["contrast"] = score_contrast(
        answer.get("cited_indices") or [], gold.get("expected_contrast_ids")
    )
    return scores


_SCORER_NAMES = ["routed", "verdict", "false_absence", "kind", "locus", "contrast"]


def aggregate(rows: list[dict]) -> dict:
    """Mean per scorer, None-excluded; count of contributing questions.
    Note: false_absence mean is a RATE — lower is better."""
    out = {}
    for name in _SCORER_NAMES:
        vals = [r["scores"][name] for r in rows if r["scores"][name] is not None]
        out[name] = {
            "mean": (sum(vals) / len(vals)) if vals else None,
            "n": len(vals),
        }
    return out


# ─── Answerers ──────────────────────────────────────────────────────

Answerer = Callable[..., dict]


def stub_answerer(scenario: dict, question: dict, *, detector: str = "stub") -> dict:
    """The floor: always routes absence, always claims absent, names no
    locus, cites nothing. Exists so the eval runs TODAY and pins the
    red bar (false_absence_rate 1.000) that the real recall path must
    turn green. Deterministic by construction."""
    return {
        "route": "absence",
        "verdict": "absent",
        "kind": "unknown",
        "locus": None,
        "cited_indices": [],
    }


_ANSWERERS: dict[str, Answerer] = {"stub": stub_answerer}


def _load_answerer(spec: str) -> Answerer:
    """Resolve --answerer: a registry name ("stub") or "module:attr"
    dotted path to a callable with the answerer signature."""
    if spec in _ANSWERERS:
        return _ANSWERERS[spec]
    if ":" not in spec:
        raise SystemExit(
            f"unknown answerer {spec!r} — use one of {sorted(_ANSWERERS)} "
            f"or a 'module:attr' path"
        )
    mod_name, attr = spec.split(":", 1)
    return getattr(importlib.import_module(mod_name), attr)


# ─── Harness ────────────────────────────────────────────────────────


def run_scenario(scenario: dict, *, answerer: Answerer, detector: str) -> list[dict]:
    """Ask the answerer each of one scenario's questions, normalize its
    answer into artifacts, score. Returns one row per question (with
    artifacts for re-scoring). The answerer owns any ingestion — the
    harness never touches Memory, so the stub path is model-free."""
    rows: list[dict] = []
    for q in scenario["questions"]:
        t0 = time.time()
        ans = answerer(scenario, q, detector=detector)
        dt = time.time() - t0
        answer = {
            "route": ans.get("route"),
            "verdict": ans.get("verdict"),
            "kind": ans.get("kind"),
            "locus": ans.get("locus"),
            "cited_indices": list(ans.get("cited_indices") or []),
        }
        rows.append({
            "scenario_id": scenario["id"],
            "family": scenario["family"],
            "question": q["q"],
            "answer": answer,
            "seconds": round(dt, 2),
            "scores": score_question(answer, q["gold"]),
        })
    return rows


def rescore_rows(rows: list[dict], scenarios_by_id: dict[str, dict]) -> list[dict]:
    """Re-apply the current rubric to persisted artifacts (no re-run)."""
    out = []
    for r in rows:
        scenario = scenarios_by_id[r["scenario_id"]]
        q = next(qq for qq in scenario["questions"] if qq["q"] == r["question"])
        out.append({**r, "scores": score_question(r["answer"], q["gold"])})
    return out


# ─── CLI ────────────────────────────────────────────────────────────


def _load_scenarios(path: Path, include_heldout: bool) -> list[dict]:
    scenarios = []
    sealed = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            if s.get("heldout") and not include_heldout:
                sealed += 1
                continue
            scenarios.append(s)
    if sealed:
        print(
            f"refused {sealed} sealed held-out scenarios "
            f"(pass --include-heldout for a RELEASE REPORT run only — "
            f"never between tuning iterations)",
            file=sys.stderr,
        )
    return scenarios


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="AbsenceEval runner")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--detector", default="stub")
    p.add_argument(
        "--answerer", default="stub",
        help="Registry name ('stub') or 'module:attr' path to an "
             "answerer callable (scenario, question, *, detector) -> dict.",
    )
    p.add_argument(
        "--include-heldout", action="store_true",
        help="Unseal held-out scenarios. Release reports ONLY. "
             "(No held-out split exists yet — dev only for now.)",
    )
    p.add_argument(
        "--rescore", type=Path, default=None,
        help="Re-apply the current rubric to a prior results JSON "
             "(reads artifacts; does not re-run the system).",
    )
    p.add_argument("--max-scenarios", type=int, default=None)
    args = p.parse_args(argv)

    scenarios = _load_scenarios(args.data, args.include_heldout)
    if args.max_scenarios:
        scenarios = scenarios[: args.max_scenarios]

    if args.rescore:
        prior = json.loads(args.rescore.read_text())
        by_id = {s["id"]: s for s in scenarios}
        rows = rescore_rows(prior["rows"], by_id)
    else:
        answerer = _load_answerer(args.answerer)
        rows = []
        t0 = time.time()
        for i, s in enumerate(scenarios, 1):
            rows.extend(run_scenario(s, answerer=answerer, detector=args.detector))
            print(
                f"  [{i}/{len(scenarios)}] {s['id']} ({s['family']})",
                file=sys.stderr,
            )
        print(f"ran {len(scenarios)} scenarios in {time.time()-t0:.0f}s",
              file=sys.stderr)

    overall = aggregate(rows)
    families = sorted({r["family"] for r in rows})
    by_family = {
        fam: aggregate([r for r in rows if r["family"] == fam])
        for fam in families
    }

    def fmt(a: dict) -> str:
        return "  ".join(
            f"{name}={a[name]['mean']:.3f}({a[name]['n']})"
            if a[name]["mean"] is not None else f"{name}=—(0)"
            for name in _SCORER_NAMES
        )

    print(f"\nAbsenceEval rubric {RUBRIC_VERSION} — {len(rows)} questions")
    print("NOTE: false_absence is a RATE (lower is better); "
          "all other scorers are higher-better.")
    print(f"overall: {fmt(overall)}")
    for fam in families:
        print(f"  {fam:24s} {fmt(by_family[fam])}")
    fa = overall["false_absence"]
    if fa["mean"] is not None and fa["mean"] > 0:
        print(
            f"!! false_absence_rate {fa['mean']:.3f} over {fa['n']} trap "
            f"questions — a wrong 'you never decided' is worse than no feature."
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "rubric_version": RUBRIC_VERSION,
            "data": str(args.data),
            "detector": args.detector,
            "answerer": args.answerer,
            "n_questions": len(rows),
            "overall": overall,
            "by_family": by_family,
            "rows": rows,
        }, indent=2))
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
