"""RouterEval — the cross-pramāṇa routing confusion matrix.

With six question classes (retrieval, synthesis, narrative,
composition, absence, analogy) the router IS the system: a misroute
fails silently even when every downstream primitive is perfect, because
each pramāṇa produces a fluent answer of the wrong KIND. This
instrument measures routing alone, over all six classes, as a
confusion matrix. Question format + the frozen scoring rubric are
documented in eval/router_data/README.md. Rubric changes require a
version bump and a re-report — never silent edits.

Rubric history:
- v1 (frozen 2026-07-08): exact / acceptable per question; confusion
  matrix + per-class precision/recall; boundary-case verdict table.

Ground truth: `Memory.recall()` consults its gates in a fixed order —
gaṇita (synthesis) → narrative → retrieval, with a gaṇita backstop on
the retrieval path (src/patha/__init__.py). Three of the six routes
exist today. Composition/absence/analogy questions carry their
INTENDED route so the instrument leads the implementation (roadmap
items 3-5): their per-class recall is 0.0 by construction until those
gates ship, and the confusion matrix shows where they land meanwhile.

Design notes:
- The router under test is any callable `question -> route label`. The
  default `intent_router` mirrors recall()'s gate ORDER using the
  production detectors on the bare question (detect_aggregation first,
  then detect_narrative + extract_theme). It deliberately measures
  routing INTENT, assuming each pramāṇa's index could serve the
  question; store-dependent fall-throughs (empty tuple index, sparse
  walk) are degradation behaviour owned by the per-pramāṇa
  instruments, not by routing. `recall_router(memory)` adapts a live
  `patha.Memory` — it calls `recall()` and maps the strategy field —
  when store-conditioned routing is what you want to measure.
- Scorers are pure functions over (predicted, gold, secondary) —
  unit-testable without models. Artifacts (the predicted route per
  question) are persisted in the results JSON so runs can be re-scored
  under future rubric versions without re-running the router.
- Deterministic: the default router is regex-only; identical inputs
  give identical predictions run-to-run.

Usage:
    uv run python -m eval.router_eval \\
        --data eval/router_data/dev_questions.jsonl \\
        --output runs/router/dev-baseline.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

RUBRIC_VERSION = "v1"

# The six question classes, in canonical (and printing) order.
ROUTES = (
    "retrieval", "synthesis", "narrative",
    "composition", "absence", "analogy",
)

# recall() strategy field → route label. Strategies recall() emits
# today, plus forward mappings for the pramāṇas the roadmap adds
# (items 3-5) so `recall_router` keeps working as gates ship.
_STRATEGY_TO_ROUTE = {
    "ganita": "synthesis",
    "narrative": "narrative",
    "direct_answer": "retrieval",
    "structured": "retrieval",
    "raw": "retrieval",
    # Not emitted yet — forward mappings only:
    "composition": "composition",
    "abhava": "absence",
    "upamana": "analogy",
}


def route_from_strategy(strategy: str) -> str:
    """Map a `Recall.strategy` value to a route label.

    Unknown strategies map to "retrieval" — recall()'s own final
    fallback — so a future strategy string degrades the same way the
    system does rather than crashing the eval.
    """
    return _STRATEGY_TO_ROUTE.get(strategy, "retrieval")


# Routes the default router can actually emit today. Questions whose
# gold is outside this set are misrouted by construction; the runner
# reports supported-gold accuracy separately so the two failure kinds
# (wrong gate vs missing gate) never blur together.
# History: ("retrieval", "synthesis", "narrative") until 2026-07-08;
# "absence" added when the anupalabdhi gate shipped in recall();
# "composition" added the same day when the time-series gate shipped;
# "analogy" added when the upamana gate shipped — full six-class
# coverage.
INTENT_ROUTER_COVERAGE = (
    "retrieval", "synthesis", "narrative", "absence", "composition",
    "analogy",
)


def intent_router(question: str) -> str:
    """Default router: recall()'s intent-gate order on the bare question.

    Mirrors src/patha/__init__.py recall(): the composition gate
    (detect_composition — more specific than plain aggregation) is
    consulted first, then the synthesis gate (detect_aggregation), then
    the absence gate (detect_absence_question — anupalabdhi), then the
    narrative gate (detect_narrative + a resolvable theme), else
    retrieval. The store-dependent parts of the real gates (gaṇita only
    wins when tuples match; composition needs ≥2 buckets; the walk only
    wins with ≥2 beats) are intentionally NOT modelled — this measures
    the routing decision on phrasing alone. Pure regex; deterministic;
    no models."""
    from patha.belief.anupalabdhi import detect_absence_question
    from patha.belief.composition import detect_composition
    from patha.belief.ganita import detect_aggregation
    from patha.belief.itihasa import detect_narrative, extract_theme
    from patha.belief.upamana import detect_analogy_question

    if detect_composition(question) is not None:
        return "composition"
    if detect_aggregation(question) is not None:
        return "synthesis"
    if detect_absence_question(question) is not None:
        return "absence"
    if detect_analogy_question(question):
        return "analogy"
    if detect_narrative(question) is not None and extract_theme(question):
        return "narrative"
    return "retrieval"


def recall_router(memory):
    """Adapter factory: route via a live `patha.Memory.recall()` call.

    Store-conditioned — the same question can route differently on
    different stores (that is the point of using this adapter). Not
    the CLI default because dev_questions.jsonl carries no store."""
    def route(question: str) -> str:
        return route_from_strategy(memory.recall(question).strategy)
    return route


# ─── Scorers (frozen rubric — see router_data/README.md) ────────────


def score_question(
    predicted: str,
    gold: str,
    acceptable_secondary: str | None = None,
) -> dict:
    """Per-question rubric: exact (predicted == gold) and acceptable
    (predicted ∈ {gold, acceptable_secondary}). For non-boundary
    questions the two are identical by construction."""
    exact = 1.0 if predicted == gold else 0.0
    acceptable = exact
    if acceptable_secondary is not None and predicted == acceptable_secondary:
        acceptable = 1.0
    return {"exact": exact, "acceptable": acceptable}


def confusion_matrix(rows: list[dict]) -> dict[str, dict[str, int]]:
    """gold route → predicted route → count, over all six classes
    (zero-filled so absent pairs are visible, not missing)."""
    m = {g: {p: 0 for p in ROUTES} for g in ROUTES}
    for r in rows:
        m[r["gold_route"]][r["predicted_route"]] += 1
    return m


def per_class_metrics(matrix: dict[str, dict[str, int]]) -> dict:
    """Per class: support, times predicted, precision, recall.
    Precision is None when the router never predicted the class;
    recall is None when the class has no gold questions (undefined,
    not zero — the aggregate() None-exclusion convention)."""
    out = {}
    for c in ROUTES:
        support = sum(matrix[c].values())
        predicted_n = sum(matrix[g][c] for g in ROUTES)
        tp = matrix[c][c]
        out[c] = {
            "support": support,
            "predicted": predicted_n,
            "recall": (tp / support) if support else None,
            "precision": (tp / predicted_n) if predicted_n else None,
        }
    return out


def boundary_verdict(
    predicted: str, gold: str, acceptable_secondary: str | None,
) -> str:
    """'gold' | 'secondary' | 'off' for one boundary question."""
    if predicted == gold:
        return "gold"
    if acceptable_secondary is not None and predicted == acceptable_secondary:
        return "secondary"
    return "off"


def boundary_table(rows: list[dict]) -> list[dict]:
    """The adversarial-case table: one entry per boundary question with
    the routing verdict. Pure projection over scored rows."""
    out = []
    for r in rows:
        if not r.get("boundary"):
            continue
        out.append({
            "id": r["id"],
            "question": r["question"],
            "gold_route": r["gold_route"],
            "acceptable_secondary": r.get("acceptable_secondary"),
            "predicted_route": r["predicted_route"],
            "verdict": boundary_verdict(
                r["predicted_route"], r["gold_route"],
                r.get("acceptable_secondary"),
            ),
        })
    return out


_SCORER_NAMES = ["exact", "acceptable"]


def aggregate(rows: list[dict]) -> dict:
    """Mean per scorer, None-excluded; count of contributing questions."""
    out = {}
    for name in _SCORER_NAMES:
        vals = [r["scores"][name] for r in rows if r["scores"][name] is not None]
        out[name] = {
            "mean": (sum(vals) / len(vals)) if vals else None,
            "n": len(vals),
        }
    return out


# ─── Harness ────────────────────────────────────────────────────────


def run_questions(questions: list[dict], router) -> list[dict]:
    """Route every question, score against gold. Returns one row per
    question (with the predicted route as the re-scorable artifact)."""
    rows: list[dict] = []
    for q in questions:
        t0 = time.time()
        predicted = router(q["question"])
        dt = time.time() - t0
        rows.append({
            "id": q["id"],
            "family": q["family"],
            "question": q["question"],
            "gold_route": q["gold_route"],
            "acceptable_secondary": q.get("acceptable_secondary"),
            "boundary": q.get("boundary", False),
            "predicted_route": predicted,
            "seconds": round(dt, 4),
            "scores": score_question(
                predicted, q["gold_route"], q.get("acceptable_secondary"),
            ),
        })
    return rows


def rescore_rows(rows: list[dict], questions_by_id: dict[str, dict]) -> list[dict]:
    """Re-apply the current rubric (and current gold labels) to
    persisted artifacts — the predicted routes — without re-running."""
    out = []
    for r in rows:
        q = questions_by_id[r["id"]]
        out.append({
            **r,
            "gold_route": q["gold_route"],
            "acceptable_secondary": q.get("acceptable_secondary"),
            "boundary": q.get("boundary", False),
            "scores": score_question(
                r["predicted_route"], q["gold_route"],
                q.get("acceptable_secondary"),
            ),
        })
    return out


# ─── CLI ────────────────────────────────────────────────────────────


def _load_questions(path: Path, include_heldout: bool) -> list[dict]:
    questions = []
    sealed = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            if q.get("heldout") and not include_heldout:
                sealed += 1
                continue
            questions.append(q)
    if sealed:
        print(
            f"refused {sealed} sealed held-out questions "
            f"(pass --include-heldout for a RELEASE REPORT run only — "
            f"never between tuning iterations)",
            file=sys.stderr,
        )
    return questions


def _print_matrix(matrix: dict[str, dict[str, int]]) -> None:
    w = 8
    header = "gold \\ pred".ljust(14) + "".join(r[:7].rjust(w) for r in ROUTES)
    print(header)
    for g in ROUTES:
        cells = "".join(str(matrix[g][p]).rjust(w) for p in ROUTES)
        print(g.ljust(14) + cells)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="RouterEval runner")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument(
        "--router", default="intent", choices=["intent"],
        help="Router adapter. 'intent' mirrors recall()'s gate order "
             "on the bare question (three-class coverage today). Live "
             "store-conditioned routing is available programmatically "
             "via recall_router(memory).",
    )
    p.add_argument(
        "--include-heldout", action="store_true",
        help="Unseal held-out questions. Release reports ONLY.",
    )
    p.add_argument(
        "--rescore", type=Path, default=None,
        help="Re-apply the current rubric to a prior results JSON "
             "(reads artifacts; does not re-run the router).",
    )
    p.add_argument("--max-questions", type=int, default=None)
    args = p.parse_args(argv)

    questions = _load_questions(args.data, args.include_heldout)
    if args.max_questions:
        questions = questions[: args.max_questions]

    coverage = INTENT_ROUTER_COVERAGE

    if args.rescore:
        prior = json.loads(args.rescore.read_text())
        by_id = {q["id"]: q for q in questions}
        rows = rescore_rows(prior["rows"], by_id)
    else:
        router = intent_router
        t0 = time.time()
        rows = run_questions(questions, router)
        print(
            f"routed {len(rows)} questions in {time.time() - t0:.2f}s",
            file=sys.stderr,
        )

    overall = aggregate(rows)
    supported = aggregate([r for r in rows if r["gold_route"] in coverage])
    boundary_rows = [r for r in rows if r.get("boundary")]
    boundary_agg = aggregate(boundary_rows)
    families = sorted({r["family"] for r in rows})
    by_family = {
        fam: aggregate([r for r in rows if r["family"] == fam])
        for fam in families
    }
    matrix = confusion_matrix(rows)
    per_class = per_class_metrics(matrix)
    btable = boundary_table(rows)

    def fmt(a: dict) -> str:
        return "  ".join(
            f"{name}={a[name]['mean']:.3f}({a[name]['n']})"
            if a[name]["mean"] is not None else f"{name}=—(0)"
            for name in _SCORER_NAMES
        )

    print(
        f"\nRouterEval rubric {RUBRIC_VERSION} — {len(rows)} questions, "
        f"router={args.router} (coverage: {', '.join(coverage)})"
    )
    print(f"overall:        {fmt(overall)}")
    print(f"supported gold: {fmt(supported)}   "
          f"(gold ∈ router coverage — the wrong-gate signal)")
    print(f"boundary:       {fmt(boundary_agg)}")
    for fam in families:
        print(f"  {fam:24s} {fmt(by_family[fam])}")

    print("\nconfusion matrix:")
    _print_matrix(matrix)

    print("\nper-class precision/recall:")
    for c in ROUTES:
        m = per_class[c]
        prec = f"{m['precision']:.3f}" if m["precision"] is not None else "—"
        rec = f"{m['recall']:.3f}" if m["recall"] is not None else "—"
        print(f"  {c:12s} precision={prec:>6s}  recall={rec:>6s}  "
              f"support={m['support']:3d}  predicted={m['predicted']:3d}")

    print("\nboundary cases:")
    for b in btable:
        sec = b["acceptable_secondary"] or "—"
        print(f"  [{b['verdict']:>9s}] {b['id']}  gold={b['gold_route']}  "
              f"secondary={sec}  predicted={b['predicted_route']}  "
              f"{b['question']!r}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "rubric_version": RUBRIC_VERSION,
            "data": str(args.data),
            "router": args.router,
            "router_coverage": list(coverage),
            "n_questions": len(rows),
            "overall": overall,
            "overall_supported_gold": supported,
            "boundary": boundary_agg,
            "by_family": by_family,
            "per_class": per_class,
            "confusion_matrix": matrix,
            "boundary_table": btable,
            "rows": rows,
        }, indent=2))
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
