"""CompositionEval — scorers + harness + CLI runner.

Measures the FOURTH question class (docs/roadmap.md §3): composition —
time-series-of-sums. "How has my spending on the bike evolved?" wants
neither one scalar (gaṇita) nor prose beats (narrative): it wants
per-period buckets of procedural arithmetic plus a trend statement.
Scenario format + the frozen scoring rubric are documented in
eval/composition_data/README.md. Rubric changes require a version bump
and a re-report — never silent edits.

Rubric history:
- v1 (frozen 2026-07-08): routed / bucket_periods / bucket_values /
  receipts / trend / scalar. Frozen BEFORE belief/composition.py exists
  (working-protocol rule 2: instrument first) — the eval runs red on
  every composition-gold question today, by design.

Gold rules (encoded here and verified by tests/test_composition_eval.py):
- Gap rule: a period appears in expected_buckets iff ≥1 contributing
  proposition exists in it — missing months are ABSENT, never zero
  buckets, for every op including count. A fabricated zero has no
  belief-id receipt behind it, which violates the preserved-facts
  contract arithmetic rests on.
- Currency rule: bucket values are USD-only. Non-USD amounts are never
  converted or silently mixed; gold excludes them
  (``excluded_currency_indices``). A bucket value that equals the
  silently-mixed sum fails cent-exact matching by construction.
- Count rule: for op=count each contributing proposition asserts one
  countable event; the bucket value is the number of contributing
  propositions in the period (mirrors gaṇita's count-of-tuples).
- Trend rule: gold trend labels are not vibes — they equal
  ``classify_trend()`` (frozen below) applied to the expected bucket
  values in period order. The data-integrity tests recompute them.

Answerer-callable pattern:
  ``run_scenario`` takes an ``answerer_factory(mem, belief_to_idx)``
  returning a callable ``(question: str) -> dict`` that produces a
  normalized answer::

      {"route": "composition"|"ganita"|"narrative"|"retrieval",
       "strategy_raw": str,
       "buckets": [{"period": "2026-03", "value": 120.0, "unit": "USD",
                    "contributing": [proposition indices]}] | None,
       "trend": "rising"|"falling"|"flat"|"spike" | None,
       "scalar_value": float | None}

  The default factory (``make_recall_answerer``) wraps the public
  ``Memory.recall()``. When the composition primitive ships it is
  expected to surface ``Recall.composition`` (attribute or dict) with
  ``buckets`` (each carrying period / value / unit /
  contributing_belief_ids) and ``trend``; the adapter below already
  reads that shape, so the instrument turns green without modification.
  Until then every composition question scores routed=0.0.

Design notes (same discipline as eval/evolution_eval.py):
- Scoring is by exact belief-id → proposition-index mapping captured at
  ingest. No fuzzy text matching anywhere.
- Scorers are pure functions over (route, buckets, trend, scalar, gold)
  — unit-testable without any model. Only the harness touches Memory.
- Artifacts (route + buckets + trend + scalar per question) are
  persisted in the results JSON so runs can be re-scored under future
  rubric versions without re-running the system (--rescore).
- Values compare cent-exact after normalization (|a − b| < half a cent).

Usage:
    uv run python -m eval.composition_eval \\
        --data eval/composition_data/dev_scenarios.jsonl \\
        --output runs/composition/dev-baseline.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

RUBRIC_VERSION = "v1"

TREND_LABELS = ("rising", "falling", "flat", "spike")

# Cent-exactness: values match iff they agree to within half a cent
# after normalization. Tight enough that a silently-mixed-currency sum
# or a leaked distractor amount can never pass.
_CENT = 0.005


def _cent_equal(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) < _CENT


# ─── Frozen trend rule ───────────────────────────────────────────────


def classify_trend(values: list[float]) -> str | None:
    """The frozen v1 trend vocabulary over bucket values in period order.

    - "spike": one bucket dominates (> 2.5× the largest of the rest)
      while the rest are roughly level (spread ≤ 50% of their max).
      Needs ≥ 3 buckets.
    - "flat": total spread ≤ 10% of the max value.
    - "rising": non-decreasing and last > first.
    - "falling": non-increasing and last < first.
    - None: < 2 buckets (no series → no trend; the degradation
      contract routes these to plain gaṇita), or an ambiguous series —
      gold data must never be ambiguous (integrity-tested).

    Checked in this order; first match wins. Gold labels in
    composition_data are REQUIRED to equal this function applied to the
    expected bucket values — the rule, not the author, is the referee.
    """
    n = len(values)
    if n < 2:
        return None
    if n >= 3:
        peak = max(values)
        rest = sorted(values)[:-1]
        rest_max, rest_min = max(rest), min(rest)
        if (
            rest_max > 0
            and peak > 2.5 * rest_max
            and (rest_max - rest_min) <= 0.5 * rest_max
        ):
            return "spike"
    hi, lo = max(values), min(values)
    if hi > 0 and (hi - lo) <= 0.10 * hi:
        return "flat"
    nondecreasing = all(values[i + 1] >= values[i] for i in range(n - 1))
    nonincreasing = all(values[i + 1] <= values[i] for i in range(n - 1))
    if nondecreasing and values[-1] > values[0]:
        return "rising"
    if nonincreasing and values[-1] < values[0]:
        return "falling"
    return None


# ─── Scorers (frozen rubric — see composition_data/README.md) ────────


def score_routed(route: str, expected_route: str) -> float:
    """1.0 iff the answer came off the expected path. The gate: every
    other scorer is None when routing missed (routed owns that miss).
    Negative controls make this bidirectional — a plain "how much
    total" question expects "ganita" and a composition route STEALING
    it scores 0.0 here."""
    return 1.0 if route == expected_route else 0.0


def score_bucket_periods(
    returned: list[dict], expected: list[dict],
) -> float | None:
    """Jaccard |R ∩ E| / |R ∪ E| over period strings. 1.0 only on exact
    set equality — a fabricated zero-bucket for a gap month grows the
    union and costs score (the gap rule, enforced). None when there are
    no expected buckets (not a composition gold)."""
    if not expected:
        return None
    r = {b["period"] for b in returned}
    e = {b["period"] for b in expected}
    return len(r & e) / len(r | e)


def score_bucket_values(
    returned: list[dict], expected: list[dict],
) -> float | None:
    """Over periods present in BOTH sets: fraction whose values match
    cent-exact. None when no common period exists (bucket_periods owns
    that failure) or when there are no expected buckets."""
    if not expected:
        return None
    rv = {b["period"]: b.get("value") for b in returned}
    common = [b for b in expected if b["period"] in rv]
    if not common:
        return None
    hits = sum(
        1 for b in common
        if rv[b["period"]] is not None
        and _cent_equal(float(rv[b["period"]]), float(b["value"]))
    )
    return hits / len(common)


def score_receipts(
    returned: list[dict], expected: list[dict],
) -> float | None:
    """Over periods present in BOTH sets: fraction where the returned
    contributing set equals the gold contributing set exactly. A bucket
    with missing/empty receipts scores 0 for that period — per-bucket
    receipts are the contract, not a nicety. None when no common
    period exists or there are no expected buckets."""
    if not expected:
        return None
    rc = {
        b["period"]: set(b.get("contributing") or []) for b in returned
    }
    common = [b for b in expected if b["period"] in rc]
    if not common:
        return None
    hits = sum(
        1 for b in common if rc[b["period"]] == set(b["contributing"])
    )
    return hits / len(common)


def score_trend(returned: str | None, expected: str | None) -> float | None:
    """Exact label match against the frozen vocabulary. None when the
    gold has no trend (degradation cases, negative controls)."""
    if expected is None:
        return None
    return 1.0 if returned == expected else 0.0


def score_scalar(returned: float | None, expected: float | None) -> float | None:
    """Cent-exact scalar for gaṇita-gold questions (degradation cases
    and plain-aggregation negative controls). None when the gold has no
    scalar expectation."""
    if expected is None:
        return None
    if returned is None:
        return 0.0
    return 1.0 if _cent_equal(float(returned), float(expected)) else 0.0


def score_question(
    route: str,
    buckets: list[dict] | None,
    trend: str | None,
    scalar: float | None,
    question: dict,
) -> dict:
    """Apply the full frozen rubric to one question's outcome.

    Gating mirrors EvolutionEval: content scorers only apply when the
    answer arrived on the expected path — otherwise they are None and
    the routed scorer owns the failure. (Scoring buckets produced by
    the wrong path would blur "routed wrong" into "computed wrong".)
    """
    expected_route = question["expected_route"]
    scores: dict[str, float | None] = {
        "routed": score_routed(route, expected_route),
        "bucket_periods": None,
        "bucket_values": None,
        "receipts": None,
        "trend": None,
        "scalar": None,
    }
    if expected_route == "composition" and route == "composition":
        eb = question.get("expected_buckets") or []
        rb = buckets or []
        scores["bucket_periods"] = score_bucket_periods(rb, eb)
        scores["bucket_values"] = score_bucket_values(rb, eb)
        scores["receipts"] = score_receipts(rb, eb)
        scores["trend"] = score_trend(trend, question.get("expected_trend"))
    if expected_route == "ganita" and route == "ganita":
        scores["scalar"] = score_scalar(scalar, question.get("expected_value"))
    return scores


_SCORER_NAMES = [
    "routed", "bucket_periods", "bucket_values", "receipts", "trend",
    "scalar",
]


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


# ─── Default answerer (wraps the public Memory.recall) ──────────────


def _payload_get(obj, key: str, default=None):
    """Read a field off an attribute-style or dict-style payload."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def make_recall_answerer(
    mem, belief_to_idx: dict[str, int],
) -> Callable[[str], dict]:
    """The default answerer: ask ``Memory.recall`` and normalize.

    Route normalization:
    - a ``Recall.composition`` payload (or strategy == "composition")
      → "composition", with buckets' contributing_belief_ids mapped to
      proposition indices;
    - strategy "ganita" → "ganita" with the GanitaResult scalar;
    - strategy "narrative" → "narrative";
    - anything else → "retrieval".
    """

    def _answer(question: str) -> dict:
        rec = mem.recall(question)
        comp = getattr(rec, "composition", None)
        if comp is not None or rec.strategy == "composition":
            buckets: list[dict] = []
            for b in _payload_get(comp, "buckets", None) or []:
                ids = _payload_get(b, "contributing_belief_ids", None) or []
                buckets.append({
                    "period": _payload_get(b, "period"),
                    "value": _payload_get(b, "value"),
                    "unit": _payload_get(b, "unit"),
                    "contributing": sorted(
                        belief_to_idx[i] for i in ids if i in belief_to_idx
                    ),
                })
            return {
                "route": "composition",
                "strategy_raw": rec.strategy,
                "buckets": buckets,
                "trend": _payload_get(comp, "trend", None),
                "scalar_value": None,
            }
        if rec.strategy == "ganita" and rec.ganita is not None:
            return {
                "route": "ganita",
                "strategy_raw": rec.strategy,
                "buckets": None,
                "trend": None,
                "scalar_value": float(rec.ganita.value),
            }
        if rec.strategy == "narrative":
            return {
                "route": "narrative",
                "strategy_raw": rec.strategy,
                "buckets": None,
                "trend": None,
                "scalar_value": None,
            }
        return {
            "route": "retrieval",
            "strategy_raw": rec.strategy,
            "buckets": None,
            "trend": None,
            "scalar_value": None,
        }

    return _answer


# ─── Harness ────────────────────────────────────────────────────────


def run_scenario(
    scenario: dict,
    *,
    detector: str,
    shared_embedder=None,
    answerer_factory: Callable[..., Callable[[str], dict]] | None = None,
) -> list[dict]:
    """Ingest one scenario into a fresh store, ask its questions through
    the answerer, score. Returns one row per question (with artifacts
    for re-scoring)."""
    import patha

    factory = answerer_factory or make_recall_answerer
    rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="composition-eval-") as td:
        mem = patha.Memory(
            path=Path(td) / "beliefs.jsonl",
            detector=detector,
            enable_phase1=True,
        )
        # Share one embedder across scenarios (evolution_eval pattern);
        # the retriever lazily creates its own only if _embedder is None.
        if shared_embedder is not None and mem._phase1_retriever is not None:
            mem._phase1_retriever._embedder = shared_embedder

        from datetime import datetime

        belief_to_idx: dict[str, int] = {}
        for i, prop in enumerate(scenario["propositions"]):
            ev = mem.remember(
                prop["text"],
                asserted_at=datetime.fromisoformat(prop["asserted_at"]),
                session_id=prop.get("session"),
                source_id=f"cmp:{scenario['id']}#{i}",
            )
            bid = ev["belief_id"] if isinstance(ev, dict) else ev.new_belief.id
            belief_to_idx[bid] = i

        answer = factory(mem, belief_to_idx)
        for q in scenario["questions"]:
            t0 = time.time()
            ans = answer(q["q"])
            dt = time.time() - t0
            rows.append({
                "scenario_id": scenario["id"],
                "family": scenario["family"],
                "question": q["q"],
                "route": ans["route"],
                "strategy_raw": ans.get("strategy_raw"),
                "buckets": ans.get("buckets"),
                "trend": ans.get("trend"),
                "scalar_value": ans.get("scalar_value"),
                "seconds": round(dt, 2),
                "scores": score_question(
                    ans["route"], ans.get("buckets"), ans.get("trend"),
                    ans.get("scalar_value"), q,
                ),
            })
    return rows


def rescore_rows(rows: list[dict], scenarios_by_id: dict[str, dict]) -> list[dict]:
    """Re-apply the current rubric to persisted artifacts (no re-run)."""
    out = []
    for r in rows:
        scenario = scenarios_by_id[r["scenario_id"]]
        q = next(
            qq for qq in scenario["questions"] if qq["q"] == r["question"]
        )
        out.append({
            **r,
            "scores": score_question(
                r["route"], r.get("buckets"), r.get("trend"),
                r.get("scalar_value"), q,
            ),
        })
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
    p = argparse.ArgumentParser(description="CompositionEval runner")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--detector", default="stub")
    p.add_argument(
        "--include-heldout", action="store_true",
        help="Unseal held-out scenarios. Release reports ONLY. "
             "(The current data is dev-only; the flag exists so the "
             "seal machinery is in place before a sealed batch is.)",
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
        # One embedder for all scenarios (each Memory would otherwise
        # load MiniLM separately).
        from patha.models.embedder_st import SentenceTransformerEmbedder
        shared = SentenceTransformerEmbedder()
        rows = []
        t0 = time.time()
        for i, s in enumerate(scenarios, 1):
            rows.extend(
                run_scenario(s, detector=args.detector, shared_embedder=shared)
            )
            print(
                f"  [{i}/{len(scenarios)}] {s['id']} "
                f"({s['family']})", file=sys.stderr,
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

    print(f"\nCompositionEval rubric {RUBRIC_VERSION} — {len(rows)} questions")
    print(f"overall: {fmt(overall)}")
    for fam in families:
        print(f"  {fam:28s} {fmt(by_family[fam])}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "rubric_version": RUBRIC_VERSION,
            "data": str(args.data),
            "detector": args.detector,
            "n_questions": len(rows),
            "overall": overall,
            "by_family": by_family,
            "rows": rows,
        }, indent=2))
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
