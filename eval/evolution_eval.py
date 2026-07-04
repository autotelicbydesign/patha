"""EvolutionEval — scorers + harness + CLI runner.

Measures whether Patha's narrative path reconstructs how a theme
evolved: ordered beats, origin, revision tagging, distractor exclusion.
Scenario format + the frozen scoring rubric (v1) are documented in
eval/evolution_data/README.md. Rubric changes require a version bump
and a re-report — never silent edits.

Design notes:
- Scoring is by exact belief-id → proposition-index mapping captured at
  ingest. No fuzzy text matching anywhere.
- Scorers are pure functions over (returned_indices, statuses, gold) —
  unit-testable without any model. The harness is the only part that
  touches Memory/embedders.
- Artifacts (returned indices + supersession statuses per question) are
  persisted in the results JSON so runs can be re-scored under future
  rubric versions without re-running the system (eval/rescore.py
  pattern).

Usage:
    uv run python -m eval.evolution_eval \\
        --data eval/evolution_data/dev_scenarios.jsonl \\
        --output runs/evolution/dev-baseline.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

RUBRIC_VERSION = "v1"

_REVISED = ("revised-from", "superseded")


# ─── Scorers (frozen rubric v1 — see evolution_data/README.md) ──────


def score_coverage(returned: list[int], gold: list[int]) -> float:
    """|returned ∩ gold| / |gold| — recall of gold beats."""
    if not gold:
        return 1.0
    r = set(returned)
    return sum(1 for g in gold if g in r) / len(gold)


def score_precision(returned: list[int], gold: list[int]) -> float | None:
    """|returned ∩ gold| / |returned| — distractor exclusion.
    None when nothing was returned (undefined, not zero)."""
    if not returned:
        return None
    g = set(gold)
    return sum(1 for r in returned if r in g) / len(returned)


def score_ordering(returned: list[int], gold: list[int]) -> float | None:
    """Concordant-pair fraction (Kendall tau mapped to [0,1]) over the
    gold beats that WERE returned, compared in returned order vs gold
    order. None if fewer than 2 gold beats were returned."""
    gold_pos = {g: i for i, g in enumerate(gold)}
    present = [r for r in returned if r in gold_pos]
    if len(present) < 2:
        return None
    concordant = 0
    total = 0
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            total += 1
            if gold_pos[present[i]] < gold_pos[present[j]]:
                concordant += 1
    return concordant / total


def score_origin(returned: list[int], expected_origin: int) -> float | None:
    """1.0 iff the FIRST returned beat is the expected origin.
    None when nothing was returned."""
    if not returned:
        return None
    return 1.0 if returned[0] == expected_origin else 0.0


def score_supersession(
    returned: list[int],
    statuses: dict[int, str],
    expected_pairs: list[list[int]],
) -> float | None:
    """Over expected [old, new] pairs where BOTH ends were returned:
    fraction where the old end is tagged revised/superseded. None if no
    pair has both ends returned (or there are no expectations)."""
    r = set(returned)
    applicable = [p for p in expected_pairs if p[0] in r and p[1] in r]
    if not applicable:
        return None
    hits = sum(
        1 for old, _new in applicable
        if statuses.get(old) in _REVISED
    )
    return hits / len(applicable)


def score_question(
    routed: bool,
    returned: list[int],
    statuses: dict[int, str],
    question: dict,
) -> dict:
    """Apply the full frozen rubric to one question's outcome."""
    if not routed:
        return {
            "routed": 0.0, "coverage": None, "precision": None,
            "ordering": None, "origin": None, "supersession": None,
        }
    gold = question["expected_beat_order"]
    return {
        "routed": 1.0,
        "coverage": score_coverage(returned, gold),
        "precision": score_precision(returned, gold),
        "ordering": score_ordering(returned, gold),
        "origin": score_origin(returned, question["expected_origin"]),
        "supersession": score_supersession(
            returned, statuses, question.get("expected_supersessions", []),
        ),
    }


_SCORER_NAMES = [
    "routed", "coverage", "precision", "ordering", "origin", "supersession",
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


# ─── Harness ────────────────────────────────────────────────────────


def run_scenario(scenario: dict, *, detector: str, shared_embedder=None) -> list[dict]:
    """Ingest one scenario into a fresh store, ask its questions, map
    returned beats to proposition indices, score. Returns one row per
    question (with artifacts for re-scoring)."""
    import patha

    rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="evolution-eval-") as td:
        mem = patha.Memory(
            path=Path(td) / "beliefs.jsonl",
            detector=detector,
            enable_phase1=True,
        )
        # Share one embedder across scenarios: 50+ scenario runs would
        # otherwise each load MiniLM. Injection point documented — the
        # retriever lazily creates its embedder only if _embedder is None.
        if shared_embedder is not None and mem._phase1_retriever is not None:
            mem._phase1_retriever._embedder = shared_embedder

        from datetime import datetime

        belief_to_idx: dict[str, int] = {}
        for i, prop in enumerate(scenario["propositions"]):
            ev = mem.remember(
                prop["text"],
                asserted_at=datetime.fromisoformat(prop["asserted_at"]),
                session_id=prop.get("session"),
                source_id=f"evo:{scenario['id']}#{i}",
            )
            bid = ev["belief_id"] if isinstance(ev, dict) else ev.new_belief.id
            belief_to_idx[bid] = i

        for q in scenario["questions"]:
            t0 = time.time()
            rec = mem.recall(q["q"], include_history=True)
            dt = time.time() - t0

            routed = rec.strategy == "narrative" and rec.narrative is not None
            returned: list[int] = []
            statuses: dict[int, str] = {}
            theme = None
            if routed:
                theme = rec.narrative.theme
                for b in rec.narrative.beats:
                    idx = belief_to_idx.get(b.belief_id)
                    if idx is None:
                        continue  # shouldn't happen in a fresh store
                    returned.append(idx)
                    statuses[idx] = b.supersession_status

            rows.append({
                "scenario_id": scenario["id"],
                "family": scenario["family"],
                "question": q["q"],
                "strategy": rec.strategy,
                "theme": theme,
                "expected_theme": q.get("expected_theme"),
                "returned_indices": returned,
                "statuses": {str(k): v for k, v in statuses.items()},
                "seconds": round(dt, 2),
                "scores": score_question(routed, returned, statuses, q),
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
        routed = bool(r["returned_indices"]) or r["scores"]["routed"] == 1.0
        statuses = {int(k): v for k, v in r.get("statuses", {}).items()}
        out.append({
            **r,
            "scores": score_question(
                routed, r["returned_indices"], statuses, q,
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
    p = argparse.ArgumentParser(description="EvolutionEval runner")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--detector", default="stub")
    p.add_argument(
        "--include-heldout", action="store_true",
        help="Unseal held-out scenarios. Release reports ONLY.",
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

    print(f"\nEvolutionEval rubric {RUBRIC_VERSION} — {len(rows)} questions")
    print(f"overall: {fmt(overall)}")
    for fam in families:
        print(f"  {fam:24s} {fmt(by_family[fam])}")

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
