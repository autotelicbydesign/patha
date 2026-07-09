"""KaranaEval — tuple-extraction quality: scorers + harness + CLI runner.

The instrument for docs/roadmap.md §1: extraction quality at ingest
bounds every synthesis claim (LongMemEval multi-session 0.857 is
synthesis-bounded; `ganita_synthesis_smoke` scores 0/8). This eval
measures the extractors DIRECTLY — tuple-level precision/recall on
hand-labeled dense-conversation text — so extractor configs can be
compared before any of them is wired as default.

Division of labour with ganita_synthesis_smoke.py (deliberate):
- KaranaEval: tuple-level P/R/F1 on 25 authored cases with hand-checked
  labels (this file). Every label is verifiable by reading one line.
- ganita_synthesis_smoke: end-to-end answers on the 8 real LongMemEval
  synthesis-bounded questions (the 0/8 → ≥6/8 definition-of-done gate).
The roadmap's original sketch put the LongMemEval turns inside this
gold set too; they stay in the smoke test instead — hand-copying long
real-conversation turns invites mislabeled golds, and a wrong gold is
worse than a missing case. Revisit if tuple-level diagnosis of the
LME-8 is ever needed.

Rubric history:
- v1 (frozen 2026-07-08): precision / recall / f1 / forbidden_hit.
  Time is NOT scored (no extractor labels it yet); gold `time` fields,
  where present, wait for a v2.

Matching (tolerant, all normalization frozen here):
- value: exact after float coercion (|a−b| < 0.005);
- unit: normalized via _UNIT_MAP (usd/$/dollar→USD, items/pieces→item…);
- entity: canonical match via ganita's _canonicalize_entity — a
  predicted tuple's entity OR any of its entity_aliases must
  canonically equal one of the gold's `acceptable` tokens;
- golds matched greedily, each at most once (age-series cases repeat
  (entity, unit) with different values).

Scorers are pure functions over plain dicts; predicted GanitaTuples are
normalized to dicts by the harness. Artifacts persist per case for
--rescore. Deterministic for the regex extractor.

Usage:
    uv run python -m eval.karana_eval \\
        --data eval/karana_data/gold_cases.jsonl \\
        --extractor regex \\
        --output runs/karana/dev-regex.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time as _time
from pathlib import Path

from patha.belief.ganita import _canonicalize_entity

RUBRIC_VERSION = "v1"

_UNIT_MAP = {
    "usd": "USD", "$": "USD", "dollar": "USD", "dollars": "USD",
    "eur": "EUR", "euro": "EUR", "euros": "EUR",
    "gbp": "GBP", "pound": "GBP", "pounds": "GBP", "£": "GBP",
    "item": "item", "items": "item", "piece": "item", "pieces": "item",
    "count": "item", "unit": "item", "units": "item",
    "km": "km", "kilometre": "km", "kilometres": "km",
    "kilometer": "km", "kilometers": "km",
    "hour": "hours", "hours": "hours", "hrs": "hours", "h": "hours",
    "year": "years", "years": "years", "yo": "years",
}


def normalize_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    return _UNIT_MAP.get(str(unit).strip().lower(), str(unit).strip())


def _value_equal(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) < 0.005
    except (TypeError, ValueError):
        return False


def _entity_matches(pred: dict, acceptable: list[str]) -> bool:
    accept = {_canonicalize_entity(a) for a in acceptable}
    cands = [pred.get("entity") or ""]
    cands += list(pred.get("entity_aliases") or [])
    return any(_canonicalize_entity(c) in accept for c in cands if c)


def match_tuples(
    predicted: list[dict], golds: list[dict],
) -> list[tuple[int, int]]:
    """Greedy one-to-one matching: for each predicted tuple in order,
    claim the first unclaimed gold it matches on (entity-acceptable,
    value, unit). Returns (pred_idx, gold_idx) pairs."""
    claimed: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for pi, p in enumerate(predicted):
        for gi, g in enumerate(golds):
            if gi in claimed:
                continue
            if not _value_equal(p.get("value"), g["value"]):
                continue
            if normalize_unit(p.get("unit")) != normalize_unit(g["unit"]):
                continue
            if not _entity_matches(p, g["acceptable"]):
                continue
            claimed.add(gi)
            pairs.append((pi, gi))
            break
    return pairs


def score_case(predicted: list[dict], case: dict) -> dict:
    """Apply the frozen rubric to one case's extraction output.

    - precision: matched / |predicted|; None when nothing was predicted
      (silence has no precision — recall owns misses, and on
      forbidden-only cases silence is the CORRECT behaviour).
    - recall: matched / |golds|; None when the case has no golds.
    - f1: harmonic mean when both defined and > 0; None when either is
      None; 0.0 when defined but degenerate.
    - forbidden_hit: 1.0 iff any predicted value equals a forbidden
      value (unit-insensitive — fabricated precision is the sin, not
      its labeling); None when the case lists none.
    """
    golds = case["gold_tuples"]
    forbidden = case["forbidden_tuples"]
    pairs = match_tuples(predicted, golds)
    matched = len(pairs)

    precision = (matched / len(predicted)) if predicted else None
    recall = (matched / len(golds)) if golds else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    if forbidden:
        forb_values = [f["value"] for f in forbidden]
        hit = any(
            _value_equal(p.get("value"), fv)
            for p in predicted for fv in forb_values
        )
        forbidden_hit = 1.0 if hit else 0.0
    else:
        forbidden_hit = None

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "forbidden_hit": forbidden_hit,
    }


_SCORER_NAMES = ["precision", "recall", "f1", "forbidden_hit"]


def aggregate(rows: list[dict]) -> dict:
    """Mean per scorer, None-excluded; count of contributing cases."""
    out = {}
    for name in _SCORER_NAMES:
        vals = [r["scores"][name] for r in rows if r["scores"][name] is not None]
        out[name] = {
            "mean": (sum(vals) / len(vals)) if vals else None,
            "n": len(vals),
        }
    return out


# ─── Harness ────────────────────────────────────────────────────────


def tuple_to_dict(t) -> dict:
    """Normalize a GanitaTuple (or anything duck-typed like one) to the
    plain dict the scorers consume."""
    get = (lambda k, d=None: t.get(k, d)) if isinstance(t, dict) \
        else (lambda k, d=None: getattr(t, k, d))
    return {
        "entity": get("entity"),
        "entity_aliases": list(get("entity_aliases") or []),
        "attribute": get("attribute"),
        "value": get("value"),
        "unit": get("unit"),
    }


def run_case(case: dict, extractor) -> dict:
    t0 = _time.time()
    raw = extractor.extract(case["text"], belief_id=f"karana-eval:{case['id']}")
    dt = _time.time() - t0
    predicted = [tuple_to_dict(t) for t in raw]
    return {
        "case_id": case["id"],
        "family": case["family"],
        "predicted": predicted,
        "seconds": round(dt, 3),
        "scores": score_case(predicted, case),
    }


def rescore_rows(rows: list[dict], cases_by_id: dict[str, dict]) -> list[dict]:
    """Re-apply the current rubric to persisted artifacts (no re-run)."""
    return [
        {**r, "scores": score_case(r["predicted"], cases_by_id[r["case_id"]])}
        for r in rows
    ]


# ─── Extractor factory ──────────────────────────────────────────────


def _probe_ollama(host: str) -> bool:
    import urllib.error
    import urllib.request
    try:
        urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=3)
        return True
    except (urllib.error.URLError, OSError):
        return False


def build_extractor(name: str, *, ollama_model: str, ollama_host: str):
    """regex | ollama | hybrid | a "module:Class" import path (so future
    extractors — DepParseKaranaExtractor — plug in without touching this
    file). Ollama-backed configs REFUSE to run when the host is
    unreachable instead of silently scoring their regex fallback as if
    it were the LLM."""
    if name == "regex":
        from patha.belief.karana import RegexKaranaExtractor
        return RegexKaranaExtractor()
    if name in ("ollama", "hybrid"):
        if not _probe_ollama(ollama_host):
            raise RuntimeError(
                f"--extractor {name} requires a reachable ollama at "
                f"{ollama_host}; refusing to run (the in-library fallback "
                f"would silently score regex under the LLM's name)."
            )
        from patha.belief.karana import (
            HybridKaranaExtractor,
            OllamaKaranaExtractor,
        )
        cls = OllamaKaranaExtractor if name == "ollama" else HybridKaranaExtractor
        return cls(model=ollama_model, host=ollama_host)
    if ":" in name:
        import importlib
        module, cls_name = name.rsplit(":", 1)
        return getattr(importlib.import_module(module), cls_name)()
    raise ValueError(f"unknown --extractor: {name!r}")


# ─── CLI ────────────────────────────────────────────────────────────


def _load_cases(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines() if line.strip()
    ]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="KaranaEval runner")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument(
        "--extractor", default="regex",
        help="regex | ollama | hybrid | module:Class import path",
    )
    p.add_argument("--ollama-model", default="qwen2.5:7b-instruct")
    p.add_argument("--ollama-host", default="http://localhost:11434")
    p.add_argument(
        "--rescore", type=Path, default=None,
        help="Re-apply the current rubric to a prior results JSON "
             "(reads artifacts; does not re-run the extractor).",
    )
    p.add_argument("--max-cases", type=int, default=None)
    args = p.parse_args(argv)

    cases = _load_cases(args.data)
    if args.max_cases:
        cases = cases[: args.max_cases]

    if args.rescore:
        prior = json.loads(args.rescore.read_text())
        by_id = {c["id"]: c for c in cases}
        rows = rescore_rows(prior["rows"], by_id)
        extractor_name = prior.get("extractor")
    else:
        extractor = build_extractor(
            args.extractor,
            ollama_model=args.ollama_model,
            ollama_host=args.ollama_host,
        )
        extractor_name = args.extractor
        rows = []
        t0 = _time.time()
        for i, c in enumerate(cases, 1):
            rows.append(run_case(c, extractor))
            print(f"  [{i}/{len(cases)}] {c['id']} ({c['family']})",
                  file=sys.stderr)
        print(f"ran {len(cases)} cases in {_time.time()-t0:.0f}s",
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

    print(f"\nKaranaEval rubric {RUBRIC_VERSION} — {len(rows)} cases, "
          f"extractor={extractor_name}")
    print(f"overall: {fmt(overall)}")
    for fam in families:
        print(f"  {fam:24s} {fmt(by_family[fam])}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "rubric_version": RUBRIC_VERSION,
            "data": str(args.data),
            "extractor": extractor_name,
            "n_cases": len(rows),
            "overall": overall,
            "by_family": by_family,
            "rows": rows,
        }, indent=2))
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
