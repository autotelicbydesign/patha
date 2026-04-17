# Phase 2 v0.1 — BeliefEval results

**Status:** v0.1 measurement complete
**Date:** 2026-04-18
**Dataset:** 20 hand-crafted scenarios (24 questions), `eval/belief_eval_data/seed_scenarios.jsonl`
**Detector:** `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (default NLI)

---

## Headline numbers

| Metric | Value |
|---|:---:|
| **Overall accuracy** | **0.833** (20/24) |
| Preference supersession | 0.571 (4/7) |
| Factual supersession | 0.875 (7/8) |
| Temporally bounded | 1.000 (9/9) |
| Current-belief questions | 0.733 (11/15) |
| Validity-at-time questions | 1.000 (9/9) |

## Stub-detector baseline (for comparison)

Same dataset, stub heuristic detector (no ML model):

| Metric | Value |
|---|:---:|
| Overall accuracy | 0.375 (9/24) |
| Current-belief questions | 0.000 (0/15) |
| Validity-at-time questions | 1.000 (9/9) |

**The NLI detector contributes +0.458 absolute** (+11 of 15 current-belief questions recovered).
Validity works perfectly on both detectors — it's orthogonal, driven by rule-based extraction.

## Failure analysis

Each of the 4 failed questions falls into one of two categories.

### Category A — Scoring-methodology artifacts (2 failures)

**`facts-05`: "Where does the user work now?"**
- Current belief: *"I left Canva and now run my own design studio"*
- Superseded: *"I work at Canva as a lead designer"*
- The scoring rule rejects any `expected_superseded_contains` term ("Canva") appearing in the current belief. But the current belief mentions Canva *because* it describes the transition away from it ("I left Canva..."). A human reader would rate this correct.

**`prefs-07`: "Where does the user work out currently?"**
- Current belief: *"I only run outdoors now, gym membership cancelled"*
- Superseded: *"I mostly work out in the gym"*
- Same pattern — the current belief mentions "gym" only in the phrase "gym membership cancelled", i.e., describing the cancellation.

**These are methodology failures, not system failures.** The BeliefEval scoring heuristic assumes current beliefs will not reference the superseded term at all. Real language frequently describes change by naming the old state ("I left X and now do Y", "I used to X, now Y"). The scoring rule is too strict.

### Category B — Genuine NLI limitations (2 failures)

**`prefs-01`: "What does the user currently believe about sushi?"**
- Propositions: *"I love sushi and eat it every week"* → *"I am avoiding raw fish on my doctor's advice"*
- NLI verdict: NEUTRAL (below supersession threshold)
- Diagnosis: Requires commonsense reasoning (sushi ⊆ raw fish). Pure NLI doesn't carry this inference.

**`prefs-03`: "Does the user currently eat fish?"**
- Propositions: *"I am vegetarian"* → *"I started eating fish again after a medical advice"*
- NLI verdict: NEUTRAL
- Diagnosis: Requires the lexical fact that vegetarians don't eat fish.

**These are real system errors.** NLI alone cannot bridge commonsense gaps of this shape. This is exactly the failure mode `v0.2`'s LLM fallback (D1 Option D) is designed to address: escalate to an LLM judge when NLI returns NEUTRAL with significant entity overlap.

## Reading the numbers honestly

| Classification | Count | % of 24 |
|---|:---:|:---:|
| Correct | 20 | 83.3% |
| Scoring artifact (system was right, metric wrong) | 2 | 8.3% |
| True failure (NLI limitation) | 2 | 8.3% |

- **Raw reported accuracy: 83.3%** — the number the runner outputs.
- **If scoring methodology is corrected to tolerate transition-describing current beliefs: ~91.7%** — but this adjustment must be stated openly, not baked in silently. Any future BeliefEval iteration should either improve the scoring rule or split the metric into `current_belief_contains` and `superseded_not_referenced` as separate checks.
- **Mechanism-pure score (excluding scoring artifacts): 22/24 = 91.7%** is the honest read of v0.1's actual mechanism capability.

## What v0.1 demonstrated

- **Temporal validity extraction works cleanly** (9/9 = 100%). Rule-based patterns catch explicit markers; validity filtering at query time correctly drops expired beliefs.
- **NLI-driven supersession works for surface-level contradictions** (Canva → own studio, Sydney → Sofia, pricing change, age change, company rename, gym → outdoors, coffee → green tea, etc.).
- **The store correctly preserves supersession lineage.** `include_history=True` surfaces the full chain for "when did that change?" queries.
- **Compression is intact.** Current-only summaries are ~21 tokens on average across 20 scenarios — far below what raw retrieval of 4-10 propositions would cost.

## Known failure modes — v0.2 targets

1. **Commonsense-gap contradictions.** NLI misses sushi↔raw fish, vegetarian↔fish. v0.2 gates an LLM-judge call behind an uncertainty band (NLI NEUTRAL + significant lexical overlap → LLM fallback).
2. **Scoring-methodology brittleness.** Scoring rule should tolerate current beliefs that descriptively name the superseded state. Simple fix: check negation context of the reference ("left X", "cancelled X", "used to X") before flagging a leak.
3. **Sample size.** 20 scenarios is a rigorous floor, not a publishable ceiling. v0.2 expands to ~150 with LLM-assisted generation + human QA, and aims for peer-reviewed benchmark-track submission (NeurIPS D&B or ICLR benchmarks).
4. **Token-economy curves not yet measured.** v0.1 reports a point estimate (~21 tokens/summary). v0.2 publishes the curve as memory grows (100 → 1,000 → 10,000 beliefs).

## Reproducing

```bash
# Stub baseline (no model download)
uv run python -m eval.belief_eval --detector stub

# NLI run (downloads DeBERTa-v3-large ~1.7 GB on first call)
uv run python -m eval.belief_eval --detector nli \
    --output runs/belief_eval/nli_v01.json
```

Runtime on Apple Silicon: ~3-5 minutes after model download, ~10-15 minutes including download.

## Files

- Benchmark data: `eval/belief_eval_data/seed_scenarios.jsonl`
- Runner: `eval/belief_eval.py`
- Results JSON: `runs/belief_eval/nli_v01.json`
- Spec: `docs/phase_2_spec.md`
- Literature survey: `docs/phase_2_literature_survey.md`
