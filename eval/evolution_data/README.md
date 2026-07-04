# EvolutionEval — can a memory system reconstruct how thinking evolved?

The first benchmark for **narrative evolution** in AI memory. Existing
benchmarks don't measure this: LongMemEval's `temporal-reasoning` stratum is
timestamp arithmetic ("how many days between X and Y"), and `knowledge-update`
is single-fact replacement. Nothing scores whether a system can return **the
ordered beats of a theme across time** — origin, revisions, and the exclusion
of interleaved off-theme noise. EvolutionEval does.

## What a scenario looks like

```json
{
  "id": "pr-001",
  "family": "progressive_revelation",
  "propositions": [
    {"text": "...", "asserted_at": "2025-01-05T09:00:00", "session": "s1"},
    ...
  ],
  "questions": [{
    "q": "how has my thinking about woodworking evolved?",
    "type": "evolution",
    "expected_theme": "woodworking",
    "expected_beat_order": [0, 2, 4, 6],   // prop indices, gold chronological arc
    "expected_origin": 0,                   // the true first engagement
    "distractor_indices": [1, 3, 5],        // off-theme props that must NOT appear
    "expected_supersessions": [[0, 6]]      // [old, new] pairs the system should tag
  }]
}
```

The harness ingests propositions chronologically into a fresh store (capturing
`belief_id` per index — scoring is by exact id mapping, never fuzzy text
match), asks each question through the public `Memory.recall()`, and scores
the returned `Recall.narrative` beats.

## The four families (each grounded in an observed failure — docs/phase_4_dogfood.md)

| family | shape | targets |
|---|---|---|
| `progressive_revelation` | vague noticing → first step → deepening → mastery | **F6**: origin identification |
| `multi_factor_change` | initial routine → *off-theme cause* → changed routine | composition headroom: the causal beat has no theme token |
| `perspective_shift` | event + first read → dwelling → reinterpretation → settled view | the N1 finding: **reversal without lexical contradiction** — detectors don't fire on reinterpretation |
| `reversed_belief_chain` | X → not-X → X′-with-nuance | nonmonotonic ordering; the middle beat must survive |

Every scenario interleaves dated off-theme distractors so precision is
measurable, and each family's `expected_supersessions` encode which revisions
a fully-capable system would tag. Some expectations are **deliberate headroom**
(documented above) — a benchmark the current system aces on day one would be
a benchmark that measures nothing.

## Dev / held-out split — the seal

| file | count | authorship | tuning allowed? |
|---|---|---|---|
| `dev_scenarios.jsonl` | 36 | `generate_scenarios.py` — deterministic templates × fixed slot tables; byte-identical regeneration | yes |
| `heldout_scenarios.jsonl` | 16 | hand-written, **disjoint surface domains**, varied structure (different beat counts, distractor placements) | **NEVER** |

Split ratio 36/16 ≈ 70/30, committed before the first reported run.

**The seal, operationally:** every held-out scenario carries `"heldout": true`
and the runner **refuses** to score them unless `--include-heldout` is passed.
Protocol: run held-out only for release-report numbers, never between tuning
iterations. Publish dev and held-out numbers side by side; the dev→held-out
gap is the honest generalization signal. This is the structural answer to the
BeliefEval lesson (tuned to 1.000 on its own scenarios; the external number
was 0.885 — see docs/benchmarks.md "Honest caveat").

**Provenance discipline:** the walker was frozen (commit `5ca7738`) *before*
these scenarios were authored. The scoring rubric below was frozen in
`eval/evolution_eval.py` *before* the first reported run. Walker-knob sweeps
(`PATHA_TOPIC_THRESHOLD`, budgets) run against **dev only**.

## Frozen scoring rubric (v1 — changes require a version bump + re-report)

All scorers operate on the returned beat sequence mapped to proposition
indices. Per question:

- **routed** — did `recall()` route `strategy="narrative"` at all? (gate; all
  other scorers are `None` when routing failed)
- **coverage** — `|returned ∩ gold| / |gold|` (recall of gold beats)
- **precision** — `|returned ∩ gold| / |returned|` (distractor exclusion)
- **ordering** — concordant-pair fraction over gold beats present, i.e.
  Kendall's tau mapped to [0,1]; `None` if fewer than 2 gold beats returned
- **origin** — 1.0 iff the *first* returned beat is `expected_origin`
- **supersession** — over `expected_supersessions` pairs with both ends
  returned: fraction where the old end is tagged `revised-from`/`superseded`;
  `None` if no pair has both ends returned

Aggregation: mean per scorer per family and overall, `None`s excluded;
routed-fraction reported separately. Artifacts (returned indices + statuses
per question) are persisted in the results JSON so any run can be re-scored
under a future rubric version without re-running the system
(`eval/rescore.py` pattern).

## Running

```bash
# dev set (tuning allowed)
uv run python -m eval.evolution_eval \
    --data eval/evolution_data/dev_scenarios.jsonl \
    --output runs/evolution/dev-baseline.json

# held-out (release reports ONLY)
uv run python -m eval.evolution_eval \
    --data eval/evolution_data/heldout_scenarios.jsonl \
    --include-heldout \
    --output runs/evolution/heldout-report.json
```

Default detector is `stub` (deterministic, no downloads); the embedder is the
real MiniLM (the system under test is the real walk over the real graph).
