# RouterEval — does the question reach the right pramāṇa?

With six question classes, the router IS the system: a misroute fails
*silently*, because every pramāṇa produces a fluent answer of the wrong
KIND (a sum where a timeline was wanted; a top-K snippet where a
qualified "you never decided that" was wanted). The per-pramāṇa
instruments (EvolutionEval, BeliefEval, the gaṇita smoke) each check
their own gate in isolation; nothing measured routing *across all
classes* until this instrument. RouterEval scores routing alone, as a
confusion matrix — no downstream primitive quality leaks into it.

Three of the six routes exist in `Memory.recall()` today
(gaṇita → narrative → retrieval, src/patha/__init__.py). The other
three — composition, absence, analogy — are roadmap items 3–5. Their
questions are labelled with the INTENDED route **on purpose**: the
instrument leads the implementation. Until those gates ship, their
per-class recall is 0.0 by construction and the confusion matrix shows
*where those questions land meanwhile* — which is precisely the
routing-theft baseline the roadmap items need ("measure routing
confusion rather than assuming it", roadmap item 4).

## What a question looks like

```json
{
  "id": "rt-bnd-01",
  "family": "boundary",
  "boundary": true,
  "question": "have I spent anything on the bike?",
  "gold_route": "absence",              // one of the six classes
  "acceptable_secondary": "synthesis",  // only where genuinely ambiguous
  "source": "authored",
  "notes": "existence-of-spending: a NO is grounded in absence ..."
}
```

- `gold_route` ∈ {retrieval, synthesis, narrative, composition,
  absence, analogy}.
- `acceptable_secondary` is set ONLY on boundary questions that are
  legitimately near two classes; scoring reports `exact` (gold only)
  and `acceptable` (gold or secondary) separately, so leniency is
  always visible, never silent.
- `notes` is the defensibility argument for the gold label. A wrong
  gold is worse than a missing question; every label must survive
  this field being read aloud.

## The families (90 questions, dev — tuning allowed)

| family | n | gold route | phrasing provenance |
|---|---|---|---|
| `retrieval_plain` | 14 | retrieval | first-person adaptations of BeliefEval seed phrasings + the recall() docstring example |
| `synthesis_plain` | 14 | synthesis | gaṇita smoke-test shapes (sum/count/average/max/min/difference) |
| `narrative_plain` | 14 | narrative | EvolutionEval question phrasings (evolution/origin/throughline), fresh domains |
| `composition_plain` | 12 | composition | authored from roadmap item 3 (aggregation × evolution co-occurrence) |
| `absence_plain` | 12 | absence | authored from roadmap item 4 (anupalabdhi phrasings) |
| `analogy_plain` | 9 | analogy | authored from roadmap item 5 (upamāna phrasings); includes two designed traps for the gaṇita "most" marker |
| `boundary` | 15 | mixed | adversarial: questions legitimately near two classes, each with a gold AND (where genuinely ambiguous) an acceptable secondary |

Phrasings were adapted from the existing corpora, never scenarios —
RouterEval sees bare questions only; there is no store.

**Dev-only.** No held-out split exists yet. The loader already
enforces the seal convention (`"heldout": true` questions are refused
without `--include-heldout`), so a future sealed batch — authored
after the composition/absence/analogy gates freeze — drops in with no
runner change. Per the working protocol, that batch must be authored
AFTER the gates it judges are frozen, committed before its first run,
and spent once.

## Frozen scoring rubric (v1 — changes require a version bump + re-report)

Per question (pure functions over predicted/gold/secondary — no
models, no store):

- **exact** — 1.0 iff `predicted == gold_route`.
- **acceptable** — 1.0 iff predicted ∈ {gold, acceptable_secondary}.
  Identical to exact for non-boundary questions.

Per run:

- **confusion matrix** — gold route × predicted route counts,
  zero-filled over all six classes.
- **per-class precision/recall** — precision `None` when the router
  never predicted the class; recall `None` when the class has no gold
  questions (undefined, not zero — the None-exclusion convention).
- **supported-gold aggregate** — exact/acceptable restricted to
  questions whose gold is in the router's declared coverage. This
  separates the two failure kinds: *wrong gate* (a supported-class
  miss) vs *missing gate* (an unimplemented class, misrouted by
  construction).
- **boundary table** — per adversarial question: verdict
  `gold` / `secondary` / `off`.

Aggregation: mean per scorer, `None`s excluded, per family and
overall. Artifacts (the predicted route per question) are persisted in
the results JSON so any run can be re-scored under a future rubric
version — or corrected gold labels — without re-running the router
(`--rescore`).

### Rubric version history

- **v1** (frozen 2026-07-08): exact/acceptable, confusion matrix,
  per-class precision/recall, boundary verdicts. Frozen before the
  first reported run.

## The default router adapter — what it does and does not measure

`intent_router` mirrors `recall()`'s gate ORDER using the production
detectors on the bare question: `detect_aggregation` first, then
`detect_narrative` + a resolvable theme, else retrieval. It measures
the routing *decision on phrasing alone*, assuming each pramāṇa's
index could serve the question. The store-dependent parts of the real
gates (gaṇita only wins when tuples match; the walk only wins with ≥2
beats) are intentionally not modelled — those fall-throughs are
degradation behaviour, owned by the per-pramāṇa instruments. When
store-conditioned routing is the question, use
`recall_router(memory)` programmatically: it calls the real
`recall()` and maps the `strategy` field
(`ganita`→synthesis, `narrative`→narrative, else retrieval, with
forward mappings for the roadmap strategies).

## Running

```bash
uv run python -m eval.router_eval \
    --data eval/router_data/dev_questions.jsonl \
    --output runs/router/dev-baseline.json

# re-score a prior run's artifacts under the current rubric/golds
uv run python -m eval.router_eval \
    --data eval/router_data/dev_questions.jsonl \
    --rescore runs/router/dev-baseline.json
```

Deterministic run-to-run: the intent router is regex-only.
