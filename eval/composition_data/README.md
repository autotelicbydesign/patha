# CompositionEval — can a memory system return a time-series of sums?

The instrument for the FOURTH question class (docs/roadmap.md §3):
**composition** — "how has my spending on the bike evolved?" wants narrative
*shape* over synthesis *content*. Neither existing path alone answers it:
gaṇita collapses to one number, the narrative walk returns prose beats. The
correct answer is per-period buckets of procedural arithmetic plus a trend
statement, each bucket carrying its contributing beliefs as receipts.

**Instrument-first (working-protocol rule 2):** this eval was authored and
frozen on 2026-07-08, *before* `src/patha/belief/composition.py` exists. It
runs **red today on every composition-gold question by design** — routed goes
to gaṇita ("per month" phrasings answer with one wrong total), narrative, or
retrieval. The routing-negative controls run green today and become the
routing-theft guards the moment a composition gate ships.

## What a scenario looks like

```json
{
  "id": "cmp-001",
  "family": "monthly_sum",
  "propositions": [
    {"text": "spent $40 on a new chain for the bike",
     "asserted_at": "2026-01-08T09:00:00", "session": "s1"},
    ...
  ],
  "questions": [{
    "q": "how has my spending on the bike evolved?",
    "type": "composition",                  // composition | aggregation | narrative
    "expected_route": "composition",        // composition | ganita | narrative
    "op": "sum",                            // sum | count | avg | null
    "period_granularity": "month",          // month | year | null
    "expected_buckets": [                   // ordered by period
      {"period": "2026-01", "value": 40.0, "contributing": [0]},
      {"period": "2026-02", "value": 65.0, "contributing": [2]},
      ...
    ],
    "expected_trend": "rising",             // rising | falling | flat | spike | null
    "expected_value": null,                 // scalar gold for ganita-route questions
    "expected_value_contributing": [],      // receipts for that scalar
    "distractor_indices": [1, 3],           // off-entity props that must not leak
    "excluded_currency_indices": []         // non-USD amounts (see currency rule)
  }]
}
```

Propositions use the EvolutionEval proposition schema (text / asserted_at /
session), ingested chronologically into a fresh store; scoring is by exact
belief-id → proposition-index mapping, never fuzzy text match.

## Gold rules (frozen with rubric v1; recomputed by the data-integrity tests)

- **Gap rule** — a period appears in `expected_buckets` iff ≥ 1 contributing
  proposition falls in it. Missing months are ABSENT, never zero buckets, for
  every op *including count*: a fabricated zero has no belief-id receipt
  behind it, and receipts are the contract (gaṇita's preserved-facts
  principle). A system that pads gaps with zeros loses `bucket_periods`
  score by construction (Jaccard over period sets).
- **Currency rule** — bucket values are USD-only. Non-USD amounts are never
  converted and never silently mixed; gold lists them in
  `excluded_currency_indices`. A silently-mixed bucket value fails the
  cent-exact match by construction. (Flagging the excluded amount to the
  user is desired behaviour but not scored in v1.)
- **Count rule** — for `op: count`, each contributing proposition asserts one
  countable event; the bucket value is the number of contributing
  propositions in the period (mirrors gaṇita's count-of-tuples semantics).
- **Trend rule** — gold trend labels are not authorial vibes: each equals
  `classify_trend()` (frozen in `eval/composition_eval.py`) applied to the
  expected bucket values in period order. spike = one bucket > 2.5× the max
  of the rest with the rest level within 50%; flat = spread ≤ 10% of max;
  rising/falling = monotone with a real endpoint change. The integrity tests
  recompute every label.
- **Degradation rule** — < 2 non-empty buckets is not a series. Gold for
  those questions is `expected_route: "ganita"` with a cent-exact
  `expected_value` (the same graceful-degradation contract the narrative
  walk honours).
- **Value provenance** — every sum/avg gold value is recomputed by the tests
  from the contributing propositions *via the production extractor*
  (`patha.belief.ganita.extract_tuples`), one USD amount per contributing
  proposition. A wrong gold is worse than a missing scenario.

## The nine families (21 scenarios, 21 questions)

| family | n | shape | targets |
|---|---|---|---|
| `monthly_sum` | 6 | rising / falling / flat / spike / multi-contributing buckets | the core primitive; distractor amounts on other entities (rent, coat) must not leak |
| `yearly_sum` | 2 | "year over year" | period-granularity parsing beyond the month default |
| `gap_months` | 2 | missing interior months | the gap rule — no fabricated zeros |
| `multi_currency` | 2 | euros interleaved with USD | the currency rule — skip, never mix |
| `count_series` | 2 | one event per proposition (one with a gap month) | op=count semantics + gap rule interaction |
| `avg_series` | 1 | per-month average | op=avg per bucket |
| `single_bucket_degradation` | 2 | composition phrasing, one month of data | fall back to plain gaṇita with the correct scalar |
| `routing_negative_ganita` | 2 | plain "how much total/altogether" | composition must NOT steal plain aggregation |
| `routing_negative_narrative` | 2 | amount-free evolution/origin questions | composition must NOT steal pure narrative |

**Deliberate headroom, documented:** `cmp-004` (identical recurring $15.99
charges) cannot be answered by the current system even after a composition
gate ships — `GanitaIndex.has_equivalent` de-duplicates equal
(entity, attribute, value, unit) tuples at ingest regardless of their
differing `time` fields, so the series is never preserved. The gold is the
truth (four renewals happened); period-aware dedup is part of the
composition work. Also headroom: `cmp-007` (travel) needs entity aliasing
across "flights / hotels / trip" — the known alias-table limitation
documented in `ganita.py`.

## Dev-only — no held-out split yet

`dev_scenarios.jsonl` is **dev data: tuning against it is allowed.** There
is deliberately no sealed held-out batch yet — protocol rule 5 says held-out
scenarios are authored AFTER the code under test freezes, and the code under
test does not exist. The first sealed composition batch gets authored fresh
once `belief/composition.py` + the routing gate freeze, committed before its
first run. No scenario in this file carries `"heldout": true`; the runner's
`--include-heldout` seal machinery is already in place for that day.

## Frozen scoring rubric (v1 — changes require a version bump + re-report)

Per question, gated on the expected route (content scorers are `None` when
routing missed — `routed` owns that failure):

- **routed** — did the answer come off the expected path? Bidirectional:
  negative controls expect `ganita`/`narrative`, so composition stealing
  them scores 0.0 too.
- **bucket_periods** — Jaccard between returned and expected period sets;
  1.0 only on exact set equality (fabricated gap-buckets cost score).
- **bucket_values** — over periods present in both: fraction cent-exact
  (|a − b| < $0.005 after normalization).
- **receipts** — over periods present in both: fraction whose contributing
  belief set (mapped to proposition indices) equals gold exactly; missing
  receipts on a matched bucket score 0 for that bucket.
- **trend** — exact label match against the frozen vocabulary; `None` when
  gold has no trend.
- **scalar** — cent-exact value for ganita-gold questions (degradation +
  negative controls); `None` otherwise.

Aggregation: mean per scorer per family and overall, `None`s excluded.
Artifacts (route, buckets, trend, scalar per question) persist in the
results JSON so any run can be re-scored under a future rubric version
without re-running the system (`--rescore`).

### Rubric version history

- **v1** (frozen 2026-07-08): the six scorers above + the frozen
  `classify_trend` rule. Frozen before the capability exists. Known
  non-goals of v1, candidates for v2: op-identification scoring (the answer
  payload's op is not compared), week granularity, currency-flagging credit,
  and receipt partial credit.

## Running

```bash
# dev set (tuning allowed) — runs red on composition golds today, by design
uv run python -m eval.composition_eval \
    --data eval/composition_data/dev_scenarios.jsonl \
    --output runs/composition/dev-baseline.json

# re-score persisted artifacts under the current rubric (no re-run)
uv run python -m eval.composition_eval \
    --data eval/composition_data/dev_scenarios.jsonl \
    --rescore runs/composition/dev-baseline.json
```

Default detector is `stub` (deterministic, no downloads); the embedder is
the real MiniLM. The default answerer wraps the public `Memory.recall()`;
when the composition primitive ships it should surface `Recall.composition`
with `buckets` (period / value / unit / contributing_belief_ids) and
`trend` — the adapter already reads that shape, so the instrument turns
green without modification.
