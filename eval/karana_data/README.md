# KaranaEval — tuple-extraction quality (the synthesis ceiling)

The instrument for docs/roadmap.md §1. Extraction at ingest bounds every
synthesis claim: LongMemEval multi-session (0.857) is synthesis-bounded,
and `eval/ganita_synthesis_smoke.py` scores 0/8 on its synthesis-bounded
questions. KaranaEval measures the extractors **directly** — tuple-level
precision/recall on 25 hand-labeled dense-conversation cases — so
extractor configs can be compared before any is made default.

**Division of labour** (deliberate deviation from the roadmap sketch,
documented in `eval/karana_eval.py`): the 8 real LongMemEval cases stay
in the smoke test (end-to-end, the 0/8 → ≥6/8 definition-of-done gate);
this gold set is authored text where every label is verifiable by
reading one line. A wrong gold is worse than a missing case.

## Gold rules (v1)

- **GOLD** = numerical facts a user could later aggregate under gaṇita
  semantics: spend (money OUT), counts, distances, ages. Stated exactly.
- **FORBIDDEN** = values an extractor might emit that fabricate facts:
  range endpoints, hypotheticals, colloquial quantities ("a couple
  hundred"), money IN (refunds/discounts — mirrors gaṇita's `_NEGATIVE`
  veto contract), numeric distractors (clock times, temperatures,
  versions) typed as money.
- `acceptable` lists every entity token that legitimately identifies a
  fact (part→whole aliases included: chain→bike); matching is canonical
  via gaṇita's `_canonicalize_entity`, so the rule and the production
  code can't drift.
- Time is **not scored** in v1 (no extractor labels it yet).

## Case schema

```json
{"id": "ka-ms-01", "family": "multi_amount",
 "text": "ended up spending $40 on the pump and another $85 on the saddle bag…",
 "gold_tuples": [{"entity": "pump", "acceptable": ["pump","bike","cycling"],
                  "value": 40.0, "unit": "USD"}],
 "forbidden_tuples": [{"value": 1200.0, "reason": "range endpoint"}]}
```

14 families, 26 gold tuples, 18 forbidden values. Dev-only.

## Frozen rubric (v1 — changes require a version bump + re-report)

- **precision** — matched predictions / all predictions (`None` on
  silence: on forbidden-only cases silence is the *correct* behaviour)
- **recall** — matched golds / all golds (`None` when a case has none)
- **f1** — harmonic mean where both defined
- **forbidden_hit** — 1.0 iff any predicted value equals a forbidden
  value, unit-insensitive (fabricated precision is the sin)

Matching: value exact (±0.005), unit normalized (USD/EUR/GBP/item/km/
hours/years), entity canonical-or-alias, greedy one-to-one.

## Baseline (first direct measurement, 2026-07-08)

`regex` extractor: **precision 0.560 / recall 0.719 / forbidden_hit
0.636**. Strong on plain dollar amounts (multi_amount, dated, dense
1.000); broken on currency symbols (£/euros missed), age series (0.0),
and — the headline — it **extracts ranges, hypotheticals, and
temperatures as money** (forbidden_hit 1.0 on those families): the
gaṇita veto families act at aggregation, not extraction. That is the
karaṇa-v2 fix list, now measured.

## Running

```bash
uv run python -m eval.karana_eval \
    --data eval/karana_data/gold_cases.jsonl \
    --extractor regex \
    --output runs/karana/dev-regex.json
# ollama/hybrid configs refuse to run when the ollama host is down
# (the in-library fallback would silently score regex under the LLM's
# name). Future extractors plug in via --extractor module:Class.
```

Provenance: authored in `author_dev.py` (validates invariants,
serializes; not a generator). Regeneration must be byte-identical.
