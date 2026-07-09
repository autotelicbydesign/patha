# AbsenceEval — can a memory system reason about what is NOT there?

The instrument for **anupalabdhi** (non-perception as evidence — roadmap
item 4). Existing evals score what a system retrieves; nothing scores
whether it can *assert absence responsibly*: "you haven't decided yet",
"you never tried acupuncture — you HAVE tried physio and ibuprofen",
"no, you stopped the Italian class in May". The catastrophic failure
mode is unique to this pramāṇa: a confident **false "absent"** ("you
never decided") when a positive belief exists. This eval scores that
failure separately from everything else.

**DEV ONLY.** Every scenario carries `"split": "dev"`; tuning against
this file is allowed. No held-out split exists yet — a sealed batch gets
authored AFTER the absence recall path freezes (working-protocol rule 5).
The runner already refuses `"heldout": true` rows without
`--include-heldout`, so the seal mechanism is in place for that day.

## What a scenario looks like

```json
{
  "id": "ab-aty-001",
  "family": "absent_atyantabhava",
  "split": "dev",
  "propositions": [
    {"text": "...", "asserted_at": "2025-03-30T18:00:00", "session": "s1"},
    ...
  ],
  "questions": [{
    "q": "have I ever been to Japan?",
    "type": "absence",
    "scope": "ever",
    "gold": {
      "expected_route": "absence",
      "expected_kind": "atyantabhava",
      "expected_verdict": "absent",
      "expected_locus": "japan",
      "expected_contrast_ids": [0, 2]
    }
  }]
}
```

Proposition schema is EvolutionEval's (`text` / `asserted_at` /
`session`). Contrast ids are proposition indices; a real answerer maps
cited belief ids back to indices at ingest (the EvolutionEval
belief_id→index pattern). Gold loci are canonical fixed points of
`absence_eval.canonicalize_locus` (asserted at authoring time and by the
integrity tests).

## The six families (24 scenarios, one question each)

| family | n | gold | targets |
|---|---|---|---|
| `absent_atyantabhava` | 3 | absent, "never" | "have I ever…" — qualified never + what you HAVE done |
| `absent_pragabhava` | 3 | absent, "not yet" | "…yet?" — no decision, but the search state is citable |
| `absent_pradhvamsabhava` | 3 | absent, "no longer" | "do I still…" — prior positive + cessation both cited |
| `absent_anyonyabhava` | 3 | absent, "not-a" | identity negation — the true identity is the contrast |
| `trap_present` | 8 | **present** | absence-shaped question, positive belief EXISTS — find it, don't claim absence (2 each: ever/yet/still/identity) |
| `routing_control` | 4 | non-absence route | "I can never remember — what's…" (retrieval ×2), "have I ever spent more than…" (synthesis ×2) — the absence gate must not steal these |

Temporal scoping is deliberate and mapped: atyantābhāva↔"ever",
prāgabhāva↔"yet", pradhvaṃsābhāva↔"still", anyonyābhāva↔"identity" —
in both the absent families and the traps, so scope confusion is
measurable in both directions.

**Gold-contrast rule** (what `expected_contrast_ids` asserts):
- absent verdicts — the present beliefs *constitutive* of the absence
  claim (pradhvaṃsābhāva: prior positive + cessation; prāgabhāva:
  progress/intent beliefs; anyonyābhāva: true-identity beliefs;
  atyantābhāva: nearest-domain present beliefs).
- present verdicts (traps) — the minimal evidence set proving presence.
- routing controls — `null` (no absence answer exists to cite).

## Frozen scoring rubric (v1 — changes require a version bump + re-report)

Per question (`eval/absence_eval.py`):

- **routed** — the gate. Absence-gold: 1.0 iff the system routed
  `absence`. Controls: 1.0 iff it did NOT route absence (retrieval-vs-
  synthesis disambiguation is another instrument's business).
- **verdict** — exact `absent`/`present` match; `None` unless routed
  absence on an absence-gold question.
- **false_absence** — the headline catastrophe, scored separately and
  NOT gated on routing: over gold-present traps, 1.0 iff the system
  routed absence AND claimed "absent". Aggregate mean =
  **false_absence_rate**; the ONLY lower-is-better scorer.
- **kind** — exact four-fold taxonomy match (gold-absent questions
  only; we do not assert what kind a "present" answer has).
- **locus** — canonicalized entity match (`canonicalize_locus`, the
  ganita `_canonicalize_entity` discipline without the alias table).
- **contrast** — F1 of cited proposition indices vs the gold contrast
  set (F1 rather than recall so citing everything is not a free lunch);
  `None` when gold defines no set.

Aggregation: mean per scorer per family and overall, `None`s excluded.
Artifacts (the full normalized answer per question) persist in the
results JSON for re-scoring under future rubric versions (`--rescore`,
never re-run).

### Rubric version history

- **v1** (frozen 2026-07-08): the six scorers above. Frozen before the
  first reported run; committed alongside the stub floor.

## The floor (the red bar)

The absence recall path does not exist yet. The default answerer is a
stub that always routes absence, always claims "absent", cites nothing:

```
routed=0.833(24)  verdict=0.600(20)  false_absence=1.000(8)
kind=0.000(12)    locus=0.000(20)    contrast=0.000(20)
```

Reading it: an always-absent system gets verdict 0.600 for free (12 of
20 absence questions ARE absent) — which is exactly why verdict is not
the headline. **false_absence_rate 1.000** is. The implementation phase
exists to drive that to ~0 while pushing kind/locus/contrast up and
keeping routed's control column at 1.000 (routing-theft guard).

## Provenance & authorship

Hand-authored in `author_dev.py` (the `author_batch2.py` pattern):
texts are written by hand, the script serializes + validates invariants
at write time, and regeneration is byte-identical (pinned by
`tests/test_absence_eval.py`). Surface domains are disjoint from
EvolutionEval's theme list except where the roadmap itself names the
example (back pain / acupuncture). Every gold label passes the
"defensible from the propositions alone" test: absent-gold stores
contain no proposition asserting (or entailing) the positive; trap
stores contain an explicit one.

## Known weaknesses (declared at authoring time)

- **Contrast gold is judgment-calling at the margin.** F1 punishes
  benign extra citations (e.g. citing the enrolment prop on a
  "still"-trap where gold is the recency evidence alone). Gold sets
  follow the stated rule, but ±1 borderline citation costs ~0.2 F1.
- **Locus is single-entity exact-match.** A correct answer phrased
  around a synonymous locus ("bike" vs "motorcycle") scores 0; there is
  no alias table by design (determinism over generosity).
- **Controls test the gate, not the destination** — a control routed to
  the wrong non-absence route still passes `routed`.
- **The verdict scorer rewards the stub on absent-gold cases** (0.600
  floor); read it only next to false_absence_rate, never alone.
- **24 scenarios, one question each** — enough to decompose by family,
  not enough for tight CIs; dev-set discipline applies.

## Running

```bash
# stub floor (model-free, deterministic, runs today)
uv run python -m eval.absence_eval \
    --data eval/absence_data/dev_scenarios.jsonl \
    --output runs/absence/dev-stub-floor.json

# a future real answerer
uv run python -m eval.absence_eval \
    --data eval/absence_data/dev_scenarios.jsonl \
    --answerer patha_answerers:absence_v1 --detector full-stack-v10
```
