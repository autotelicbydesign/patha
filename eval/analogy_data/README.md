# AnalogyEval — can a memory system do analogical recall (upamāna)?

The instrument for the sixth question class (docs/roadmap.md §5):
*"what does this remind me of?"* / *"have I been in a situation like X
before?"* — given a NEW situation in the question, return the
structurally-matching past SITUATION (a multi-proposition episode,
identified by session id) and name the shared structure.

Built instrument-first (working-protocol rule 2): no production analogy
route exists yet. The runner scores any *answerer*; the committed floor
is a deterministic random-session stub.

## The defining invariant: content-word disjointness

An analogy question shares **≤ 2 content tokens** with each gold
analogue session's full text — structure is shared, vocabulary is not.
That is what makes this analogy rather than retrieval: a top-K search
over the same store cannot find the gold by lexical or embedding
proximity to the question's *words*; it has to match the *shape* of the
situation. The invariant is measured, not aspirational —
`content_tokens()` (shared by the weakest scorer and the data tests, so
rule and referee can't drift) enforces it in
`tests/test_analogy_eval.py`, and `author_dev.py` asserts it at write
time.

## Scenario schema (dev-only; no held-out split yet)

```json
{"id": "an-core-01", "family": "core_analogy",
 "propositions": [{"session": "s-joboffer", "text": "...", "asserted_at": "ISO"}],
 "questions": [{
   "q": "…new situation… — have I been in a situation like this before?",
   "type": "analogy", "expected_route": "analogy",
   "gold_analogue_sessions": ["s-joboffer"],      // ranked, primary first
   "shared_structure": ["hard time limit forcing a choice", "..."],
   "surface_trap_session": "s-rungear"            // surface_trap family only
 }]}
```

## Families (16 scenarios)

- **core_analogy (6)** — one gold episode, one structurally-unrelated
  foil episode, two bland filler sessions.
- **surface_trap (4)** — the trap session shares ≥ 3 content tokens with
  the question and strictly more than the gold does; the
  lexically-seductive answer is the wrong one. Trap performance is
  scored separately (`trap_resistance`) so it can't hide in the hit rate.
- **multi_candidate (2)** — two defensible analogues, ranked; the
  structurally-closest must come first.
- **routing_negative (4)** — near-analogy phrasings that belong to other
  pramāṇa (gaṇita / retrieval / narrative); gold is *declining* the
  analogy route.

Every analogy scenario carries **two filler sessions** so the
random-stub chance floor stays informative (measured floor: hit@1
0.250, hit@2 0.583, trap_resistance 0.250, structure_overlap 0.000).

## Frozen scoring rubric (v1 — changes require a version bump + re-report)

- **routed** — routing claim matches gold (bidirectional: claiming
  analogy on a negative control fails)
- **analogue_hit_1 / analogue_hit_2** — hit@k over the gold session set;
  `None` on negative controls; 0.0 on claimed-route-empty-return
- **trap_resistance** — hit@1 on surface-trap scenarios only
- **structure_overlap** — fraction of gold structure phrases "named"
  (≥ half of a phrase's content tokens appear in the answer).
  **The weakest scorer, by design and by documentation**: a lexical
  proxy that misses paraphrase and can be keyword-stuffed. Coarse
  comparison only; never a headline number.

Aggregation: mean per scorer per family and overall, `None`s excluded.
Artifacts persist per question for `--rescore`.

## Running

```bash
uv run python -m eval.analogy_eval \
    --data eval/analogy_data/dev_scenarios.jsonl \
    --answerer stub \
    --output runs/analogy/dev-stub-floor.json
```

Provenance: authored by hand in `author_dev.py` (not a generator — the
script validates invariants and serializes). Regenerating must be
byte-identical to the committed JSONL.
