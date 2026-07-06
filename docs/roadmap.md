# Patha roadmap — executable designs

*Written 2026-07-06, at the close of the v0.11 arc. Each item below is written
to be started cold: motivation from measured data, design, files, tests, the
instrument that judges it, and a definition of done. Read
[The working protocol](#the-working-protocol) first — it is why the numbers in
this repo can be trusted, and it is binding on everything below.*

**Priority order** (by measured impact):

1. [Karaṇa extraction v2](#1-karaṇa-extraction-v2) — the multi-session 0.857
2. [Supersession precision](#2-supersession-precision-the-v012-program) — the 0.230
3. [Composition](#3-composition-time-series-of-sums) — the fourth question class
4. [Anupalabdhi](#4-anupalabdhi-absence-queries) — the fifth
5. [Upamāna & arthāpatti](#5-upamāna--arthāpatti-sketches) — the hexalogy
6. [Frontier Articulation Bridge run](#6-frontier-articulation-bridge-run) — one command, needs a key

---

## The working protocol

The discipline that produced v0.10 → v0.11. It is not decoration; every time
it was followed it caught something (a hash-seed reproducibility bug, a false
supersession-edge class, an over-tagging blind spot), and the one time it was
skipped — hand-waving a precision dip as "benign folding" — a real defect hid
under the hand-wave until a direct question exposed it.

1. **Measure → decompose → fix → guard. Never fix → hope.** No detector or
   walker change without a failing measurement first, a per-family/per-class
   decomposition of *why*, and a regression guard after.
2. **Instrument-first.** Before fixing a behaviour, make sure a scorer can
   *see* the behaviour (the supersession over-tagging existed for months;
   it became fixable the day rubric v2 could measure it). If the fix's effect
   is invisible to every current scorer, build the scorer first.
3. **Frozen stacks.** Published detector stacks (`full-stack-v7/v8/v9`) never
   change behaviour. Fixes ship as a NEW stack in
   `src/patha/belief/detector_factory.py`; old numbers stay reproducible.
4. **Versioned rubrics.** EvolutionEval scorers change only with a
   `RUBRIC_VERSION` bump + re-report. Old runs re-score from persisted
   artifacts (`--rescore`); they are never re-run.
5. **Sealed held-out sets, spent once.** Author scenarios AFTER freezing the
   code under test; commit the sealed file BEFORE the first run (the commit
   hash is the seal); run once per reported config; publish as-run, including
   the numbers you don't like. A spent set folds into dev and never serves as
   evidence again. Batches so far: b1 (16, spent 2026-07-05), b2 (20, spent
   2026-07-06). Next is batch 3, authored fresh after the next fix wave.
6. **Publish the gap.** Dev vs held-out deltas, weakest-stratum numbers, and
   known failure modes go in `docs/benchmarks.md` and the README *at claim
   time*, not when someone asks. The credibility of every strong number in
   this repo rests on the visibly published weak ones.
7. **Keys never through chat; uploads gated.** API keys arrive via env or a
   gitignored `.env` only. TestPyPI first; real PyPI only on an explicit
   "go pypi".

---

## 1. Karaṇa extraction v2

**Motivation (measured).** LongMemEval-S multi-session is the sole weak
stratum (0.857) and `eval/multisession_diagnosis.py` proves it is
**synthesis-bounded, not retrieval-bounded**: retrieval surfaces the gold
sessions; the numeric answer just isn't extracted as tuples. The direct
instrument agrees: `eval/ganita_synthesis_smoke.py` scores **0/8** on the
synthesis-bounded LongMemEval questions (gold answers like $185, $270,
59.6 avg). Extraction quality at ingest bounds every synthesis claim.

**Current state (verified 2026-07-06).**
- `src/patha/belief/karana.py` — `KaranaExtractor` protocol
  (`extract(text, *, belief_id, time) -> list[GanitaTuple]`) with three
  implementations: `RegexKaranaExtractor` (default fallback, weak on dense
  conversational text by its own docstring), `OllamaKaranaExtractor`
  (qwen2.5:7b, temp 0), `HybridKaranaExtractor` (regex finds every amount,
  LLM only labels; 4,000-char chunks). Wired at
  `src/patha/__init__.py:259-263`; CLI `--karana-mode`.
- Gaṇita's false-positive filters (keep them): `_RANGE`, `_HYPOTHETICAL`,
  `_NEGATIVE` marker families with a 50-char window (`ganita.py:129-156`).
- **`src/patha/query/entities.py` (`EntityEnricher`, spaCy
  `en_core_web_sm`) exists but is dead code in the production path** —
  `extract_entity_hints()` is regex+stopwords only. The NER/dependency
  machinery for a better extractor is already a dependency; it's just
  unwired.

**Design — measure first, then three candidate extractors.**
1. **KaranaEval before any fix** (protocol rule 2): hand-author gold tuple
   sets — the 8 synthesis-bounded LongMemEval questions plus ~20 authored
   dense-conversation paragraphs — scoring tuple-level precision/recall
   per config. The three existing configs have *never been benchmarked
   against each other*; that table is the first deliverable and may change
   the plan (if hybrid-14b already clears the bar, the work is packaging,
   not research).
2. **Dependency-parse extractor** (the no-LLM bet): wire the dormant
   `EntityEnricher`; attach amounts to entities via dependency paths
   (nsubj/dobj/prep-of between money/quantity tokens and their governing
   nominals), dates via the existing frontmatter/ISO machinery. Keep the
   gaṇita veto families on top.
3. **Propositions-not-chunks** synergy: the supersession program's
   propositionization (item 2, lever 3) shortens karaṇa inputs too —
   sentence-scale inputs are where the regex path already works.

**Files**: `karana.py` (new `DepParseKaranaExtractor`), `query/entities.py`
(wake the dead code), `eval/karana_eval.py` (new), `__init__.py` (config).

**Instrument & definition of done**: KaranaEval tuple-P/R published per
config; `ganita_synthesis_smoke` ≥ 6/8 with a no-LLM or documented-cost
config; LongMemEval-S multi-session stratum re-run ≥ 0.90 with the winning
config (full battery: the other strata must not move); `_RANGE` /
`_HYPOTHETICAL` / `_NEGATIVE` regression tests stay green.

## 2. Supersession precision — the v0.12 program

**Motivation (measured 2026-07-06).** Rubric v2's `supersession_precision`
scorer: of the beats a timeline tags `revised-from`/`superseded`, the fraction
that are old-ends of expected pairs. Results for `full-stack-v9` (recommended
stack): **dev 0.475, held-out batch 2 0.230**, `progressive_revelation` 0.000
on both. Recall is excellent (0.885 dev / **1.000 held-out**); the stack's
*claims* are the problem — it says "you changed your mind" on arcs that only
*refined* (~4 of 5 held-out claims unwarranted). Same behaviour in v8; not a
v9 regression. Real-data corroboration: the v0.11.0 audit found the same
class on essay imports (1 false / 1 defensible of 2 symmetric adoptions).

**Failure classes, from row-level inspection** (`runs/evolution/v2/*.json`,
`runs/evolution/heldout2-*.json`):

- **Refinement read as revision** (dominant): "been thinking about making
  things with my hands" → a specific craft; "started running twice a week" →
  "signed up for a 10k". NLI sees specificity change as stance change.
- **Additive-phrasing gaps**: sequential/additive vetoes
  (`sequential_detector.py`'s `has_additive_marker`) don't cover these
  phrasings; extend the marker families from the observed rows.
- **Chunk-scale NLI** (essay imports): DeBERTa is sentence-scale; ~2,000-char
  chunks are out-of-domain. Whole-stack issue (17 inherited + 2 symmetric
  edges on the writeups corpus).

**Design (three independent levers, in order of expected value):**

1. **Specificity-increase veto** (new wrapper, or a layer inside the next
   stack): before accepting CONTRADICTS between A (older) and B (newer),
   test whether B *elaborates* A rather than opposes it. Cheap signals that
   need no new models: token-containment / entity-hint overlap where B adds
   modifiers, embedding similarity high (same viṣaya) + NLI *entailment*
   score in the A→B direction competitive with contradiction, no negation or
   cessation markers in B. Architecture: same wrapper pattern as
   `AdhyasaAwareDetector` — veto-only, cannot create edges, so recall is
   structurally protected except where the veto misfires (measure it).
2. **Additive/arrangement marker extension**: harvest the exact false-claim
   sentences from the persisted artifacts (they are all in
   `runs/evolution/**.json` rows), classify their phrasings, extend the
   additive-marker families. Zero-model, low-risk.
3. **Propositionize long chunks at import** (fixes the essay class): split
   imported chunks into atomic propositions before NLI (sentence split +
   coreference-light merging), or skip NLI supersession above a length
   threshold and record `related-to` instead. Import-path only — MCP
   atomic-fact ingest is untouched.

**Ship as `full-stack-v10`** (protocol rule 3). Files:
`src/patha/belief/refinement_veto.py` (new), `sequential_detector.py`
(marker families), `importers.py` (propositionization),
`detector_factory.py` (v10 registration + describe).

**Instrument & validation battery** (all pre-existing):
- EvolutionEval dev under rubric v2: `supersession_precision` is the target
  (v9 baseline 0.475); `supersession` recall must not drop below v9's 0.885
  by more than the veto's measured, documented cost.
- BeliefEval 300-scenario (`uv run python -m eval.belief_eval --detector
  full-stack-v10`): 347/347 is the bar (v9 baseline).
- False-contradiction eval: FP rate ≤ v9's 0.0625.
- Real-data ingest sanity on the writeups corpus (v0.11 audit procedure:
  instrumented detector proxy, judge every new edge semantically).
- **Batch 3** (author fresh AFTER v10 freezes; 5 pr-style refinement arcs
  minimum): held-out verdict on both axes.

**Definition of done**: dev supersession_precision ≥ 0.75 with recall ≥ 0.85
and every guard green, then batch-3 held-out precision materially above 0.230
published as-run. If the veto can't clear that without recall collapse,
publish the trade-off curve and stop at the best point — that result is
also shippable.

## 3. Composition — time-series of sums

**The question class**: "how has my spending on the bike evolved?" —
narrative *shape* over synthesis *content*. Neither existing path alone:
gaṇita answers one number; the narrative walk returns prose beats.

**Current state (verified 2026-07-06) — the data is already there.**
- `GanitaTuple.time` is populated at ingest (`__init__.py:358`,
  `time=at.isoformat()`) and **used by nothing**. Per-period grouping needs
  zero new ingest work.
- `detect_aggregation()` handles six ops (sum/count/avg/max/min/difference)
  in an ordered marker list (`ganita.py:582-601`);
  `answer_aggregation_question(..., restrict_to_belief_ids=...)` already
  supports scoped computation — per-bucket reuse is a parameter away.
- Routing order in `recall()`: gaṇita → narrative → retrieval → backstop
  gaṇita (`__init__.py:440-580`).

**Design.**
1. **Intent**: composition = aggregation marker AND evolution marker
   co-occur ("how has my spending changed/evolved", "spending over time",
   "per month"). Detector in `belief/itihasa.py` style; checked *before*
   the plain gaṇita gate (more specific wins).
2. **Primitive**: group the `(entity, attribute)` tuple list by period
   (month default; parse "per week/month/year" from the question), run the
   detected op per bucket via the existing machinery, emit one
   `NarrativeBeat` per period (value + contributing tuples as receipts) and
   render the through-line deterministically ("rising since March; spike in
   June — 3 purchases"). Zero LLM, same as both parents.
3. **Degradation**: <2 non-empty buckets → fall through to plain gaṇita
   (same graceful-degradation contract as the narrative walk).

**Files**: `belief/composition.py` (new), `ganita.py` (period bucketing on
`GanitaIndex`), `itihasa.py` (numeric beat rendering), `__init__.py`
(routing gate).

**Instrument & definition of done**: a small authored CompositionEval
(dev-only first; EvolutionEval's rubric stays frozen — new capability, new
instrument) scoring routed / bucket-correctness / per-bucket-arithmetic /
trend-statement. Routing-theft guards: EvolutionEval routed stays 1.000,
BeliefEval 347/347, plain-sum questions still route plain gaṇita.
Deterministic run-to-run. Dogfood on a real store before claiming.

## 4. Anupalabdhi — absence queries

**The question class**: "what have I *not* decided about the move?", "have
I ever lived abroad?", "is there anything I haven't tried for the back
pain?" — answers grounded in *absence* of belief, the pramāṇa where
non-perception is itself the evidence.

**Current state (verified 2026-07-06).** The philosophy is built; the
wiring is zero:
- `src/patha/belief/abhava.py` — `classify_abhava(proposition)` implements
  the full four-fold Nyāya taxonomy (prāgabhāva "not yet" /
  pradhvaṃsābhāva "no longer" / anyonyābhāva "I am not a" / atyantābhāva
  "never") with `referenced_state` extraction. Exported, unit-tested
  (`tests/belief/test_abhava.py`), **called by nothing in the recall
  path**. BeliefEval's 12 `abhava_negation` cases score it as plain
  retrieval.

**Design.**
1. **Absence index at ingest**: classify every proposition; persist
   `(kind, referenced_state_canonical, belief_id)` as a JSONL sidecar —
   the exact `GanitaIndex` pattern (`.abhava.jsonl`).
2. **Intent gate** in `recall()`: absence questions are lexically
   distinctive ("what have I not…", "have I ever…", "anything I
   haven't…"). Route after gaṇita, before narrative; measure routing
   confusion rather than assuming it.
3. **The epistemics are the design** (this is the part that makes it a
   pramāṇa and not a feature): *anupalabdhi asserts absence only after
   qualified search*. Concretely: answer "you have not decided X" only
   when (a) the absence index has no positive current belief for X's
   locus, AND (b) the exhaustive scan (same guarantee gaṇita makes)
   confirms no current belief entails X. Temporal scoping by kind:
   atyantābhāva answers "ever" questions; prāgabhāva answers "yet";
   pradhvaṃsābhāva answers "still".
4. **Answer shape**: state the absence, its kind, and the nearest
   *present* beliefs as contrast ("no decision on the flat itself; you
   HAVE decided the budget cap and the postcode").

**Files**: `belief/abhava_index.py` (new, sidecar), `abhava.py` (canonical
locus extraction reusing `_canonicalize_entity`), `__init__.py` (gate),
`belief/layer.py` (exhaustive-scan verification).

**Instrument & definition of done**: AbsenceEval authored dev set (absence
questions with gold kind + gold scoping + trap cases where a positive
belief EXISTS and the correct answer is "yes you have — here it is").
Routing-theft guards green (EvolutionEval routed 1.000, BeliefEval
347/347, gaṇita smoke unchanged). The trap cases are the bar that matters:
a wrong "you never decided" is worse than no feature.

## 5. Upamāna & arthāpatti — sketches

**Status: design sketches only — do not build before 1–4.** They complete the
six-pramāṇa arc (the hexalogy) but have no measured failure driving them yet;
per protocol rule 2, their *benchmarks* must exist before their
implementations.

- **Upamāna (analogical recall)** — "what does this remind me of?" /
  "have I been in a situation like X before?" Routing: comparison-intent
  detector (same shape as `detect_narrative`). Primitive: embedding-space
  nearest-neighbour over *situation* representations (v1 pada vectors of
  episode summaries, possibly session-pooled), returning ranked analogues
  with the shared structure named (common entities/predicates). The honest
  hard part is evaluation: an AnalogyEval needs hand-authored
  gold-analogue sets — start there.
- **Arthāpatti (abductive postulation)** — "what must be true for these two
  beliefs to coexist?" / gap-filling between an observed outcome and prior
  beliefs. Primitive candidate: constrained search over the songline graph
  for minimal connecting assumptions; likely the first pramāṇa that
  genuinely requires an LLM call at recall time — if so, it goes through the
  Articulation Bridge pattern (Patha assembles the structured context, the
  user's LLM postulates, Patha stores the postulate *marked as inferred*,
  never as asserted). The provenance marking is the design's soul: inferred
  beliefs must carry `pramana: arthapatti` and lower confidence, and
  supersession from an asserted belief must always beat an inferred one.

## 6. Frontier Articulation Bridge run

Blocked only on credentials. When `ANTHROPIC_API_KEY` exists (env or
gitignored `.env` — never through chat):

```bash
uv pip install anthropic       # not in the default env (tests skip without it)
set -a; source .env; set +a    # if using .env
uv run python -m eval.run_answer_eval \
    --data data/longmemeval_ku_78.json \
    --llm claude --scorer overlap \
    --output runs/answer_eval/ku78-claude-overlap.json
```

~10 min, <$2. Publish beside the 0.308 local-14B floor in
`docs/benchmarks.md` + README (the floor exists to be beaten; the qwen
number without the frontier number *reads* as weakness). Repeat with
`--scorer judge --judge-llm ollama` optionally for scorer triangulation.

---

*Memory-system state at handoff: v0.11.0 live on PyPI + TestPyPI; 882 tests
green; EvolutionEval at rubric v2 with batches 1–2 spent and published;
detector stacks v7/v8/v9 frozen; `writeups/` (launch drafts) intentionally
uncommitted.*
