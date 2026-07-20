# Benchmarks — full tables and honest caveats

This file holds the detailed benchmark numbers that used to live in the README. The headline summary is in the main README; this is the long-form.

## Quick numbers

### Claim A: Phase 1 retrieval — session-level R@5

| System | R@5 on LongMemEval-KU (78q) |
|---|:---:|
| **Patha Phase 1** | **1.000 (78/78)** |

This is the **retrieval-quality claim.** "Did Phase 1 rank the gold session in the top-5?" Patha Phase 1 gets this right on every one of the 78 questions in the LongMemEval-KU public subset.

### Claim C: Unified `patha.Memory` on full 500q LongMemEval-S (answer-recall)

Phase 1 retrieval + Phase 2 belief layer run together through `patha.Memory()`. Session-level ingest, stub detector, the full 500q LongMemEval-S.

| Configuration | 500q LongMemEval-S |
|---|:---:|
| **Patha unified — answer-recall** | **0.952 (472/496)** |

**Metric:** *answer-recall* — does the gold answer (or one of its synonyms) appear as a substring in Patha's emitted summary text? **No LLM is involved in scoring.** This measures what the Belief Layer surfaces; it does *not* measure end-to-end answer accuracy through an LLM. For that, see Phase 3 / Articulation Bridge below.

4 of 500 questions skipped due to a datetime-tz edge case; scored over 496.

Reproducible: `eval/longmemeval_integrated.py --data data/longmemeval_s_cleaned.json --granularity session`.

Per-stratum on the 500q run:

| Stratum | Patha | Note |
|---|:---:|---|
| single-session-assistant | 1.000 (55/55) | perfect |
| single-session-preference | 1.000 (30/30) | perfect |
| **knowledge-update** | **0.987** (76/77) | strong |
| single-session-user | 0.986 (69/70) | near-perfect |
| temporal-reasoning | 0.977 (128/131) | strong |
| **multi-session** | **0.857** (114/133) | **sole weakness — drags overall down** |

**Five of six strata are 0.977–1.000.** The remaining gap is entirely in the multi-session stratum.

### Songline walks — tried it, didn't help. Here's why.

The Phase 1 bridge was originally calling `retrieve()` without a `songline_graph`, silently disabling Pillar 2 (Aboriginal songline traversal) in the unified pipeline. Hypothesis: enabling it should lift multi-session retrieval by walking shared-entity / shared-session edges across the haystack.

**Re-ran 500q with songlines enabled. Score: 472/496 = 0.952, identical to the pre-songline run. Every stratum unchanged, including multi-session (114/133 = 0.857).**

Why it didn't help: the multi-session failures **aren't retrieval failures**. Inspecting the 19 multi-session misses, they're all **arithmetic-synthesis questions**:

- "How many weeks did it take me to watch all the MCU and Star Wars?" → gold: 3.5 weeks
- "How much total money have I spent on bike expenses since the start of the year?" → gold: $185
- "How many hours of jogging and yoga last week?" → gold: 0.5 hours

The gold answer **never appears verbatim in the source text**. "$185" isn't written anywhere — it has to be derived by adding "$50 saddle + $75 helmet + $30 lights + $30 gloves" across 4 sessions.

Our scoring does token-overlap on the answer. No retrieval improvement can surface a string that doesn't exist in the data. Patha's unified pipeline retrieves all 4 sessions correctly, but the summary doesn't compute the sum. We don't have an LLM synthesis step.

### How v0.10 closes this honestly

The synthesis-bounded gap motivates the v0.10 architectural distinction: **Patha separates retrieval queries from synthesis queries.**

- **Retrieval** (perception, *pratyakṣa*): "what did I say about the saddle?" → Phase 1 → Phase 2 → summary.
- **Synthesis** (inference, *anumāna*): "how much have I spent on bikes total?" → gaṇita queries the belief store directly. Pure deterministic arithmetic over preserved tuples. No LLM call at recall.

The architectural correctness is independent of extractor quality. Quality scales with the karaṇa extractor — and as of 2026-07-08 that scaling is **measured, not assumed**:

### KaranaEval — the first extractor head-to-head (2026-07-08)

Tuple-level precision/recall on 26 hand-labeled dense-conversation cases across 15 failure families (`eval/karana_eval.py`; gold rules in `eval/karana_data/README.md`). `forbidden_hit` = fraction of cases where the extractor fabricated a value from a range, hypothetical, refund, colloquial quantity, or numeric distractor — lower is better.

| config | precision | recall | F1 | forbidden_hit |
|---|---|---|---|---|
| regex (v0.11 default) | 0.560 | 0.719 | 0.768 | 0.636 |
| ollama qwen2.5-14B | 0.750 | 0.912 | 0.892 | 0.333 |
| hybrid-14B | 0.769 | 0.559 | 0.967 | 0.250 |
| **depparse (new, zero-LLM)** | **1.000** | **1.000** | **1.000** | **0.000** |

`DepParseKaranaExtractor` (karaṇa v2): dependency-parse attachment (amounts claim their nearest prep objects; copular subjects; charge-verbs prefer their subject) with **clause-level vetoes** — the regex extractor's forbidden_hit 0.636 was a character-window problem, and clauses are the right window. Two honest caveats: the depparse line was iterated against this dev set (read it as "no known misses", not "generalises" — the BeliefEval lesson); the LLM configs were NOT tuned, so their gap is honest. Even a 14B model fabricates facts from hypotheticals a third of the time; the parser vetoes them structurally.

**The external gate, as-run** (`ganita_synthesis_smoke`, 8 synthesis-bounded LongMemEval questions the extractor has never seen): regex 0/8 → depparse **1/8 hits + 2 near**. The first run exposed assistant-speech contamination (advisory amounts — "free shipping over $75 is a great strategy" — summed into user totals); fixed by a speaker gate (karaṇa extracts from USER turns only; tuple noise fell 459 → 82 per store). NOTE the deliberate asymmetry with the Claim-D ingest fix below: belief INGESTION keeps both speakers (retrieval needs assistant answers — the 0.841 bug), while tuple EXTRACTION is user-only (assistant illustrations are nobody's spending). Remaining failure classes, named from the run artifacts: user-pasted third-party content ("rewrite this article…" bringing foreign numbers), cross-turn count semantics (cumulative "23 pieces" vs summed snapshots), and under-recall on dense turns. The ≥6/8 definition-of-done stays OPEN — depparse ships opt-in (`PATHA_KARANA=depparse`, `--karana depparse`), regex stays default until the gate clears.

See `docs/innovations.md` for the full architectural explanation and `docs/phase_3_plan.md` for the end-to-end answer-evaluation plan.

### Claim D: Stratified 300q LongMemEval-S (subset of 500q)

Same eval on a 300q stratified sample (reproducible with `eval/make_stratified.py --n 300`). Result: **0.950 (283/298)** — consistent with the 500q at 0.952.

**Note on an earlier eval bug:** a prior 300q run scored 0.841 because it ingested only USER turns, missing the `single-session-assistant` stratum where the gold fact was stated by the assistant (e.g. "what did you recommend for dinner?"). Fixed in commit `d44a223` by ingesting both sides of the conversation; the 0.950 / 0.952 numbers above are post-fix. The old number was an artifact of our pipeline, not the architecture.

### Claim B: Unified Patha answer-recall — `patha.Memory` public API

The public developer API (what you get when you `import patha; patha.Memory()`) ingesting every user turn, then answering the question through Retrieval Layer → Belief Layer → structured summary. **The metric is answer-recall** (gold answer appears as substring in Patha's emitted summary), **not end-to-end through an LLM.** For end-to-end-through-an-LLM see Phase 3 / Articulation Bridge.

| Configuration | Answer-recall (78q) | Notes |
|---|:---:|---|
| Session-level ingest (one belief per session) | **0.987 (76/77)** | through the public developer API |
| Turn-level ingest, `phase1_top_k=100` | 0.455 (35/77) | loses signal to reranker on fragmented turns |
| Turn-level ingest, default `phase1_top_k=20` | 0.325 (25/77) | early over-trimming |
| Stub baseline (no supersession, keep everything) | 0.795 (62/78) | lexical-overlap upper bound, from v0.7 |

**How to reproduce: `uv run python -m eval.longmemeval_integrated --granularity session`.**

### How to pick ingest granularity

- **Use session-level** when you're ingesting whole conversations you already have (transcripts, Slack channels, LongMemEval haystacks). Each session's text becomes one belief. Phase 1 retrieval gets session-aware chunks. `memory.remember(session_text, session_id=...)`.
- **Use turn-level** when the user is asserting individual facts over time (the personal-memory / MCP case — "I live in Lisbon", "I'm vegetarian"). Each fact is its own belief; Phase 2's supersession/contradiction handling shines here. This is the default shape of `memory.remember(one_fact)`.
- **Use both**: ingest the concatenated conversation for retrieval, and individual extracted facts for belief management. Hybrid chunking.

The 53pp gap between turn-level (0.455) and session-level (0.987) on LongMemEval-KU is entirely explained by **chunk size matching the benchmark's assumption.** LongMemEval is authored as session-level retrieval; turn-level splits the "charity 5K + 25:50 personal best" context across 7 competing chunks. Session-level keeps them together.

On a benchmark that tests belief-revision (like our BeliefEval), turn-level wins because that's the right granularity for supersession. On LongMemEval-KU, session-level wins because that's the right granularity for retrieval. Neither granularity is "wrong" — the unified pipeline just lets the caller pick.

### The gap between A and B is real and honest

The two claims differ by **54.5 percentage points** on the same benchmark. Three compounding reasons:

1. **Granularity mismatch.** Phase 1's 100% was measured with one embedding per SESSION (~10 turns packed together). The unified `Memory` stores one belief per TURN. The gold session's "charity 5K" + "25:50 personal best" context gets split across 7 separate beliefs that now compete against each other and against similar turns from unrelated sessions. LongMemEval is authored assuming session-chunking.

2. **Reranker can rank the gold below the bait.** On Q1 traced in detail: the cross-encoder reranks 7 turns from the gold session highly (scores 4–7), but the specific turn containing "25:50" scores -10.998, pushed to rank 14 — behind near-duplicate phrasings from other sessions. Even at top_k=100 the answer-containing turn sometimes falls outside the returned set.

3. **Phase 2's value isn't measured here.** LongMemEval tests retrieval, not belief supersession or contradiction handling. The contradiction-detection machinery (adhyāsa, numerical, sequential, learned classifier) adds no signal for these questions because no belief contradicts any other — users aren't revising their 5K time.

### Vedic gaṇita (procedural arithmetic) — scaffolding, honest about its limit

Most multi-session failures aren't retrieval failures — they're synthesis failures. "$185 total bike spend" never appears in the source; it's $50 + $75 + $30 + $30 across 4 sessions. The Vedic tradition has a principled answer: *gaṇita* (auxiliary mathematics from the Sulbasūtras). Procedural rule-application on preserved facts, not interpretation. Aboriginal songlines have a parallel: increase-walks include totalling sites where the songkeeper recounts everything traversed along the path.

`patha.Memory` ships a gaṇita layer (`src/patha/belief/ganita.py`) with this shape:

- **Ingest-time:** regex-based extraction of (entity, attribute, value, unit, time) tuples. Currency, durations, percentages, counts. Sentence-scoped entity binding to avoid cross-topic contamination.
- **Sidecar JSONL index** keyed by (entity, attribute). Append-only, mirrors the BeliefStore's persistence pattern.
- **Query-time:** detect aggregation operator from question wording; restrict to retrieved-belief tuples (Phase 1 scopes topic); run procedural arithmetic; return value + contributing belief_ids.

**Status: works on clean inputs, doesn't yet help on dense conversational benchmarks.**

- 24/24 unit tests pass.
- End-to-end test on the canonical case (4 hand-crafted bike-expense beliefs) returns $185.0 USD with all 4 source belief ids. ✓
- Smoke test on the 8 synthesis-bounded LongMemEval-S questions: **0/8**. The regex extractor pulls in too many irrelevant currency mentions because real conversational sessions contain many "I spent $X on Y" sentences across topics.

What would make gaṇita actually close the LongMemEval synthesis gap:

1. **NER + dependency parsing** to bind currency/values to syntactic objects of verbs (vs. nearby nouns by coincidence). ~2-3 weeks of engineering. Tradition-aligned, no LLM.
2. **An LLM extractor** for the tuples (cleaner extraction). Brings back the LLM dependency we explicitly want to avoid.

Honest position: the layer is *architecturally right* (the right shape for tradition-aligned synthesis without an LLM), but the *regex implementation* is the wrong fidelity for dense natural-language haystacks. Shipped as scaffolding; the real research problem is robust no-LLM tuple extraction.

### Token economy (when paired with Claude or any LLM)

Measured per-query (`eval/token_economy.py` — 4-char/token approximation, calibrated against tiktoken):

| Strategy | Tokens per query | Compression vs naive RAG |
|---|:---:|:---:|
| **naive_rag** (dump raw conversation context to LLM) | 285.9 | 1.0× baseline |
| **structured** (Patha's compressed summary) | 64.6 | **4.5× reduction** |
| **direct_answer** including gaṇita (no LLM call needed) | **0** | **∞** |

For aggregation questions, Patha's gaṇita layer returns the answer directly without an LLM call — token cost on those questions drops to **zero**. For other questions where a structured summary is sent to the LLM, ~4.5× reduction in input tokens vs dumping raw history.

### Plasticity (neuroplasticity-inspired) — still on by default

All five plasticity mechanisms are wired and active in the unified pipeline:

| Mechanism | Role | Default |
|---|---|:---:|
| LTP (long-term potentiation) | Reinforce confidence on repeated assertion | on |
| LTD (long-term depression) | Decay unused beliefs over time | on |
| Hebbian association | Co-retrieval edges between beliefs | on |
| Homeostatic regulation | Bound max/min confidence ratio | on |
| Synaptic pruning | Archive deeply-superseded chains | on |

Verified end-to-end with `tests/belief/test_plasticity_wiring.py` (4 tests) and `tests/belief/test_plasticity.py` (10 tests).

### Honest summary

- **Phase 1 retrieval alone, session-level R@5: 1.000.** Perfect retrieval on the LongMemEval-KU public subset.
- **`patha.Memory` answer-recall, session-level ingest: 0.987.** Gold answer appears in Patha's summary in 76 of 77 KU questions. *(This was previously labelled "end-to-end" — corrected: no LLM is involved in scoring.)*
- **`patha.Memory` answer-recall, turn-level ingest: 0.455.** Substantially worse, because LongMemEval assumes session-level chunks. Turn-level is the right shape for personal-memory / MCP use, which LongMemEval doesn't measure.
- **BeliefEval (our supersession benchmark), turn-level: 1.000.** Different test, different granularity match.
- **Articulation Bridge end-to-end through an LLM, KU 78q (qwen2.5:14b local, token-overlap ≥0.6 — LongMemEval-S official scorer): 0.308 (24/78).** First real-LLM measurement; frontier-LLM measurement pending.

Reproduce:
```bash
uv run make eval-ku                                                       # Phase 1 retrieval (R@5)
uv run python -m eval.longmemeval_integrated --granularity session        # answer-recall, session
uv run python -m eval.longmemeval_integrated                              # answer-recall, turn
uv run python -m eval.belief_eval                                         # BeliefEval
uv run python -m eval.run_answer_eval --data data/longmemeval_ku_78.json \
    --llm ollama --ollama-model qwen2.5:14b-instruct --scorer overlap \
    --output runs/answer_eval/ku-qwen14b-overlap.json                      # Articulation Bridge end-to-end
```

## EvolutionEval — narrative evolution (Phase 4; category-defining)

**The first benchmark measuring whether a memory system can reconstruct how a theme evolved** — ordered beats, origin identification, revision tagging, distractor exclusion. Nothing external measures this (LongMemEval's temporal-reasoning is timestamp arithmetic; knowledge-update is single-fact replacement). Instrument documentation, scenario schema, the four families, and the frozen rubric (v2 — version history inside): [eval/evolution_data/README.md](../eval/evolution_data/README.md).

**Protocol**: 36 templated dev scenarios (tuning allowed) / 16 hand-written sealed held-out scenarios in disjoint domains (release reports only; runner refuses them without `--include-heldout`). Walker frozen before scenario authoring; rubric frozen before the first reported run. Run-to-run determinism verified (two full runs, identical scores; the benchmark itself caught and forced the fix of a hash-seed-dependent anchor-ordering bug on day one).

**Dev set, current numbers** (rubric v1, `stub` detector, real MiniLM + real songline graph):

| family | routed | coverage | precision | ordering | origin | supersession |
|---|---|---|---|---|---|---|
| progressive_revelation | 1.000 | 0.775 | 0.775 | **1.000** | **1.000** | — |
| multi_factor_change | 1.000 | **1.000** | 0.750 | **1.000** | **1.000** | 0.000 |
| perspective_shift | 1.000 | **1.000** | **1.000** | **1.000** | **1.000** | 0.000 |
| reversed_belief_chain | 1.000 | 0.917 | 0.688 | **1.000** | **1.000** | 0.000 |
| **overall** | **1.000** | 0.919 | 0.799 | **1.000** | **1.000** | 0.000 (26) |

Honest reading:
- **ordering = 1.000 and origin = 1.000 across all 36 questions** — when the walk returns a timeline, the sequence is always right and the first beat is always the true origin. The core temporal claims hold on authored scenarios.
- **supersession = 0.000 is the quantified headroom, by design.** The scenarios' expected revisions are *reinterpretive* (perspective shifts) or *causal* (multi-factor) — no lexical contradiction — and the stub detector never fires on them. This is the dogfood N1 finding turned into a number. **The detector sweep below answers it.**
- **precision 0.75–0.80 outside perspective_shift** — distractor leakage; the knob-sweep target. **The threshold sweep below answers it.**
- Held-out numbers are **deliberately unreported** here — the set stays sealed until a release report (v0.11), where dev and held-out publish side by side and the gap is the honest generalization signal.

### Sweeps (dev-only, per protocol)

**Experiment 1 — topic-cluster similarity threshold** (`PATHA_TOPIC_THRESHOLD`, stub detector held fixed):

| threshold | coverage | precision | ordering | origin |
|---|---|---|---|---|
| 0.65 | 0.912 | 0.792 | 1.000 | 1.000 |
| 0.55 (launch baseline) | 0.919 | 0.799 | 1.000 | 1.000 |
| 0.45 | 0.942 | 0.819 | 1.000 | 1.000 |
| **0.35** | **0.951** | **0.826** | **1.000** | **1.000** |
| 0.25 | 0.965 | 0.832 | 1.000 | **0.972** ⚠ |

Coverage and precision improve **together, monotonically** as the threshold loosens — looser clusters merge more paraphrase beats into anchor clusters, and those true beats displace distractors from the fixed beat budget (no coverage/precision trade-off in this regime). The curve cracks at 0.25: **origin identification is the first casualty** (1.000 → 0.972) as clusters broaden enough to pull a wrong first beat. **Default set to 0.35** — the best point that preserves both perfect temporal claims. (Shipped: `topics.py` / `phase1_bridge.py` defaults.)

**Experiment 2 — detector sweep** (`stub` → `full-stack-v8`, threshold held at 0.55): **the first measurement of reinterpretation-detection in a memory system.**

| family | supersession (stub) | supersession (v8) |
|---|---|---|
| perspective_shift (reinterpretive reversals — the dogfood N1 finding) | 0.000 | **0.875** |
| multi_factor_change (causal revisions) | 0.000 | **0.900** |
| reversed_belief_chain (nonmonotonic X→Y→X′) | 0.000 | **0.688** |
| **overall** | **0.000** | **0.827** |

Nothing else regressed: ordering/origin/routed stayed 1.000, coverage ticked up (0.919 → 0.926). The "reversal without lexical contradiction" gap is real, measurable, and mostly closed by the production NLI stack; nonmonotonic chains remain the hardest case.

**Recommended production config — the wins compose** (`full-stack-v8` + threshold 0.35):

| routed | coverage | precision | ordering | origin | supersession |
|---|---|---|---|---|---|
| 1.000 | **0.965** | 0.811 | **1.000** | **1.000** | **0.808** |

Highest coverage of any measured config, both temporal claims perfect, and revision tagging at 0.808 — vs the all-stub launch baseline's 0.919 / 0.000. All artifacts under `runs/evolution/`; every row reproducible via the commands above with `--detector` / `PATHA_TOPIC_THRESHOLD` set accordingly.

### The held-out reveal (unsealed 2026-07-04 — one shot, numbers frozen as-run)

The 16 hand-written scenarios in disjoint domains, never run before this date, never tuned against, run exactly once per reported config under the frozen rubric. **Whatever they showed, they publish.** This is the honest-generalization number BeliefEval never had (its 1.000 dev collapsed to 0.885 external because the detectors were tuned on the benchmark).

| config | set | routed | coverage | precision | ordering | origin | supersession |
|---|---|---|---|---|---|---|---|
| stub @ 0.35 (shipped default) | dev | 1.000 | 0.951 | 0.826 | 1.000 | 1.000 | 0.000 |
| stub @ 0.35 | **held-out** | **1.000** | **0.944** | **0.828** | **1.000** | **1.000** | 0.000 |
| v8 @ 0.35 (recommended) | dev | 1.000 | 0.965 | 0.811 | 1.000 | 1.000 | 0.808 |
| v8 @ 0.35 | **held-out** | **1.000** | **0.944** | **0.791** | **1.000** | **1.000** | **0.625** |

**The generalization verdict:**
- **The temporal core generalizes with ZERO gap.** Routing, ordering, and origin identification are perfect on all 52 questions across both sets — dev and held-out, both configs. The structural overfit guards (walker frozen before authoring, rubric frozen before running, sealed split) did their job.
- **On the shipped default config, the dev/held-out gap is statistical noise**: coverage −0.007, precision +0.002 (held-out slightly *higher*).
- **The one real gap is v8 supersession: 0.808 → 0.625 (−0.183)**, concentrated in `multi_factor_change` (0.900 → 0.500).

**Failure decomposition (the v0.12 fix list):**
1. **Causal-revision detection is phrasing-sensitive** (2 of 4 mf pairs missed): "I play tennis with Dad every Saturday" → "Saturdays are doubles with Dad coaching; I hit with the ball machine Wednesdays" is a revision with no lexical contradiction. The NLI stack needs arrangement-change semantics, not just negation.
2. **The nonmonotonic return-link misses** (every rb chain scored exactly 1 of 2 pairs): X→Y fires ("quit coffee" contradicts "coffee non-negotiable"), but Y→X′ doesn't — "back on coffee, one cup before noon" doesn't lexically contradict "quit entirely." Return-with-nuance is a distinct detection class.
3. **Bug caught by the unseal**: theme canonicalization false-plural stemming — "tennis" → "tenni", "thesis" → "thesi" (`_canonicalize_entity` strips trailing *s* from non-plurals). The substring gate survived by accident ("tenni" ⊂ "tennis"); entity-channel exact-match lookups would not.

**Protocol note.** These held-out numbers are frozen as-run for this report. The fixes above will be built from the general failure classes, measured on dev, and validated on a **future held-out batch 2** authored after the fixes ship. This held-out set is now spent as an unseen instrument and will be folded into dev for future work.

### The fixes — `full-stack-v9` (post-reveal; v7/v8 frozen)

All three failure classes from the reveal, shipped as a new detector stack (published v8 numbers stay reproducible):

1. **Stemming bug** — `_canonicalize_entity` no longer strips the trailing *s* from -is/-us/-ss words ("tennis", "thesis", "status"); real plurals still normalize.
2. **`SymmetricContradictionDetector`** — wraps the NLI core only. Belief contradiction is symmetric; NLI is premise/hypothesis-asymmetric (diagnosed pairs scored CONTRADICTS ≥ 0.94 in one direction, NEUTRAL ≥ 0.95 in the other). Reverse-direction contradictions are adopted at a conservative ≥ 0.90 bar. Direction-*dependent* outer detectors (sequential, numerical, learned) stay one-way by design.
3. **`RevisionPatternDetector`** — three marker families the stack had no coverage for: **resumption** (cessation → "back on / X again now"), **settlement** ("landed on / on my own terms"), **arrangement** ("is/are now", "we do X now"). Same additive architecture as the sequential detector: embedding topic-overlap gate, additive-marker veto, negated-resumption veto, confidence 0.84.

**Validation (dev + guards).** The first v9 iteration showed a precision dip (0.811 → 0.797) that was initially mis-read as benign "supersession folding." Beat-set diffing traced it to a real defect: **ungated symmetric-NLI reverse adoptions created false supersession edges between distractors and on-theme beliefs** — DeBERTa is confidently wrong on some completely unrelated pairs (measured: "I finally fixed the squeaky hinge" vs an on-theme critique reflection, CONTRADICTS 0.992 in reverse) — and the lineage fold then pulled the falsely-superseded distractors into timelines. A confidence bar cannot filter a model that is *sure*; the fix is the codebase's standard **topic-overlap gate** (embedding sim ≥ 0.35, same as sequential/revision detectors) on reverse adoptions: contradiction presupposes a shared locus (*virodha* requires a common *viṣaya*). The gate honestly costs ~0.06 supersession (a few true reinterpretation pairs whose surface content diverges below the gate) in exchange for eliminating store-corrupting false lineage — the right trade, and the threshold is the codebase-wide standard, not tuned to this set.

| measurement | v8 | v9 ungated | **v9 (gated, final)** |
|---|---|---|---|
| EvolutionEval dev — supersession | 0.808 | 0.942 ⚠ (incl. false edges) | **0.885** (mf 1.000 · ps 0.875 · rb 0.750) |
| EvolutionEval dev — coverage | 0.965 | 0.972 | **0.972** |
| EvolutionEval dev — precision | 0.811 | 0.797 ⚠ (distractor leak) | **0.812** |
| EvolutionEval dev — ordering / origin / routed | 1.000 | 1.000 | **1.000** |
| BeliefEval (regression guard) | 1.000 | 1.000 | **1.000** |
| False-contradiction FP rate (guard) | 0.0625 | 0.0625 | **0.0625 — identical**: same single pre-existing FP (fc-07, v7-era sequential detector); the v9 additions introduced **zero** new false positives |

**v9 (gated) strictly dominates v8 on every metric.** Generalization claims for v9 await **held-out batch 2** (to be authored fresh, after this ships) — the spent batch cannot be reused as evidence, per protocol.

### Rubric v2 — supersession *precision* (2026-07-06)

The v0.11.0 real-data audit found supersession edges the system created that **no scorer could see**: rubric v1 measured supersession *recall* only (were expected revisions tagged?), so unexpected edges were invisible. Rubric v2 adds `supersession_precision`: of the beats a timeline *tags* as revised/superseded, what fraction are old-ends of expected pairs? Per protocol this is a version bump; all v1 scorers are byte-identical, and the numbers below are **re-scored from persisted run artifacts** (deterministic, verified twice) — nothing was re-run, no held-out scenario was touched.

| config | set | supersession (recall) | **supersession_precision (new)** |
|---|---|---|---|
| stub @ 0.35 (shipped default) | dev | 0.000 | — (never tags) |
| stub @ 0.35 | held-out b1 | 0.000 | — (never tags) |
| v8 @ 0.35 | dev | 0.808 | **0.449** |
| v8 @ 0.35 | held-out b1 | 0.625 | **0.494** |
| v9-gated @ 0.35 (recommended) | dev | 0.885 | **0.475** |

**The finding, stated plainly: the NLI stack over-tags revision on arcs that never reversed.** Family decomposition (v9-gated, dev): `reversed_belief_chain` 0.750, `multi_factor_change` 0.658, `perspective_shift` 0.512, `progressive_revelation` **0.000**. The pr family has *zero* expected pairs by design — refinement ("been thinking about making things with my hands" → specific craft) is not revision — yet the stack tags refinement beats `revised-from` on nearly every pr timeline, and in one case tagged the off-theme distractor ("the dentist moved my appointment to Thursday"). Sequential additive events ("started running twice a week" → "signed up for a 10k") also draw edges: phrasings the additive/sequential vetoes don't cover.

Three honest observations:

1. **This is not a v9 regression** — v9-gated scores *above* v8 on the new axis (0.475 vs 0.449 dev), and the symmetric-adoption topic gate holds. The over-tagging is inherited base-NLI supersession behavior, present in every published config, measurable only now.
2. **The false-contradiction guard (FP 0.0625) did not generalize to this distribution.** That eval's 20 hand-crafted pairs cover marker-driven classes; scenario-corpus refinement arcs are a different distribution, and the instrument that sees them is this scorer.
3. **User-visible meaning**: a narrative timeline may claim "you changed your mind" where you actually *sharpened* it. Recall of true revisions is strong (0.885); precision of the claim is roughly a coin flip (0.475). Both numbers are the product truth; quoting the first without the second would be cherry-picking.

**Fix program (v0.12, instrument-first — no detector changes ship today):** refinement-vs-revision discrimination (specificity-increase veto), additive-phrasing coverage, and chunk-scale propositionization, each measured against this scorer on dev and validated on held-out batch 2+. v7/v8/v9 stay frozen; fixes ship as v10.

### Held-out batch 2 — the v9 generalization verdict (2026-07-06)

20 hand-written scenarios (5 per family) in domains disjoint from dev *and* batch 1, authored **after** the v9 stack froze (v0.11.0) and **after** rubric v2 froze (`fddea0b`), sealed at commit `93e8765` before any run, run exactly once per reported config. The batch was designed to probe exactly what batch 1's failure decomposition named: resumption/settlement/arrangement phrasings in unseen domains (v9's new detectors' first held-out test), harder token-less paraphrase origin beats, and zero-expected-pair refinement arcs (the supersession-precision probes). Numbers as-run:

| config | routed | coverage | precision | ordering | origin | supersession (recall) | supersession_precision |
|---|---|---|---|---|---|---|---|
| stub @ 0.35 (shipped default) | 1.000 | 0.848 | 0.975 | 1.000 | **0.850** | 0.000 | — (never tags) |
| v8 @ 0.35 | 1.000 | 0.925 | 0.784 | 1.000 | 1.000 | 0.967 | 0.230 |
| **v9-gated @ 0.35 (recommended)** | 1.000 | 0.925 | 0.784 | 1.000 | 1.000 | **1.000** | 0.233 |

**The verdict, in order of importance:**

1. **The v9 fixes generalize — the fix loop is closed.** Batch 1 exposed v8 supersession recall collapsing 0.808 → 0.625; the three fixes were built from that decomposition, validated on dev, and batch 2 is their first unseen test: **v9 recall 1.000** (15 applicable pairs), including every reversed-belief-chain return-link (rb 1.000 vs v8's 0.900 — precisely the resumption/settlement class `RevisionPatternDetector` was built for, firing correctly in domains it has never seen). Decompose → fix → validate-on-fresh-sealed-data: the whole discipline paid off in one number.
2. **The temporal core holds for the third consecutive set.** Routing 1.000 and ordering 1.000 on all 20 questions, every config — now demonstrated on dev (36q), batch 1 (16q), and batch 2 (20q).
3. **Batch 2 found the shipped default config's edge — publishing it.** With `stub` (no supersession edges), coverage drops to 0.848 and **origin identification cracks to 0.850 overall / 0.600 on progressive_revelation**: batch 2's token-less paraphrase origin beats (e.g. "I keep stopping on the roof to stare at the sky" for an astronomy arc) are genuinely harder than batch 1's, and the topic-channel walk alone misses some of them. The NLI configs score origin 1.000 partly *because* their (often false) supersession edges pull origin beats into the walk — false lineage accidentally helps recall-side metrics while beat precision pays for it (0.784 vs stub's 0.975). Batch 2 is the first set to cleanly price this trade.
4. **Supersession precision degrades further out-of-domain: 0.230** (dev was 0.475), with claims on all 20 questions and progressive_revelation at 0.000 again — on fresh refinement arcs, roughly 4 of 5 revision claims are unwarranted. This is now a held-out-validated systemic finding and **the v0.12 fix program's target number**: refinement-vs-revision discrimination, measured by this scorer, dev-first, validated on a future batch 3.

**Protocol note.** Batch 2 is now spent as an unseen instrument (folded into dev for future work). Batch 3, when needed, gets authored fresh after the next fix wave ships.

Reproduce:
```bash
uv run python -m eval.evolution_eval \
    --data eval/evolution_data/dev_scenarios.jsonl \
    --output runs/evolution/dev.json
# re-score any stored artifact under the current rubric (no re-run):
uv run python -m eval.evolution_eval \
    --data eval/evolution_data/dev_scenarios.jsonl \
    --rescore runs/evolution/dev-v9b-thr035.json \
    --output runs/evolution/v2/dev-v9b-thr035.v2.json
```

## Phase 3 — End-to-end answer evaluation

Phase 1 measures retrieval (R@k) and Phase 2 measures supersession (did the right belief end up `current`?). Both are surrogates. The product question is: **given Patha's output, does the user's LLM produce the right answer?**

That's what Phase 3 measures. Plan: `docs/phase_3_plan.md`. Engine: `eval/answer_eval.py`. Runner: `eval/run_answer_eval.py`.

### Three knobs

- **LLM** — `null` (deterministic baseline), `claude` (Anthropic Messages API, needs `ANTHROPIC_API_KEY`), `ollama` (local).
- **Prompt template** — what fields of `Recall` go to the LLM: `{question}`, `{summary}`, `{ganita}`, `{current}`, `{answer}`. Default template is in `eval/run_answer_eval.py` (`DEFAULT_TEMPLATE`).
- **Scorer** — `normalised`, `numeric` (5% tol, falls back to normalised), `overlap` (token overlap ≥0.6, LongMemEval-style), `embedding` (MiniLM cosine ≥0.85), `judge` (LLM-as-judge with one-word MATCH / NO_MATCH verdict).

### Floor — NullTemplateLLM baseline on LongMemEval-KU

`NullTemplateLLM` is a deterministic stub: it echoes the first dollar amount, the first number, or the start of the memory text. It cannot reason — it's the **floor** that any real LLM should beat.

| Scorer    | KU (78q) accuracy | Wall time |
|-----------|:-----------------:|:---------:|
| numeric   | **5/78 = 0.064**  | 11.7 s    |
| overlap   | **2/78 = 0.026**  | 15.1 s    |

Per-strategy on the numeric run:
- ganita (synthesis intent): 4/41 = 0.098
- structured (retrieval intent): 1/37 = 0.027

The numeric floor (6.4%) is what NullTemplateLLM gets by accidentally echoing matching numbers; the overlap floor (2.6%) is stricter because it requires the *content tokens* of the gold (e.g. "minutes", "suburbs") to appear in the candidate, which a number-echoer rarely produces.

Reproduce:
```bash
uv run python -m eval.run_answer_eval \
    --data data/longmemeval_ku_78.json \
    --llm null --scorer numeric \
    --output runs/answer_eval/ku-null-numeric.json
```

### First real-LLM measurement — qwen2.5:14b on LongMemEval-KU

**Setup:** local Ollama running `qwen2.5:14b-instruct` (Q4_K_M quantization, ~9 GB, GPU-resident). 78 questions, default Articulation Bridge prompt template, Patha's structured summary + gaṇita context piped to the model. Wall time 12 minutes (10 s/question).

| Scorer | Accuracy | Notes |
|---|:---:|---|
| numeric (5% tol) | 12/78 = 0.154 | strict; fails on non-numeric gold paraphrased correctly |
| normalised_match | 2/78 = 0.026 | strictest; fails on any prose wrapping |
| **token_overlap ≥0.6 (LongMemEval-S official)** | **24/78 = 0.308** | **the canonical apples-to-apples number** |
| token_overlap ≥0.4 | 34/78 = 0.436 | looser overlap threshold |
| embedding_cosine ≥0.85 | 6/78 = 0.077 | over-strict at default threshold |
| embedding_cosine ≥0.55 | 36/78 = 0.462 | semantic match, threshold tuned for short answers |

Per-strategy on the canonical token-overlap run:
- ganita (synthesis intent): 5/41 = 0.122
- structured (retrieval intent): 19/37 = 0.514

**What this number tells us, and what it doesn't:**
- The Articulation Bridge measurement framework runs end-to-end, exits cleanly, and produces a real number above the NullTemplateLLM floor. The plumbing works.
- 0.308 with a 14B local model is a *floor for "real LLMs in the loop,"* not a ceiling. qwen2.5:14b is small relative to frontier models (Claude Sonnet 4, GPT-4o, Gemini 2.5 Pro), and inspection of failures shows two clusters: (a) hallucinated specific values (e.g. gold "$400,000" → answer "$350,000") and (b) prose verbosity that defeats strict scorers despite a correct fact (e.g. gold "Three times a week" → answer correctly mentions "2-3 times" inside a longer paragraph). A frontier-class model would likely reduce both.
- **Frontier-LLM measurement pending.** This will likely lift the number substantially; it'll be published in v0.11.

Reproduce (full run — re-calls the LLM, ~12 min, needs Ollama + qwen2.5:14b):
```bash
uv run python -m eval.run_answer_eval \
    --data data/longmemeval_ku_78.json \
    --llm ollama --ollama-model qwen2.5:14b-instruct \
    --scorer overlap \
    --output runs/answer_eval/ku-qwen14b-overlap.json
```

Or re-score the existing numeric run's stored answers under the overlap
scorer (instant, no LLM — the model's answers don't depend on the
scorer, so this reproduces 24/78 = 0.308 deterministically):
```bash
uv run python -m eval.rescore \
    --in runs/answer_eval/ku-qwen14b-numeric.json \
    --scorer overlap \
    --out runs/answer_eval/ku-qwen14b-overlap.json
```

(The first time these numbers were published, only the numeric artifact
had been persisted; the overlap variant was computed ad-hoc. `eval/rescore.py`
exists so the overlap number is reproducible from the committed numeric
answers without a fresh 12-minute LLM run. `runs/` is gitignored, so the
artifacts live locally; the tool + data + this command are the reproducibility path.)

### What Phase 3 doesn't yet measure (deferred to v0.11+)

- **Frontier-LLM runs** (Claude Sonnet 4, GPT-4o) on KU and on the 500q full set. Engine + runner are wired; one local-model run shipped (above); the frontier-class runs are deferred pending API access.
- **Karaṇa-quality correlation** (`regex < ollama-7b < hybrid-14b` on the synthesis-bounded subset). Requires running the same questions through three karaṇa configurations and reporting the spread.
- **BeliefEval (300 supersession scenarios)** through the answer-eval engine. Requires a small adapter from BeliefEval's per-scenario shape to the question-list shape `run_answer_eval` expects.

The scaffolding ships with v0.10.2; the full battery is part of the v0.11 milestone.

## Phase 1 — LongMemEval retrieval

| Benchmark | R@5 | Notes |
|-----------|:---:|:------|
| LongMemEval S — 100q stratified sample | **0.989** | Full pipeline with `rrf_blend=0.2` |
| LongMemEval-KU — full 78-question subset | **1.000** (78/78) | Knowledge-update stratum |
| Full 500q LongMemEval S | *not yet run* | Needs >32 GB RAM for session cache |

These are Patha Phase 1's measured numbers. Cross-system comparison to other published results is left to readers; metric definitions (session-level vs turn-level R@5, scoring methodology) are documented above so any comparison can be made on like-for-like terms.

### Per-stratum R@5 on the 100q stratified sample

| Stratum | R@5 |
|---------|:---:|
| Single-session | 1.000 |
| Multi-session | 1.000 |
| Knowledge update | 1.000 |
| Temporal reasoning | 0.957 |

### Phase 1 ablation matrix

Each configuration run on the same 100-question stratified sample, same seed, identical protocol.

| Configuration | R@5 | Δ vs baseline |
|---|:---:|:---:|
| **Baseline (full pipeline, rrf_blend=0.2)** | **0.989** | — |
| No cross-encoder | 0.950 | **−0.039** |
| No songline | 0.990 | +0.001 |
| Single view (v1 only) | 0.979 | −0.011 |
| Two views (v1 + v4) | 0.989 | 0.000 |
| No reranker + no songline | 0.979 | −0.011 |

Reading the ablations honestly:

- The **cross-encoder is the single largest contributor** (+3.9 points).
- The songline graph adds essentially zero on this sample. Likely a benchmark-fit issue (LongMemEval favours intra-session retrieval) rather than an architecture problem, but worth being honest about.
- Two views capture almost all the benefit of seven. The Vedic multi-view framing is valid, but two views are the working minimum on this benchmark.
- Pure hybrid retrieval (BM25 + dense + reranker, no songline, no extra views) reaches 0.979 on its own.

The RRF blend — blending 20% of the upstream RRF rank score into the cross-encoder's output — is the single architectural fix that took the pipeline from R@5 = 0.989 (one question missed in validation) to R@5 = 1.000. Principle: no single downstream model should silently override a multi-view consensus.

## Phase 2 — BeliefEval (our own benchmark)

| Set | Detector | Accuracy |
|-----|---------|:--------:|
| 20-scenario seed (v0.1) | hybrid NLI + scripted LLM | 0.958 (23/24) |
| 150-scenario templated | adhyasa-nli | 1.000 (180/180) |
| 125-scenario hand-curated | adhyasa-nli | 0.897 (122/136) |
| 300-scenario combined | adhyasa-nli | 0.960 (333/347) |
| 300-scenario combined | live-ollama-hybrid (gemma4:8B) | 0.963 (334/347) |
| 300-scenario combined | full-stack (v0.6) | 0.986 (342/347) |
| **300-scenario combined** | **full-stack-v7 (v0.7)** | **1.000 (347/347)** |

Per-family on the combined 300-scenario set at v0.6 (before the sequential detector closed the last gaps):

| Family | Accuracy | Notes |
|---|:---:|:---|
| temporally_bounded | 1.000 | Validity windows + rule-based extraction |
| abhava_negation | 1.000 | Nyāya four-fold negation taxonomy |
| reinforcement | 1.000 | Multi-source corroboration chains |
| preference_supersession | 0.992 | Adhyāsa rewrite lifts commonsense cases |
| factual_supersession | 0.978 | Numerical + value-replacement detectors |
| pramana_sublation | 1.000 | Pramāṇa-hierarchy-aware resolution |
| context_scoped | 1.000 | Context filter + ingest-time scoping |
| multi_step_chain | 0.600 | Closed to 1.000 in v0.7 via sequential detector |

**Honest caveat on the 1.000 v0.7 number:** the SequentialEventDetector added in v0.7 was built specifically to cover the five failures v0.6 left behind in the `multi_step_chain` and `factual_supersession` families. So 1.000 on *this* benchmark should be read as "all known failure modes addressed" — not "generalises to unseen benchmarks." The external test below is the fairer read.

## Phase 2 — LongMemEval-KU external benchmark

Measures "does the gold answer text appear in the belief-layer's summary" on the same 78 KU questions Phase 1 scores R@5 on.

| Detector / mode | Accuracy | Avg current beliefs | Avg tokens/summary |
|---|:---:|:---:|:---:|
| stub (no supersession, keep-all null baseline) | 0.795 (62/78) | 79.5 | 2454 |
| v0.6 full-stack, current-only summary | 0.359 (28/78) | 13.1 | 480 |
| v0.7 full-stack-v7, current-only summary | 0.359 (28/78) | 12.6 | 467 |
| **v0.7 full-stack-v7, current + history** | **0.885 (69/78)** | 12.6 current + ~80 history | 2456 |

Interpretation (see [phase_2_v07_results.md](phase_2_v07_results.md) for the full write-up):

1. v0.7 current-only alone scores 35.9% — that's the surface reading and it's misleading.
2. Diagnostic (`eval/longmemeval_diagnose.py`) confirmed **34/34 of stub's wins over v0.7 are recoverable from v0.7's superseded store**. Zero information loss; just reorganised.
3. With `include_history=True`, v0.7 scores 88.5% and is a **proper superset** of stub's correct answers — it gets every question stub gets, plus 7 more (mostly abstention questions where the semantic separation between current and superseded helps).
4. **+9 points over the null baseline is the honest Phase-2-alone contribution.** Not +52 (current-only surface reading) and not "Phase 2 beats Phase 1" (different metrics).

## Phase 2 — non-commutativity measurement

Built to test whether Patha's belief evolution is genuinely order-dependent (quantum-cognition-inspired) or effectively commutative in practice.

```
240 scenarios tested (those with ≥2 propositions)
  Non-commutative: 230/240 (95.8%)
  Mean divergence (forward vs reversed): 0.906

  By family:
    preference_supersession:  127/127 (100.0%),  div=0.943
    factual_supersession:      92/92  (100.0%),  div=0.957
    multi_step_chain:           5/5   (100.0%),  div=0.867
    pramana_sublation:          5/8   ( 62.5%),  div=0.531
    context_scoped:             1/4   ( 25.0%),  div=0.250
    reinforcement:              0/4   (  0.0%),  div=0.000  ← correctly commutative
```

Reinforcement scenarios correctly come out 0% non-commutative (repeating the same claim in any order produces the same single belief). The method distinguishes real order-sensitivity from order-invariant operations — not just "everything is order-sensitive." This is to my knowledge the first empirical order-dependent-belief benchmark for an AI memory system.

Reproduce: `uv run python -m eval.non_commutative_eval`.

## Phase 2 — plasticity on real logs

10 LongMemEval knowledge-update conversations ingested into a fresh belief layer with full plasticity wiring. Measured at query time.

```
mean ingested:         75.6 props / question
mean final_current:    15.8
mean supersession:     30.4 events
mean reinforcement:     3.4 events
mean conf mean:         0.949
mean conf std:          0.106   ← LTD is producing real spread
mean Hebbian edges:   149.8
max  Hebbian degree:   31
mean archived:          0.0   ← pruning doesn't fire on short KU conversations
```

Honest interpretation:

- **LTD is doing real work** (0.106 std isn't zero — old beliefs decay).
- **Hebbian network emerges** (150 edges / conversation from co-retrieval).
- **LTP is now wired** (was dead code through v0.6). Fires on 4.5% of ingests (the reinforcement events).
- **Pruning didn't fire** — correct for short conversations; not a bug. Will matter on longer personal-memory logs.

Reproduce: `uv run python -m eval.plasticity_on_real_logs`.

## Phase 2 — false-contradiction rate

The spec's "dangerous failure mode." Measured on 20 hand-crafted pairs (4 real contradictions, 16 non-contradictions covering reinforcement, parallel activities, past context, marker-present-but-different-topic, etc.).

| Stack | FP rate | Recall |
|---|:---:|:---:|
| full-stack-v7 without additive veto | 25% | 100% |
| **full-stack-v7 with additive veto (current default)** | **6%** | **100%** |

Additive-veto pattern: when p2 contains markers like "also", "too", "as well", "in addition", "still", "another", block supersession even if a state-change marker is present. Distinguishes "I now listen to classical too" (expansion) from "I now listen to classical instead" (replacement).

Reproduce: `uv run python -m eval.false_contradiction_eval`.

## Plasticity-stressing benchmark

6/6 mechanistic tests pass:

- LTP reinforcement: 5 distinct-source reinforcements → confidence 0.916, vāsanā crystallised
- LTD decay: after 2× half-life → confidence 0.25
- Homeostasis: max/min confidence ratio bounded
- Synaptic pruning: depth-3 chain ancestors archived
- Hebbian association: co-retrieval → edge weight grows linearly
- Vāsanā preservation: effective_confidence survives heavy surface decay

Reproduce: `uv run python -m eval.plasticity_benchmark`.

## Test suite

```
uv run pytest tests/ -q
→ 602 passed, 1 skipped (wordnet), 9 deselected (slow)
```

449 belief-layer tests, 149 Phase 1 tests, 4 plasticity-wiring tests, 8 sequential-detector tests, 25 numerical-detector tests, 4 counterfactual-reingest tests.
