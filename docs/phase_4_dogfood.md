# Phase 4 dogfood — the narrative path on real data

**Corpus**: `writeups/` — 18 reflective first-person files Stefi wrote *while building Patha* (2026-04-13 → 2026-04-20, dated via YAML frontmatter derived from `build_diary.py`'s editorial Day-N labels anchored to file mtimes; Day 1 = 2026-04-13). 77 beliefs after import splitting. This is the strongest available ground truth: the author can personally verify whether a generated timeline is correct.

**Setup**: fresh store at `/tmp/patha-dogfood`, `Memory(detector="stub", enable_phase1=True)`, real MiniLM embedder, real spaCy `en_core_web_sm` entities, real songline graph. **No topic channel** (doesn't exist yet) — this run is the *before* baseline for the Step-3/4 comparison.

Reproduce:
```bash
uv run python -m spacy download en_core_web_sm    # once
rm -rf /tmp/patha-dogfood
PATHA_STORE_PATH=/tmp/patha-dogfood uv run patha import folder writeups/
# then run the questions below through patha.Memory(path="/tmp/patha-dogfood/beliefs.jsonl", enable_phase1=True)
```

## Questions asked

| id | question | routed | verdict |
|---|---|---|---|
| N1 | how has my thinking about the **songline graph** evolved? | `narrative` ✓ | arc present; origin imperfect (see F6) |
| N2 | how has my thinking about **memory** evolved? | `narrative` ✓ | plausible arc; theme too broad for this corpus (F5) |
| N3 | when did I **first start doubting the benchmark** numbers? | `narrative` ✓ | **origin correct** — the true arc |
| N4 | trace my thinking on the **Vedic framing** | `narrative` ✓ | thread present; padded with weakly-related beats (F5) |
| C1 | what did I say about the ablations? | `structured` ✓ | control — correctly NOT narrative |
| C2 | how much have I spent on bikes total? | `structured` ✓ | control — correctly NOT narrative (no bike tuples → graceful) |
| C3 | what do I currently believe about the cross-encoder? | `direct_answer` ✓ | control — correctly NOT narrative |

**Routing: 7/7 correct.** All four narrative phrasings routed `narrative`; all three controls stayed on their own paths.

**The N3 result is the headline.** Asked *"when did I first start doubting the benchmark numbers?"*, the walk returned origin = *"The romance of building in public is a lie. Hour 47."* (Apr 15) followed by *"The ablations humbled me. I just hit 98.9%… The table did not say what I wanted it to say"* (Apr 17) — which **is the true story** of how the benchmark skepticism developed, in the right order, with real dates. No LLM was called to produce this ordering.

## Bugs found and fixed during this dogfood (would never have surfaced in synthetic tests)

| # | Bug | Root cause | Fix |
|---|---|---|---|
| F1 | Frontmatter ingested as belief text; dates ignored | `_split_frontmatter` was gated on `obsidian=True`; plain `folder`/`file` imports skipped it | `importers.py`: parse frontmatter unconditionally (it's a generic Markdown convention); only wikilink/tag entity-hints stay Obsidian-gated |
| F2 | All 18 files collapsed into ONE session → session channel became a 91-node near-clique (11,832 edges) | `import_folder` used the *folder* name as session for root-level files | Root-level files now get per-file sessions (`path.stem`); nested files keep parent-folder grouping |
| F3 | Theme extraction returned `doubting` for N3 (gerund, not topic) | mental-verb inflections weren't in the narrative stopword list | `itihasa.py`: added doubt/wonder/realize/believe/… inflection families; N3 now resolves theme=`benchmark` |

After F1+F2: 77 beliefs (was 91 — frontmatter-block "beliefs" gone), 18 sessions, 8 distinct real dates, graph 3,928 edges (was 11,832).

## Open findings (the failure-mode inventory that seeds Step 3 and the evolution benchmark)

- **F4 — Entity channel cannot anchor abstract themes (predicted, now confirmed empirically).** `songline`, `memory`, `vedic`, `benchmark` are all absent from the entity channel (`in_entity_channel=False` on every narrative question); spaCy NER extracted proper nouns (`Aboriginal`, `Claude`, `Rig Veda`, `LongMemEval`…). Anchoring survived *only* via the Phase-1 semantic-seed half of the anchor union. **The topic channel (Step 3) is the principled fix.**
- **F5 — Beat precision saturates on theme-dense corpora.** Every narrative question returned exactly `max_beats=24` — the cap, not a natural arc size. With ~2,000-char chunks and a corpus *about* building a memory system, substring gating ("memory" in chunk) passes for most of the corpus; weakly-related beats (e.g. the Twitter-thread mechanics chunk) leak into the `vedic`/`songline` timelines. Topic clusters should tighten this: the cluster-shared gate admits only chunks in *anchor* clusters, which separates "songline-as-subject" chunks from "songline-mentioned-once" chunks. **Step 4 measures exactly this.**
- **F6 — Origin identification is recall-limited.** N1's origin beat is *"The ablations humbled me"* (Apr 17), but the true origin of the songline arc is update_02 (Apr 14, where songlines were introduced with conviction). That chunk contains "songlines" and passes the gate — it simply wasn't *reached* (anchor set + hop/branch budget). Early, short, pre-pivot chunks are the hardest to reach from semantically-central anchors. → the evolution benchmark needs an **origin-identification scorer**, not just ordering/coverage.
- **F7 — `max_beats` truncation policy is arc-preserving but not relevance-preserving.** `_temporal_thin` keeps endpoints + revisions and strides the middle; on saturated walks this keeps weakly-related middle beats while dropping strongly-related ones. Revisit after the topic gate lands (F5 may mostly dissolve it).

## What Stefi should verify (the ground-truth check)

1. **N3 (benchmark doubt)**: does *Hour 47 (Apr 15) → ablations-humbled (Apr 17) → honest-caveats (Apr 20)* match your memory of when the skepticism actually started? *(Model's read: yes — this is the arc.)*
2. **N1 (songline)**: the walk says the arc runs Apr 17 → Apr 20 with no reversals. Your actual arc: introduced-with-conviction (Apr 14) → ablations-showed-≈0 (Apr 17) → redeemed-as-philosophy (Apr 20). The Apr-14 origin is missing (F6) and the "no reversals" through-line is wrong for this theme — the ablation *was* the reversal. The through-line currently only detects reversals via supersession edges, which don't exist inside an imported one-way corpus. → benchmark scenario family: *reversal detection without explicit supersession*.
3. Sanity: do the N2/N4 timelines read as *your* thinking, or as generic chunks that mention the word? (Grounds the F5 precision finding.)

## Step 4 — topic channel A/B (before/after)

Same store, same four narrative questions, `PATHA_TOPICS=off` vs `on`.

**Structural result (unambiguous):**

| | topics OFF | topics ON |
|---|---|---|
| graph edges | 3,928 | **4,310** (+382 topic edges) |
| channels | entity 117 · temporal 18 · session 18 | + **topic 16** (16 real clusters over MiniLM v1 embeddings) |

**Capability result (verified):** the cluster-share gate admits paraphrased on-theme beliefs that the substring gate excluded — proven by unit regression (`test_temporal_edge_to_cluster_mate_passes_gate`) and observed on the corpus: with topics ON, N2 recovered its **true origin** — *"Memory was designed before writing was invented"* (update_02, Apr 14) — the exact F6-class miss the gate was built for.

**Honest composition result (mixed, knob-sensitive):** both configs saturate `max_beats=24`, so widening the gate *substitutes* beats rather than strictly adding — and which true beats survive is sensitive to the walk budgets. Two walker refinements came out of this A/B (shipped): score-aware thinning (`_temporal_thin` keeps the strongest-connected middle beats instead of an even stride — F7 fix) and a graph-size-scaled frontier budget (`max_branch` defaults to `node_count/4`, floored at 8 — a fixed 8 gave 4×8=32 visits against a 76-node graph, structurally starving the walk).

**Discipline stop.** After those two structural fixes, further knob-tuning against these 4 questions would be fitting the dogfood set — the BeliefEval-overfit pattern this plan explicitly guards against. Per-question beat composition is hereby handed to the Step-5 evolution benchmark: coverage/ordering/origin-identification scorers on held-out scenarios are the instrument that can adjudicate knob values; four eyeballed questions are not.

## Post-fix addendum — v9 on real data (ingest sanity, 2026-07-06)

Re-ingested the corpus with `full-stack-v9` (all prior dogfoods used `stub`). Findings:
- **The new v9 components are quiet on real writing**: `RevisionPatternDetector` fired **0** times (no false positives in the wild); `SymmetricContradictionDetector` adopted 2 reverse edges, rejected 0 as off-topic (the gate had nothing to catch here — its value was demonstrated on the scenario corpus).
- **17 of the 19 supersession edges come from the inherited v8 stack** running NLI over ~2,000-char essay chunks. Some are sensible draft-consolidation (later "Three things no other system ships" superseding the earlier "Fork it" claims); some are weak ("The silent veto" superseding "What I'm building"). **Known behavior, pre-existing**: chunk-level supersession on long-form imports is noisy; supersession semantics are cleanest on atomic facts. Documented as a limitation for imported-essay corpora, not a v9 regression.
- Rubric-v2 note: EvolutionEval currently scores supersession *recall* only; a supersession-**precision** scorer (penalizing unexpected edges between golds) is queued so false-edge inflation can never hide again.

## Next (per the approved plan)

- **Step 5**: evolution benchmark authored from the verified failure modes above (F5/F6/F7 + reversal-without-supersession) in scenario families (`progressive_revelation`, `multi_factor_change`, `perspective_shift`, `reversed_belief_chain`), with the 70/30 sealed split, frozen rubric, and the walk-knob sweep (threshold 0.45/0.55/0.65, budgets) run against the dev set only.
- **Gate**: Stefi's ground-truth verification of the Step-2 timelines (the three checks above) feeds scenario authoring.
