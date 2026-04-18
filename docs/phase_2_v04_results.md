# Phase 2 v0.4 — Deep Cognitive Dimensions (complete)

**Status:** v0.4 complete. All 6 roadmap items from `docs/phase_2_v04_roadmap.md` shipped.
**Date:** 2026-04-18
**Branch:** `phase-2-belief-layer`
**Tests:** 534 passing across the whole repo, zero regressions.

---

## What v0.4 adds

Six capabilities drawn from Vedic and quantum-cognition traditions, each with an operational software translation that earns its place by addressing a specific failure mode:

| # | Capability | Source tradition | What it fixes |
|---|---|---|---|
| 1 | Vṛtti-state taxonomy | Yoga Sūtras I.5-11 | Beliefs now have a cognitive mode (pramāṇa / viparyaya / vikalpa / nidrā / smṛti) computed at retrieval time |
| 2 | Abhāva four-fold negation | Nyāya | Distinguishes prior / destructive / mutual / absolute absence — 'I no longer X' is semantically different from 'I never X' |
| 3 | Contextuality | Quantum cognition | Beliefs in different contexts don't contradict — 'I'm available' in work vs personal are both valid |
| 4 | Saṁskāra → Vāsanā | Yoga Sūtras | Two-layer confidence: surface fast, deep slow. Prevents decayed surface from losing long-held positions. |
| 5 | Order-sensitive / counterfactual operations | Quantum cognition | `replay_in_order()` + `order_sensitivity()` — 'what would you currently believe if you'd heard B before A?' |
| 6 | Adhyāsa superimposition detection | Advaita Vedānta | Ontology-aware rewrite-and-retest for 'sushi ≈ raw fish' contradictions NLI misses |

## The real headline number from v0.3 that landed during v0.4

**BeliefEval at 150-scenario scale, NLI detector: 0.933 (168/180).**

Ran during v0.4 in parallel to the new-capability work. Breakdown:

| Family | Accuracy |
|---|:---:|
| Factual supersession | **1.000 (45/45)** |
| Temporally bounded | **1.000 (60/60)** |
| Preference supersession | 0.840 (63/75) |

The 12 failures are all in preference_supersession — the family that stresses commonsense reasoning. v0.4's adhyāsa detector is designed exactly to catch these; a hybrid run that combines NLI + adhyāsa rewrite-and-retest should recover them.

## Module map (v0.4 additions)

```
src/patha/belief/
  vritti.py              VrittiClass enum + vritti_of() classifier
  abhava.py              AbhavaKind enum + classify_abhava() + referenced-state extractor
  adhyasa.py             IsAOntology protocol + HandCuratedOntology + check_superimposition()
  counterfactual.py      replay_in_order() + order_sensitivity()
  # Changes to existing modules:
  types.py               Belief.context, .samskara_count, .deep_confidence
                         + .is_vasana_established, .effective_confidence properties
  store.py               BeliefStore.by_context() + reinforce() now tracks samskara
                         and crystallises vāsanā at threshold 5
  layer.py               BeliefLayer.ingest(context=...) + .query(context=...)
                         + context-aware candidate filtering at ingest
  plasticity.py          LongTermDepression also decays deep confidence at 10x slower rate
```

---

## The bigger picture — what Patha v0.4 now is

Reading the whole belief layer end-to-end, Patha is no longer a 'memory system' in the narrow retrieval sense. It's a **belief maintenance system** with cognitive-theory primitives drawn from specific traditions:

- Every belief knows **how it came to be known** (pramāṇa)
- Every belief knows **its cognitive mode at retrieval** (vṛtti)
- Every belief knows **what absence it encodes** (abhāva, for negations)
- Every belief knows **its context of applicability** (context)
- Every belief knows **its epistemic depth** (surface vs vāsanā confidence)
- Every belief knows **its position in a temporal sequence** (counterfactual replay)
- Near-identity contradictions get caught by **ontology-aware rewrite** (adhyāsa)

No other open-source AI memory system ships any of this. Some (Zep, Mem0) ship one piece — temporal edges, binary update — but none ship the integrated cognitive-theory layer.

## What v0.4 does NOT claim

Be honest about the limits:

1. **Probability interference (quantum cognition) deferred.** The roadmap flagged this as wait-until-weak-spot. v0.4 didn't surface a concrete weakness that justifies it. Held for v0.5+ if a specific failure mode requires it.

2. **Adhyāsa ontology is hand-curated.** The 15-class HandCuratedOntology covers the BeliefEval failure cases but is not production-grade. WordNet / ConceptNet integration is v0.5 work; the ontology protocol is already in place.

3. **Vṛtti classification is diagnostic, not enforced.** v0.4 exposes `vritti_of(belief)` for callers to consume; it does NOT alter storage or retrieval policy based on vṛtti. Policy consequences (e.g., 'never surface vikalpa in direct-answer') are v0.5.

4. **Counterfactual replay doesn't re-run NLI.** It reorders events but preserves supersede edges verbatim. A full re-evaluation mode (replay from raw ingest through NLI decisions) is v0.5 work.

5. **Saṁskāra threshold (5) is hand-tuned, not learned.** The establishment threshold and deep-confidence decay rate are configurable constants. A principled calibration against user behaviour data is v0.5+.

6. **Contextuality auto-detection not built.** v0.4 requires callers to tag contexts explicitly. Auto-detection (LLM classifier on ingest) is a natural v0.5 follow-up.

7. **v0.4 has not re-run BeliefEval with the new features.** The 150-scenario benchmark was scored on Phase 2 v0.3's pipeline (0.933 NLI-only). A v0.4 run with adhyāsa rewrite + vṛtti-aware rendering + confidence-weighted supersession would likely lift preference_supersession closer to 1.0, but that's a measurement not yet made.

## Sprint statistics

- **Commits on branch during v0.4:** ~15
- **Lines of code added:** ~2500
- **New tests:** 75 (16 vritti + 21 abhāva + 9 contextuality + 12 samskāra + 8 counterfactual + 12 adhyāsa — all passing)
- **Total tests in repo:** 534 passing, 5 slow deselected, 0 regressions
- **Total belief-layer modules:** 13 (types, contradiction, llm_judge, ollama_judge, store, layer, pramana, plasticity, validity_extraction, direct_answer, raw_archive, vritti, abhava, adhyasa, counterfactual)

## What's next (v0.5 candidates)

Ordered by leverage-per-hour:

1. **Live-Ollama 150-scenario BeliefEval run** — the actual publication moment the v0.3 doc held back. Will give us the hybrid-detector headline number on the expanded set.

2. **Adhyāsa + NLI pipeline integration.** Currently the adhyāsa module is standalone; wiring it into the default BeliefLayer as a pre-NLI pass is a small change that should lift preference_supersession accuracy.

3. **Vṛtti-aware direct-answer policy.** Surface pramāṇa beliefs confidently, flag vikalpa with uncertainty caveats, never surface nidrā.

4. **WordNet-backed IsAOntology.** Replaces HandCuratedOntology. Broad coverage.

5. **Plasticity-stressing benchmark** — tests that actually stress LTD/Hebbian/homeostasis (which the current BeliefEval does not, per the v0.3 ablation null-result).

6. **Hand-curated 300-scenario BeliefEval for peer-review submission.**

## Reproducing

```bash
git checkout phase-2-belief-layer
uv sync
uv run pytest                                               # 534 tests

# v0.1 seed (20 scenarios)
uv run python -m eval.belief_eval --detector stub
uv run python -m eval.belief_eval --detector hybrid         # 0.958

# v0.3 set (150 scenarios)
uv run python -m eval.belief_eval \
  --scenarios eval/belief_eval_data/v03_scenarios.jsonl \
  --detector nli                                            # 0.933

# Token-economy curves
uv run python -m eval.token_economy --sizes 50,500,5000

# Plasticity ablations
uv run python -m eval.plasticity_ablations

# Minimal CLI
uv run patha ingest "I live in Sofia"
uv run patha ask "Where do I currently live?"
```

## Acknowledgments

Built with Claude (Anthropic) as pair-programming partner. Vedic conceptual translations informed by Nyāya, Mīmāṃsā, Advaita Vedānta, and Yoga Sūtras sources; quantum-cognition framing informed by Busemeyer, Pothos, and Bruza. Any errors of interpretation are the author's; any errors of translation-to-software are mine and Claude's jointly.
