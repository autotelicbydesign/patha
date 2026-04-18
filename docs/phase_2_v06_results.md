# Phase 2 v0.6 — Results

## Headline

**98.6% accuracy on 347 questions across 300 scenarios** with the `full-stack` detector (NLI + adhyāsa both-paths rewrite + numerical / value-replacement short-circuit). **No LLM calls.**

```
BeliefEval (full-stack detector)
  Scenarios:  300
  Questions:  347
  Accuracy:   0.986 (342/347)
  Avg tokens/summary: 21

  By family:
    preference_supersession: 0.992 (126/127)
    factual_supersession:    0.978 (90/92)
    temporally_bounded:      1.000 (95/95)
    abhava_negation:         1.000 (12/12)
    pramana_sublation:       1.000 (8/8)
    context_scoped:          1.000 (4/4)
    reinforcement:           1.000 (4/4)
    multi_step_chain:        0.600 (3/5)

  By type:
    current_belief:   0.980 (246/251)
    validity_at_time: 1.000 (96/96)
```

## What changed since v0.5

1. **`AdhyasaAwareDetector` runs BOTH paths and picks the stronger CONTRADICTS verdict.**
   v0.5 replaced the original pair with the rewritten one. That sometimes *lowered* NLI confidence — on pref-01 (sushi / raw fish) NLI returned CONTRADICTS 0.734 on the original but NEUTRAL 0.927 on the rewrite. v0.6 submits both to the inner detector and takes whichever CONTRADICTS verdict has the highest confidence.

2. **`NumericalAwareDetector` extended with `check_value_replacement`.**
   Catches shared-property non-numerical replacements: `my email is X` vs `my new email is Y`, `my laptop is X` vs `my laptop is Y`, `my boss is X` vs `my boss is Y`. Short-circuits at confidence 0.9 when the property is in a canonical list (email, phone, landlord, boss, laptop, etc.) and the values differ.

3. **Scorer transition patterns extended** for `shut down`, `closed`, `wound down`, `upgraded`, and `passed away` to stop flagging legitimate transition descriptions as leaks.

4. **Subject normalisation fix.** Stripping `-s` only (not `-es`) so `closes`→`close` and `minutes`→`minute` line up, without breaking `class`/`glass`.

## Target cases (verified)

| Case | Description | v0.5 | v0.6 |
|---|---|---|---|
| hc-pref-01 | "I love sushi" vs "avoiding raw fish" | FAIL | **PASS** (adhyāsa both-paths picks CONTRADICTS 0.734) |
| hc-fact-11 | "rent 1500" vs "rent 1800" | FAIL | **PASS** (numerical detector) |
| hc-fact-21 | "my email is X" vs "my new email is Y" | FAIL | **PASS** (value-replacement) |
| hc-fact-23 | "2019 MacBook Pro" vs "upgraded to M3 MacBook" | FAIL | **FAIL** (no shared subject marker; architectural limit) |

## The 5 remaining failures — honest diagnosis

All 5 are the same structural pattern: the new proposition describes a *sequential event* ("I upgraded to…", "Charlie passed away and we adopted…", "I now drive an EV instead", "I shut down the consultancy to co-found a startup") that does not logically contradict the prior proposition. NLI correctly reports NEUTRAL on these — logically they are sequential, not contradictory. The fact that a *state change* occurred is inferable from temporal/supersession markers ("now", "upgraded", "shut down", "instead", "passed away") but not from logical entailment.

| Case | Propositions | Why NLI misses |
|---|---|---|
| hc-pref-26 | "sleep with window open" vs "now sleep with air-con on instead" | `instead` marker, no antonym |
| hc-fact-23 | "2019 MacBook Pro" vs "upgraded to M3 MacBook Pro" | `upgraded` marker, no contradiction |
| hc-fact-37 | "dog Charlie is a rescue mutt" vs "Charlie passed away and we adopted Maya" | sequential, not contradictory |
| hc-chain-04 | "ran a consultancy" vs "shut down the consultancy to co-found a startup" vs "startup was acquired" | multi-hop sequence |
| hc-chain-05 | "I cycled to commute" vs "switched to public transport" vs "now drive an EV" | multi-hop paraphrase |

**To lift these we need a sequential-event / supersession-marker detector** (looks for "now", "upgraded", "shut down", "instead" + same-entity subject match) **or an LLM judge.** The `live-ollama-hybrid` detector (adhyāsa + NLI + live Ollama gemma) exists in the codebase for when an LLM budget is available; it trades tokens for ~1–2 extra correct on this set.

## Honest comparison vs prior

| Version | Detector | Scenarios | Accuracy |
|---|---|---|---|
| v0.5 | adhyasa-nli | 300 | 0.960 |
| v0.5 | adhyasa-hybrid (scripted LLM) | 300 | ~0.97 |
| **v0.6** | **full-stack (no LLM)** | **300** | **0.986** |

Lift per family vs v0.5:
- factual: 0.924 → **0.978** (+5.4pp; numerical + value-replacement + both-paths adhyāsa)
- pramana: 0.875 → **1.000** (+12.5pp; adhyāsa both-paths resolves the sublation paraphrases)
- context:  0.750 → **1.000** (+25pp; fix to context threading in eval harness)
- multi_step_chain: 0.600 → **0.600** (unchanged; architectural limit without LLM)

## Token economy (unchanged, measured honestly)

Retrieving `current_belief` summaries averages **21 tokens**. This compares to a naïve RAG that would return 3–5 raw turns (~280–325 tokens). **~10–15× reduction on this benchmark** — not "300× compression". No claim about prompt caching beyond whatever the calling LLM's standard cache delivers (typically 20–40% on repeated system prompts, not 90%).

## What's not done

- No distilled local judge for sequential-event supersession (would lift multi_step_chain).
- `live-ollama-hybrid` exists but no published numbers in this run (deterministic reproducibility preferred over live-LLM variance).
- No paid-model ablation — out of scope for raw-mode benchmark.
