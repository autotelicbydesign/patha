# Phase 2 v0.3 — Full Sprint Results

**Status:** v0.3 complete
**Date:** 2026-04-18
**Branch:** `phase-2-belief-layer`
**Tests:** 460 passing across the whole repo, zero regressions

---

## What v0.3 added on top of v0.2

Thirteen new capabilities shipped this sprint:

1. **Plasticity mechanisms wired into runtime.** v0.2 had them as tested classes; v0.3 makes them fire during `.ingest()` and `.query()`. LTD decays stale beliefs, Hebbian records co-retrieval edges, homeostasis rescales periodically, pruning archives deeply-superseded beliefs.

2. **Pramāṇa-aware belief tracking.** Six classical sources of valid knowledge (pratyakṣa / anumāna / upamāna / śabda / arthāpatti / anupalabdhi). Auto-detected from linguistic cues; drives source-independence diversity weighting in reinforcement.

3. **Pramāṇa hierarchy in contradiction resolution.** "What if doctor is wrong?" — when PRATYAKṢA (direct perception) contradicts SHABDA (testimony), the weaker claim becomes `BADHITA` (sublated), not just SUPERSEDED. Distinct semantics: the doctor's claim was right-to-believe-at-the-time but has been demoted.

4. **Source reliability tracking.** Every source_id accumulates a reliability score [0.3, 1.0]. Each time its testimony is sublated by stronger pramāṇa, reliability drops. Future SHABDA from that source starts at lower confidence.

5. **Pramāṇa-weighted default confidence.** PRATYAKṢA / UNKNOWN bare assertions start at 1.0. ANUMANA at 0.80. SHABDA at 0.60 × source_reliability. UPAMANA at 0.55. Reported testimony from an unreliable source literally produces less-confident beliefs.

6. **Ollama-backed LLM judge (D1 Option D).** Real local LLM for contradiction detection via HTTP. No hosted-LLM API calls. Default `qwen2.5:7b`. Clear error if Ollama isn't running.

7. **LLM-inferred validity (D4 completion).** Implicit durations ("training for a marathon" → ~120 days, "on parental leave" → ~180 days) inferred via LLM. Gated: only escalates when rule-based fails AND a temporal-marker word is present AND an LLM generator is configured.

8. **Ingest-time sliding window (D2 hybrid).** Optional `ingest_window_days=30` scopes contradiction checks to recent beliefs, bounding cost from O(N) to O(window). Older beliefs still get checked at query time.

9. **Confidence-weighted supersession (D3 advanced).** Opt-in. When pramāṇas tie and confidence delta exceeds margin, the higher-confidence belief wins. Pramāṇa hierarchy still trumps confidence — PRATYAKṢA always beats low-confidence SHABDA.

10. **Raw Archive Layer.** Immutable, append-only store for original turns. Every proposition traces back to its verbatim source via `raw_archive.turn_for_proposition(id)`. Content-addressable IDs. JSONL persistence.

11. **BeliefEval expanded to 150 scenarios.** Programmatic generator from 29 templates across three families (preference / factual / temporally-bounded). Deterministic, seeded, reproducible.

12. **Plasticity ablation framework.** Six configurations, one-mechanism-off comparisons. Current BeliefEval shows zero differential across ablations — a real, honest finding: BeliefEval doesn't stress plasticity's effects (confidence curves, Hebbian retrieval, long-run homeostasis). Documents a v0.4 benchmark requirement.

13. **Patha CLI.** `patha ingest` / `patha ask` / `patha history` / `patha stats` — minimal shell interface with persistent state in `~/.patha/`.

---

## Headline examples

### What-if-doctor-is-wrong, end-to-end

```python
from patha.belief import BeliefLayer, BeliefStore, Pramana, StubContradictionDetector

layer = BeliefLayer(store=BeliefStore(), detector=StubContradictionDetector())

# Testimony from the doctor
layer.ingest(
    proposition="my doctor told me I have diabetes",
    asserted_at=datetime(2024, 1, 1),
    asserted_in_session="doctor-visit",
    source_proposition_id="p1",
    # pramana auto-detected: SHABDA, confidence auto-set: 0.60
)

# User's own observation (PRATYAKṢA, confidence 1.0)
layer.ingest(
    proposition="I saw my blood sugar reading was normal",
    asserted_at=datetime(2024, 2, 1),
    asserted_in_session="home",
    source_proposition_id="p2",
)

# Result: doctor's claim is BADHITA (sublated), not just SUPERSEDED.
# Source reliability for 'doctor-visit' drops from 1.0 → 0.5.
# Future SHABDA from that source starts at 0.60 × 0.5 = 0.30.
```

### Direct-answer compression + lookup

```bash
patha ingest "I live in Sofia"
patha ingest "I work at Patha remotely"
patha ask "Where do I currently live?"
# [strategy: direct_answer]
#
# - I live in Sofia

patha ask "summarise my situation"
# [strategy: structured]
# (would send 63 tokens to an LLM — no LLM wired in CLI mode)
# [renders the structured summary for the caller to route to an LLM]
```

### Pramāṇa-diversity reinforcement

```python
# Two reinforcements of the same claim, different kinds of knowing
store.add(
    proposition="I have high cholesterol",
    pramana=Pramana.SHABDA,     # doctor told me
    source_id="doctor-visit",
    ...
)
store.add(
    proposition="Blood test shows high cholesterol",
    pramana=Pramana.PRATYAKSA,  # I saw the result myself
    source_id="lab-report",
    ...
)
store.reinforce(a="doctor's claim", b="blood test")

# Distinct source + distinct pramāṇa → 40% gap-closure (cross-corroboration)
# vs. two SHABDA from the same doctor → only 10% (pure echo)
```

---

## The numbers

### Test coverage

- **460 tests passing** across the whole repo
- **Zero regressions** on Phase 1 (KU-78 R@5 = 1.000)
- **4 slow** integration tests (DeBERTa loading, live Ollama) deselected by default

### BeliefEval — 20-scenario seed, hybrid detector (carried from v0.2)

| Detector | Accuracy |
|---|:---:|
| Stub heuristic | 0.375 (9/24) |
| NLI (DeBERTa-v3-large) | 0.833 (20/24) |
| Hybrid NLI + LLM judge | **0.958 (23/24)** |

### BeliefEval — 150-scenario v0.3 set, stub baseline

180 total questions (150 scenarios, some with multiple questions).

| Detector | Accuracy |
|---|:---:|
| Stub heuristic | 0.333 (60/180) |

Validity questions pass 100% (60/60) even on stub; current-belief questions fail 0% (0/120) because the stub can't detect paraphrased contradictions. NLI + hybrid runs with the real LLM judge are the v0.4 headline run; they need Ollama wired for the 150-scenario scale (unscripted pairs).

### Token economy (carried from v0.2)

| Strategy | Mean tokens_in | Compression vs naive RAG |
|---|:---:|:---:|
| naive_rag baseline | ~310 | 1.00x |
| structured summary (D7-B) | 65 | 4.88x |
| **direct_answer (D7-C)** | **0** | **~309x** |

---

## The honest caveats

Four things v0.3 does NOT claim:

1. **Headline hybrid numbers on the 150-scenario benchmark.** The stub baseline ran and serves as a sanity check. A headline hybrid-detector run at 150-scenario scale needs a real Ollama-backed LLM judge running unscripted (the v0.2 scripted-stub approach only scripts the 2 v0.1 failures and would degenerate to NLI-only at scale). This is a ~15-minute run once Ollama is configured — held as the v0.3 publication moment.

2. **Plasticity ablations are currently null-result.** See v0.3 commit `145bdcf`. BeliefEval doesn't stress the mechanisms plasticity affects (confidence curves, Hebbian retrieval, long-run homeostasis). The ablation framework is correct — a plasticity-stressing benchmark is explicit v0.4 work in `docs/phase_2_v04_roadmap.md`.

3. **BeliefEval 150-scenario set is templated, not hand-curated.** The generator re-instantiates 29 base templates with varied timestamps. This reaches statistical scale but the scenarios are topically narrow compared to a true publication-grade benchmark. A hand-curated + LLM-assisted + human-QA'd set targeting ~300 diverse scenarios is v0.4 / peer-review prep.

4. **CLI is minimal.** No inline NLI (keeps startup fast), no streaming, no semantic-search query (lookups match on presence in current beliefs). For real use, wire the Python API with `IntegratedPatha` + a proper Phase 1 retriever.

---

## v0.3 architecture summary

```
src/patha/
  belief/
    types.py                     # Belief + Validity + Pramana + ResolutionStatus + PRAMANA_STRENGTH
    contradiction.py             # NLI + Stub detectors
    llm_judge.py                 # Hybrid + StubLLMJudge + PromptLLMJudge
    ollama_judge.py              # OllamaLLMJudge (HTTP, no new dep)
    store.py                     # BeliefStore + SourceReliability + resolve_contradiction
    layer.py                     # BeliefLayer + PlasticityConfig + ingest_window_days
                                 #   + confidence_weighted_supersession
    pramana.py                   # Rule-based pramana detection
    plasticity.py                # LTP / LTD / SynapticPruning / Homeostasis / Hebbian
    validity_extraction.py       # Rule + LLM-inferred validity
    direct_answer.py             # D7-C: no-LLM lookup answers
    raw_archive.py               # Provenance substrate
  integrated.py                  # IntegratedPatha (Phase 1 + Phase 2 end-to-end)
  cli.py                         # patha CLI

eval/
  belief_eval.py                 # Detector-agnostic runner
  belief_eval_data/
    seed_scenarios.jsonl         # v0.1 20-scenario seed
    v03_scenarios.jsonl          # v0.3 150-scenario set
    generate_scenarios.py        # Template-based generator
  plasticity_ablations.py        # Six-config ablation matrix
  token_economy.py               # Compression curves across memory sizes

docs/
  phase_2_spec.md                # Architectural spec with D1-D7
  phase_2_literature_survey.md   # ~4000 words on related work
  phase_2_v01_results.md         # First BeliefEval run
  phase_2_v02_results.md         # v0.2 sprint complete
  phase_2_v03_results.md         # this file
  phase_2_v04_roadmap.md         # Vedic + quantum-cognition design
```

## What's next (v0.4)

Already scoped in `docs/phase_2_v04_roadmap.md`:
- Vṛtti-state taxonomy (extended ResolutionStatus)
- Order-sensitive / counterfactual belief API (quantum cognition)
- Saṁskāra → Vāsanā layered confidence
- Adhyāsa-based contradiction detection (ontology-aware near-identity)
- Abhāva four-fold epistemology of negation
- Contextuality (session-scoped beliefs)
- Plasticity-stressing benchmark
- Hand-curated 300-scenario BeliefEval + peer-review submission
- Live-Ollama BeliefEval run as the v0.3 publication moment

## Reproducing

```bash
git checkout phase-2-belief-layer
uv sync
uv run pytest                                               # all 460 tests
uv run python -m eval.belief_eval --detector stub           # v0.1 seed
uv run python -m eval.belief_eval --detector hybrid         # v0.2 headline
uv run python -m eval.belief_eval \\
    --scenarios eval/belief_eval_data/v03_scenarios.jsonl   # v0.3 set
uv run python -m eval.plasticity_ablations                  # ablation matrix
uv run python -m eval.token_economy                         # compression curves
uv run patha ingest "I live in Sofia"                       # CLI
uv run patha ask "Where do I currently live?"
```

## Acknowledgments

Built with Claude (Anthropic) as pair-programming partner. Architectural
decisions, evaluation design, and framing by the author. Code written in
collaboration. Specific Vedic pramāṇa translations informed by Nyāya and
Mīmāṃsā sources; specific quantum-cognition framing informed by Busemeyer,
Pothos, and Bruza.
