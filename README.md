# Patha

> *The way. The recitation.*

An AI memory system built on multi-view redundant encoding, narrative traversal, and cognitive-theory primitives drawn from Vedic and quantum-cognition traditions. Runs fully local with **zero hosted-LLM API calls**.

Patha has two phases, both shipped:

- **Phase 1 — Retrieval.** 7-view redundant encoding (Vedic recitation), songline graph traversal (Aboriginal narrative), hybrid BM25 + dense + RRF-blended cross-encoder reranking.
- **Phase 2 — Belief layer.** Non-destructive supersession, pramāṇa-aware reinforcement, vāsanā layered confidence, vṛtti-classified retrieval, adhyāsa superimposition detection, neuroplasticity-inspired maintenance (LTD/LTP/Hebbian/homeostasis/pruning).

## Results

### Phase 1 — LongMemEval (retrieval)

| Benchmark | R@5 | Notes |
|-----------|:---:|:------|
| **LongMemEval S — 100q stratified sample** | **0.989** | Full pipeline with `rrf_blend=0.2` |
| **LongMemEval-KU — full 78-question subset** | **1.000** | Beats Mem0 (ECAI 2025, 0.934) by **+6.6 points** |
| Full 500q LongMemEval S | *not yet run* | Needs >32 GB RAM for session cache |

### Phase 2 — BeliefEval (belief maintenance)

| Set | Detector | Accuracy |
|-----|---------|:--------:|
| 20-scenario seed (v0.1) | hybrid NLI + scripted LLM | 0.958 (23/24) |
| 150-scenario templated | adhyasa-nli | 1.000 (180/180) |
| **125-scenario hand-curated** | adhyasa-nli | 0.897 (122/136) |
| **300-scenario combined** | adhyasa-nli | 0.960 (333/347) |
| **300-scenario combined** | **live-ollama-hybrid** (gemma4:8B) | **0.963** (334/347) |

Per-family on the combined 300-scenario set:

| Family | Accuracy | Notes |
|---|:---:|:---|
| temporally_bounded | 1.000 | Validity windows + rule-based extraction |
| abhava_negation | 1.000 | Nyāya four-fold negation taxonomy |
| reinforcement | 1.000 | Multi-source corroboration chains |
| preference_supersession | 0.976 | Adhyāsa rewrite lifts commonsense cases |
| factual_supersession | 0.924 | NLI weak on numerical supersessions |
| pramana_sublation | 0.875 | Pramāṇa-hierarchy-aware contradiction resolution |
| context_scoped | 0.750 | v0.6 target: tighter context filter |
| multi_step_chain | 0.600 | v0.6 target: transitive supersession |

### Plasticity-stressing benchmark

6/6 mechanistic tests pass:
- LTP reinforcement: 5 distinct-source reinforcements → confidence 0.916, vāsanā crystallised
- LTD decay: after 2× half-life → confidence 0.25
- Homeostasis: max/min confidence ratio bounded
- Synaptic pruning: depth-3 chain ancestors archived
- Hebbian association: co-retrieval → edge weight grows linearly
- Vāsanā preservation: effective_confidence survives heavy surface decay

### Comparison on LongMemEval-KU

| System | R@5 | Source |
|---|:---:|---|
| **Patha Phase 1** | **1.000** (78/78) | This repo |
| Mem0 (ECAI 2025) | 0.934 | [arXiv:2504.19413](https://arxiv.org/abs/2504.19413) |

Patha Phase 1 beats Mem0 by **+6.6 points** on the subset that specifically stresses knowledge update and supersession — before any belief layer is implemented (see Roadmap).

### Per-stratum R@5 on the 100q stratified sample

| Stratum | R@5 |
|---------|:---:|
| Single-session | 1.000 |
| Multi-session | 1.000 |
| Knowledge update | 1.000 |
| Temporal reasoning | 0.957 |

## Ablation table

Each configuration run on the same 100-question stratified sample, same seed, identical protocol.

| Configuration | R@5 | Δ vs baseline |
|---|:---:|:---:|
| **Baseline (full pipeline, rrf_blend=0.2)** | **0.989** | — |
| No cross-encoder | 0.950 | **−0.039** |
| No songline | 0.990 | +0.001 |
| Single view (v1 only) | 0.979 | −0.011 |
| Two views (v1 + v4) | 0.989 | 0.000 |
| No reranker + no songline | 0.979 | −0.011 |

### Reading the ablations honestly

- The **cross-encoder is the single largest contributor** (+3.9 points).
- The songline graph adds essentially zero on this sample. Likely a benchmark-fit issue (LongMemEval favours intra-session retrieval) rather than an architecture problem, but worth being honest about.
- Two views capture almost all the benefit of seven. The Vedic multi-view framing is valid, but two views are the working minimum on this benchmark.
- Pure hybrid retrieval (BM25 + dense + reranker, no songline, no extra views) reaches 0.979 on its own.

The RRF blend — blending 20% of the upstream RRF rank score into the cross-encoder's output — is the single architectural fix that took the pipeline from an R@5 = 0.989 that missed one question in validation to an R@5 = 1.000 validation. Principle: no single downstream model should silently override a multi-view consensus.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     PHASE 1 — Retrieval                          │
│                                                                  │
│  raw turns ─▶ Raw Archive (immutable, content-addressed)         │
│     ↓                                                            │
│  propositionize ─▶ 7 view fingerprints ─▶ proposition store      │
│     ↓                                                            │
│  songline graph (entity / time / topic / speaker edges)          │
│                                                                  │
│  query ─▶ hybrid candidates (7-view RRF + BM25)                  │
│         ─▶ cross-encoder rerank (rrf_blend=0.2)                  │
│         ─▶ songline walk  ─▶ MMR  ─▶ top-k                       │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼ candidate proposition ids
┌─────────────────────────────────────────────────────────────────┐
│                     PHASE 2 — Belief Layer                       │
│                                                                  │
│  Belief store (non-destructive)                                  │
│    ├─ types:     Belief + Validity + Pramana + Vṛtti             │
│    ├─ statuses:  CURRENT, SUPERSEDED, BADHITA, COEXISTS,         │
│    │             DISPUTED, AMBIGUOUS, ARCHIVED                   │
│    ├─ edges:     supersedes, coexists_with, disputed_with,       │
│    │             reinforced_by                                   │
│    └─ layers:    surface confidence + deep vāsanā confidence     │
│                                                                  │
│  Contradiction detection pipeline:                               │
│    adhyāsa rewrite (ontology-aware)                              │
│      ─▶ NLI primary                                              │
│         ─▶ LLM judge fallback (live Ollama or scripted)          │
│            ─▶ pramāṇa-hierarchy resolution                       │
│               (supersede / sublate / coexist / dispute)          │
│                                                                  │
│  Plasticity (runs continuously):                                 │
│    ├─ LTD decay on query              (time-based)               │
│    ├─ Hebbian co-retrieval edges      (association)              │
│    ├─ Homeostasis on ingest           (normalisation)            │
│    └─ Synaptic pruning                (archival)                 │
│                                                                  │
│  Query → direct-answer (no LLM) for lookup queries               │
│         → structured summary for reasoning queries               │
│         → raw propositions as fallback                           │
└─────────────────────────────────────────────────────────────────┘
```

## Design Philosophy

### Pillar 1 — Vedic Redundant Encoding

The Vedic oral tradition preserved large bodies of sacred text for thousands of years without writing, using multiple interlocking recitation patterns (*pada*, *krama*, *jaṭā*, *ghana*) that encoded each word in structurally redundant ways. The mechanism functions as an error-correcting code: if one thread frays, others hold.

Patha applies this as a retrieval principle. Each proposition is embedded in **7 overlapping views** — isolated, forward-paired, backward-paired, bidirectional triple, entity-anchored, reframed, and temporally-anchored. A query that misses one view catches another. This makes retrieval robust to paraphrase, restatement, and indirect reference without query rewriting or LLM calls.

*Honest caveat: ablations show two views (v1 + v4) perform nearly as well as seven on this benchmark. The redundancy is valid but has diminishing returns beyond two views on LongMemEval-shaped data.*

### Pillar 2 — Aboriginal Songline Traversal

Aboriginal songlines encode knowledge as walked narratives through country, binding geography, ecology, ancestry, and obligation into continuous paths. Retrieval becomes traversal, not point lookup.

Patha builds a multi-modal graph over propositions — edges weighted by shared entities, temporal proximity, session locality, speaker identity, and topic clusters. At query time, after initial retrieval and reranking, a weighted graph walk from the top anchors surfaces evidence that no single-vector search can reach.

*Honest caveat: on LongMemEval, songline walks contribute approximately zero to R@5 (ablation above shows 0.990 with/without). LongMemEval questions are typically satisfiable from a single session, so graph traversal doesn't have much to do. The songline pillar is expected to earn its keep on cross-session, multi-hop scenarios — including the Phase 2 BeliefEval benchmark we're constructing.*

## Quick Start

```bash
# Install
uv sync

# Run tests (140 tests)
make test

# Smoke test: 10 questions
make eval-quick

# Full eval on LongMemEval S (warning: needs >32 GB RAM for full 500q)
make eval
```

### Prerequisites

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- ~4 GB disk for model weights (downloaded automatically)
- No GPU required (CPU inference works; GPU optional for speed)

## Retrieval Pipeline

```
query
  → PRF expansion (RM3 over BM25 top-10)
  → hybrid candidates: 7 dense views + BM25 → RRF         [2000]
  → pointwise cross-encoder rerank, blended with RRF rank  [100]
  → songline walks from top-3 anchors                      [~140]
  → MMR diversity pass (λ=0.7, per-session cap=2)          [30]
  → top-5 output
```

**Current models (all local, no API calls):**
- **Embedder:** all-MiniLM-L6-v2 (384-dim, sentence-transformers)
- **Reranker:** cross-encoder/ms-marco-MiniLM-L-6-v2, with `rrf_blend=0.2`
- **NER:** spaCy en_core_web_sm

## The 7 Views (Patha Scheme)

| View | Vedic Analogue | Content |
|------|---------------|---------|
| v1 | *pada* (isolated) | Proposition alone |
| v2 | *krama* (forward pair) | Proposition + next |
| v3 | *reverse krama* | Previous + proposition |
| v4 | *jaṭā* (bidirectional) | Previous + proposition + next |
| v5 | *ghana* (entity-anchored) | Entities + surrounding triple |
| v6 | Reframed | "Fact about {entity}: " + proposition |
| v7 | Temporally-anchored | "{timestamp}: " + proposition |

## Reproduction

```bash
git clone https://github.com/autotelicbydesign/patha.git
cd patha
uv sync

# Download the LongMemEval S dataset (not in this repo — get from upstream)
make setup-data
# Or manually: download from https://github.com/xiaowu0162/long-mem-eval
# and place at data/longmemeval_s_cleaned.json

# Download spaCy model
uv run python -m spacy download en_core_web_sm

# Run tests
make test

# Run the 100q stratified eval (matches the headline R@5 = 0.989 number)
make eval-100

# Run the full ablation matrix
make ablation

# Run just the LongMemEval-KU subset (78 questions, the Mem0 comparison)
# Filter LongMemEval-KU questions then pass to runner with --eval-checkpoint
```

**Model versions:**
- `sentence-transformers/all-MiniLM-L6-v2`
- `cross-encoder/ms-marco-MiniLM-L-6-v2`
- `spacy/en_core_web_sm` ≥ 3.7

## Project Structure

```
src/patha/
  chunking/, indexing/, retrieval/,   # Phase 1 — retrieval pipeline
  query/, models/                       (7-view RRF, BM25, songline, MMR)
  belief/                              # Phase 2 — belief maintenance
    types.py              Belief + Validity + Pramana + ResolutionStatus
    contradiction.py      NLI + stub detectors (Protocol-based)
    llm_judge.py          HybridContradictionDetector
    ollama_judge.py       OllamaLLMJudge (real local LLM backend)
    adhyasa.py            superimposition detection + HandCuratedOntology
    adhyasa_detector.py   AdhyasaAwareDetector wrapper
    wordnet_ontology.py   WordNet-backed ontology (optional, needs nltk)
    store.py              BeliefStore (non-destructive, JSONL-persistent)
    layer.py              BeliefLayer (+ PlasticityConfig)
    pramana.py            Nyāya source-of-knowledge classifier
    plasticity.py         LTP + LTD + pruning + homeostasis + Hebbian
    validity_extraction.py rule-based + LLM-inferred validity windows
    direct_answer.py      D7-C: no-LLM lookup answers + vṛtti policy
    raw_archive.py        immutable provenance substrate
    vritti.py             Patañjali cognitive-mode taxonomy
    abhava.py             Nyāya four-fold negation classifier
    counterfactual.py     order-sensitive / replay-in-alt-order API
  integrated.py                        # Phase 1 + Phase 2 end-to-end
  cli.py                               # minimal patha ingest/ask CLI

eval/
  runner.py, ablations.py, metrics.py  # Phase 1 evaluation
  belief_eval.py                        # Phase 2 BeliefEval runner
  token_economy.py                      # compression-curve measurement
  plasticity_ablations.py               # plasticity-feature ablations
  plasticity_benchmark.py               # plasticity-stressing tests
  belief_eval_data/
    seed_scenarios.jsonl               # v0.1 20-scenario seed
    v03_scenarios.jsonl                # v0.3 150-scenario templated
    v05_hand_curated.jsonl             # v0.5 125-scenario hand-curated
    v05_combined_300.jsonl             # hand-curated + templated = 300

docs/
  phase_2_spec.md                      # architectural spec (D1–D7)
  phase_2_literature_survey.md         # ~4000-word survey of related work
  phase_2_v01_results.md               # first BeliefEval run
  phase_2_v02_results.md               # ablations + v0.2 sprint
  phase_2_v03_results.md               # integration + real KU-78 1.000
  phase_2_v04_results.md               # Vedic + quantum cognition additions
  phase_2_v04_roadmap.md               # original v0.4 design doc
  phase_2_v05_results.md               # 1.000 on templated + this work
```

## Phase 2 — Belief Layer (in production)

Phase 2 handles the *dynamic* aspects of memory that current AI memory systems fumble by dumping retrieved context into an LLM and hoping it sorts things out:

- **Contradiction detection** — DeBERTa-v3-large NLI + adhyāsa rewrite-and-retest + optional live-LLM judge (Ollama) for commonsense cases
- **Non-destructive supersession** — old beliefs stay queryable; pramāṇa hierarchy decides whether a claim temporally supersedes (SUPERSEDED) or is sublated by stronger evidence (BADHITA)
- **Temporal validity** — beliefs have lifespans (permanent / dated range / inferred duration / decay); rule-based extraction with LLM fallback
- **Pramāṇa-aware reinforcement** — six Nyāya sources of valid knowledge (perception, inference, testimony, comparison, postulation, non-perception); cross-source / cross-pramāṇa corroboration weighted higher than same-source repetition
- **Vāsanā layered confidence** — surface confidence decays fast; deep vāsanā confidence crystallises after 5+ reinforcements and decays 10× slower
- **Vṛtti classification** — every surfaced belief tagged with its Patañjali cognitive mode (pramāṇa / viparyaya / vikalpa / nidrā / smṛti)
- **Abhāva handling** — four Nyāya kinds of absence distinguished ('I never X' vs 'I no longer X' vs 'I am not a X' vs 'haven't X yet')
- **Contextuality** — beliefs carry contexts; work-scoped belief doesn't contradict personal-scoped belief
- **Plasticity mechanisms** — LTD time decay, Hebbian co-retrieval edges, homeostatic regulation, synaptic pruning all firing during normal ingest/query
- **Raw archive** — every belief traces back to its verbatim source via content-addressed IDs
- **Order-sensitive belief evolution** — counterfactual replay API: "what would you currently believe if you'd heard B before A?"

**Token economy:** direct-answer compression for belief-lookup queries spends zero LLM input tokens; structured summary compresses ~4.88× vs naive RAG; published curves across memory sizes (50 → 5000 beliefs).

See [docs/phase_2_spec.md](docs/phase_2_spec.md) and [docs/phase_2_v05_results.md](docs/phase_2_v05_results.md) for details.

## Roadmap

### Phase 1 (shipped)

- [x] 100-question stratified LongMemEval S eval + full ablation matrix
- [x] LongMemEval-KU 78-question subset: R@5 = 1.000 (beats Mem0 +6.6 pts)
- [x] RRF blend fix (no single downstream model vetoes multi-view consensus)
- [x] Atomic checkpointing + per-question eval resume
- [ ] Full 500q LongMemEval S eval — pending >32 GB RAM
- [ ] End-to-end answer accuracy eval (generation over retrieved context)
- [ ] LanceDB persistent store

### Phase 2 (shipped through v0.5)

- [x] BeliefStore with non-destructive supersession, JSONL persistence
- [x] NLI-based contradiction detection + hybrid LLM fallback
- [x] Pramāṇa-aware belief tracking + source reliability + BADHITA status
- [x] Plasticity wired into runtime (LTD/LTP/Hebbian/homeostasis/pruning)
- [x] Vṛtti cognitive-mode taxonomy + vāsanā layered confidence
- [x] Abhāva four-fold negation classifier
- [x] Adhyāsa superimposition detection + ontology-aware rewrite
- [x] Order-sensitive / counterfactual belief operations
- [x] Contextuality (session-scoped beliefs)
- [x] Raw archive integration (end-to-end provenance)
- [x] Live-Ollama hybrid detector (real local LLM judge)
- [x] WordNet IsAOntology (optional, via nltk)
- [x] Vṛtti-aware direct-answer policy (filter vikalpa, flag viparyaya)
- [x] BeliefEval 300-scenario benchmark (125 hand-curated + 175 generated)
- [x] Plasticity-stressing benchmark (6/6 tests pass)

### Phase 2 v0.6 (next)

- [ ] Hand-curated 300-scenario benchmark with inter-annotator agreement
- [ ] multi_step_chain transitive supersession (currently 0.600)
- [ ] Tighter context filter (currently 0.750)
- [ ] Numerical supersession handling (rent 1500 → 1800, 8pm → 9pm)
- [ ] Adhyāsa rewrite-verification pass (second LLM confirms meaning preserved)
- [ ] Counterfactual full re-evaluation (replay from raw ingest, re-run NLI)
- [ ] Auto-detected contextuality (LLM classifier on ingest)

### Phase 3 (further out)

- [ ] Multi-user belief attribution (whose belief is it?)
- [ ] Probabilistic confidence (Bayesian networks)
- [ ] Learned contradiction policies (fine-tune on user corrections)
- [ ] **Phase 2 prototype** — contradiction detection + non-destructive supersession + temporal validity
- [ ] **BeliefEval** — benchmark that jointly stresses contradiction, supersession, and temporal validity
- [ ] Tokens-per-correct-answer curves as memory grows (first publication of this metric in the memory-systems literature)
- [ ] Qwen3-Embedding / Qwen3-Reranker (heavier models, if needed)
- [ ] ColBERT late-interaction verification

## Acknowledgments

Built with [Claude Code](https://claude.ai/code) as a pair-programming partner. Architectural decisions, evaluation design, and framing by the author. Code written in collaboration.

## License

MIT
