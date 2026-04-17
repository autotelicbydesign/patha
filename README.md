# Patha

> *The way. The recitation.*

An AI memory system built on multi-view redundant encoding and narrative traversal. Inspired by Vedic recitation and Aboriginal songlines. Runs fully local with **zero hosted-LLM API calls**.

## Results

### Headline numbers

| Benchmark | R@5 | R@10 | Notes |
|-----------|:---:|:----:|:------|
| **LongMemEval S — 100q stratified sample** | **0.989** | 0.989 | Baseline pipeline with `rrf_blend=0.2` |
| **LongMemEval-KU — full subset (all 78 questions)** | **1.000** (78/78) | 1.000 | Knowledge-update subset |
| Full 500q LongMemEval S | *not yet run* | — | Requires >32 GB RAM for session cache |

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
                    ┌──────────────────────────────────────┐
      raw           │         INGESTION (Patha)            │
   conversations ──▶│  propositionize → 7 view fingerprints│
                    │       → proposition store             │
                    └────────────────┬─────────────────────┘
                                     │
                                     ▼
                    ┌──────────────────────────────────────┐
                    │       SONGLINE GRAPH BUILD            │
                    │  bind propositions to entity / time / │
                    │  topic / speaker modalities → edges   │
                    └────────────────┬─────────────────────┘
                                     │
 query ─▶ hybrid candidates ─▶ rerank ─▶ songline walk ─▶ MMR ─▶ top-5
         (BM25 + 7 dense views)  (CE)      (graph)
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
  chunking/
    propositionizer.py       # deterministic rule-based proposition splitter
    views.py                 # 7 Vedic view constructors
  indexing/
    store.py                 # in-memory proposition store
    bm25_index.py            # BM25 sparse retrieval
    songline_graph.py        # multi-modal graph (entity/time/topic edges)
    ingest.py                # ingestion orchestrator
  retrieval/
    hybrid_candidates.py     # 7-view dense + BM25 → RRF fusion
    reranker.py              # cross-encoder with RRF blend
    songline_walker.py       # 3-hop weighted graph walks
    mmr.py                   # MMR diversity with session cap
    pipeline.py              # full query orchestrator
  query/
    prf.py                   # RM3 pseudo-relevance feedback
    temporal.py              # temporal expression extraction
    entities.py              # spaCy NER enrichment
  models/
    embedder.py              # embedder protocol + stub
    embedder_st.py           # sentence-transformers wrapper

eval/
  runner.py                  # LongMemEval evaluation harness
  ablations.py               # ablation matrix runner
  metrics.py                 # R@K, NDCG@K, per-stratum breakdowns
  viewer/                    # Streamlit results viewer

docs/
  phase_2_spec.md                    # Phase 2 (belief layer) design — DRAFT
  phase_2_literature_survey.md       # Phase 2 literature survey — DRAFT
```

## Phase 2 — Belief Layer (in design)

Phase 1 (this repo, above) handles retrieval. Phase 2 adds a belief-maintenance layer on top of retrieval that tracks contradiction, supersession, and temporal validity — the dynamic aspects of memory that current AI memory systems handle by dumping everything into the LLM's context window and hoping it sorts things out.

See [docs/phase_2_spec.md](docs/phase_2_spec.md) for the full design. Three capabilities:

1. **Contradiction detection** — pairwise NLI with LLM fallback for ambiguous cases.
2. **Supersession** — non-destructive belief replacement with full lineage preserved. When a new assertion conflicts with an existing one, the old one is marked superseded, not deleted. Queries return the current belief; history is available on request.
3. **Temporal validity** — beliefs have lifespans (permanent, dated range, inferred duration, decay by default). Rule-based extraction + LLM fallback.

**Token economy as a first-class evaluation axis.** Phase 2 is explicitly designed to *reduce* tokens per correct answer, not increase them. A belief state compresses a chain of assertions (e.g., five preference updates) into a single current belief with optional lineage — often a 5-10× compression on its own. See [docs/phase_2_spec.md §4.3](docs/phase_2_spec.md) and [docs/phase_2_literature_survey.md §G](docs/phase_2_literature_survey.md).

## Roadmap

- [x] 100-question stratified eval with full ablation matrix
- [x] LongMemEval-KU full subset evaluation (78/78 correct, beats Mem0 by 6.6 pts)
- [x] RRF blend fix preventing single-reranker veto of multi-view consensus
- [x] Atomic checkpointing + per-question eval resume (for long-running evals)
- [x] Phase 2 design spec and literature survey
- [ ] Full 500q LongMemEval S eval — pending a machine with >32 GB RAM
- [ ] End-to-end answer accuracy eval (generation layer over retrieved context)
- [ ] LanceDB persistent store
- [ ] **Phase 2 prototype** — contradiction detection + non-destructive supersession + temporal validity
- [ ] **BeliefEval** — benchmark that jointly stresses contradiction, supersession, and temporal validity
- [ ] Tokens-per-correct-answer curves as memory grows (first publication of this metric in the memory-systems literature)
- [ ] Qwen3-Embedding / Qwen3-Reranker (heavier models, if needed)
- [ ] ColBERT late-interaction verification

## Acknowledgments

Built with [Claude Code](https://claude.ai/code) as a pair-programming partner. Architectural decisions, evaluation design, and framing by the author. Code written in collaboration.

## License

MIT
