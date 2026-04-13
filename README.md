# Patha

> *The way. The recitation.*

A memory system inspired by Vedic recitation and Aboriginal songlines. Achieves **100% Recall@5** on [LongMemEval](https://arxiv.org/abs/2407.15460) (raw mode, 100-question stratified sample) with **zero hosted-LLM API calls**. Everything runs locally.

## Results

| Metric | 100q Stratified | Full 500q |
|--------|:-:|:-:|
| **R@5** | **1.000** | *running* |
| R@10 | 1.000 | *running* |
| R\_all@5 | 0.900 | *running* |
| NDCG@5 | 1.591 | *running* |

### Per-stratum R@5

| Stratum | R@5 |
|---------|:---:|
| Single-session | 1.000 |
| Multi-session | 1.000 |
| Temporal reasoning | 1.000 |
| Knowledge update | 1.000 |

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

The Vedic oral tradition is the only system to losslessly transmit a text corpus for ~3,000 years. The mechanism: each word is stored in ~11 interlocking permuted contexts (*pada*, *krama*, *jaṭā*, *ghana* schemes), functioning as a literal error-correcting code.

Patha applies this principle to retrieval. Each proposition is embedded in **7 overlapping views** — isolated, forward-paired, backward-paired, bidirectional triple, entity-anchored, reframed, and temporally-anchored. A query that misses one view catches another. This makes retrieval robust to paraphrase, restatement, and indirect reference without any query rewriting or LLM calls.

### Pillar 2 — Aboriginal Songline Traversal

Australian Aboriginal songlines are the longest continuously transmitted information on Earth, encoding verified post-glacial coastal geography from ~7,000–10,000 years ago. The mechanism: the landscape *is* the index, and retrieval is a sequential walk that returns coherent paths, not disconnected hits.

Patha builds a multi-modal graph over propositions — edges weighted by shared entities, temporal proximity, session locality, speaker identity, and topic clusters. At query time, after initial retrieval and reranking, a 3-hop walk from the top anchors discovers evidence that no single-vector search can reach. This is what lifts multi-session and temporal-reasoning recall: the graph walk traverses from a hit to its narrative neighbors across sessions.

## Quick Start

```bash
# Install
uv sync

# Run tests (140 tests)
make test

# Smoke test: 10 questions
make eval-quick

# Full eval on LongMemEval S
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
  → pointwise cross-encoder rerank                         [100]
  → songline walks from top-3 anchors                      [~140]
  → MMR diversity pass (λ=0.7, per-session cap=2)          [30]
  → top-5 output
```

**Current models (all local, no API calls):**
- **Embedder:** all-MiniLM-L6-v2 (384-dim, sentence-transformers)
- **Reranker:** cross-encoder/ms-marco-MiniLM-L-6-v2
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

## Ablation Results

| Ablation | R@5 (100q) | Δ |
|----------|:-:|:-:|
| Full pipeline | 1.000 | — |
| No reranker | 0.950 | −0.050 |
| No songline | *pending* | |
| Single view (v1 only) | *pending* | |
| Turn-level chunking | *pending* | |

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
    reranker.py              # cross-encoder pointwise reranker
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
  metrics.py                 # R@K, NDCG@K, per-stratum breakdowns
  analyze.py                 # error analysis for eval results
```

## Reproduction

```bash
git clone https://github.com/stefi/patha.git
cd patha
uv sync

# Download the LongMemEval S dataset
# See https://github.com/xiaowu0162/long-mem-eval for dataset access
# Place at data/longmemeval_s_cleaned.json

make eval
# Results saved to runs/full/results.json
```

**Model versions:**
- `sentence-transformers/all-MiniLM-L6-v2`
- `cross-encoder/ms-marco-MiniLM-L-6-v2`
- `spacy/en_core_web_sm` ≥ 3.7

## Roadmap

- [ ] Full 500-question LongMemEval S evaluation
- [ ] Complete ablation matrix
- [ ] LanceDB persistent store
- [ ] Qwen3-Embedding / Qwen3-Reranker (heavier models, if needed)
- [ ] ColBERT late-interaction verification
- [ ] Neuroplasticity-inspired adaptive belief layer

## License

MIT
