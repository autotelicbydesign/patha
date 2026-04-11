# Patha

> *The way. The recitation.*

An AI long-term memory system targeting **≥99% R@5 on LongMemEval (raw mode, zero hosted-LLM calls)**.

Patha is built around two co-equal pillars borrowed from the two most rigorous human memory traditions ever documented:

1. **Vedic redundant encoding** — every atomic proposition is stored in seven interlocking embedding views modeled on the *pada / krama / jata / ghana* recitation schemes that preserved the Rig Veda losslessly for ~3,000 years. Paraphrase robustness is a structural property of the index, not a hoped-for emergent one.
2. **Aboriginal songlines** — retrieval is a weighted graph walk along shared entity / temporal / session / speaker / topic edges, returning a narrative *path* through memory instead of five disconnected hits. This is the only mechanism known to preserve multi-hop geographic/historical information across 10,000+ years.

The project is deliberately *not* a Greek memory palace. That metaphor was optimized for short-lived rhetorical recall, not lossless long-term preservation.

## Status

Early scaffolding. See `/Users/stefi/.claude/plans/abundant-tinkering-comet.md` for the full architecture plan.

## Stack

Python 3.11 + `uv`. LanceDB (not ChromaDB). `bm25s` for sparse. Qwen3-Embedding-4B primary + Stella-1.5B secondary (Hafiz-style ensemble check). Qwen3-Reranker-4B pointwise + listwise. ColBERTv2 / PLAID as a late-interaction verification layer. spaCy + dateparser for deterministic NER / temporal parsing.

No hosted LLM API is invoked anywhere in the pipeline.

## Layout

```
src/patha/
  chunking/     # propositionizer + 7 view constructors
  indexing/     # LanceDB, BM25, ColBERT, songline graph, ingest orchestrator
  retrieval/    # hybrid candidates, rerankers, colbert verifier, songline walker, pipeline
  query/        # PRF, temporal parsing, entity extraction
  models/       # embedder + reranker wrappers
eval/           # LongMemEval harness + ablation matrix + read-only Streamlit viewer
configs/        # default.yaml (the 99% config) + per-ablation configs
third_party/    # long-mem-eval submodule
```

## Development

```bash
uv sync --extra dev
uv run pytest
```

(ColBERT and vLLM extras are gated behind `--extra colbert` and `--extra vllm` and are not required for unit tests or early scaffolding work.)

## License

MIT
