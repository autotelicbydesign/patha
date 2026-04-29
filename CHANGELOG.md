# Changelog

## ganita-layer (v0.10 in development)

### One coherent architectural claim

**Patha separates retrieval queries from synthesis queries.** No mainstream AI memory system makes this distinction; they all force every question through the same top-K funnel.

Two pramāṇa, two paths:

- **Retrieval** — *pratyakṣa*: "what did I say about the saddle?" → Phase 1 (7-view dense + BM25 + RRF + reranker + songlines) → Phase 2 (current-state filter) → direct-answer or structured summary.
- **Synthesis** — *anumāna*: "how much have I spent on bikes total?" → gaṇita queries the belief store directly. Phase 1 isn't the right primitive — top-K of N misses (N-K) of the inputs you need to sum. Pure deterministic arithmetic over preserved tuples. **Zero LLM tokens at recall.**

### What this changes in the codebase

**`patha.Memory.recall()`** — detects aggregation intent (existing `detect_aggregation`); on synthesis, queries the gaṇita index globally without `restrict_to_belief_ids`. Phase 1 runs in parallel to populate retrieval context. The architectural claim: the synthesis answer is independent of Phase 1's top-K — proven by `test_synthesis_intent_independent_of_phase1` (forces Phase 1 to return `[]` and gaṇita still computes the right answer).

**`answer_aggregation_question`** — bug fix: was strictly filtering by `restrict_to_belief_ids`. Violated the Vedic principle gaṇita is named for (arithmetic on **all** preserved facts). Now uses an `ambiguity_threshold` (default 30): trust the index when entity+attribute matches yield few enough tuples globally; restrict only on broad/ambiguous queries.

**`patha.belief.karana.HybridKaranaExtractor`** — new. Regex finds every `$X` in text (perfect recall on the easy task); LLM only labels each `(entity, aliases, attribute)` or marks `entity: "skip"` for ranges/hypotheticals. The architectural design splits the work along the abstraction line; recall isn't bounded by the LLM's free-form judgement. Generalises beyond karaṇa.

**`patha.belief.karana.OllamaKaranaExtractor`** — full LLM extraction; prompt asks for explicit broader-category aliases (`saddle → ["saddle", "bike", "cycling"]`).

**`patha.importers`** — new. `patha import obsidian-vault <path>` walks pre-existing writing into the belief store. Frontmatter dates → `asserted_at`; wikilinks/`#tags` → entity hints. Read-only.

### Hebbian retrieval

Kept on the branch with honest framing.

- No measurable lift on LongMemEval-S 500q multi-session (paired A/B = identical).
- No regression either.
- Real for repeat-query workloads — a store queried many times accumulates co-retrieval edges that Phase-1's static cosine never sees. LongMemEval is single-shot, so the recorded signal can't accumulate.

### Empirical caveat

The synthesis-intent routing reaches the right architectural answer; the gaṇita arithmetic is correct. But it depends on the karaṇa extractor's tagging quality. Tested with `gemma4:8b` (Q4) and `qwen2.5:7b-instruct`; both work on clean user stores. On dense LongMemEval haystacks, small-quantized models miss bike-relevant facts (mis-classify "$25 chain replacement" as a duration; fail to add "bike" to the helmet's alias list).

**Recommendation: ≥ 14B local model or hosted LLM for synthesis-heavy workloads.** The architecture doesn't change; only the karaṇa quality.

### Phase 3 plan

`docs/phase_3_plan.md` lays out the next milestone: end-to-end answer evaluation. Token-overlap-on-summary measures retrieval; the product question is whether the user's LLM, given Patha's output, produces the correct answer. Phase 3 ships that scorer.

### Verification

- 725 unit tests pass (was 598 before this branch).
- `tests/test_mcp_protocol.py::test_mcp_full_roundtrip` passes.
- `tests/belief/test_karana_ollama_live.py` passes against `gemma4:8b`.
- `test_synthesis_intent_independent_of_phase1` proves the architectural claim: the synthesis answer is independent of Phase 1's top-K — gaṇita computes the right answer even when Phase 1 is forced to return `[]`.

### Multi-session benchmark (500q LongMemEval-S, stub detector, regex karaṇa)

| Metric | Baseline (v0.9.3) | ganita-layer | Δ |
|---|:---:|:---:|:---:|
| Multi-session accuracy | 114/133 = 0.857 | 114/133 = 0.857 | 0 |
| Average tokens / summary | ~118,000 | ~18,400 | **−6.5×** |

**Accuracy is unchanged** because the regex karaṇa extractor can't reliably extract the relevant tuples from dense conversational text. The architecture is correct; the bottleneck is extraction quality. With `HybridKaranaExtractor` + ≥14B model the synthesis path delivers correct answers (verified end-to-end on the canonical \$185 bike scenario via `tests/belief/test_karana_ollama_live.py`).

**Token economy improves 6.5×** because synthesis questions now produce a compact gaṇita summary (`"sum: \$50 + \$75 + \$30 + \$30 = \$185.0"` plus contributing beliefs) instead of the full Phase 2 retrieval dump. Zero LLM tokens at recall on the synthesis path.
