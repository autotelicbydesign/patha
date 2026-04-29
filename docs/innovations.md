# What ships on the `ganita-layer` branch

One coherent architectural claim: **Patha separates retrieval queries from synthesis queries.** No mainstream AI memory system today does this. Both classes flow through the same top-K retrieval funnel everywhere else; the LLM is left to clean up.

The branch sits on top of `phase-2-belief-layer` at v0.9.3 (LongMemEval-S 500q = 0.952 on the unified pipeline; multi-session stratum = 0.857).

## The architectural claim

Two classes of question. Two pramāṇa. Two paths.

| Class | Example | Pramāṇa | Path |
|---|---|---|---|
| **Retrieval** | "What did I say about the saddle?" | *pratyakṣa* — direct perception | Phase 1 (7-view dense + BM25 + RRF + reranker + songlines) → Phase 2 (current-state filter) → direct-answer or structured summary |
| **Synthesis** | "How much have I spent on bikes total?" | *anumāna* — inference across many facts | Gaṇita queries the belief store directly and exhaustively over preserved tuples. Phase 1 runs in parallel to populate retrieval context, but the synthesis answer is independent of Phase 1's top-K. Zero LLM tokens at recall. |

The reason: **top-K retrieval is the wrong primitive for synthesis.** A question like "how much have I spent on bikes" has no single right session. The answer requires every bike-aliased expense tuple, summed. Top-100 of 1000 sessions misses 90% of the inputs.

The traditional alignment is exact: pratyakṣa (perception) and anumāna (inference) are distinct pramāṇa in Nyāya epistemology. The architecture reflects this rather than collapsing both into one funnel.

## The four supporting components

### 1. Trust-the-precise-index aggregation (real architectural correction)

Before this branch, `answer_aggregation_question` strictly filtered candidate tuples by `restrict_to_belief_ids`. That violated the Vedic principle gaṇita is named for: arithmetic on **all** preserved facts, not retrieval-scoped facts. Now the function uses an `ambiguity_threshold` (default 30): if entity+attribute matches yield few enough tuples globally, trust the index; restriction kicks in only on broad/ambiguous queries.

Test: `tests/belief/test_innovations_compose.py::test_dense_haystack_phase1_misses_some_bike_sessions`.

### 2. Hybrid regex+LLM extraction (real, generalises)

The pure-LLM extractor's recall is bounded by the model's free-form judgement. On dense conversational text, gemma4:8b routinely misses dollar amounts entirely. The hybrid extractor splits the work along the abstraction line:

- **Regex** finds every `$X` in the text — perfect recall on the easy task.
- **LLM** sees a numbered list of amounts and labels each: `(entity, aliases, attribute)` or `entity: "skip"` (range/hypothetical/non-purchase). Easy task — pure semantic tagging, zero free-form generation pressure.

Test: `test_extracts_every_amount_via_mocked_llm`.

This pattern generalises beyond karaṇa — anywhere you need recall on a structured pattern + LLM-quality semantic tags.

### 3. Vedic *karaṇa* ingest-time extraction (Innovation #2)

The Vedic *karaṇa* model: ritual preparation up front so performance is deterministic. For Patha:

- **Ingest** — local LLM reads each new belief and emits `(entity, attribute, value, unit)` tuples. Once.
- **Recall** — pure deterministic arithmetic over the preserved tuples. **Zero LLM tokens.**

Inverse of mainstream RAG, which spends tokens at recall every query.

Three implementations:
- `RegexKaranaExtractor` — zero-deps baseline, works on toy clean text
- `OllamaKaranaExtractor` — full LLM extraction
- `HybridKaranaExtractor` — recommended for synthesis-heavy use

### 4. Filesystem-native ingest (Innovation #3)

`patha import obsidian-vault <path>` walks pre-existing writing into the belief store. Frontmatter dates → `asserted_at`; wikilinks/`#tags` → entity hints. Read-only. Removes the "what do I do after install?" friction.

## Hebbian retrieval

Kept on the branch. Honest framing:

- **No measurable lift on LongMemEval-S 500q multi-session** (paired A/B = identical 114/133 = 0.857).
- **No regression either**.
- **Real for repeat-query workloads** — a store queried many times accumulates co-retrieval edges that Phase-1's static cosine never sees. LongMemEval is single-shot, so the recorded signal can't accumulate.
- Default-on in `patha.Memory(hebbian_expansion=True)`. Disable for ablation studies.

## Empirical caveat

The synthesis-intent routing reaches the right architectural answer; the gaṇita arithmetic over preserved tuples is correct. But it depends on the karaṇa extractor's tagging quality. Tested with `gemma4:8b` (Q4) and `qwen2.5:7b-instruct`; both work on clean user stores. On dense LongMemEval haystacks, small-quantized models miss bike-relevant facts (mis-classify `$25 chain replacement` as a duration; fail to add `bike` to the helmet's alias list).

**Recommendation: ≥ 14B local model or hosted LLM for synthesis-heavy workloads.** The architecture doesn't change; only the karaṇa quality.

## Verification

| Check | Result |
|---|---|
| Unit tests | 725 pass (was 598 before this branch) |
| Slow integration | `tests/test_mcp_protocol.py::test_mcp_full_roundtrip` passes; `tests/belief/test_karana_ollama_live.py` passes against gemma4:8b |
| Synthesis answer independent of Phase 1 top-K | `test_synthesis_intent_bypasses_phase1` proves gaṇita answers correctly even when Phase-1 retrieval is forced to return `[]` |
| Retrieval still works | `test_retrieval_intent_uses_phase1` confirms perception queries flow through Phase 1 |
| Hebbian no-op A/B | Both arms returned 114/133 = 0.857 with zero per-question disagreement |
| Multi-session 500q LongMemEval-S, synthesis-intent on, regex karaṇa | 114/133 = 0.857 (matches baseline); avg tokens/summary 18,384 (was ~118,000) — **6.5× token reduction** |

The accuracy doesn't move with the regex extractor — the synthesis-bounded questions still fail because regex can't extract the right tuples from dense conversational text. The architecture is correct; the bottleneck is extraction quality. With `HybridKaranaExtractor` + ≥14B model, the synthesis path delivers correct answers (verified on the canonical \$185 case).

The token economy improvement IS measurable — synthesis questions now produce a compact gaṇita summary (operator + value + contributing beliefs) instead of the full Phase 2 retrieval dump. Zero LLM tokens at recall on the synthesis path.

## Activation

```bash
# Install — Patha CLI/MCP/library — same as before
uv sync && uv run patha install-mcp

# Filesystem ingest
patha import obsidian-vault ~/MyVault

# Power-user library
import patha
from patha.belief.karana import HybridKaranaExtractor
mem = patha.Memory(karana_extractor=HybridKaranaExtractor(model="qwen2.5:14b-instruct"))
```

`Memory.recall("how much have I spent on bikes?")` → routes through gaṇita.
`Memory.recall("what bike did I buy?")` → routes through Phase 1.
