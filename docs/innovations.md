# What ships on the `ganita-layer` branch

One coherent architectural claim: **Patha separates retrieval queries from synthesis queries.** No mainstream AI memory system today does this. Both classes flow through the same top-K retrieval funnel everywhere else; the LLM is left to clean up.

The branch sits on top of `phase-2-belief-layer` at v0.9.3 (LongMemEval-S 500q = 0.952 on the unified pipeline; multi-session stratum = 0.857).

## The architectural claim

Different questions are different epistemic acts. **The *pramāṇa* — the Nyāya taxonomy of valid means of knowledge — is a near-complete map of the operations a memory must support, and Patha routes `recall()` by which one a question demands.** This is the load-bearing idea; the encoding mechanics (below) are supporting design philosophy.

| Class | Example | Pramāṇa | Path | Status |
|---|---|---|---|---|
| **Retrieval** | "What did I say about the saddle?" | *pratyakṣa* — perception | Phase 1 (dense + BM25 + RRF + reranker) → current-state filter → direct-answer or structured summary | shipped |
| **Synthesis** | "How much have I spent on bikes total?" | *anumāna* — inference | Gaṇita over preserved tuples, exhaustive, zero LLM tokens at recall (top-K runs in parallel for context but the answer is independent of it) | shipped |
| **Narrative** | "How has my thinking on agency evolved?" | *itihāsa* (emplotment of *śabda*) | Songline walk — a temporally-ordered traversal of a theme, ordered beats + supersession structure | wired, validating |

The reason: **top-K retrieval is the wrong primitive for synthesis *and* narrative.** "How much have I spent on bikes" has no single right session — it needs every bike-aliased tuple, summed; top-100 of 1000 misses 90%. "How has my thinking evolved" needs *sequence* — a ranked bag has no notion of order. Each pramāṇa is a primitive the others can't serve.

The alignment is not loose: pratyakṣa, anumāna, and śabda are distinct pramāṇa in Nyāya, and they correspond to genuinely distinct memory operations. The remaining pramāṇa map just as cleanly to operations Patha hasn't built yet (see "The epistemology roadmap" below) — which is why the philosophy is a *blueprint*, not ornament.

### Narrative synthesis (itihāsa) — now a first-class path

Earlier versions of this doc listed narrative synthesis as deferred. It is now **wired end-to-end** (`belief/itihasa.py` intent detector + `retrieval/narrative_walk.py` walker + a routing gate in `Memory.recall()`), and exercised by unit + end-to-end tests. It is not yet *validated on real data* — the songline graph's **topic channel** isn't populated (only entity/temporal/session/speaker edges exist today), and there's no authored evolution benchmark yet. So it's reachable but pre-release; it ships as a headline once those land.

The mechanism: a narrative question resolves a *theme*, anchors on the theme's beliefs (union of Phase-1 semantic top-K and direct entity-channel members — the gaṇita lesson, don't let top-K bound a synthesis query), then walks the songline graph *staying on-theme* (entity/topic edges always followed; temporal/session edges only to on-theme endpoints — the inverse of Phase 1's diversity-seeking `songline_walk`). It folds in each surfaced belief's supersession lineage (the "used to think X, now Y" beats), orders by time, and renders a deterministic through-line — **zero LLM tokens for the selection/ordering/supersession-tagging**; an LLM only verbalizes the pre-structured timeline.

This is also where the **songline graph finally earns its place.** Ablations show it contributes ≈0 to top-K retrieval R@5 — because top-K never needed graph traversal. Narrative is the first recall strategy where traversal is the *only* right primitive, so the graph investment becomes load-bearing exactly here.

### The epistemology roadmap — the remaining pramāṇa

The taxonomy predicts what's left to build, and one piece is already half-built:

- **Anupalabdhi** (non-apprehension) → *absence* queries: "what have I **not** decided about X?" The `abhāva` modules exist but are dormant; wiring them into a recall path is the next clean primitive.
- **Upamāna** (comparison) → *analogical* recall: "what does this remind me of? have I faced this before?" Similarity-across-difference, distinct from top-K-by-embedding.
- **Composition** → chaining pramāṇa: a *time-series of sums* (narrative + synthesis), e.g. "how has my spending evolved?" — a primitive no other memory system has.
- **Arthāpatti** (postulation) → *abductive* gap-fill: "what must be true, given X and Z?" Research-stage.

*Śabda* (testimony) isn't a query path — it's the substrate: every belief in the store **is** the user's own recorded word. The narrative path (itihāsa) is the emplotment of that testimony, which is why its epistemic grounding is śabda rather than a pramāṇa of its own.

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
| Synthesis answer independent of Phase 1 top-K | `test_synthesis_intent_independent_of_phase1` proves gaṇita answers correctly even when Phase-1 retrieval is forced to return `[]` |
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
