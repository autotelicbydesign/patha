# Patha Phase 2 — Literature Survey

**Status:** DRAFT
**Purpose:** Ground Phase 2 design decisions (D1–D7) in prior work. Identify capability gaps we can honestly claim.
**Last updated:** 2026-04-17

---

## Executive summary

Five findings that should directly shape Phase 2 design:

1. **Contradiction handling is the clearest capability gap in open-source memory systems.** MemGPT, Mem0, Letta, and LangMem all support "update memory" via LLM calls but none publish a benchmark of *contradiction detection accuracy* or *supersession correctness*. Zep is the only system that even maintains a temporal edge model (bi-temporal validity via Graphiti), but its contradiction handling is implicit in the graph-update prompt, not a measured capability. This is the hole Phase 2 can credibly fill.

2. **The formal belief-revision literature (AGM, TMS) is mature and mostly ignored by modern LLM memory papers.** Patha gets cheap legitimacy from citing it, and more importantly the AGM rationality postulates (Success, Consistency, Preservation, Minimal change) give us a checklist for whether supersession is behaving correctly, not just "does the demo look good."

3. **Temporal validity is consistently underspecified.** Existing work treats "time" as a timestamp on a memory row; nothing in the production memory-system literature models *validity intervals with inference* (e.g., "training for a marathon" → ~4-month duration). Temporal KG literature (TimeR4, TempoQA) has the machinery but hasn't been plugged into conversational memory systems. Phase 2's Validity object is a direct port.

4. **Token economy is the unexamined axis.** (See section G.) No open-source memory system publishes tokens-per-query curves as memory grows. The dominant implicit assumption — "more retrieved context = better answers" — is load-bearing and unverified at scale. If Patha publishes a compression-ratio curve from 100 → 10 000 entries and shows constant or sublinear tokens-per-correct-answer, that alone is a novel contribution, independent of the belief-layer story.

5. **Belief-as-compression is an unclaimed framing.** Prompt compression (LLMLingua, Selective Context) compresses *text*. Memory summarization (MemGPT's recursive summary) compresses *history*. No one has framed a belief state (current claim + supersession lineage + validity window) as a compression primitive that answers queries without surfacing raw history. This is both a research claim and a product differentiator.

---

## A. Memory systems for LLMs (the direct competition)

### MemGPT (Packer et al., 2023, UC Berkeley)
- **Core idea:** OS-style virtual memory for LLMs. Main context + external "archival memory" with function-call interface.
- **Belief state handling:** none explicitly. Memory is written and rewritten via LLM-invoked `core_memory_append` / `core_memory_replace`. Contradictions are resolved by whatever the LLM decides to write.
- **Temporal model:** timestamps on messages; no validity inference; no supersession graph.
- **Benchmark:** Deep Memory Retrieval (DMR, synthetic) + document QA. No contradiction benchmark.
- **Gap for Phase 2:** no principled contradiction detection, no supersession, no validity. A belief layer on top of MemGPT's architecture would be strictly additive.

### Letta (formerly MemGPT, 2024)
- Productised MemGPT. Same architectural assumptions. Adds agent framework but no new belief semantics.

### Mem0 (Chhikara et al., 2024, updated arXiv:2504.19413, Apr 2025, ECAI 2025)
- **Core idea:** extract "memory facts" from conversations, store in vector DB + optional graph (Mem0g).
- **Belief state handling:** an `UPDATE` operation is performed at ingest: new fact is compared against top-k similar facts and LLM decides ADD / UPDATE / DELETE / NOOP. Mem0g adds an "update resolver" that marks conflicting graph relationships obsolete and tracks edge validity intervals.
- **This is the closest flat-store system comes to contradiction+supersession.** But: the UPDATE decision is a single LLM call with no published contradiction precision/recall numbers, no separation of contradiction detection from supersession policy, and DELETE is destructive in the flat variant.
- **Benchmark (2025 paper):** 91.6% on LoCoMo, 93.4% on LongMemEval, 91% p95 latency reduction vs full-context. Numbers are real and strong; the failure modes (destructive DELETE on false-positive contradictions) are invisible in these metrics.
- **Gap for Phase 2:** Mem0 collapses detection, policy, and storage into one LLM call. Patha splits them (D1, D3, D7) and measures each. Patha also avoids destructive supersession.

### Zep / Graphiti (Rasmussen et al., arXiv:2501.13956, Jan 2025)
- **Core idea:** temporal knowledge graph of entities and edges. Bi-temporal edges (valid-time via `valid_at`/`invalid_at` + system/ingestion time).
- **Belief state handling:** when a new edge is extracted, semantically/keyword-similar existing edges are retrieved and an LLM "invalidation prompt" asks whether the new edge contradicts any of them (per source at `graphiti_core/utils/maintenance/edge_operations.py`). On conflict, the old edge's `invalid_at` is set; old edge remains queryable as history. This is the closest prior work to what Phase 2 proposes.
- **Limits:** contradiction detection is an LLM call *per edge* at ingest — expensive at scale. No published precision/recall on contradiction. Temporal validity is interval-based but not inferred duration or decay.
- **Gap for Phase 2:** Zep demonstrates the shape of the solution (bi-temporal edges, non-destructive supersession) but doesn't publish rigour (measured contradiction accuracy, inferred validity), and incurs one LLM call per ingested edge. Patha can differentiate on (a) NLI-first detection with bounded LLM cost, (b) richer validity modes (decay, duration), (c) measured contradiction F1 on BeliefEval.

### LangMem, LlamaIndex memory modules, OpenAI Memory
- Similar pattern: add/replace memory via LLM, no belief semantics, no benchmark.

### Spatial-metaphor memory systems (2025)
- Recent academic work that achieves strong retrieval numbers on LongMemEval via hierarchical clustering + rerank, but treats memory ≡ retrieval; belief layer not in scope. We don't compete on this axis directly because our claim (separates retrieval from synthesis, non-destructive supersession) addresses a different surface.

---

## B. Formal belief revision (the ignored tradition)

### AGM (Alchourrón, Gärdenfors, Makinson, 1985)
- The canonical rational-agent belief revision framework. Three operations: **expansion** (add consistent belief), **revision** (add potentially conflicting belief, restore consistency), **contraction** (remove belief).
- Eight rationality postulates (Success, Inclusion, Vacuity, Consistency, Extensionality, …). These are a free test suite for Patha's supersession behaviour.
- **Direct mapping to Phase 2:** our `supersedes` edge is AGM revision. Phase 2 should at minimum check the Success postulate (new belief is present after revision) and Consistency postulate (active belief set has no direct contradictions).

### Truth maintenance systems (Doyle 1979, de Kleer 1986 ATMS)
- Each belief has a justification; when a justification fails, dependent beliefs are retracted.
- **Direct mapping:** `reinforced_by` is a weak justification link. An ATMS-style dependency tracker would let Phase 2 answer "why do you believe X?" — a natural Phase 3 extension but out of scope now.

### Belief base vs belief set
- Hansson (1999): revise the *base* (explicit assertions), derive the *set*. Matches Patha's separation: propositions are the base, beliefs are the derived current state.

### Non-monotonic reasoning, default logic (Reiter 1980)
- Relevant for validity decay: a belief holds "by default" absent contradicting information. The `decay` Validity mode is operationally a default-logic rule with a half-life.

---

## C. Contradiction detection (mechanisms for D1)

### NLI models (Option A in D1)
- **Default:** `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (HF). DeBERTa-v3-large fine-tuned on MNLI + FEVER-NLI + ANLI + LingNLI + WANLI. Reported SOTA across each constituent at release; ~435M params, ~60–150 ms/pair GPU.
- **Lighter alternative:** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (184M, ~15–30 ms GPU, small accuracy drop). `cross-encoder/nli-deberta-v3-small` and `-xsmall` from SBERT for edge deployment (22–70M params), with meaningful accuracy drop on adversarial NLI.
- DeBERTa-v3-base reports ~90% MNLI-matched accuracy; DeBERTa-v3-large on ANLI ≈ +8.3pp over ALBERT-XXL (DeBERTa-v3 paper). Contradiction-class F1 typically 2–4pp below overall accuracy.
- **Calibration caveat:** NLI models are better-calibrated than LLMs-as-judge but still overconfident on OOD inputs ("Lost in Inference," arXiv:2411.14103). Expect to need temperature scaling on a held-out slice before thresholding.
- Weakness: trained on premise-hypothesis pairs from crowdsourced data. Conversational contradictions with implicit context (same speaker, different sessions) are out of distribution.
- **Recommendation:** use as a fast pre-filter; expect ~80% recall on clear conversational contradictions, lower on context-dependent ones.

### Structured predicate extraction (Option B)
- OpenIE / CompactIE extract (subject, predicate, object). Comparing at the tuple level avoids surface-form noise.
- Weakness: extraction quality is the bottleneck; reported F1 on conversational text is 60-75%.

### LLM-as-judge (Option C)
- GPT-4-class: 90%+ agreement with human on pairwise contradiction (Zheng et al. 2023).
- Small local (Qwen2.5-7B): ~82% on similar tasks.
- Cost: 10-100x an NLI call. Latency: seconds.
- **Known calibration failures** (surveyed in "A Survey on LLM-as-a-Judge," arXiv:2411.15594 and "Overconfidence in LLM-as-a-Judge," arXiv:2508.06225): overconfident on uncertain pairs, position bias (verdict flips when argument order is swapped), and non-transitivity (A > B, B > C, C > A). Mitigate with position-swapped double evaluation and pinned-model reproducibility.

### Hybrid (Option D)
- NLI pre-filter → high-confidence pass-through, low-confidence → LLM. Used in production fact-checking pipelines (Google's FEVER-derived systems).

**Recommendation for Phase 2 prototype:** A (DeBERTa-MNLI) with LLM fallback scaffolded for later. D for production.

---

## D. Temporal validity and reasoning (mechanisms for D4)

### Temporal knowledge graphs
- TimeR4 (Jin et al. 2024), TempoQA: reason over (subject, predicate, object, timestamp) quadruples. Answer "what was X at time T?"
- These handle *time-indexed* facts, not *validity-bounded* facts. "Born in Sofia" has no end; "training for marathon" has an inferred end. TKG literature doesn't model the latter.

### Temporal expression extraction
- HeidelTime, SUTime: extract "tomorrow", "next month", "for 3 weeks" from text. Mature (~85% F1).
- Directly usable for Option C (rule-based heuristics in D4).

### Event duration modeling
- Pan et al. 2011 (TempEval), McTaco (2019): predict typical durations of events. "Marathon training" → months.
- Could seed the `duration` inference for Validity.

### Temporal logics
- Linear Temporal Logic, Allen's interval algebra. More expressive than Patha needs; useful vocabulary for documenting the spec.

**Recommendation for D4:** rule-based (HeidelTime-style patterns) + small LLM fallback for ambiguous cases. Matches D4 Option C.

---

## E. Neuroplasticity as framing (risk register)

- Elastic Weight Consolidation (Kirkpatrick et al. 2017): slow plasticity on "important" weights. Loosely maps to Patha's confidence-weighted supersession — but this is a neural-network training technique, not a memory-system mechanism.
- Complementary Learning Systems theory (McClelland, O'Reilly 1995): fast episodic + slow semantic. Maps cleanly to proposition store (episodic) + belief layer (semantic). **This is the mapping that earns its keep.**
- Synaptic consolidation / reconsolidation: each retrieval re-encodes memory. `reinforced_by` is a weak analogue.
- **Honest framing:** call Phase 2 a "belief maintenance layer inspired by complementary learning systems" — defensible. Avoid claiming plasticity mechanisms unless the mapping is mechanistic.

---

## F. Evaluation benchmarks

- **LongMemEval** (Wu et al., arXiv:2410.10813, ICLR 2025): 500 questions, 5 abilities: information extraction, multi-session reasoning, temporal reasoning, **knowledge updates**, abstention. **The knowledge-updates slice does explicitly test supersession** — the earlier draft of this survey underplayed it. Commercial assistants report 20–85% on KU; Mem0 (Apr 2025) reports 93.4% overall. Before building BeliefEval, Patha should report its score on LongMemEval-KU specifically — if Phase 1 already scores high there, that reshapes the Phase 2 claim.
- **LoCoMo** (Maharana et al., ACL 2024, arXiv:2402.17753): ~35-session conversations, QA + summarisation + multimodal. Annotators explicitly *remove contradictions* during dataset construction → LoCoMo is a coherence benchmark, not a contradiction benchmark. Useful as a secondary axis.
- **ReviseQA** (OpenReview Z4KBiAYXlI, 2025): **the most direct prior benchmark for belief revision** — multi-turn logical reasoning where each turn adds/retracts facts and the model must revise. Operates on abstract logical facts, not personal memory. BeliefEval should position itself as the personal-memory/temporal-validity analogue and cite ReviseQA as precedent.
- **MemoryBench** (arXiv:2510.17281, Oct 2025), **MemBench** (arXiv:2506.21605), **MemoryAgentBench** (ICLR 2026): newer benchmarks for memory/continual learning; none specifically contradiction-targeted.
- **TimeBench** (ACL 2024, arXiv:2311.17667), **Test of Time** (arXiv:2406.09170), **TempReason**, **TimeQA**: temporal reasoning suites. Secondary axis for D4 validity validation.
- **No public benchmark jointly stresses contradiction + supersession + validity on personal-memory data.** BeliefEval has a genuine niche, but its pitch should be "jointly stresses what LongMemEval-KU and ReviseQA each test in isolation," not "first benchmark of its kind."
- **Token-economy benchmarks:** none. LongBench-v2 reports context-length costs but not per-query tokens under a memory system. This is an open lane.

---

## G. Token economy and memory-as-compression

### G.1 Who actually measures this?

Almost nobody. Surveying the systems in section A:

- **MemGPT:** reports latency and function-call counts, not tokens per query. Acknowledges context overflow as the motivating problem but doesn't publish compression-ratio numbers.
- **Zep / Graphiti:** publishes retrieval latency and answer accuracy (DMR, LongMemEval). No tokens-per-query curve as graph grows.
- **Mem0:** the Mem0 paper reports latency improvements over full-context baselines and claims "~90% token reduction" vs. sending the full conversation — but this is vs. the no-memory baseline, not vs. naive RAG, and is not broken out by memory size.
- **Letta, LangMem, LlamaIndex memory, OpenAI Memory:** no published token-economy numbers.
- **Spatial-metaphor systems on LongMemEval:** retrieval metrics only (R@5, R@10, MRR).

The only numbers close to what Phase 2 needs come from *long-context benchmark papers* (LongBench, RULER) which measure cost per query but don't involve a memory system — they just vary context length.

**Finding:** tokens-per-query and compression-ratio-as-memory-grows are essentially unmeasured in the public memory-systems literature. Publishing these curves is a cheap, differentiated contribution.

### G.2 De-facto numbers for naive RAG

For a naive top-k RAG baseline with k=5 chunks of ~200 tokens each plus ~500 tokens of system/question prompt: **~1500 tokens-in per query**, roughly constant with memory size (retrieval cost is the hidden variable, not prompt tokens).

But "naive RAG" in many production LLM-memory stacks is *not* top-k — it's "dump last N messages + top-k retrievals". At 1000-entry memory (≈ a month of heavy use), the last-N-messages tail alone is commonly 2000–4000 tokens; at 10 000 entries, session-level context windows start ballooning to 8–20 k tokens if no summarization is applied. This is where the crossover (§G.4) lives.

Quantitative anchor points from the literature:
- Liu et al. "Lost in the Middle" (2023): context scaling beyond ~10 k tokens *degrades* answer accuracy on multi-doc QA — so more retrieval ≠ better answers.
- LongBench-v2 (2024): typical input 8 k–128 k, answer quality plateaus or drops past 32 k for most models.
- Mem0 paper (2024): claims 26× token reduction vs full-conversation baseline at unspecified memory size, on LOCOMO.

### G.3 Memory-as-compression literature

Four strands, none integrated:

1. **Prompt compression.**
   - **LLMLingua / LongLLMLingua** (Jiang et al. 2023): perplexity-based token pruning. 2–20× compression, 1–5% accuracy loss. Operates on already-retrieved prompts, not on memory structure.
   - **Selective Context** (Li et al. 2023): similar perplexity-filtering approach.
   - **Recurrent Context Compression / AutoCompressor** (Chevalier et al. 2023): train soft-token summaries of context. ~4× at inference.
   - These are orthogonal to belief-layer work — they could be stacked on top of whatever Patha sends to the LLM.

2. **Sketch-based memory.**
   - **Compressive Transformer** (Rae et al. 2019): compressed representations of old context.
   - **Memorizing Transformers** (Wu et al. 2022): kNN over key-value cache.
   - Operate in embedding space, not symbolic. Orthogonal to belief semantics.

3. **Hierarchical memory with summarization.**
   - **MemGPT recursive summarization:** when main context overflows, older content is summarized into archival. Reduces tokens but loses fidelity; no supersession semantics.
   - **A-MEM, HippoRAG, RAPTOR** (Sarthi et al. 2024): hierarchical clustering + summarization of retrieved chunks. Improves retrieval + reduces tokens, but treats memory as text, not belief.

4. **Retrieval + reranking + summarization pipelines.**
   - Standard in production: retrieve 20, rerank to 5, summarize to a paragraph. Reduces tokens-in but is lossy and not belief-aware.

**None of these frame memory itself as compression-by-belief.** The closest is Zep's graph, which implicitly compresses history (you query the current edge, not the edit log), but Zep does not publish compression-ratio numbers or frame the system this way.

### G.4 Does memory increase or decrease tokens?

Honest answer: **for most production memory systems, total tokens-per-correct-answer initially increases vs. a short-conversation baseline, then decreases only after the conversation gets long enough that full-history replay is prohibitive.**

- At 10 turns of conversation (≈ 2 k tokens): full-history baseline costs ~2 k in, memory system costs ~1.5 k in (retrieved chunks + prompt scaffolding) — small saving.
- At 100 turns (≈ 20 k tokens): full-history costs 20 k; memory system still ~1.5 k. 13× saving.
- At 1000 turns: full-history exceeds context window; memory system still ~1.5 k in principle, though naïve top-k starts missing important facts.

The crossover is roughly the point where conversation history exceeds ~2× the retrieval prompt budget, which with k=5×200 tokens lands around **5 k–8 k tokens of conversation**, or ~25–40 turns.

**The honest claim Patha can make:** below the crossover, memory systems are overhead; above it, they are essential. Patha's belief layer should *not* inflate the retrieval prompt (it should replace it for belief-lookup queries, per D7 Option C), otherwise it raises the crossover point.

### G.5 Belief-layer compression as a novel primitive

**Search result: no prior work explicitly treats belief state (current belief + supersession lineage) as a compression primitive.**

Adjacent work:
- **Temporal KG QA** (TimeR4, CronKGQA): answers time-indexed queries without surfacing full history — structurally similar to "return current belief, hide supersession chain by default." But these systems answer queries *about* time, not *using* time to compress.
- **SPARQL-over-temporal-KG:** can answer "latest value of property P for entity E" in one query. Operationally identical to a belief-lookup. Just never framed as compression.
- **Event sequence summarization** (timeline summarization, Martschat & Markert 2018): produces condensed timelines from news streams. Closer in spirit but operates on documents, not on assertions by a single agent.

**The unclaimed framing:** a belief is a compressed representation of its supersession lineage. Instead of sending "User said X on day 1; User said Y on day 30 superseding X; User said Z on day 45 superseding Y" (≈ 60 tokens) to the LLM, send "Current belief: Z (as of day 45)" (≈ 10 tokens). That is 6× compression *on a single belief chain*, compounding as chains lengthen.

This framing has three testable consequences BeliefEval should measure:
1. **Tokens per correct lookup** for current-belief queries should be strictly less than naive RAG (target: ≥5× reduction).
2. **Compression ratio** should grow with memory age — older memories have longer lineages, hence more compression.
3. **Accuracy must not drop** on history-requesting queries; those must surface the full lineage on demand.

If BeliefEval shows all three, Patha has a defensible claim that belief-layer ≠ prettier RAG — it is fundamentally a compression mechanism that happens to also be correct.

---

## H. What Phase 2 can cite vs. what it must originate

**Cite freely:**
- AGM / TMS as the formal backbone
- Zep / Graphiti as prior art on bi-temporal edges
- Mem0 as prior art on ingest-time update decisions
- NLI / HeidelTime / temporal KG literature for component mechanisms
- Complementary Learning Systems for the one honest neuroscience mapping

**Must originate (or at minimum, publish first clean measurement of):**
- A benchmark that jointly stresses contradiction + supersession + validity (BeliefEval)
- Tokens-per-correct-answer curves as memory grows
- The belief-as-compression framing and its empirical validation

---

## I. Recommendations for D1–D6 (grounded in the above)

### D1 — Contradiction detection mechanism
**Option D (hybrid), concretely specified.** Primary: `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`; threshold contradiction logit at τ_high (auto-flag) and τ_low (auto-reject); pairs in [τ_low, τ_high] escalate to a small LLM judge with position-swapped double evaluation. Calibrate τ on a held-out BeliefEval split. This is cheaper per ingest than Graphiti's per-edge LLM call, bounds latency predictably, and preserves flexibility for hard cases (§A Zep, §C NLI+LLM calibration). For v0.1 prototype: NLI alone is fine; scaffold the LLM-fallback path for v0.2.

### D2 — Ingest-time vs. query-time
**Hybrid with a sliding window.** Ingest-time detection against the last 30 days of beliefs (bounded cost; matches Graphiti's default); query-time fallback for older beliefs when they surface in retrieval. This gives the "you said the opposite last Tuesday" demo without O(N) ingest cost. Graphiti and Mem0 both run ingest-time unconditionally — the 30-day window is a concrete differentiator on cost per ingest at scale.

### D3 — Supersession opinionation
**Neutral default, Confidence-weighted behind a flag.** For v0.1, store both propositions with timestamps and return newer-first with an explicit supersession pointer; do not delete. Mem0 flat's destructive DELETE is precisely the silent-failure mode to avoid — a false-positive contradiction permanently erases correct belief. The confidence-weighted variant maps onto Hansson's non-prioritized revision (§B) and should be the v0.2 upgrade. This is also where Patha visibly differs from Mem0 flat in demos.

### D4 — Validity assignment
**Option C: rule-based + LLM fallback.** LLMs are demonstrably bad at temporal reasoning (§D, TimeBench, Test of Time — even GPT-4 scores 66% on TRACIE). Ship a hand-written rule set for common patterns ("until X", "for N weeks/months", "through <date>", "this weekend", "next quarter", tense-shift detection). Call an LLM only when no rule fires *and* a temporal marker word is present. Default `Validity = permanent` when nothing fires. HeidelTime / SUTime are the right references for the rule engine. Explicit user markers override everything.

### D5 — Capability demo
**"What do you currently believe about X, and when did that change?"** Build a ~10-step preference-shift scenario: user states a preference ("I love sushi"), later changes it ("avoiding raw fish for six months"), returns after supersession window. Demo shows Patha returning current belief *and* visible supersession lineage with timestamps. Contrast: Mem0 flat (destructive delete — no history), Letta (LLM-overwrite — no lineage), Zep (bi-temporal edges, but no explicit lineage query surface). Single concrete outcome; reproducible; beats the field because no open-source system ships this demo end-to-end.

### D6 — BeliefEval initial scope
**~150 scenarios, hand-curated seeds with LLM-assisted expansion, human-QA'd.** Three families (50 each): (a) preference supersession, (b) factual supersession with temporal overlap, (c) temporally-bounded assertions (with and without explicit end). Modelled on ReviseQA's structure. Release: construction methodology, inter-annotator agreement, baselines on Mem0, Zep, LangMem, and Patha Phase-1-only. Submit to NeurIPS D&B or ICLR benchmark track — do not self-publish only. Inflated headline numbers that don't survive external review are the cautionary precedent.

---

## J. References

### Papers
- Alchourrón, Gärdenfors, Makinson, "On the Logic of Theory Change: Partial Meet Contraction and Revision Functions," *J. Symbolic Logic* 50(2):510–530, 1985.
- Chhikara et al., "Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory," arXiv:2504.19413, Apr 2025. https://arxiv.org/abs/2504.19413
- de Kleer, "An Assumption-Based TMS," *Artificial Intelligence* 28(2):127–162, 1986.
- Doyle, "A Truth Maintenance System," *Artificial Intelligence* 12(3):231–272, 1979.
- Gärdenfors, *Knowledge in Flux*, MIT Press 1988.
- Graves et al., "Neural Turing Machines," arXiv:1410.5401, 2014.
- Graves et al., "Hybrid computing using a neural network with dynamic external memory," *Nature* 538:471–476, 2016.
- Hansson, "In Defense of Base Contraction," *Synthese* 91(3):239–245, 1992.
- Hansson, "A Survey of Non-Prioritized Belief Revision," *Erkenntnis* 50:413–427, 1999.
- Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks," *PNAS* 114(13):3521–3526, 2017. https://arxiv.org/abs/1612.00796
- Maharana et al., "Evaluating Very Long-Term Conversational Memory of LLM Agents," ACL 2024. https://arxiv.org/abs/2402.17753
- McClelland, McNaughton, O'Reilly, "Why there are complementary learning systems in the hippocampus and neocortex," *Psychological Review* 102(3), 1995.
- Nebel, "A Knowledge Level Analysis of Belief Revision," KR 1989.
- Packer et al., "MemGPT: Towards LLMs as Operating Systems," arXiv:2310.08560, 2023.
- Rasmussen et al., "Zep: A Temporal Knowledge Graph Architecture for Agent Memory," arXiv:2501.13956, Jan 2025. https://arxiv.org/abs/2501.13956
- Reiter, "A Logic for Default Reasoning," *Artificial Intelligence* 13, 1980.
- Wu et al., "LongMemEval," ICLR 2025, arXiv:2410.10813. https://arxiv.org/abs/2410.10813
- Cai et al., "A Survey on Temporal Knowledge Graph," arXiv:2403.04782.
- Chu et al., "TimeBench," ACL 2024, arXiv:2311.17667.
- Chen et al., "Test of Time," arXiv:2406.09170, Jun 2024.
- Li et al., "A Survey on LLM-as-a-Judge," arXiv:2411.15594, Nov 2024.
- "Overconfidence in LLM-as-a-Judge," arXiv:2508.06225.
- "Lost in Inference: Rediscovering the Role of NLI for LLMs," arXiv:2411.14103.
- "MemoryBench," arXiv:2510.17281.
- "ReviseQA," OpenReview Z4KBiAYXlI, 2025. https://openreview.net/forum?id=Z4KBiAYXlI
- "How Should Rational Belief Revision Work in LLMs?", arXiv:2406.19354.
- Stanford Encyclopedia of Philosophy, "Logic of Belief Revision." https://plato.stanford.edu/entries/logic-belief-revision/

### Repositories
- Letta: https://github.com/letta-ai/letta
- Zep / Graphiti: https://github.com/getzep/graphiti
- Mem0: https://github.com/mem0ai/mem0
- LangMem: https://github.com/langchain-ai/langmem
- LongMemEval: https://github.com/xiaowu0162/LongMemEval
- LoCoMo: https://github.com/snap-research/locomo
- TimeBench: https://github.com/zchuz/TimeBench
- MemoryAgentBench: https://github.com/HUST-AI-HYZ/MemoryAgentBench
- JTMS reference: https://github.com/hbeck/jtms

### HuggingFace models
- `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`
- `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
- `cross-encoder/nli-deberta-v3-small` / `-xsmall`

---

## Appendix — Gaps worth tracking

- Multi-user belief (whose belief is it?) — deliberate out-of-scope, flag for Phase 3
- Probabilistic belief (Bayesian nets) — out of scope, but if Phase 2 confidence scalars prove insufficient, this is the upgrade path
- Learned contradiction policies (fine-tune on user's historical corrections) — out of scope, but this is where personalization compounds
