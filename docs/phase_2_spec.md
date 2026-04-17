# Patha Phase 2 — Belief Layer Specification

**Status:** DRAFT — decisions pending
**Owner:** Stefi (architecture) + Claude Code (implementation)
**Last updated:** 2026-04-17

---

## 1. Purpose and scope

Phase 1 of Patha is a retrieval system: given a question, return the most relevant propositions from memory. Phase 2 adds a *belief layer* that tracks how propositions relate, update, and expire over time.

The Phase 2 claim is that current AI memory systems handle the *static retrieval* problem reasonably well but fail on the *dynamic belief* problem — what does the user currently believe, what used to be true, what is contradicted by newer evidence, what has an expired validity window. A system that handles these is solving a capability gap, not just improving retrieval precision.

**Phase 2 is not:** another retrieval improvement, a better reranker, a larger benchmark.
**Phase 2 is:** a layer on top of retrieval that reasons about belief state across time.

## 2. Three capabilities, defined precisely

### 2.1 Contradiction detection

**Informal:** given two propositions in memory, determine if they conflict.

**Formal target:** a function `contradicts(P1, P2) -> (label: {CONTRADICTS, ENTAILS, NEUTRAL}, confidence: [0,1], rationale: string)`.

**Examples (positive):**
- P1: "I love sushi" / P2: "I'm avoiding raw fish for six months" → CONTRADICTS (for the overlap period)
- P1: "Our pricing is $29/mo" / P2: "We raised the price to $39/mo in Q3" → CONTRADICTS (with supersession)
- P1: "Living in Sydney" / P2: "Moved to Sofia last month" → CONTRADICTS (with supersession)

**Examples (negative — must NOT flag):**
- P1: "I enjoy spicy food" / P2: "I had a mild curry for lunch" → NEUTRAL
- P1: "Meeting at 2pm" / P2: "Meeting at 3pm" (different meetings) → NEUTRAL

**Hard cases the spec needs to resolve:**
- Context-dependent contradictions (same proposition, different scopes)
- Implicit temporal markers ("I'm training for a marathon" — has implicit duration)
- Unit/measurement differences ("6 feet" vs "1.83m") — not contradictions
- Personal pronoun resolution ("I" in one session vs. "I" in another — same speaker?)

**Decision needed (D1):** What detection mechanism?
- **Option A — NLI model.** Fine-tuned entailment model (e.g., DeBERTa-MNLI). Runs over (P1, P2) pairs. Fast, interpretable, bounded accuracy.
- **Option B — Structured predicate extraction.** Extract (subject, predicate, object, qualifiers) from each proposition, compare at the structured level. More principled but depends on reliable extraction.
- **Option C — LLM-as-judge.** Small local LLM (e.g., Gemma/Qwen 7B) asked "do these contradict?". Most flexible, highest cost, hardest to reproduce.
- **Option D — Hybrid.** NLI pre-filter + structured check + LLM fallback for hard cases.

Recommendation pending survey: probably A for the prototype, D for production.

### 2.2 Supersession

**Informal:** when a newer proposition contradicts an older one, the newer one "wins" by default, but the older one isn't deleted.

**Formal target:** each belief has a `supersedes: list[belief_id]` and `superseded_by: list[belief_id]`. The system maintains a partial order of supersession. Queries return *current* beliefs unless explicitly asked for history.

**Data model sketch:**
```python
@dataclass
class Belief:
    id: str
    proposition: str
    asserted_at: datetime
    asserted_in_session: str
    confidence: float  # [0, 1]
    validity: Validity  # see 2.3
    supersedes: list[str]  # belief_ids
    superseded_by: list[str]  # belief_ids
    reinforced_by: list[str]  # belief_ids (repeat assertions)
    source_proposition_id: str  # link to Phase 1 proposition
```

**Decision needed (D2):** When do we detect contradictions — ingest-time or query-time?
- **Ingest-time:** check new propositions against existing at the moment of storage. Expensive (O(N) per ingest). Enables "you said the opposite — which one?" prompts. Changes the ingest surface.
- **Query-time:** only check when a query could be affected. Cheaper. Late detection of important contradictions.
- **Hybrid:** ingest-time for "recent window" (last 30 days?), query-time for older.

**Decision needed (D3):** How opinionated is supersession?
- **Aggressive:** newer claim wins, older marked superseded, not returned by default. (Most visible as a capability, most risky to get wrong.)
- **Neutral:** both stored, surfaced with timestamps, user resolves. (Most honest, least impressive demo.)
- **Confidence-weighted:** whichever claim has stronger evidence/repetition wins. (Most sophisticated, hardest to implement correctly.)

### 2.3 Temporal validity

**Informal:** beliefs have lifespans. Some are permanent ("born in Sofia"). Some have fixed durations ("training for a marathon" ≈ months). Some have explicit endpoints ("until June").

**Formal target:** each belief carries a `Validity` object:

```python
@dataclass
class Validity:
    mode: Literal["permanent", "dated_range", "duration", "decay"]
    start: datetime | None = None
    end: datetime | None = None  # explicit endpoint
    half_life_days: float | None = None  # for "decay" mode
    source: Literal["explicit", "inferred", "default"] = "default"
```

**Examples:**
- "I was born in Sofia" → `permanent`
- "I'm on holiday until Friday" → `dated_range(end=this_friday, source=explicit)`
- "I'm training for a marathon" → `duration(half_life=120d, source=inferred)`
- "I prefer oat milk" (no temporal markers) → `decay(half_life=365d, source=default)`

**Decision needed (D4):** How are validities assigned?
- **Option A — Explicit only.** System only uses validities the user provides. Silent decay by default. Simple, limited.
- **Option B — LLM-inferred at ingest.** Small LLM extracts temporal markers from each proposition. More capable, more work, more chances to be wrong.
- **Option C — Rule-based heuristics + LLM fallback.** Hard-coded patterns (e.g., "until X", "for N months") + LLM for complex cases. Reasonable first cut.

### 2.4 How the three capabilities compose

At **ingest time** (pending D2):
1. Proposition P enters the system
2. Validity is computed (D4)
3. Candidate conflicting beliefs are retrieved (small window via existing retrieval layer)
4. For each candidate, `contradicts(P, candidate)` is run (D1)
5. If a contradiction is found, supersession is applied (D3): P supersedes candidate, or vice versa, or flagged for user
6. P is stored as a Belief with links to superseded/reinforced prior beliefs

At **query time**:
1. Query Q arrives
2. Normal retrieval runs as in Phase 1
3. Returned propositions are filtered/re-ranked by belief state:
   - Superseded beliefs are demoted (or excluded)
   - Expired-validity beliefs are demoted (or excluded)
   - Reinforced beliefs are boosted
4. If belief history is requested, return chain: current belief + supersession lineage

## 3. Integration with existing Patha

The belief layer sits **above** Phase 1 retrieval, not inside it. Architecturally:

```
[Query] → [Phase 1 retrieval: views + BM25 + RRF + songline + rerank]
           ↓
         [Candidate propositions + songline paths]
           ↓
         [NEW: Belief-layer filter/reorder]
           ↓
         [Top-K current beliefs + optional history]
```

This means Phase 1 remains unchanged. Phase 2 is additive. If the belief layer fails, fall back to Phase 1 behaviour.

**Storage implications:**
- New table/file: `beliefs` — one row per belief
- New edges: `supersedes`, `reinforced_by`, `contradicts`
- Existing `propositions` table unchanged (referenced by `Belief.source_proposition_id`)

## 4. Evaluation

**Phase 2 is not valid without an evaluation it can fail.** The retrieval benchmark we used for Phase 1 (LongMemEval) is not sufficient — it doesn't stress contradiction, supersession, or temporal validity. It also does not measure the thing that most determines whether Patha is actually useful in production: **token economy**.

We construct a new benchmark: **BeliefEval** (working name).

### 4.1 BeliefEval structure

A hand-constructed dataset of ~100-200 scenarios, each containing:
- A sequence of propositions asserted over time (with explicit timestamps)
- At least one contradiction, supersession, or validity transition
- A set of questions with ground-truth answers:
  - "What does the user currently believe about X?" → current belief
  - "What did the user previously believe about X?" → superseded belief(s)
  - "When did the user's view change?" → timestamp
  - "Is this claim currently valid?" → yes/no with confidence

### 4.2 Correctness metrics

- **Current-belief accuracy:** % of "what do they currently believe?" questions answered correctly
- **Supersession recall:** when a superseded belief exists, is it surfaced when asked for history?
- **Temporal accuracy:** % of "is this still valid?" questions answered correctly
- **Contradiction detection P/R:** precision and recall of pairwise contradiction labelling
- **False-contradiction rate:** how often the system incorrectly flags unrelated propositions as contradictory (the dangerous failure mode)

### 4.3 Token-economy metrics (FIRST-CLASS — not optional)

A memory system that increases total token usage is not memory — it is a search interface that inflates cost. Patha must demonstrably *reduce* tokens per correct answer versus naive RAG, or its product claim collapses.

Metrics to report on every run:

- **Tokens-in per query** — input tokens sent to the downstream LLM for each question
- **Tokens-out per query** — output tokens generated (mostly LLM-dependent but affected by prompt quality)
- **Tokens per correct answer** — tokens-in ÷ (accuracy on that question type). Penalises wasteful context.
- **Compression ratio** — total memory size (tokens of raw ingested content) ÷ tokens sent to LLM per query. Higher = more compression. A real memory system should hit 100x-1000x at scale.
- **Marginal token cost as memory grows** — how does tokens-per-query scale as the memory store grows from 100 → 1000 → 10000 propositions? A bad system scales linearly. A good one stays flat or sublinear.

### 4.4 Baselines

- **No memory (LLM alone):** directly asked the question without any retrieved context. Baseline for "does adding memory help at all?"
- **Naive RAG:** dump top-K retrieved chunks into the LLM context, ask the question. The behaviour of current systems. Baseline for "are we better than the default?"
- **LongMemEval state-of-the-art + current-belief prompting:** best published retrieval system + prompt engineering for "pick the latest." Baseline for "are we beating the field?"
- **Patha Phase 1 only (ablation):** retrieval layer with no belief layer. Baseline for "what does Phase 2 add?"

Each baseline reported on BOTH correctness AND token-economy axes.

## 5. The neuroplasticity mapping

The framing needs to hold up under scrutiny. Each plasticity mechanism should map to an operational feature, not just rhetorical analogy.

| Neural plasticity | Patha belief layer |
|---|---|
| Long-term potentiation | Repeated assertions boost belief confidence |
| Long-term depression | Non-reinforced beliefs decay in confidence over time |
| Synaptic pruning | Superseded beliefs are marked dormant (not surfaced by default) |
| Hebbian association | Co-asserted propositions gain associative edges |
| Homeostatic plasticity | Total active belief count stays bounded; new beliefs compete with old |
| Experience-dependent rewiring | Contradiction resolution updates supersession graph |

**Risk:** the mapping is either trivial (long-term memory systems already do reinforcement) or overclaimed (we're not actually doing plasticity, we're doing bookkeeping with a prettier name). The spec should err toward precision — call it a "belief maintenance layer with decay, reinforcement, and supersession" when we write about it. Reserve the plasticity language for when the mapping is mechanistically earned.

## 6. Prototype scope (first sprint)

To prove Phase 2 has legs, the smallest possible prototype:

1. NLI-based contradiction detection over pairs (D1 = Option A)
2. Simple supersession with explicit timestamp ordering (D3 = Neutral for v0.1)
3. Validity = binary `valid_until` field, default `None` (permanent) (D4 = Option A)
4. Belief-layer filter at query time only (D2 = Query-time for v0.1)
5. Storage: new `beliefs.jsonl` file, alongside existing proposition store
6. Benchmark: 50 hand-crafted scenarios, minimum

This version makes none of the bolder architectural choices. It is a baseline. If v0.1 doesn't beat Phase-1-only on BeliefEval, we rethink the whole approach before investing more.

## 7. Open questions and risks

- **Risk of feature creep.** Belief maintenance is a deep problem (decades of KR research). We need to scope tightly to what demonstrates the capability gap, not solve knowledge representation.
- **Risk of mismatched benchmark.** If BeliefEval is too easy, we learn nothing. If too hard, we fail early. Calibration is hard; expect to iterate.
- **Risk of overclaiming.** The neuroplasticity framing must be earned, not asserted. If the mapping is thin, drop it.
- **Risk of re-inventing.** Belief revision has 40 years of formal literature (AGM, belief bases, truth maintenance systems). Phase 2 should know that literature and cite it.
- **Risk of displacing what already works.** Phase 1 retrieval is solid. Adding a belief layer must not degrade Phase 1 behaviour on the original benchmark.

## 8. Out of scope for Phase 2

- Multi-user belief systems (whose belief is it?)
- Causal reasoning (beyond temporal ordering)
- Uncertainty quantification beyond a single confidence scalar
- Probabilistic/Bayesian belief networks
- Learning contradiction rules from data (fine-tuning on user patterns)

These are all interesting. None of them are Phase 2.

## 9. Decisions to make before coding

This document is incomplete until the following are answered:

- [ ] **D1:** Contradiction detection mechanism (A / B / C / D)
- [ ] **D2:** Ingest-time vs. query-time detection (ingest / query / hybrid)
- [ ] **D3:** Supersession opinionation (aggressive / neutral / confidence-weighted)
- [ ] **D4:** Validity assignment (explicit / LLM-inferred / rule+fallback)
- [ ] **D5:** The capability demo for Phase 2 — what's the single visible outcome that earns the writeup?
- [ ] **D6:** BeliefEval initial scope — 50 scenarios? 200? Hand-built or LLM-generated with human QA?
- [ ] **D7:** Compression architecture — how do retrieved beliefs reach the downstream LLM?
  - **Option A — Raw propositions to LLM.** Send retrieved propositions verbatim. Current default. Highest tokens, highest fidelity.
  - **Option B — Structured belief summary.** Send belief state as structured object (current belief, supersedes list, confidence, validity). ~10x compression, may lose nuance.
  - **Option C — Direct answer for lookups, raw for generation.** If the query is a factual lookup ("what do I currently believe about X?"), Patha answers directly from belief state without invoking LLM. Only generation-heavy queries get LLM + context. Maximum compression, requires robust query-type detection.
  - **Option D — Tiered.** Direct answer for lookups, structured summary for reasoning, raw propositions for open-ended generation. Most flexible, most complex.

Each of these has downstream implications on architecture, timeline, and claim strength. Work through them deliberately, don't default.

---

## Appendix A — References and prior work to survey

(To be populated by `docs/phase_2_literature_survey.md`.)

- AGM belief revision framework (Alchourrón, Gärdenfors, Makinson, 1985)
- Truth maintenance systems (Doyle 1979; de Kleer 1986)
- MemGPT — handling memory overflow, but not contradiction
- Zep — session-based memory with temporal metadata
- Letta / Mem0 — current open-source memory frameworks
- Natural Language Inference models (DeBERTa-MNLI, RoBERTa-MNLI)
- Knowledge graph update literature
- Continual learning in neural networks (EWC, elastic weight consolidation) — the plasticity analogy

## Appendix B — Changelog

- 2026-04-17: Initial draft. Decisions D1–D6 pending.
