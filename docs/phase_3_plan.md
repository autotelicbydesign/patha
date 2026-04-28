# Phase 3 — End-to-End Answer Evaluation

The first two phases measure intermediate signals: Phase 1 measures retrieval (R@k, did the gold session surface?), Phase 2 measures supersession (did the right belief end up `current`?). Both are useful, both are surrogates.

The actual product-relevant question is: **given Patha's output, does the user's LLM produce the right answer to the user's question?**

That's what Phase 3 measures.

## Why this matters now

Three reasons:

1. **The synthesis-intent path produces *computed* answers.** The current LongMemEval scorer checks if gold-answer tokens appear in `rec.current`. For a synthesis question with gold "$185", no source session contains "185" — it's the SUM. The architecture is right; the metric is wrong. Phase 3 evaluates the actual answer text, not the retrieval intermediate.

2. **The field is moving toward this.** RAG benchmarks have been reporting recall@k for years; that's no longer differentiating. The current bar is end-to-end response correctness — judged by humans, by stronger LLMs, or by exact-match on canonical answers.

3. **Patha's claims need this metric.** "Patha separates retrieval from synthesis" is an architectural claim. To support it empirically, we need a metric that can distinguish a retrieval pass+miss from a synthesis pass+miss. Token-overlap-on-summary can't.

## What Phase 3 measures

For each question in a benchmark:

```
1. memory.recall(question)        # Patha produces a Recall
2. answer = llm(prompt(rec))       # the user's LLM produces an answer
                                   # using Patha's summary as context
3. score(answer, gold)             # exact match, semantic match, or LLM-judge
```

Three knobs:
- **The LLM** — Claude (3.5 Sonnet, 4 Sonnet), GPT-4o, local Llama / Qwen, or a placeholder template renderer for deterministic tests
- **The prompt template** — what fields of `Recall` go into the LLM prompt: `summary`, `current`, `history`, `ganita.explanation`, `answer`
- **The scorer** — strict (`gold == answer`), normalised (case+punct insensitive), embedding-cosine, or LLM-as-judge

## Concrete scope (v0.10 milestone)

Land the minimum useful version:

### `eval/answer_eval.py`

```python
@dataclass
class AnswerEvalConfig:
    llm: Callable[[str], str]   # prompt → answer
    prompt_template: str        # Jinja-style with {{question}}, {{rec.summary}}
    scorer: Callable[[str, str], bool]   # (answer, gold) → bool

def run_answer_eval(
    questions: list[dict],
    memory_factory: Callable[[], patha.Memory],
    config: AnswerEvalConfig,
) -> AnswerEvalReport:
    ...
```

### Scorers shipped with Phase 3

- `exact_match` — strict
- `normalised_match` — case + whitespace + punctuation
- `numeric_match(tol=5%)` — for numeric gold answers
- `embedding_cosine_match(threshold=0.85)` — uses MiniLM
- `llm_judge` — pass-through to a stronger LLM with a judge prompt

### LLM adapters

- `claude_messages_api` — Anthropic API (require key in env)
- `ollama_chat` — local Ollama
- `null_render` — template renders the prompt as the "answer" (for deterministic tests)

## Honest open questions

- **LLM-as-judge variance** — different judges produce different rankings. Mitigate with a small ensemble + standard prompt.
- **Prompt template choice biases the result.** Document the template used in every report; allow comparison across templates (which Patha output is more useful — `summary`, `summary + current`, `summary + history`, etc.).
- **Cost** — running an end-to-end eval against Claude/GPT for 500 questions is real money. Default the eval to local models for CI; opt into hosted for the headline numbers.

## Out of scope for Phase 3

- New retrieval mechanisms.
- New extractor variants.
- Adversarial evaluation.

## Success criterion

Phase 3 ships when:

1. `eval/answer_eval.py` runs on a fresh checkout against:
   - LongMemEval-S 500q (synthesis-bounded subset measured separately)
   - BeliefEval (300 supersession scenarios)
2. Reports a single number per (LLM × scorer) pair, plus per-question detail.
3. The number on synthesis-bounded questions correlates with karaṇa quality (regex < ollama-7b < hybrid-14b).
4. Documented in `docs/benchmarks.md` alongside the existing R@k numbers, not replacing them — both metrics are useful for different things.
