"""Phase 3 — End-to-end answer evaluation.

The first two phases measure intermediate signals: Phase 1 measures
retrieval (R@k, did the gold session surface?), Phase 2 measures
supersession (did the right belief end up `current`?). Both useful,
both surrogates.

The actual product question: **given Patha's output, does the user's
LLM produce the right answer to the user's question?**

That's what this module measures. Plan in `docs/phase_3_plan.md`.

Three knobs:
  - LLM       — Claude / GPT / Ollama / null-template (deterministic CI)
  - Prompt    — what fields of `Recall` go into the prompt to the LLM
  - Scorer    — exact / normalised / numeric / embedding / LLM-judge

Usage:

    from eval.answer_eval import (
        AnswerEvalConfig, NullTemplateLLM, normalised_match, run_answer_eval,
    )
    cfg = AnswerEvalConfig(
        llm=NullTemplateLLM(),
        prompt_template="{question}\\nMemory:\\n{summary}\\nAnswer:",
        scorer=normalised_match,
    )
    report = run_answer_eval(questions=[...], memory_factory=..., config=cfg)

This is intentionally small. The scope is the v0.10 milestone; we
ship the scaffolding, the deterministic NullTemplateLLM for tests,
and one or two real adapters. LLM-as-judge is reserved for a follow-up.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol


# ─── LLM Protocol ────────────────────────────────────────────────────


class LLM(Protocol):
    """A callable that takes a prompt string and returns an answer string."""

    def __call__(self, prompt: str) -> str: ...


class NullTemplateLLM:
    """Deterministic 'LLM' for CI / tests.

    Renders a tiny canned response from the prompt's structure: looks
    for a "Memory:" block followed by an "Answer:" trailing marker
    and just echoes the relevant numbers/text from the memory block.

    For real evaluation, swap with a real LLM adapter. This exists so
    the eval scaffolding can be tested without API access.
    """

    def __call__(self, prompt: str) -> str:
        # Pull dollar amounts and numbers from the memory section
        # (everything between "Memory:" and "Answer:").
        memory_match = re.search(
            r"Memory:(.*?)(?:Question:|Answer:|$)",
            prompt,
            flags=re.DOTALL,
        )
        if memory_match:
            memory = memory_match.group(1)
        else:
            memory = prompt
        # Prefer the value after a "computed:" marker (the gaṇita
        # synthesis result) — that's the canonical answer when synthesis
        # is involved.
        computed = re.search(
            r"computed:\s*\$?\s*(\d+(?:\.\d+)?)\s*([A-Za-z%]+)?",
            memory,
            flags=re.IGNORECASE,
        )
        if computed:
            value = computed.group(1)
            unit = computed.group(2) or ""
            # Render with $ prefix for currency-like units, else value+unit
            if unit.upper() == "USD":
                return f"${value}"
            return f"{value} {unit}".strip()
        # Otherwise fall through: first dollar amount, then first number.
        money = re.search(r"\$\s*\d+(?:\.\d+)?", memory)
        if money:
            return money.group(0).strip()
        num = re.search(r"\b\d+(?:\.\d+)?\b", memory)
        if num:
            return num.group(0)
        return memory.strip()[:140]


# ─── Scorers ─────────────────────────────────────────────────────────

Scorer = Callable[[str, str], bool]


def exact_match(answer: str, gold: str) -> bool:
    """Strict equality after stripping leading/trailing whitespace."""
    return answer.strip() == gold.strip()


def normalised_match(answer: str, gold: str) -> bool:
    """Case + whitespace + punctuation insensitive."""
    def norm(s: str) -> str:
        s = s.lower()
        # Replace punctuation with space, then collapse
        s = re.sub(r"[^\w\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    return norm(answer) == norm(gold)


def numeric_match(tol: float = 0.05) -> Scorer:
    """Returns a scorer that matches numeric values within `tol` (5%)
    or 1 absolute unit, whichever is larger."""

    def _num(s: str) -> float | None:
        cleaned = s.replace("$", "").replace(",", "").strip()
        m = re.search(r"-?\d+(?:\.\d+)?", cleaned)
        if not m:
            return None
        try:
            return float(m.group(0))
        except ValueError:
            return None

    def _scorer(answer: str, gold: str) -> bool:
        a = _num(answer)
        g = _num(gold)
        if a is None or g is None:
            # Fall back to normalised match on non-numeric questions.
            return normalised_match(answer, gold)
        return abs(a - g) <= max(abs(g) * tol, 1.0)

    return _scorer


def token_overlap_match(threshold: float = 0.6) -> Scorer:
    """Matches if ≥`threshold` fraction of gold's content tokens
    appear in the answer. Mirrors the LongMemEval-S original scoring
    methodology, kept here for backward-comparable runs."""
    stopwords = {
        "a", "an", "the", "is", "are", "was", "were", "to", "of",
        "in", "on", "at", "for", "with", "and", "or", "but", "not",
        "i", "you", "my", "we", "our", "your",
    }

    def _toks(s: str) -> list[str]:
        return [
            t.lower() for t in re.findall(r"[A-Za-z0-9]+", s)
            if t.lower() not in stopwords
        ]

    def _scorer(answer: str, gold: str) -> bool:
        g_toks = _toks(gold)
        if not g_toks:
            return False
        a_lower = answer.lower()
        present = sum(1 for t in g_toks if t in a_lower)
        return (present / len(g_toks)) >= threshold

    return _scorer


# ─── Eval engine ─────────────────────────────────────────────────────


@dataclass
class AnswerEvalConfig:
    llm: LLM
    prompt_template: str  # uses {question}, {summary}, {ganita}, {current}
    scorer: Scorer
    include_history: bool = False


@dataclass
class QuestionOutcome:
    question_id: str
    question: str
    gold: str
    answer: str
    correct: bool
    strategy: str  # patha's recall strategy: "ganita" / "structured" / ...
    summary_tokens: int


@dataclass
class AnswerEvalReport:
    n: int
    correct: int
    outcomes: list[QuestionOutcome] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0

    def by_strategy(self) -> dict[str, tuple[int, int]]:
        """{strategy: (correct, total)}"""
        d: dict[str, list[int]] = {}
        for o in self.outcomes:
            ent = d.setdefault(o.strategy, [0, 0])
            ent[1] += 1
            if o.correct:
                ent[0] += 1
        return {k: (c, t) for k, (c, t) in d.items()}


def render_prompt(template: str, question: str, rec) -> str:
    """Render `template` with the recall fields. Available placeholders:
       {question}, {summary}, {ganita}, {current}, {answer}."""
    ganita_block = ""
    if rec.ganita is not None:
        ganita_block = (
            f"computed: {rec.ganita.value} {rec.ganita.unit} "
            f"(via {rec.ganita.operator}); "
            f"{rec.ganita.explanation}"
        )
    current_block = "\n".join(
        f"  - {c['proposition']}" for c in rec.current
    )
    return template.format(
        question=question,
        summary=rec.summary or "",
        ganita=ganita_block,
        current=current_block,
        answer=rec.answer or "",
    )


def run_answer_eval(
    questions: Iterable[dict],
    memory_factory: Callable[[], Any],
    config: AnswerEvalConfig,
) -> AnswerEvalReport:
    """Run the eval over `questions`. Each question dict needs at least
    `question_id`, `question`, `answer` (gold). For LongMemEval-shaped
    inputs, the caller is responsible for ingesting the haystack into
    each fresh memory before recall."""
    outcomes: list[QuestionOutcome] = []
    correct_count = 0
    n = 0
    for q in questions:
        n += 1
        memory = memory_factory()
        # The caller is responsible for ingest (haystack-shaped inputs
        # live outside this module's concern).
        if "_ingested_memory" in q:
            memory = q["_ingested_memory"]
        rec = memory.recall(
            q["question"], include_history=config.include_history,
        )
        prompt = render_prompt(config.prompt_template, q["question"], rec)
        answer = config.llm(prompt)
        ok = bool(config.scorer(answer, str(q["answer"])))
        outcomes.append(QuestionOutcome(
            question_id=q.get("question_id", ""),
            question=q["question"],
            gold=str(q["answer"]),
            answer=answer,
            correct=ok,
            strategy=getattr(rec, "strategy", "?"),
            summary_tokens=getattr(rec, "tokens", 0),
        ))
        if ok:
            correct_count += 1
    return AnswerEvalReport(
        n=n, correct=correct_count, outcomes=outcomes,
    )


__all__ = [
    "LLM",
    "NullTemplateLLM",
    "AnswerEvalConfig",
    "AnswerEvalReport",
    "QuestionOutcome",
    "exact_match",
    "normalised_match",
    "numeric_match",
    "token_overlap_match",
    "render_prompt",
    "run_answer_eval",
]
