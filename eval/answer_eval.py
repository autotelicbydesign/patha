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


class _LazyMiniLMEmbedder:
    """Lazy-loaded MiniLM embedder used by `embedding_cosine_match`.

    Mirrors the pattern in `patha.belief.semantic_filter` so the
    scorer ships with no extra dependency: sentence-transformers is
    already a core Patha dependency. The first call loads the model
    (~80 MB download on first ever use, then cached locally).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None

    def __call__(self, texts: list[str]):
        # Imported here so that just importing answer_eval doesn't pull
        # sentence_transformers into the import graph for users who
        # never use the embedding scorer (NLI eval / numeric tests etc.).
        import numpy as np

        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return np.asarray(
            self._model.encode(texts, normalize_embeddings=True),
            dtype=np.float32,
        )


def embedding_cosine_match(
    threshold: float = 0.85,
    *,
    embedder: Callable[[list[str]], Any] | None = None,
) -> Scorer:
    """Returns a scorer that matches if cos(embed(answer), embed(gold)) ≥ threshold.

    Useful for free-form answers where exact / normalised / numeric
    matching is too strict but `llm_judge_match` is too expensive
    (no network, deterministic, ~5 ms per scorer call after first load).

    Parameters
    ----------
    threshold:
        Cosine similarity in [-1, 1]. 0.85 is a sensible MiniLM default
        for "the same fact, paraphrased". Lower (0.7) accepts looser
        paraphrase; higher (0.92) demands near-identical wording.
    embedder:
        Optional callable that takes a list of strings and returns a
        2D array of unit-norm row vectors (shape `[n, d]`). Defaults
        to lazy-loaded `all-MiniLM-L6-v2` from sentence-transformers.
        Inject a deterministic stub for tests.

    Notes
    -----
    Empty answer or empty gold → returns False (no match), since
    cosine is undefined / unstable on zero vectors.
    """
    emb = embedder if embedder is not None else _LazyMiniLMEmbedder()

    def _scorer(answer: str, gold: str) -> bool:
        a = (answer or "").strip()
        g = (gold or "").strip()
        if not a or not g:
            return False
        vecs = emb([a, g])
        # Defensive: handle list/tuple/ndarray uniformly.
        try:
            import numpy as np

            arr = np.asarray(vecs, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] < 2:
                return False
            v_a, v_g = arr[0], arr[1]
            # Assume rows are unit-norm (the default for MiniLM
            # `normalize_embeddings=True`); fall back to explicit
            # normalisation if a stub returns un-normalised vectors.
            na = float(np.linalg.norm(v_a))
            ng = float(np.linalg.norm(v_g))
            if na == 0.0 or ng == 0.0:
                return False
            cos = float(np.dot(v_a, v_g) / (na * ng))
        except ImportError:
            return False
        return cos >= threshold

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


# ─── Real LLM adapters ───────────────────────────────────────────────


@dataclass
class OllamaLLM:
    """LLM adapter pointing at a local Ollama instance.

    Uses urllib so we don't add a runtime dependency. Fails loudly
    if Ollama isn't reachable — caller should handle the exception
    or use NullTemplateLLM as a CI fallback.

    Parameters mirror :class:`patha.belief.karana.OllamaKaranaExtractor`.
    """

    model: str = "qwen2.5:14b-instruct"
    host: str = "http://localhost:11434"
    temperature: float = 0.0
    timeout_s: float = 240.0
    num_predict: int = 256

    calls: int = 0
    total_latency_s: float = 0.0

    def __call__(self, prompt: str) -> str:
        import json as _json
        import time as _time
        import urllib.error
        import urllib.request

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
        }
        req = urllib.request.Request(
            f"{self.host.rstrip('/')}/api/generate",
            data=_json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start = _time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = resp.read()
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"OllamaLLM: call failed ({self.host}, model={self.model}): {e}. "
                f"Is Ollama running and the model pulled?"
            ) from e
        finally:
            self.calls += 1
            self.total_latency_s += _time.monotonic() - start

        data = _json.loads(body)
        return str(data.get("response", "")).strip()


@dataclass
class ClaudeLLM:
    """LLM adapter pointing at the Anthropic Messages API.

    Uses the optional `anthropic` SDK. If the SDK isn't installed,
    raises a clear ImportError pointing at the install command. If
    the API key is missing, raises a clear RuntimeError naming the
    env var.

    Defaults to claude-sonnet-4 with deterministic temperature for
    reproducible eval runs. Pass `model="claude-haiku-4"` etc. to
    swap.
    """

    model: str = "claude-sonnet-4-20250514"
    api_key: str | None = None  # falls back to ANTHROPIC_API_KEY
    temperature: float = 0.0
    max_tokens: int = 256
    system: str = (
        "You are a careful assistant answering a user's question using "
        "the user's own memory as context. Answer concisely. If the "
        "memory contains a directly-computed value, return that exact "
        "value (with units if present). If the answer is not in the "
        "memory, say so."
    )

    calls: int = 0
    total_latency_s: float = 0.0

    def __call__(self, prompt: str) -> str:
        import os as _os
        import time as _time

        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise ImportError(
                "ClaudeLLM requires the anthropic SDK. Install with "
                "`pip install anthropic` or `uv pip install anthropic`."
            ) from e

        api_key = self.api_key or _os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ClaudeLLM: no API key. Set ANTHROPIC_API_KEY in your "
                "environment or pass api_key= to the constructor."
            )

        client = Anthropic(api_key=api_key)
        start = _time.monotonic()
        try:
            resp = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self.system,
                messages=[{"role": "user", "content": prompt}],
            )
        finally:
            self.calls += 1
            self.total_latency_s += _time.monotonic() - start

        # Anthropic SDK returns a list of content blocks; we expect text.
        parts = []
        for block in resp.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "".join(parts).strip()


# ─── LLM-as-judge scorer ─────────────────────────────────────────────


_LLM_JUDGE_PROMPT = (
    "You are evaluating whether a candidate answer matches a gold "
    "answer to a memory question. Two answers MATCH if a careful "
    "reader would consider them to express the same fact, even if "
    "phrased differently (e.g. '$185' matches '185 USD'; 'Lisbon' "
    "matches 'I live in Lisbon, Portugal'). Two answers DO NOT MATCH "
    "if any key fact differs (different number, different city, "
    "different person).\n"
    "\n"
    "Respond with exactly one word: MATCH or NO_MATCH.\n"
    "\n"
    "Gold answer: {gold}\n"
    "Candidate answer: {candidate}\n"
    "\n"
    "Verdict:"
)


def llm_judge_match(judge_llm: LLM, *, prompt_template: str | None = None) -> Scorer:
    """Returns a scorer that asks `judge_llm` to verdict MATCH / NO_MATCH.

    Used for free-form answers where exact / normalised / numeric
    matching is too strict (e.g. "summarise my evolving thinking on
    agency"). The judge LLM is independent of whatever LLM produced
    the candidate answer — best practice is to use a different model
    than the one being evaluated, to reduce bias.

    The judge prompt is intentionally short and constrained
    (one-word output) to make the verdict deterministic and cheap.
    Override `prompt_template` if you need to customise.
    """
    template = prompt_template or _LLM_JUDGE_PROMPT

    def _scorer(answer: str, gold: str) -> bool:
        verdict_prompt = template.format(gold=gold, candidate=answer)
        try:
            verdict = judge_llm(verdict_prompt)
        except Exception:
            # If the judge call itself fails, conservative: NO_MATCH.
            return False
        return verdict.strip().upper().startswith("MATCH")

    return _scorer


__all__ = [
    "LLM",
    "NullTemplateLLM",
    "OllamaLLM",
    "ClaudeLLM",
    "AnswerEvalConfig",
    "AnswerEvalReport",
    "QuestionOutcome",
    "exact_match",
    "normalised_match",
    "numeric_match",
    "token_overlap_match",
    "embedding_cosine_match",
    "llm_judge_match",
    "render_prompt",
    "run_answer_eval",
]
