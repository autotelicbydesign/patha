"""LLM-as-judge fallback for contradiction detection.

When NLI returns NEUTRAL but there's significant lexical or entity
overlap between the two propositions, the pair may still be a
commonsense contradiction the NLI model couldn't resolve (canonical
example: "I love sushi" vs "I'm avoiding raw fish"). For these pairs we
escalate to an LLM judge.

Design (D1 Option D completion):

- HybridContradictionDetector wraps any primary detector + any
  LLMJudge. Sends a pair to the primary first; if the primary returns
  NEUTRAL and the overlap check fires, sends the pair to the LLM.
- LLMJudge is a protocol. One default implementation is provided:
  LocalLLMJudge wraps a small local model via llama-cpp, transformers,
  or Ollama. A StubLLMJudge is provided for tests: it looks up its
  verdicts in a pre-configured dict.
- No hosted-LLM API calls. This is a core Patha constraint. The NLI +
  local-LLM stack stays fully offline.

Cost guard: the LLM is only called when the primary detector is
NEUTRAL AND the pair shares ≥ min_overlap content words. In the v0.1
BeliefEval run this would have fired only on the 2 failures, not on
every NEUTRAL pair.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from patha.belief.contradiction import ContradictionDetector
from patha.belief.types import ContradictionLabel, ContradictionResult


# ─── LLM judge protocol ──────────────────────────────────────────────

@runtime_checkable
class LLMJudge(Protocol):
    """Minimal interface for an LLM used as contradiction judge.

    The implementation is free to use any backend (Ollama, llama-cpp,
    transformers, vLLM, etc.). The protocol keeps Patha's core
    dependency surface small.
    """

    def judge(self, p1: str, p2: str) -> ContradictionResult:
        """Ask the LLM whether p1 and p2 contradict."""
        ...


# ─── Stub judge (tests / deterministic runs) ─────────────────────────

class StubLLMJudge:
    """Deterministic stub.

    Returns pre-configured verdicts for specific pairs, NEUTRAL otherwise.
    Useful for unit tests and for offline CI runs of the hybrid detector.
    """

    def __init__(self, verdicts: dict[tuple[str, str], ContradictionResult] | None = None):
        self.verdicts = verdicts or {}

    def judge(self, p1: str, p2: str) -> ContradictionResult:
        if (p1, p2) in self.verdicts:
            return self.verdicts[(p1, p2)]
        return ContradictionResult(
            label=ContradictionLabel.NEUTRAL,
            confidence=0.5,
            rationale="stub: no scripted verdict",
        )


# ─── Prompt-based local-LLM judge ────────────────────────────────────

_JUDGE_PROMPT = """You are a strict logical judge. Given two statements by the
same person, decide whether they CONTRADICT (both cannot be true of
the same person at the same time), ENTAIL (the second follows from the
first), or are NEUTRAL (neither holds).

Use commonsense. For example:
- "I love sushi" and "I'm avoiding raw fish" → CONTRADICTS (sushi is raw fish).
- "I have a golden retriever" and "I have a dog" → ENTAILS.
- "I love sushi" and "The weather is nice" → NEUTRAL.

Output exactly one token: CONTRADICTS, ENTAILS, or NEUTRAL.

Statement 1: {p1}
Statement 2: {p2}
Verdict:"""


class PromptLLMJudge:
    """Generic prompt-based LLM judge.

    Takes any callable that turns a string prompt into a string response
    (bring-your-own-backend). Parses the first line of the response for
    the verdict token.

    Parameters
    ----------
    generate
        A function: (prompt: str) -> str. The caller wires this to their
        LLM of choice (Ollama, llama-cpp, transformers, vLLM, etc.).
    """

    def __init__(self, generate):
        self._generate = generate

    def judge(self, p1: str, p2: str) -> ContradictionResult:
        prompt = _JUDGE_PROMPT.format(p1=p1, p2=p2)
        raw = self._generate(prompt).strip()
        # Take the first line / first meaningful token
        first = raw.split("\n", 1)[0].strip().upper()
        if "CONTRADICT" in first:
            label = ContradictionLabel.CONTRADICTS
            conf = 0.85
        elif "ENTAIL" in first:
            label = ContradictionLabel.ENTAILS
            conf = 0.85
        else:
            label = ContradictionLabel.NEUTRAL
            conf = 0.6
        return ContradictionResult(
            label=label, confidence=conf, rationale=f"llm:{raw[:64]}"
        )


# ─── Hybrid detector (the main product) ──────────────────────────────

_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "i", "you", "he", "she",
    "it", "we", "they", "my", "your", "his", "her", "our", "their",
    "this", "that", "these", "those", "in", "on", "at", "to", "for",
    "with", "from", "of", "by", "as", "so",
})


def _content_words(text: str) -> set[str]:
    toks = re.findall(r"[a-zA-Z]+", text.lower())
    return {t for t in toks if len(t) >= 3 and t not in _STOPWORDS}


class HybridContradictionDetector:
    """NLI primary + LLM fallback on uncertain pairs with lexical overlap.

    Flow (per pair):
      1. Call primary detector (NLI, usually).
      2. If primary is CONTRADICTS or ENTAILS: return primary.
      3. If primary is NEUTRAL but content overlap ≥ min_overlap and
         primary confidence is in the uncertainty band [lo, hi]:
         call the LLM judge, return its verdict.
      4. Otherwise: return primary (NEUTRAL).

    The overlap and confidence guards keep LLM call volume small — the
    LLM should fire on a minority of pairs per the survey's cost
    guidance.

    Parameters
    ----------
    primary
        The primary (fast, cheap) detector. Usually an NLI model.
    llm
        The LLM judge used for escalation.
    min_overlap
        Minimum number of shared content words to consider an LLM call.
        Default 1 — any shared meaningful word is enough.
    uncertainty_band
        Only escalate NEUTRAL verdicts whose confidence falls in this
        [low, high] range. Very-high-confidence NEUTRAL (the model is
        sure these are unrelated) is not escalated. Default (0.0, 0.95).
    """

    def __init__(
        self,
        primary: ContradictionDetector,
        llm: LLMJudge,
        *,
        min_overlap: int = 1,
        uncertainty_band: tuple[float, float] = (0.0, 0.95),
        escalate_low_confidence_verdicts: bool = False,
        low_confidence_threshold: float = 0.6,
    ) -> None:
        self._primary = primary
        self._llm = llm
        self._min_overlap = min_overlap
        self._uncertainty_band = uncertainty_band
        self._escalate_low_conf = escalate_low_confidence_verdicts
        self._low_conf_threshold = low_confidence_threshold
        # Metrics a caller can inspect after a run
        self.llm_calls = 0
        self.primary_calls = 0

    # ── batched hot path ────────────────────────────────────────────

    def detect(self, p1: str, p2: str) -> ContradictionResult:
        return self.detect_batch([(p1, p2)])[0]

    def detect_batch(
        self, pairs: list[tuple[str, str]]
    ) -> list[ContradictionResult]:
        if not pairs:
            return []
        self.primary_calls += len(pairs)
        primary_results = self._primary.detect_batch(pairs)

        # Decide which pairs need escalation
        lo, hi = self._uncertainty_band
        final: list[ContradictionResult] = []
        for (p1, p2), r in zip(pairs, primary_results):
            # Low-confidence CONTRADICTS/ENTAILS also escalate when
            # enabled. This catches cases where NLI has a weak signal
            # that the LLM judge can either confirm or retract.
            is_low_conf_verdict = (
                self._escalate_low_conf
                and r.label != ContradictionLabel.NEUTRAL
                and r.confidence < self._low_conf_threshold
            )

            if r.label != ContradictionLabel.NEUTRAL and not is_low_conf_verdict:
                # High-confidence non-neutral: trust primary.
                final.append(r)
                continue

            if r.label == ContradictionLabel.NEUTRAL:
                # NEUTRAL escalation gates
                if not (lo <= r.confidence <= hi):
                    final.append(r)
                    continue
                overlap = _content_words(p1) & _content_words(p2)
                if len(overlap) < self._min_overlap:
                    final.append(r)
                    continue

            # Escalate
            self.llm_calls += 1
            llm_result = self._llm.judge(p1, p2)
            final.append(llm_result)
        return final
