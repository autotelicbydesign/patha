"""Ollama-backed LLM judge for contradiction detection.

Wraps a locally-running Ollama model so the HybridContradictionDetector
can escalate uncertain NLI pairs to a real local LLM rather than a
scripted stub. No hosted-LLM API calls — this is Patha's 'zero hosted
API' constraint kept intact.

Usage:
    # User starts Ollama separately: `ollama serve` + `ollama pull qwen2.5:7b`
    from patha.belief import NLIContradictionDetector, HybridContradictionDetector
    from patha.belief.ollama_judge import OllamaLLMJudge

    judge = OllamaLLMJudge(model="qwen2.5:7b")
    hybrid = HybridContradictionDetector(
        primary=NLIContradictionDetector(),
        llm=judge,
        min_overlap=0,
        uncertainty_band=(0.0, 1.0),
    )

Defaults target Qwen2.5-7B because it scored 82% on pairwise contradiction
tasks in the survey (good enough for Phase 2 v0.3) and runs comfortably
on 16GB Macs. Swap for llama3.1:8b, mistral-nemo, or anything else via
the ``model`` parameter.

Failure mode: if Ollama isn't running, the judge raises on first call
with a clear error. Callers can catch that and fall back to a scripted
judge or the primary-only path.
"""

from __future__ import annotations

import json
from typing import Any

from patha.belief.llm_judge import PromptLLMJudge, _JUDGE_PROMPT


# ─── HTTP backend for Ollama ────────────────────────────────────────

_DEFAULT_HOST = "http://localhost:11434"
_DEFAULT_MODEL = "qwen2.5:7b"


class OllamaLLMJudge:
    """LLM judge powered by a local Ollama instance.

    Parameters
    ----------
    model
        Ollama model tag. Must already be pulled (`ollama pull <model>`).
        Default qwen2.5:7b — good balance of contradiction accuracy
        and speed on consumer hardware.
    host
        Ollama HTTP endpoint. Default http://localhost:11434.
    temperature
        Sampling temperature for the judge. Default 0.0 — contradiction
        judgements should be deterministic given the same pair.
    timeout_s
        Seconds to wait for a response before treating the call as failed.
        Default 30.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        host: str = _DEFAULT_HOST,
        *,
        temperature: float = 0.0,
        timeout_s: float = 30.0,
    ) -> None:
        self._model = model
        self._host = host.rstrip("/")
        self._temperature = temperature
        self._timeout_s = timeout_s
        # Delegate prompt construction + parsing to PromptLLMJudge
        self._inner = PromptLLMJudge(self._generate)
        # Runtime metrics
        self.calls = 0
        self.total_latency_s = 0.0

    def _generate(self, prompt: str) -> str:
        """POST to /api/generate and return the model's response text.

        Uses urllib from stdlib so we don't add a dependency. Ollama's
        /api/generate endpoint supports streaming, but for a short
        classification response non-streaming is fine.
        """
        import time
        import urllib.error
        import urllib.request

        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self._temperature,
                # Contradiction is a short single-token answer; keep
                # num_predict low to avoid paying for rambling.
                "num_predict": 8,
            },
        }
        req = urllib.request.Request(
            f"{self._host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                body = resp.read()
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Ollama call failed ({self._host}, model={self._model}): {e}. "
                "Is Ollama running? Try `ollama serve` and "
                f"`ollama pull {self._model}`."
            ) from e
        finally:
            self.calls += 1
            self.total_latency_s += time.monotonic() - start

        data = json.loads(body)
        # Ollama's /api/generate returns {"response": "...", ...}
        return str(data.get("response", "")).strip()

    # ── Delegates to inner PromptLLMJudge ──────────────────────────

    def judge(self, p1: str, p2: str):
        return self._inner.judge(p1, p2)


__all__ = ["OllamaLLMJudge", "_DEFAULT_MODEL", "_DEFAULT_HOST"]
