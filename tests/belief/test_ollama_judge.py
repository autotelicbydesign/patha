"""Tests for Ollama-backed LLM judge.

Unit tests mock the HTTP layer so they don't require Ollama running.
An optional @pytest.mark.slow integration test does hit a live Ollama
instance — skipped by default.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from patha.belief.ollama_judge import OllamaLLMJudge
from patha.belief.types import ContradictionLabel


# ─── Unit tests (no Ollama required) ────────────────────────────────

class TestOllamaLLMJudge:
    def test_constructs_with_defaults(self) -> None:
        judge = OllamaLLMJudge()
        assert judge._model == "qwen2.5:7b"
        assert "11434" in judge._host

    def test_custom_model_and_host(self) -> None:
        judge = OllamaLLMJudge(model="llama3.1:8b", host="http://gpu-box:11434")
        assert judge._model == "llama3.1:8b"
        assert judge._host == "http://gpu-box:11434"

    def test_trailing_slash_stripped(self) -> None:
        judge = OllamaLLMJudge(host="http://localhost:11434/")
        assert not judge._host.endswith("/")

    def _mock_ollama_response(self, response_text: str):
        """Helper: returns a patch context manager that makes urllib
        respond with the given text wrapped in the Ollama JSON envelope.
        """
        import json as _json

        class FakeResp:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *args):
                return False

            def read(self_inner):
                return _json.dumps({"response": response_text}).encode()

        return patch(
            "urllib.request.urlopen", return_value=FakeResp()
        )

    def test_judge_returns_contradicts_on_contradicts_response(self) -> None:
        """Simulate Ollama returning 'CONTRADICTS'."""
        judge = OllamaLLMJudge()
        with self._mock_ollama_response("CONTRADICTS"):
            r = judge.judge("a", "b")
        assert r.label == ContradictionLabel.CONTRADICTS

    def test_judge_returns_entails_on_entails_response(self) -> None:
        judge = OllamaLLMJudge()
        with self._mock_ollama_response(
            "ENTAILS\nThe first implies the second."
        ):
            r = judge.judge("I own a golden retriever", "I have a dog")
        assert r.label == ContradictionLabel.ENTAILS

    def test_judge_returns_neutral_on_unknown_response(self) -> None:
        judge = OllamaLLMJudge()
        with self._mock_ollama_response("i dunno"):
            r = judge.judge("a", "b")
        assert r.label == ContradictionLabel.NEUTRAL

    def test_metrics_track_calls_and_latency(self) -> None:
        """The judge records call count and total latency across calls."""
        judge = OllamaLLMJudge()
        with patch("urllib.request.urlopen") as mocked:
            mocked.return_value.__enter__.return_value.read.return_value = (
                b'{"response": "CONTRADICTS"}'
            )
            judge.judge("a", "b")
            judge.judge("c", "d")
        assert judge.calls == 2
        assert judge.total_latency_s >= 0  # monotonic clock, may be 0 on mock

    def test_connection_failure_raises_clear_error(self) -> None:
        """Ollama down → clear RuntimeError mentioning Ollama + model."""
        import urllib.error

        judge = OllamaLLMJudge(model="qwen2.5:7b")
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            with pytest.raises(RuntimeError, match="Ollama"):
                judge.judge("a", "b")

    def test_payload_structure(self) -> None:
        """Verify the JSON payload sent to Ollama has the expected shape."""
        import json as _json

        captured: list[bytes] = []

        class FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"response": "NEUTRAL"}'

        def fake_urlopen(req, timeout=None):
            captured.append(req.data)
            return FakeResp()

        judge = OllamaLLMJudge(model="qwen2.5:7b", temperature=0.1)
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            judge.judge("statement 1", "statement 2")

        assert len(captured) == 1
        payload = _json.loads(captured[0].decode())
        assert payload["model"] == "qwen2.5:7b"
        assert payload["stream"] is False
        assert payload["options"]["temperature"] == 0.1
        assert "statement 1" in payload["prompt"]
        assert "statement 2" in payload["prompt"]


# ─── Integration test (requires live Ollama) ────────────────────────

@pytest.mark.slow
class TestOllamaIntegration:
    """Requires: `ollama serve` running and `ollama pull qwen2.5:7b`.

    Skipped by default — run with `uv run pytest -m slow`.
    """

    def test_live_ollama_contradiction(self) -> None:
        judge = OllamaLLMJudge()
        r = judge.judge(
            "I love sushi and eat it every week",
            "I am avoiding raw fish on my doctor's advice",
        )
        # Live model may return CONTRADICTS (good) or NEUTRAL (weaker).
        # Don't hard-assert CONTRADICTS — we're confirming plumbing, not
        # model accuracy here.
        assert r.label in (
            ContradictionLabel.CONTRADICTS,
            ContradictionLabel.NEUTRAL,
            ContradictionLabel.ENTAILS,
        )
        assert judge.calls == 1
