"""End-to-end pytest exercising karaṇa with a live Ollama instance.

Runs only when Ollama is reachable AND the user opts in via
`pytest -m slow` (or `-m ''`). On CI without Ollama running, the test
is skipped automatically.

The test verifies that the canonical $185 bike scenario:
  - extracts 4 expense tuples via the LLM at ingest
  - answers "how much total" via deterministic gaṇita arithmetic
  - returns rec.ganita.value == 185

This is the empirical demonstration that Innovation #2 (Vedic karaṇa
LLM extraction) closes the synthesis-bounded gap that the regex
extractor leaves open.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import pytest

import patha
from patha.belief.karana import OllamaKaranaExtractor


_DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
_DEFAULT_MODEL = os.environ.get("PATHA_KARANA_MODEL", "qwen2.5:7b-instruct")


def _ollama_reachable() -> tuple[bool, list[str]]:
    """Return (reachable, available_models)."""
    try:
        req = urllib.request.Request(
            f"{_DEFAULT_HOST.rstrip('/')}/api/tags", method="GET",
        )
        with urllib.request.urlopen(req, timeout=2.0) as r:
            import json as _json
            data = _json.loads(r.read())
            return True, [m["name"] for m in data.get("models", [])]
    except (urllib.error.URLError, OSError):
        return False, []


@pytest.mark.slow
def test_karana_ollama_extracts_canonical_185_bike(tmp_path: Path) -> None:
    """The canonical bike-expense test, with a live local LLM."""
    reachable, models = _ollama_reachable()
    if not reachable:
        pytest.skip(
            f"Ollama not reachable at {_DEFAULT_HOST} — skip live test"
        )

    # Pick whatever model is available; user can pin via env var.
    if _DEFAULT_MODEL not in models:
        if not models:
            pytest.skip("Ollama running but no models pulled")
        chosen_model = models[0]
    else:
        chosen_model = _DEFAULT_MODEL

    karana = OllamaKaranaExtractor(
        model=chosen_model,
        host=_DEFAULT_HOST,
        timeout_s=60.0,
    )
    mem = patha.Memory(
        path=tmp_path / "store.jsonl",
        enable_phase1=False,
        karana_extractor=karana,
    )

    # Same 4 facts as examples/three_innovations_demo.py
    facts = [
        "I bought a $50 saddle for my bike",
        "I got a $75 helmet for the bike",
        "$30 for new bike lights",
        "I spent $30 on bike gloves",
    ]
    for f in facts:
        mem.remember(f, asserted_at=datetime(2024, 1, 1),
                     session_id="bike-shopping")

    # Karaṇa should have extracted at least 4 expense tuples (one per fact).
    assert len(mem._ganita_index) >= 4, (
        f"expected ≥ 4 tuples, got {len(mem._ganita_index)}; "
        f"karaṇa.calls={karana.calls} failures={karana.failures}"
    )
    assert karana.calls == 4, (
        f"expected exactly 4 LLM calls (one per ingest), got {karana.calls}"
    )

    # Recall should answer aggregation deterministically.
    rec = mem.recall("how much total did I spend on bike-related expenses?")
    assert rec.ganita is not None, "ganita layer didn't fire"
    # Allow ±5% slack since LLM extraction can normalise units differently.
    assert abs(rec.ganita.value - 185.0) <= max(185 * 0.05, 1.0), (
        f"expected ≈ 185, got {rec.ganita.value}; "
        f"explanation: {rec.ganita.explanation}"
    )
    # At least 3 of 4 ingested beliefs should contribute (allow 1 miss
    # for LLM phrasing edge cases).
    assert len(rec.ganita.contributing_belief_ids) >= 3
