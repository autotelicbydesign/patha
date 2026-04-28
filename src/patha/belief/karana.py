"""Karaṇa — Vedic ingest-time LLM extraction for the gaṇita layer.

The Vedic word *karaṇa* means "instrument" or "preparation": ritual
work done in advance so the moment of performance can be deterministic
and faithful. For Patha, the same idea splits the extractor in two:

  - **Ingest-time (karaṇa)** — a small local LLM reads each new belief,
    returns a structured list of `(entity, attribute, value, unit)`
    tuples. This is **once per belief**, costs LLM tokens up front.
  - **Recall-time (performance)** — pure deterministic arithmetic
    over the preserved tuple index. **Zero LLM tokens.**

This is the inverse of mainstream RAG architectures, which spend tokens
at recall (re-prompt with retrieved context every query). Most users
ask the same things many times; spending tokens once at ingest is a
strict win, plus it means recall is reproducible to the cent.

We don't replace the regex extractor — it stays as the zero-dependency
fallback and runs in unit tests. The karaṇa extractor is opt-in via:

    >>> from patha.belief.karana import OllamaKaranaExtractor
    >>> mem = patha.Memory(karana_extractor=OllamaKaranaExtractor())

When `OLLAMA_HOST` is reachable, every ingest goes through the LLM.
When it isn't, ingest gracefully falls back to the regex extractor.

Why Ollama / a small local LLM:
  - Patha's "no hosted API" constraint stays intact.
  - 7B-class models (Qwen2.5, Llama 3.1) handle structured extraction
    well; tested with `qwen2.5:7b-instruct` in our LongMemEval runs.
  - Latency is acceptable: ingest is offline; queries are still O(1).

Honest scope:
  - LLM extraction is not deterministic across model versions. The
    saved `~/.patha/<store>.ganita.jsonl` IS deterministic — the
    extracted tuples are immutable once written. Different LLM
    versions produce slightly different tuples on the same input,
    but that's a one-time choice at ingest, not a query-time problem.
  - Failure modes: malformed JSON output, hallucinated values,
    missing entities. We log + skip those; the regex extractor still
    catches obvious currency/duration patterns as a backstop.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterable, Protocol

from patha.belief.ganita import (
    GanitaTuple,
    _canonicalize_entity,
    extract_tuples as regex_extract_tuples,
)


# ─── Protocol ────────────────────────────────────────────────────────


class KaranaExtractor(Protocol):
    """Pluggable interface for ingest-time tuple extraction."""

    def extract(
        self,
        text: str,
        *,
        belief_id: str,
        time: str | None = None,
    ) -> list[GanitaTuple]: ...


# ─── Regex fallback ──────────────────────────────────────────────────


class RegexKaranaExtractor:
    """The zero-dependency baseline — wraps the existing regex extractor.

    Used in unit tests and when no LLM is configured. Works on
    obviously-numeric clean text (the demo cases). Documented to be
    less reliable on dense conversational text — that's why the LLM
    variant exists.
    """

    def extract(
        self, text: str, *, belief_id: str, time: str | None = None,
    ) -> list[GanitaTuple]:
        return regex_extract_tuples(text, belief_id=belief_id, time=time)


# ─── Ollama-backed LLM extractor ─────────────────────────────────────


_DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
_DEFAULT_MODEL = os.environ.get("PATHA_KARANA_MODEL", "qwen2.5:7b-instruct")


_KARANA_PROMPT = """You are a deterministic numerical-fact extractor.

From the user-supplied TEXT, extract every numerical fact that could be
the basis of a future arithmetic question (sum, count, average, etc.).
Return ONLY a JSON array. Each element MUST have these keys:

  entity     — the concrete topic the number refers to (single word or
               short noun phrase, lowercase, e.g. "saddle", "rent")
  aliases    — a JSON array of 1–3 BROADER topical categories this
               fact directly belongs to. Be precise; only include
               categories where the user might ask "how much have I
               spent on <X>" and reasonably expect this fact to count.
               If the entity itself is already a broad category, just
               repeat it. NEVER include incidental words from the
               text that aren't the fact's actual topic.
  attribute  — what the number measures: one of
               "expense" "income" "fundraising" "savings" "weight" "age"
               "duration" "count" "percentage" "value"
  value      — the numeric value as a JSON number (no commas, no
               currency symbol)
  unit       — "USD" "EUR" "hour" "minute" "day" "week" "month" "year"
               "%" "item" "kg" "lb" or any other unit literally present

If a number is not topical (e.g., a phone number, year, address) skip
it. If you can't tell which entity a number refers to, skip it.

Examples:

TEXT: "I bought a $50 saddle for my bike"
JSON: [{"entity":"saddle","aliases":["bike","cycling"],"attribute":"expense","value":50,"unit":"USD"}]

TEXT: "I spent 3.5 hours practicing yoga and donated $20 to charity"
JSON: [
  {"entity":"yoga","aliases":["yoga","fitness"],"attribute":"duration","value":3.5,"unit":"hour"},
  {"entity":"charity","aliases":["charity","donation"],"attribute":"fundraising","value":20,"unit":"USD"}
]

TEXT: "I have 4 bikes and my favourite color is blue"
JSON: [{"entity":"bike","aliases":["bike"],"attribute":"count","value":4,"unit":"item"}]

TEXT: "Paid $1500 rent on a flat near the bike path"
JSON: [{"entity":"rent","aliases":["rent","housing"],"attribute":"expense","value":1500,"unit":"USD"}]

TEXT: "user: How are you?\\nassistant: Doing fine, thanks for asking."
JSON: []

Now extract from this TEXT and return ONLY the JSON array:

TEXT: __TEXT__
JSON:"""


def _build_karana_prompt(text: str) -> str:
    """Substitute TEXT into the prompt without using str.format (which
    chokes on the JSON braces in the in-context examples)."""
    return _KARANA_PROMPT.replace("__TEXT__", text)


@dataclass
class OllamaKaranaExtractor:
    """Karaṇa extractor backed by a locally-running Ollama model.

    Parameters
    ----------
    model
        Ollama model tag. Must already be pulled
        (`ollama pull qwen2.5:7b-instruct`). Default
        `qwen2.5:7b-instruct`.
    host
        Ollama HTTP endpoint. Default `http://localhost:11434`.
    temperature
        Sampling temperature. Default 0.0 — extraction should be
        reproducible given the same input + model version.
    timeout_s
        Per-call timeout. Default 30.
    fallback_to_regex
        If the LLM call fails (network, parse error), fall through
        to the regex extractor instead of returning [] so the user
        always gets some signal. Default True.
    max_text_chars
        Truncate input text to this many chars. Default 6000 — fits
        a single LME session in a 4k-context window with headroom
        for prompt + output.
    """

    model: str = _DEFAULT_MODEL
    host: str = _DEFAULT_HOST
    temperature: float = 0.0
    timeout_s: float = 30.0
    fallback_to_regex: bool = True
    max_text_chars: int = 6000

    # Runtime stats
    calls: int = 0
    failures: int = 0
    total_latency_s: float = 0.0

    def extract(
        self,
        text: str,
        *,
        belief_id: str,
        time: str | None = None,
    ) -> list[GanitaTuple]:
        if not text.strip():
            return []

        truncated = text[: self.max_text_chars]
        prompt = _build_karana_prompt(truncated)

        try:
            raw = self._generate(prompt)
        except Exception:
            self.failures += 1
            if self.fallback_to_regex:
                return regex_extract_tuples(text, belief_id=belief_id, time=time)
            return []

        records = _parse_json_array(raw)
        if records is None:
            self.failures += 1
            if self.fallback_to_regex:
                return regex_extract_tuples(text, belief_id=belief_id, time=time)
            return []

        tuples: list[GanitaTuple] = []
        for r in records:
            t = _record_to_tuple(r, belief_id=belief_id, time=time, raw_text=truncated)
            if t is not None:
                tuples.append(t)
        return tuples

    def _generate(self, prompt: str) -> str:
        """POST to Ollama /api/generate. Returns the raw response text."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": 1024,  # enough for ~10-20 tuples
            },
        }
        req = urllib.request.Request(
            f"{self.host.rstrip('/')}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = resp.read()
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"karana: Ollama call failed ({self.host}, model={self.model}): {e}. "
                f"Is Ollama running? Try `ollama serve` and `ollama pull {self.model}`."
            ) from e
        finally:
            self.calls += 1
            self.total_latency_s += time.monotonic() - start

        data = json.loads(body)
        return str(data.get("response", "")).strip()


# ─── Helpers ────────────────────────────────────────────────────────


def _parse_json_array(raw: str) -> list[dict] | None:
    """Best-effort JSON-array extraction from a possibly-noisy LLM output.

    Handles three common forms:
      1. clean array: '[{...}, {...}]'
      2. wrapped in code fences: '```json\\n[...]\\n```'
      3. prose + array: 'Here is the JSON: [...]'

    Returns None if no parseable array is found.
    """
    s = raw.strip()
    # Strip code fences
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    s = s.strip()
    # Find the first '[' and the last ']' to handle prose-wrapped
    # arrays. Cheap heuristic; works for our zero-temperature extractor.
    if "[" not in s or "]" not in s:
        return None
    start = s.find("[")
    end = s.rfind("]")
    if end <= start:
        return None
    candidate = s[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    # Each element should be a dict; tolerate missing keys per element.
    return [r for r in parsed if isinstance(r, dict)]


def _record_to_tuple(
    record: dict,
    *,
    belief_id: str,
    time: str | None,
    raw_text: str,
) -> GanitaTuple | None:
    """Convert one LLM-emitted record into a GanitaTuple.

    Lenient: missing optional fields default; fundamentally-broken
    records (no value or no entity) return None.

    Aliases come from the LLM's explicit `aliases` field only. We do
    NOT auto-supplement from raw text noun-tokens — that adds
    incidental context words ("the bike path was nearby" pollutes a
    rent tuple with a bogus "bike" alias) which over-matches at query
    time. Trust the LLM's per-fact judgement on broader categories;
    the prompt explicitly asks for them.
    """
    entity_raw = str(record.get("entity", "")).strip().lower()
    attribute = str(record.get("attribute", "value")).strip().lower()
    unit = str(record.get("unit", "")).strip()
    if not entity_raw:
        return None
    try:
        value = float(record.get("value"))
    except (TypeError, ValueError):
        return None
    # Map common LLM-emitted attribute synonyms back to our canonical set.
    attribute = _ATTR_ALIASES.get(attribute, attribute)
    canon = _canonicalize_entity(entity_raw)
    # Aliases — canonical entity always first; LLM's explicit aliases
    # added (deduplicated, canonicalised). No auto-supplementation
    # from text words — that's noise, not signal.
    aliases = [canon]
    raw_aliases = record.get("aliases") or record.get("entity_aliases")
    if isinstance(raw_aliases, list):
        for a in raw_aliases:
            ca = _canonicalize_entity(str(a).strip().lower())
            if ca and ca not in aliases:
                aliases.append(ca)
    return GanitaTuple(
        entity=canon,
        attribute=attribute,
        value=value,
        unit=unit or "value",
        time=time,
        belief_id=belief_id,
        raw_text=raw_text[:200],
        entity_aliases=tuple(aliases),
    )


_ATTR_ALIASES = {
    "spent": "expense",
    "spending": "expense",
    "cost": "expense",
    "price": "expense",
    "earnings": "income",
    "earn": "income",
    "revenue": "income",
    "salary": "income",
    "donation": "fundraising",
    "donations": "fundraising",
    "raised": "fundraising",
    "saved": "savings",
    "deposit": "savings",
    "weight": "weight",
    "age": "age",
    "time": "duration",
    "hours": "duration",
    "percent": "percentage",
}


__all__ = [
    "KaranaExtractor",
    "RegexKaranaExtractor",
    "OllamaKaranaExtractor",
]
