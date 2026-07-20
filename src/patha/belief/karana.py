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
  aliases    — a JSON array of 2–5 broader topical categories the user
               might use to ask about this fact later. ALWAYS include
               the higher-level category the entity belongs to.

               STRICT RULES:
               - If the entity is a part, accessory, or maintenance item
                 of a larger thing, INCLUDE the larger thing.
                 Examples: saddle/helmet/chain/lights/pump/gloves/wheel/
                 tire/brake → ALWAYS include "bike" AND "cycling".
               - If the entity is a service performed on a thing,
                 INCLUDE the thing.
                 Example: "chain replacement" → include "bike".
               - If the entity is a hypothetical / range / not an actual
                 transaction, SKIP this fact entirely.
                 Example: "racks range from $100 to $500" → skip.
               - If the source text mentions the topic word casually
                 ("a flat near the bike path") and the entity is
                 unrelated (rent), do NOT include the casual word.

  attribute  — what the number measures: one of
               "expense" "income" "fundraising" "savings" "weight" "age"
               "duration" "count" "percentage" "value"
  value      — the numeric value as a JSON number (no commas, no
               currency symbol)
  unit       — "USD" "EUR" "hour" "minute" "day" "week" "month" "year"
               "%" "item" "kg" "lb" or any other unit literally present

If a number is not topical (a phone number, year, address, range, or
hypothetical) skip it. If you can't tell which entity a number refers
to, skip it.

Examples:

TEXT: "I bought a $50 saddle for my bike"
JSON: [{"entity":"saddle","aliases":["saddle","bike","cycling"],"attribute":"expense","value":50,"unit":"USD"}]

TEXT: "Got new bike lights installed for $40"
JSON: [{"entity":"lights","aliases":["lights","bike","cycling","accessories"],"attribute":"expense","value":40,"unit":"USD"}]

TEXT: "Bell Zephyr helmet from the bike shop, $120"
JSON: [{"entity":"helmet","aliases":["helmet","bike","cycling","safety"],"attribute":"expense","value":120,"unit":"USD"}]

TEXT: "Mechanic replaced my chain for $25"
JSON: [{"entity":"chain","aliases":["chain","bike","cycling","maintenance"],"attribute":"expense","value":25,"unit":"USD"}]

TEXT: "I spent 3.5 hours practicing yoga and donated $20 to charity"
JSON: [
  {"entity":"yoga","aliases":["yoga","fitness","exercise"],"attribute":"duration","value":3.5,"unit":"hour"},
  {"entity":"charity","aliases":["charity","donation","giving"],"attribute":"fundraising","value":20,"unit":"USD"}
]

TEXT: "I have 4 bikes and my favourite color is blue"
JSON: [{"entity":"bike","aliases":["bike","cycling"],"attribute":"count","value":4,"unit":"item"}]

TEXT: "Paid $1500 rent on a flat near the bike path"
JSON: [{"entity":"rent","aliases":["rent","housing"],"attribute":"expense","value":1500,"unit":"USD"}]

TEXT: "Bike racks range from $100 to $500 depending on size"
JSON: []

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


# ─── Hybrid: regex finds amounts, LLM tags them ──────────────────────


_HYBRID_PROMPT = """You are tagging numerical amounts in a TEXT.

The TEXT is a conversation. We have already located every dollar
amount with regex; for each one we need YOUR semantic judgement on
what it refers to.

For each numbered AMOUNT, return ONE entry in the JSON array. Each
entry MUST have these keys:

  index      — the AMOUNT number (1, 2, 3, ...)
  entity     — the concrete topic the amount refers to (single word
               or short noun phrase, lowercase). If the amount is a
               range, hypothetical, gift card, store credit, phone
               number, or doesn't refer to a real purchase, set
               entity to "skip" and we'll ignore it.
  aliases    — a JSON array of 2–5 broader topical categories. If
               the entity is a part/accessory/maintenance of a
               larger thing, ALWAYS include the larger thing.
               Examples: chain/helmet/saddle/lights/pump/wheel →
               always include "bike". Drivers seat/headlights/tire
               for car → always include "car". Plumber/electrician
               for home → always include "home".
  attribute  — what the amount measures: one of
               "expense" (money out), "income" (money in), "savings",
               "fundraising", "value" (descriptive, not a transaction).
               Default to "expense" for purchases.

Examples of TEXT + amounts → JSON:

TEXT: "I spent $50 on a saddle for my bike. Got a $75 helmet too."
AMOUNTS:
  1. $50
  2. $75
JSON: [
  {"index":1,"entity":"saddle","aliases":["saddle","bike","cycling"],"attribute":"expense"},
  {"index":2,"entity":"helmet","aliases":["helmet","bike","cycling","safety"],"attribute":"expense"}
]

TEXT: "Bike racks range from $100 to $500."
AMOUNTS:
  1. $100
  2. $500
JSON: [
  {"index":1,"entity":"skip","aliases":[],"attribute":"value"},
  {"index":2,"entity":"skip","aliases":[],"attribute":"value"}
]

TEXT: "The mechanic replaced my chain for $25 and I got new bike
lights for $40 the same day."
AMOUNTS:
  1. $25
  2. $40
JSON: [
  {"index":1,"entity":"chain","aliases":["chain","bike","cycling","maintenance"],"attribute":"expense"},
  {"index":2,"entity":"lights","aliases":["lights","bike","cycling","accessories"],"attribute":"expense"}
]

Now tag each AMOUNT in this TEXT and return ONLY the JSON array.

TEXT: __TEXT__
AMOUNTS:
__AMOUNTS__
JSON:"""


_HYBRID_AMOUNT_RE = re.compile(r"\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)")


@dataclass
class HybridKaranaExtractor:
    """Hybrid: regex finds every \\$X amount; LLM only labels.

    Solves the recall problem of the pure-LLM extractor: small models
    routinely miss dollar amounts entirely. By letting regex enumerate
    every match first and asking the LLM only for the semantic tag
    (entity, aliases, attribute), we never lose an amount. The LLM can
    explicitly say `entity: "skip"` for amounts that aren't real
    transactions (ranges, gift cards, phone numbers, etc.).

    Parameters mirror :class:`OllamaKaranaExtractor`.
    """

    model: str = _DEFAULT_MODEL
    host: str = _DEFAULT_HOST
    temperature: float = 0.0
    timeout_s: float = 60.0
    # Chunking: long sessions (LongMemEval haystacks routinely run
    # 10k+ chars) get split into overlapping chunks so no $X amount
    # is lost to truncation. Each chunk is one LLM call.
    chunk_chars: int = 4000
    chunk_overlap: int = 400

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
        # Chunk long texts so we don't lose any $X amount to truncation.
        chunks = self._chunk(text)
        all_tuples: list[GanitaTuple] = []
        for chunk in chunks:
            all_tuples.extend(
                self._extract_from_chunk(
                    chunk, belief_id=belief_id, time=time,
                )
            )
        return all_tuples

    def _chunk(self, text: str) -> list[str]:
        """Split text into overlapping chunks of ~chunk_chars each.
        Overlap (chunk_overlap) ensures an amount near a boundary still
        gets full context."""
        if len(text) <= self.chunk_chars:
            return [text]
        chunks = []
        step = self.chunk_chars - self.chunk_overlap
        i = 0
        while i < len(text):
            chunks.append(text[i : i + self.chunk_chars])
            i += step
        return chunks

    def _extract_from_chunk(
        self,
        text: str,
        *,
        belief_id: str,
        time: str | None = None,
    ) -> list[GanitaTuple]:
        """One LLM call: regex finds amounts in `text`; LLM tags them."""
        truncated = text  # already chunked upstream
        amounts = list(_HYBRID_AMOUNT_RE.finditer(truncated))
        if not amounts:
            return []
        # Build the AMOUNTS section
        amount_lines = []
        for i, m in enumerate(amounts, 1):
            amount_lines.append(f"  {i}. ${m.group(1)}")
        prompt = _HYBRID_PROMPT.replace("__TEXT__", truncated)
        prompt = prompt.replace("__AMOUNTS__", "\n".join(amount_lines))

        try:
            raw = self._generate(prompt)
        except Exception:
            self.failures += 1
            return []
        records = _parse_json_array(raw)
        if records is None:
            self.failures += 1
            return []

        tuples: list[GanitaTuple] = []
        for r in records:
            try:
                idx = int(r.get("index", 0))
            except (TypeError, ValueError):
                continue
            if idx < 1 or idx > len(amounts):
                continue
            entity = str(r.get("entity", "")).strip().lower()
            if not entity or entity == "skip":
                continue
            attribute = str(r.get("attribute", "expense")).strip().lower()
            attribute = _ATTR_ALIASES.get(attribute, attribute)
            aliases_raw = r.get("aliases") or []
            if not isinstance(aliases_raw, list):
                aliases_raw = []

            # Parse the value back from the regex match
            m = amounts[idx - 1]
            try:
                value = float(m.group(1).replace(",", ""))
            except ValueError:
                continue

            canon = _canonicalize_entity(entity)
            aliases = [canon]
            for a in aliases_raw:
                ca = _canonicalize_entity(str(a).strip().lower())
                if ca and ca not in aliases:
                    aliases.append(ca)

            tuples.append(GanitaTuple(
                entity=canon,
                attribute=attribute,
                value=value,
                unit="USD",  # hybrid is currency-only for now
                time=time,
                belief_id=belief_id,
                # Keep enough context so the recall-time topic-proximity
                # fallback can see topical words within the configured
                # window (default 80 chars). 100/100 gives 200 chars
                # total — enough for the typical sentence-with-context
                # scenario without bloating the index.
                raw_text=truncated[
                    max(0, m.start() - 100):
                    min(len(truncated), m.end() + 100)
                ],
                entity_aliases=tuple(aliases),
            ))
        return tuples

    def _generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": 2048,  # bigger budget — multi-tag output
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
                f"karana hybrid: Ollama call failed ({self.host}): {e}"
            ) from e
        finally:
            self.calls += 1
            self.total_latency_s += time.monotonic() - start
        data = json.loads(body)
        return str(data.get("response", "")).strip()


__all__ = [
    "KaranaExtractor",
    "RegexKaranaExtractor",
    "OllamaKaranaExtractor",
    "HybridKaranaExtractor",
]


# ─── Dependency-parse extractor (karaṇa v2 — the no-LLM bet) ─────────


_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12,
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}

_CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP"}
_CURRENCY_WORDS = {
    "dollar": "USD", "dollars": "USD", "usd": "USD",
    "euro": "EUR", "euros": "EUR", "eur": "EUR",
    "pound": "GBP", "pounds": "GBP", "gbp": "GBP",
}

_MONEY_IN_LEMMAS = {
    "refund", "reimburse", "credit", "sell", "knock", "discount",
    "waive", "cashback",
}
_IRREALIS_LEMMAS = {
    "if", "imagine", "wonder", "whether", "hope", "might", "would",
    "consider", "maybe", "probably",
}
_COLLOQUIAL_QUANTS = {"couple", "few", "several", "bunch"}
_ROUND_WORDS = {"hundred", "thousand", "million", "grand"}


class DepParseKaranaExtractor:
    """Karaṇa v2: dependency-parse extraction — no LLM, no network.

    Wakes the spaCy machinery that sat as dead code in
    `query/entities.py`: amounts attach to entities via dependency
    paths (pobj of on/for preps, copular nsubj, nummod heads) instead
    of character windows, and the veto families (ranges, hypotheticals,
    money-in, colloquial quantities, numeric distractors) run at the
    CLAUSE level — the regex extractor's measured failure mode
    (KaranaEval: forbidden_hit 0.636, ranges and temperatures extracted
    as money) is a window problem, and clauses are the right window.

    Instrument: eval/karana_eval.py. Falls back to the regex extractor
    only when the spaCy model is missing, and says so loudly via
    `parser_available` (never a silent identity swap — the lesson from
    the eval runner's ollama-probe rule).
    """

    def __init__(self) -> None:
        self._nlp = None
        self.parser_available: bool | None = None

    def _load(self):
        if self._nlp is None and self.parser_available is None:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
                self.parser_available = True
            except Exception:
                self.parser_available = False
        return self._nlp

    # ── amount classification ────────────────────────────────────

    @staticmethod
    def _num_value(tok) -> float | None:
        t = tok.text.replace(",", "")
        try:
            return float(t)
        except ValueError:
            pass
        return float(_WORD_NUMBERS[t.lower()]) if t.lower() in _WORD_NUMBERS \
            else None

    def _classify_amount(self, tok, sent):
        """Return (value, unit, kind) or None. kind ∈ money / item /
        km / years / hours."""
        text = tok.text
        # "8k" distance shorthand in a movement context
        m = re.fullmatch(r"(\d+(?:\.\d+)?)k", text.lower())
        if m is not None:
            if any(t.lemma_ in ("run", "ride", "walk", "cycle", "row")
                   for t in sent):
                return float(m.group(1)), "km", "km"
            return None
        if not tok.like_num:
            return None
        value = self._num_value(tok)
        if value is None:
            return None
        # explicit distance/duration units
        nxt = tok.nbor(1) if tok.i + 1 < len(tok.doc) else None
        if nxt is not None and nxt.lemma_ in ("km", "kilometre", "kilometer"):
            return value, "km", "km"
        if nxt is not None and nxt.lemma_ in ("hour", "hr"):
            return value, "hours", "hours"
        # currency: symbol/word neighbours
        prv = tok.nbor(-1) if tok.i > 0 else None
        if prv is not None and prv.text in _CURRENCY_SYMBOLS:
            return value, _CURRENCY_SYMBOLS[prv.text], "money"
        if nxt is not None and nxt.lower_ in _CURRENCY_WORDS:
            return value, _CURRENCY_WORDS[nxt.lower_], "money"
        # spaCy sometimes merges "$15.99" into one token
        m = re.fullmatch(r"([$€£])(\d+(?:,\d{3})*(?:\.\d+)?)", text)
        if m is not None:
            return (float(m.group(2).replace(",", "")),
                    _CURRENCY_SYMBOLS[m.group(1)], "money")
        # ages: copular construction with a person-ish subject
        head = tok.head
        if head.lemma_ == "be" or (tok.dep_ in ("attr", "acomp", "conj")
                                   and head.head.lemma_ == "be"):
            subj = [t for t in (head.lefts if head.lemma_ == "be"
                    else head.head.lefts) if t.dep_ == "nsubj"]
            if subj and subj[0].ent_type_ in ("", "PERSON") \
                    and subj[0].lemma_ not in ("it", "that", "this"):
                return value, "years", "years"
        # counts: nummod on a noun ("three succulents", "two kittens")
        if tok.dep_ == "nummod" and head.pos_ in ("NOUN", "PROPN"):
            return value, "item", "item"
        # ordinal counts: "my ninth book"
        if tok.lower_ in _WORD_NUMBERS and tok.lower_.endswith("th") \
                and head.pos_ == "NOUN":
            return value, "item", "item"
        return None

    # ── clause-level vetoes ──────────────────────────────────────

    def _vetoed(self, tok, sent) -> bool:
        # irrealis / hypothetical anywhere in the sentence
        if any(t.lemma_.lower() in _IRREALIS_LEMMAS for t in sent):
            return True
        if "'d" in sent.text or "’d" in sent.text:
            return True
        # money-in: a money-in verb governs the amount's clause
        anc = list(tok.ancestors)
        if any(a.lemma_ in _MONEY_IN_LEMMAS for a in anc):
            return True
        # range: between/from ... to/and joining two amounts
        low = sent.text.lower()
        if re.search(r"\b(?:between|somewhere|from)\b.*\d.*\b(?:and|to)\b.*\d",
                     low) or re.search(r"\$?\d[\d,.]*\s*(?:to|-|–)\s*\$?\d", low):
            return True
        # colloquial quantities: "a couple hundred", "a few thousand"
        if any(t.lemma_ in _COLLOQUIAL_QUANTS for t in sent) and (
            tok.lower_ in _ROUND_WORDS
            or any(t.lower_ in _ROUND_WORDS for t in sent)
        ):
            return True
        # numeric distractors: clock times, versions, temperatures,
        # process durations with a non-personal subject
        nxt = tok.nbor(1) if tok.i + 1 < len(tok.doc) else None
        if nxt is not None and nxt.lower_ in ("pm", "am"):
            return True
        if re.fullmatch(r"\d+(?::\d+)?(?:pm|am)", tok.lower_):
            return True
        if tok.head.lemma_ == "version" or (
            nxt is not None and nxt.lemma_ == "degree"
        ) or (tok.i > 0 and tok.nbor(-1).lemma_ == "version"):
            return True
        if nxt is not None and nxt.lemma_ in ("minute", "second"):
            # clause-scoped, not sentence-scoped: "the build takes 45
            # minutes … and WE'RE on version 2.7" must not let the
            # unrelated we-clause launder the build's duration
            v = tok
            while v.head.i != v.i and v.pos_ not in ("VERB", "AUX"):
                v = v.head
            subjs = [c for c in v.children if c.dep_ == "nsubj"]
            if not any(s.lower_ in ("i", "we") for s in subjs):
                return True
        return False

    # ── entity attachment ────────────────────────────────────────

    @staticmethod
    def _phrase(tok) -> str:
        """Noun + its compound/amod lefts: 'saddle bag', 'emergency
        fund' (articles and quantifier adjectives dropped)."""
        parts = [
            t.text for t in tok.lefts
            if t.dep_ in ("compound", "amod") and t.pos_ != "DET"
        ] + [tok.text]
        return " ".join(parts)

    def _entity_for(self, tok, sent):
        """Walk the dependency graph from the amount to its entity.
        Preference order: pobj of on/for under the governing verb (or
        the amount itself), copular nsubj, dobj sibling, nummod head."""
        head = tok.head
        # "charged $150 TOTAL": 'total' is an amount qualifier, never
        # the entity — fall through to the verb logic instead
        if tok.dep_ == "nummod" and head.pos_ in ("NOUN", "PROPN") \
                and not head.like_num and head.lemma_ not in ("total",):
            return head, self._phrase(head)
        # appositive amounts on mistagged brand nouns: "the airbnb
        # $380" — spaCy tags unknown lowercase brands ADJ; the appos
        # attachment is still the entity signal
        if tok.dep_ == "appos" and head.is_alpha and not head.like_num \
                and head.pos_ in ("NOUN", "PROPN", "ADJ", "X"):
            return head, self._phrase(head)
        # governing verb (climb through the noun the amount modifies)
        # NB: spaCy Tokens are ephemeral views — identity (`is`) fails
        # even at the root; compare indices.
        v = tok
        while v.head.i != v.i and v.pos_ not in ("VERB", "AUX"):
            v = v.head
        # prep objects: spent $40 ON THE PUMP / paid 220 euros FOR THE
        # HOTEL. A verb can carry several (…$40 on the pump and $85 on
        # the saddle bag): each amount claims the NEAREST object after
        # it, falling back to nearest overall.
        candidates = []
        for holder in (v, tok, head):
            for child in holder.children:
                if child.dep_ == "prep" and child.lemma_ in ("on", "for"):
                    for c in child.children:
                        # "replaced the chain for $28": the amount can
                        # BE the for-object — never its own entity
                        if c.dep_ == "pobj" and c.i != tok.i \
                                and not c.like_num:
                            candidates.append(c)
        if candidates:
            after = [c for c in candidates if c.i > tok.i]
            pool = after or candidates
            best = min(pool, key=lambda c: abs(c.i - tok.i))
            return best, self._phrase(best)
        # copular: THE VET VISIT was $120
        if v.lemma_ == "be":
            subjs = [c for c in v.children if c.dep_ in ("nsubj",
                                                         "nsubjpass")]
            if subjs and subjs[0].pos_ in ("NOUN", "PROPN"):
                return subjs[0], self._phrase(subjs[0])
        # verb-governed: renewed THE SUBSCRIPTION at $15.99 / THE CAT
        # SITTER charged $150 (charge-verbs: the subject IS the entity)
        if v.pos_ in ("VERB", "AUX"):
            prefer_subj = v.lemma_ in ("charge", "cost", "bill", "quote")
            deps = ("nsubj", "dobj") if prefer_subj else ("dobj", "nsubj")
            for dep in deps:
                objs = [c for c in v.children if c.dep_ == dep
                        and c.pos_ in ("NOUN", "PROPN")
                        and c.i != tok.i and not c.like_num]
                if objs:
                    return objs[0], self._phrase(objs[0])
        # verbless list fragments ("gym renewal $89, … running shoes
        # $130", "the airbnb $380"): the nearest noun to the LEFT owns
        # the amount
        for j in range(tok.i - 1, max(sent.start, tok.i - 6) - 1, -1):
            t = tok.doc[j]
            if t.pos_ in ("NOUN", "PROPN") and not t.like_num \
                    and t.lemma_ not in ("total",):
                return t, self._phrase(t)
        # noun the amount modifies (appositive)
        if head.pos_ in ("NOUN", "PROPN"):
            return head, self._phrase(head)
        return None, None

    @staticmethod
    def _aliases(ent_tok, phrase, sent) -> tuple[str, ...]:
        out: list[str] = []
        for cand in [phrase, *(
            t.text for t in sent
            if t.pos_ in ("NOUN", "PROPN") and t is not ent_tok
        )]:
            c = _canonicalize_entity(cand)
            if c and c not in out:
                out.append(c)
        # part→whole from the shared alias table (chain → bike …)
        from patha.belief.ganita import ENTITY_ALIASES
        for c in list(out):
            whole = ENTITY_ALIASES.get(c)
            if whole and whole not in out:
                out.append(whole)
        return tuple(out)

    # ── main entry ───────────────────────────────────────────────

    _ROLE_LINE = re.compile(
        r"^(user|human|assistant|system|ai)\s*:\s*",
        re.IGNORECASE | re.MULTILINE,
    )

    def _user_text(self, text: str) -> str:
        """Karaṇa preserves the USER's numerical facts. In role-tagged
        session transcripts ("user: …" / "assistant: …"), assistant
        turns are full of illustrative amounts ("free shipping over
        $75 is a great strategy", "Example: Spend $50, get 10% off")
        that are nobody's spending — the ganita smoke measured them
        summed into user totals (0/8 with 3 hard fails traced to
        exactly this class). Text with no role markers (the ordinary
        Memory.remember path) passes through untouched."""
        matches = list(self._ROLE_LINE.finditer(text))
        if not matches:
            return text
        parts: list[str] = []
        for i, m in enumerate(matches):
            if m.group(1).lower() not in ("user", "human"):
                continue
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            parts.append(text[m.end():end])
        return "\n".join(parts)

    def extract(
        self, text: str, *, belief_id: str, time: str | None = None,
    ) -> list[GanitaTuple]:
        nlp = self._load()
        if nlp is None:
            return regex_extract_tuples(text, belief_id=belief_id, time=time)
        from patha.belief.ganita import _detect_attribute

        doc = nlp(self._user_text(text))
        tuples: list[GanitaTuple] = []
        age_nouns: set[str] = set()
        # first pass: ages (so a count of the same noun can defer)
        for sent in doc.sents:
            for tok in sent:
                cls = self._classify_amount(tok, sent)
                if cls and cls[2] == "years" and not self._vetoed(tok, sent):
                    subj = [t for t in sent if t.dep_ == "nsubj"]
                    if subj:
                        age_nouns.add(_canonicalize_entity(subj[0].text))
        for sent in doc.sents:
            for tok in sent:
                cls = self._classify_amount(tok, sent)
                if cls is None:
                    continue
                value, unit, kind = cls
                if self._vetoed(tok, sent):
                    continue
                ent_tok, phrase = self._entity_for(tok, sent)
                if kind == "years":
                    subj = [t for t in sent if t.dep_ == "nsubj"]
                    if subj:
                        ent_tok, phrase = subj[0], self._phrase(subj[0])
                elif kind in ("km", "hours"):
                    # "ran 8k": the activity is the governing verb, not
                    # the measure token
                    v = tok
                    while v.head.i != v.i and v.pos_ not in ("VERB", "AUX"):
                        v = v.head
                    if v.pos_ == "VERB":
                        ent_tok, phrase = v, v.lemma_
                if ent_tok is None or not phrase:
                    continue
                entity = _canonicalize_entity(phrase)
                # a count of a noun whose AGES were extracted is context,
                # not a separate aggregable fact ("my three nephews are
                # 4, 7 and 12" — the ages are the facts)
                if kind == "item" and entity in age_nouns:
                    continue
                attr_default = {
                    "money": "expense", "item": "count", "km": "distance",
                    "years": "age", "hours": "duration",
                }[kind]
                tuples.append(GanitaTuple(
                    entity=entity,
                    attribute=_detect_attribute(sent.text,
                                                default=attr_default),
                    value=float(value),
                    unit=unit,
                    time=time,
                    belief_id=belief_id,
                    raw_text=sent.text.strip(),
                    entity_aliases=self._aliases(ent_tok, phrase, sent),
                ))
        # "one physio session $70": the money is the fact; a count of
        # the same noun in the same sentence is context, not a second
        # aggregable fact
        money_keys = {
            (t.entity, t.raw_text) for t in tuples if t.unit in
            ("USD", "EUR", "GBP")
        }
        tuples = [
            t for t in tuples
            if not (t.unit == "item" and (t.entity, t.raw_text) in money_keys)
        ]
        return tuples
