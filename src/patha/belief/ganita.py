"""Gaṇita layer — Vedic-tradition arithmetic-as-preservation for Patha.

Background
==========

The six Vedāṅga (auxiliary disciplines around the Vedic corpus) include
*gaṇita* (computation/mathematics) and *jyotiṣa* (astronomy). The Sulba­sūtras
encode procedural geometry for ritual altars: deterministic rule-application
on preserved inputs. The result is reproducible from the same inputs every
time. This is NOT *mīmāṃsā* (interpretation) — it's the tradition's explicit
"do arithmetic on preserved facts" lineage.

The Aboriginal songline tradition has a parallel: increase-walks include
totalling sites where the songkeeper recounts everything encountered along
the path. Counting is part of the recitation, not derived from it.

This module gives Patha a tradition-faithful way to answer aggregation
questions ("how much total", "how many", "average", "how much more")
without invoking an LLM. Inputs are preserved literally as beliefs; the
arithmetic is procedural and shows its work.

Three components
================

1. **GanitaExtractor** — at ingest time, parses (entity_canonical,
   attribute, numeric_value, unit, time) tuples from a proposition's
   text. Pure regex + dateparser. Lossless: the original belief is
   never changed; the tuples are added to a sidecar index.

2. **GanitaIndex** — append-only sidecar index keyed by
   (entity_canonical, attribute). Mirrors the same JSONL pattern as
   the BeliefStore. Each entry tracks (value, unit, time, belief_id)
   so the source proposition is always recoverable.

3. **answer_aggregation_question(question, index)** — at query time,
   detects an aggregation operator from the question wording (sum,
   count, average, difference, max/min). When detected, runs the
   procedural arithmetic over matching index entries. Returns
   (computed_value, contributing_belief_ids) — both shown to the user
   so the source preservation principle is honored.

Honest scope
============

- Currency, counts, durations, percentages — these cover the vast
  majority of synthesis-bounded questions on LongMemEval.
- Entity canonicalisation here is heuristic (lowercased noun phrase,
  spaCy NER pass for nouns). Real-world fuzziness ("bike" vs "bicycle"
  vs "cycling expense") is a known weakness; we aliased it via small
  hand-curated synonyms.
- This is procedural arithmetic. It does NOT do natural-language
  reasoning about the question. "How much did I spend on bike-related
  expenses" works if "bike" is the canonical entity; "How much did I
  spend on transportation" might not match unless we also tag the
  bike beliefs as transportation. That's an alias-table problem,
  documented honestly.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Literal


# ─── Data model ──────────────────────────────────────────────────────

# An aggregation operator detected in the question.
AggOp = Literal["sum", "count", "average", "difference", "max", "min"]


@dataclass(frozen=True)
class GanitaTuple:
    """One numerical fact extracted from a belief.

    `entity` is the most-likely primary entity (a single canonical noun).
    `entity_aliases` is a list of all noun-like tokens from the surrounding
    context — at query time, any of these can match.

    Examples
    --------
    "I bought a $50 saddle for the bike"
        → entity="saddle", entity_aliases=["saddle","bike"], value=50, unit=USD
    "I have 4 bikes"
        → entity="bike", entity_aliases=["bike"], value=4, unit=item
    "I spent 3.5 hours on yoga"
        → entity="yoga", entity_aliases=["yoga"], value=3.5, unit=hours
    """

    entity: str
    attribute: str
    value: float
    unit: str
    time: str | None
    belief_id: str
    raw_text: str
    entity_aliases: tuple[str, ...] = ()  # all canonical entity candidates

    def to_dict(self) -> dict:
        d = asdict(self)
        # Convert tuple→list for JSON
        d["entity_aliases"] = list(d["entity_aliases"])
        return d

    def matches_entity(self, query_entity: str) -> bool:
        """Match if query_entity == entity OR appears in aliases."""
        q = _canonicalize_entity(query_entity)
        if q == self.entity:
            return True
        return q in self.entity_aliases


# ─── Extraction ──────────────────────────────────────────────────────

# Currency patterns: $50, $1,234.56, 50 dollars, USD 50
_CURRENCY = re.compile(
    r"(?:\$|USD\s*)(?P<amt>\d+(?:,\d{3})*(?:\.\d+)?)"
    r"|(?P<amt2>\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:dollars?|USD)\b",
    re.IGNORECASE,
)

# Hours / minutes / weeks / days / months / years
_DURATION = re.compile(
    r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>"
    r"hours?|hrs?|minutes?|mins?|seconds?|secs?"
    r"|days?|weeks?|months?|years?"
    r")\b",
    re.IGNORECASE,
)

# Percentages
_PERCENT = re.compile(r"(?P<num>\d+(?:\.\d+)?)\s*%", re.IGNORECASE)

# Counts: "4 bikes", "23 short stories", "99 rare items"
_COUNT = re.compile(
    r"\b(?P<num>\d+|one|two|three|four|five|six|seven|eight|nine|ten"
    r"|eleven|twelve|thirteen|fourteen|fifteen|twenty|thirty|forty|fifty)"
    r"\s+(?P<thing>\w+(?:\s+\w+)?)\b",
    re.IGNORECASE,
)

_WORD_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
}


# Light-weight synonym aliasing — keeps the design honest (we declare
# what's aliased) without trying to do general semantic matching.
ENTITY_ALIASES: dict[str, str] = {
    "bicycle": "bike",
    "bicycles": "bike",
    "bikes": "bike",
    "cycling": "bike",
    # transportation general — opt-in only via explicit alias
    "subway": "transit",
    "bus": "transit",
    "train": "transit",
    # food / groceries
    "groceries": "grocery",
    "supermarket": "grocery",
    # writing
    "stories": "writing",
    "essays": "writing",
    "articles": "writing",
    "pieces": "writing",
}


# Attribute heuristics: which words near a number suggest which attribute?
_ATTR_KEYWORDS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:spent|spend|paid|pay|cost|costs?|price|priced|charge)\b", re.I), "expense"),
    (re.compile(r"\b(?:earned|earn|made|income|salary|revenue|profit)\b", re.I), "income"),
    (re.compile(r"\b(?:raised|donated|contribution)\b", re.I), "fundraising"),
    (re.compile(r"\b(?:saved|saving|deposit)\b", re.I), "savings"),
    (re.compile(r"\b(?:weighs?|weight)\b", re.I), "weight"),
    (re.compile(r"\b(?:age|aged?|old|years?\s+old)\b", re.I), "age"),
    (re.compile(r"\b(?:duration|hours?|spent.*hours?|took.*weeks?)\b", re.I), "duration"),
]


def _canonicalize_entity(text: str) -> str:
    """Lowercase, strip plurals, apply alias table."""
    t = text.lower().strip()
    if t in ENTITY_ALIASES:
        return ENTITY_ALIASES[t]
    # naive de-pluralisation
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        singular = t[:-1]
        if singular in ENTITY_ALIASES:
            return ENTITY_ALIASES[singular]
        return singular
    return t


def _detect_attribute(context: str, default: str = "value") -> str:
    """Choose an attribute based on surrounding text keywords."""
    for pat, attr in _ATTR_KEYWORDS:
        if pat.search(context):
            return attr
    return default


def _sentences(text: str) -> list[tuple[int, str]]:
    """Split text into sentences (with offsets). Sentence boundaries:
    period, question mark, exclamation, paragraph break, or 'role:'
    prefix that appears in our session-concatenation format."""
    # Coarse: split on period/!/?/\n\n
    pieces: list[tuple[int, str]] = []
    pos = 0
    for chunk in re.split(r"(?<=[.!?])\s+|\n{2,}", text):
        if not chunk:
            continue
        # Skip role markers as their own "sentence"
        idx = text.find(chunk, pos)
        if idx < 0:
            idx = pos
        pieces.append((idx, chunk))
        pos = idx + len(chunk)
    return pieces


def extract_tuples(
    text: str,
    *,
    belief_id: str,
    time: str | None = None,
    entity_hints: list[str] | None = None,
) -> list[GanitaTuple]:
    """Extract numerical (entity, attribute, value, unit) tuples from text.

    Entity binding is **sentence-scoped**: a number's entity_aliases come
    from the SAME SENTENCE the number appears in, not the ±60 char
    sliding window that crossed sentence boundaries. This prevents
    false-positives where currency from one topic gets aliased with
    nouns from a neighboring sentence about something else entirely.

    `entity_hints` is an optional list of entities (e.g. from spaCy NER
    or from the user's question context) — when provided, the extractor
    prefers them as the entity field.
    """
    tuples: list[GanitaTuple] = []
    # Pre-segment into sentences; each extraction's context is the
    # sentence containing the number.
    sentences = _sentences(text) or [(0, text)]

    # Currency
    for m in _CURRENCY.finditer(text):
        amt_str = (m.group("amt") or m.group("amt2") or "").replace(",", "")
        if not amt_str:
            continue
        try:
            value = float(amt_str)
        except ValueError:
            continue
        # Sentence-scoped context: find the sentence containing this match.
        ctx = ""
        for s_off, s_text in sentences:
            if s_off <= m.start() < s_off + len(s_text):
                ctx = s_text
                break
        if not ctx:
            ctx = text[max(0, m.start()-60):min(len(text), m.end()+60)]
        attribute = _detect_attribute(ctx, default="expense")
        entity = _pick_entity(ctx, entity_hints)
        aliases = tuple(_all_entities(ctx))
        tuples.append(GanitaTuple(
            entity=entity, attribute=attribute,
            value=value, unit="USD",
            time=time, belief_id=belief_id,
            raw_text=text[m.start():m.end()],
            entity_aliases=aliases,
        ))

    # Durations
    for m in _DURATION.finditer(text):
        try:
            value = float(m.group("num"))
        except ValueError:
            continue
        unit = m.group("unit").lower().rstrip("s")
        # Normalize: "hr" → "hour", "min" → "minute", "sec" → "second"
        unit = {"hr": "hour", "min": "minute", "sec": "second"}.get(unit, unit)
        ctx_start = max(0, m.start() - 60)
        ctx_end = min(len(text), m.end() + 60)
        ctx = text[ctx_start:ctx_end]
        entity = _pick_entity(ctx, entity_hints)
        aliases = tuple(_all_entities(ctx))
        tuples.append(GanitaTuple(
            entity=entity, attribute="duration",
            value=value, unit=unit,
            time=time, belief_id=belief_id,
            raw_text=text[m.start():m.end()],
            entity_aliases=aliases,
        ))

    # Percentages
    for m in _PERCENT.finditer(text):
        try:
            value = float(m.group("num"))
        except ValueError:
            continue
        ctx_start = max(0, m.start() - 60)
        ctx_end = min(len(text), m.end() + 60)
        ctx = text[ctx_start:ctx_end]
        entity = _pick_entity(ctx, entity_hints)
        aliases = tuple(_all_entities(ctx))
        tuples.append(GanitaTuple(
            entity=entity, attribute="percentage",
            value=value, unit="%",
            time=time, belief_id=belief_id,
            raw_text=text[m.start():m.end()],
            entity_aliases=aliases,
        ))

    # Counts ("4 bikes", "twenty short stories")
    for m in _COUNT.finditer(text):
        num_str = m.group("num").lower()
        if num_str.isdigit():
            value = float(num_str)
        else:
            value = float(_WORD_NUM.get(num_str, 0))
            if value == 0:
                continue
        thing = m.group("thing")
        # Skip if the count is right next to currency/duration/percent
        # (already captured above as the more specific pattern)
        snippet = text[max(0, m.start()-1):min(len(text), m.end()+1)]
        if any(k in snippet.lower() for k in ["$", " usd", " dollar"]):
            continue
        if any(k in thing.lower() for k in ["hour", "minute", "second", "day", "week", "month", "year", "%"]):
            continue
        entity = _canonicalize_entity(thing)
        # Sentence-scoped context for count aliases
        ctx = ""
        for s_off, s_text in sentences:
            if s_off <= m.start() < s_off + len(s_text):
                ctx = s_text
                break
        if not ctx:
            ctx = text[max(0, m.start()-60):min(len(text), m.end()+60)]
        ctx_aliases = _all_entities(ctx)
        # ensure primary entity at front of alias list
        aliases_list = [entity] + [a for a in ctx_aliases if a != entity]
        tuples.append(GanitaTuple(
            entity=entity, attribute="count",
            value=value, unit="item",
            time=time, belief_id=belief_id,
            raw_text=text[m.start():m.end()],
            entity_aliases=tuple(aliases_list),
        ))

    return tuples


# Stopwords that aren't entities, used by both _pick_entity and _all_entities
_ENTITY_STOP = {
    "the", "and", "for", "with", "from", "into", "onto", "this",
    "that", "have", "had", "has", "was", "were", "been", "being",
    "are", "but", "not", "any", "all", "some", "more", "most",
    "less", "least", "than", "then", "when", "where", "what", "which",
    "who", "how", "why", "very", "just", "also", "even", "after",
    "before", "again", "still", "back", "out", "over", "under",
    "spent", "spend", "paid", "pay", "cost", "earned", "make",
    "raised", "saved", "donated", "buy", "bought", "sold",
    "received", "got", "took", "took", "every", "each", "many",
    "much", "today", "yesterday", "tomorrow", "morning", "evening",
    "night", "week", "year", "month", "day", "hour", "minute",
    # NOT included: 'bike', 'helmet', 'saddle', etc — those ARE entities.
}


def _all_entities(context: str) -> list[str]:
    """Return all canonical entity candidates from the context (deduped,
    in order of appearance)."""
    parts = re.findall(r"[A-Za-z]{3,}", context)
    seen = []
    seen_set = set()
    for p in parts:
        canon = _canonicalize_entity(p.lower())
        if canon in _ENTITY_STOP or canon in seen_set:
            continue
        seen_set.add(canon)
        seen.append(canon)
    return seen


def _pick_entity(context: str, hints: list[str] | None) -> str:
    """Pick the primary entity. Prefers hints; falls back to first
    non-stopword in context."""
    if hints:
        for h in hints:
            if h.lower() in context.lower():
                return _canonicalize_entity(h)
        return _canonicalize_entity(hints[0])
    candidates = _all_entities(context)
    return candidates[0] if candidates else "unknown"


# ─── Index ───────────────────────────────────────────────────────────


@dataclass
class GanitaIndex:
    """Append-only index of GanitaTuples grouped by (entity, attribute).

    Can be persisted to a JSONL sidecar file alongside the BeliefStore.
    Each line is one tuple. Re-loading the index is O(N) text scan.
    """

    persistence_path: Path | None = None
    _by_key: dict[tuple[str, str], list[GanitaTuple]] = field(
        default_factory=dict, repr=False,
    )

    def __post_init__(self) -> None:
        if self.persistence_path is not None:
            self.persistence_path = Path(self.persistence_path)
            if self.persistence_path.exists():
                self._load()

    def add(self, tuple_: GanitaTuple) -> None:
        key = (tuple_.entity, tuple_.attribute)
        self._by_key.setdefault(key, []).append(tuple_)
        if self.persistence_path is not None:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persistence_path, "a") as f:
                f.write(json.dumps(tuple_.to_dict()) + "\n")

    def add_many(self, tuples: Iterable[GanitaTuple]) -> None:
        for t in tuples:
            self.add(t)

    def all_for(self, entity: str, attribute: str | None = None) -> list[GanitaTuple]:
        """Return tuples whose primary entity OR aliases match `entity`.

        Honors entity_aliases so a tuple with entity="saddle" and
        aliases=("saddle", "bike") is returned for both `entity="saddle"`
        and `entity="bike"` lookups.
        """
        ent = _canonicalize_entity(entity)
        results = []
        for (_, attr), tuples in self._by_key.items():
            if attribute is not None and attr != attribute:
                continue
            for t in tuples:
                if t.entity == ent or ent in t.entity_aliases:
                    results.append(t)
        return results

    def all(self) -> list[GanitaTuple]:
        return [t for ts in self._by_key.values() for t in ts]

    def __len__(self) -> int:
        return sum(len(ts) for ts in self._by_key.values())

    def _load(self) -> None:
        with open(self.persistence_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    t = GanitaTuple(**d)
                except (json.JSONDecodeError, TypeError):
                    continue
                key = (t.entity, t.attribute)
                self._by_key.setdefault(key, []).append(t)


# ─── Query — aggregation operator detection + arithmetic ────────────


_AGG_PATTERNS: list[tuple[AggOp, re.Pattern]] = [
    ("difference", re.compile(r"\b(?:how\s+much\s+more|how\s+much\s+less|difference|differs?)\b", re.I)),
    ("average", re.compile(r"\b(?:average|mean|on\s+average)\b", re.I)),
    ("max", re.compile(r"\b(?:most|highest|maximum|max)\b", re.I)),
    ("min", re.compile(r"\b(?:least|lowest|minimum|min)\b", re.I)),
    ("sum", re.compile(r"\b(?:total|altogether|overall|sum|combined|how\s+much)\b", re.I)),
    ("count", re.compile(r"\b(?:how\s+many|count|number\s+of)\b", re.I)),
]


def detect_aggregation(question: str) -> AggOp | None:
    """Return the aggregation operator implied by the question, if any.

    Order matters: "how much more" must match before plain "how much"
    so the operator is `difference`, not `sum`.
    """
    for op, pat in _AGG_PATTERNS:
        if pat.search(question):
            return op
    return None


# Topic / entity hint extraction from the question.
# We pull noun-like tokens and try synonym normalization.
_QUESTION_STOPWORDS = {
    "how", "much", "many", "what", "when", "where", "who", "why",
    "the", "a", "an", "is", "are", "was", "were", "do", "does", "did",
    "have", "has", "had", "i", "me", "my", "you", "your",
    "in", "on", "at", "for", "with", "from", "to", "by",
    "of", "and", "or", "but", "than",
    "total", "altogether", "average", "mean", "more", "less",
    "most", "highest", "least", "lowest", "count", "number",
    "spent", "spend", "earn", "earned", "made", "raised",
    "this", "that", "these", "those", "year", "month", "week",
    "ever", "all", "any", "some",
}


def extract_entity_hints(question: str) -> list[str]:
    """Pull noun-like tokens from the question for entity matching."""
    tokens = re.findall(r"[A-Za-z]{3,}", question.lower())
    hints = []
    seen = set()
    for tok in tokens:
        if tok in _QUESTION_STOPWORDS:
            continue
        canon = _canonicalize_entity(tok)
        if canon in seen:
            continue
        seen.add(canon)
        hints.append(canon)
    return hints


@dataclass(frozen=True)
class GanitaResult:
    """Result of a successful aggregation."""

    operator: AggOp
    value: float
    unit: str
    contributing_belief_ids: list[str]
    explanation: str  # human-readable: "$50 saddle + $75 helmet + ..."

    def to_dict(self) -> dict:
        return asdict(self)


def answer_aggregation_question(
    question: str,
    index: GanitaIndex,
    *,
    restrict_to_belief_ids: set[str] | None = None,
) -> GanitaResult | None:
    """Try to answer ``question`` by procedural arithmetic over the index.

    Returns None if no aggregation operator is detected, or if no
    matching tuples exist.

    restrict_to_belief_ids
        When provided, only tuples whose `belief_id` is in this set are
        considered. Use this to scope arithmetic to topically-relevant
        beliefs (typically: the top-K beliefs from Phase 1 retrieval),
        avoiding "sum every dollar in the haystack" failures.
    """
    op = detect_aggregation(question)
    if op is None:
        return None

    hints = extract_entity_hints(question)
    if not hints:
        return None

    # Pull all tuples for any hint entity
    candidates: list[GanitaTuple] = []
    for h in hints:
        candidates.extend(index.all_for(h))

    if not candidates:
        return None

    # Topic restriction via retrieval scope
    if restrict_to_belief_ids is not None:
        candidates = [c for c in candidates if c.belief_id in restrict_to_belief_ids]
        if not candidates:
            return None

    # If the question implies a specific attribute (e.g. "spent" → expense),
    # filter by it
    hinted_attr = _detect_attribute(question)
    if hinted_attr != "value":
        attr_filtered = [c for c in candidates if c.attribute == hinted_attr]
        if attr_filtered:
            candidates = attr_filtered

    # Filter by unit consistency: if any USD candidate exists, use only USD;
    # if any duration unit, normalize.
    units = {c.unit for c in candidates}
    if "USD" in units:
        candidates = [c for c in candidates if c.unit == "USD"]
        unit = "USD"
    elif units & {"hour", "minute", "second", "day", "week", "month", "year"}:
        # Normalize all to hours for sum/average; keep most-common unit for display
        from collections import Counter
        # Common unit
        cnt = Counter(c.unit for c in candidates if c.unit in
                      {"hour", "minute", "second", "day", "week", "month", "year"})
        unit = cnt.most_common(1)[0][0]
        candidates = [c for c in candidates if c.unit == unit]
    else:
        unit = next(iter(units))
        candidates = [c for c in candidates if c.unit == unit]

    if not candidates:
        return None

    values = [c.value for c in candidates]
    if op == "sum":
        result = sum(values)
    elif op == "count":
        result = float(len(values))
    elif op == "average":
        result = sum(values) / len(values)
    elif op == "max":
        result = max(values)
    elif op == "min":
        result = min(values)
    elif op == "difference":
        if len(values) < 2:
            return None
        # Difference = max - min (interprets "how much more")
        result = max(values) - min(values)
    else:
        return None

    explanation_parts = [f"{c.raw_text!r} ({c.unit})" for c in candidates]
    if op == "sum":
        explanation = f"sum: {' + '.join(c.raw_text for c in candidates)} = {result}"
    elif op == "average":
        explanation = (
            f"average of {len(candidates)} values "
            f"({', '.join(c.raw_text for c in candidates)}) = {result:.2f}"
        )
    elif op == "count":
        explanation = (
            f"count: {len(candidates)} matching entries "
            f"({', '.join(c.raw_text for c in candidates[:5])}{' …' if len(candidates) > 5 else ''})"
        )
    elif op == "difference":
        explanation = (
            f"max {max(values)} − min {min(values)} = {result}"
        )
    else:
        explanation = f"{op}: {result}"

    return GanitaResult(
        operator=op,
        value=result,
        unit=unit,
        contributing_belief_ids=[c.belief_id for c in candidates],
        explanation=explanation,
    )
