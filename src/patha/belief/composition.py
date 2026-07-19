"""Composition — time-series-of-sums, the first chained pramāṇa.

"How has my spending on the bike evolved?" wants neither one scalar
(gaṇita) nor prose beats (narrative): it wants narrative SHAPE over
synthesis CONTENT — per-period buckets of procedural arithmetic plus a
deterministic trend statement. Both parents contribute their guarantee:
gaṇita's exhaustive arithmetic over preserved tuples inside each
bucket, the narrative path's temporal ordering across buckets. Zero
LLM tokens, same as both parents.

Design (docs/roadmap.md §3): `GanitaTuple.time` is populated at ingest
and was used by nothing until this module. Candidate selection mirrors
`answer_aggregation_question`'s steps (entity hints → index.all_for →
attribute filter) so the two paths can never disagree about which
facts exist; this module only adds the period grouping.

Rules (encoded to match CompositionEval's frozen v1 gold rules —
eval/composition_data/README.md):
- Gap rule: a period exists iff ≥1 contributing tuple exists in it.
  Missing months are ABSENT, never zero-filled — a fabricated zero has
  no belief-id receipt behind it.
- Currency rule: sum/avg buckets are USD-only; non-USD tuples are
  never converted or silently mixed.
- Count rule: bucket value = number of contributing tuples in the
  period.
- Trend rule: `classify_trend` below mirrors the eval's frozen v1
  function; tests/belief/test_composition.py asserts the two agree.
- Degradation: < 2 non-empty buckets → the caller falls through to
  plain gaṇita (same graceful-degradation contract as the narrative
  walk: composition never degrades an answer plain arithmetic could
  give).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from patha.belief.ganita import (
    _detect_attribute,
    detect_aggregation,
    extract_entity_hints,
)

_USD_UNITS = frozenset({"usd", "$", "dollar", "dollars"})

# ─── Intent detection ───────────────────────────────────────────────

_PERIOD_MARKERS = re.compile(
    r"\b(?:month\s+(?:to|by|over)\s+month|per\s+month|each\s+month|"
    r"monthly|over\s+the\s+months?|by\s+month|"
    r"year\s+(?:to|by|over)\s+year|per\s+year|each\s+year|yearly|"
    r"by\s+year|over\s+the\s+(?:past\s+)?years?|over\s+time|"
    r"quarter\s+(?:to|by|over)\s+quarter|per\s+quarter|"
    r"week\s+(?:to|by|over)\s+week|per\s+week|each\s+week|weekly|"
    r"across\s+the\s+(?:seasons|months|years))\b",
    re.IGNORECASE,
)
_YEAR_MARKERS = re.compile(
    r"\b(?:year\s+(?:to|by|over)\s+year|per\s+year|each\s+year|yearly|"
    r"by\s+year|over\s+the\s+(?:past\s+)?years)\b",
    re.IGNORECASE,
)
_EVOLUTION_MARKERS = re.compile(
    r"\b(?:evolved?|evolving|changed?|changing|trended|trends?|"
    r"progressed|progression|break\s+down)\b",
    re.IGNORECASE,
)
# "changed my MIND" is a revision idiom, not a quantity trend — it
# cancels the evolution marker (explicit period markers still qualify:
# "how many times did I change my mind per month" is compositional).
_MIND_IDIOM = re.compile(
    r"\bchanged?\s+(?:my|our)\s+mind\b", re.IGNORECASE,
)
# Tight by design: 'budgeting'/'money' would steal EvolutionEval's
# narrative themes ("how has my thinking about budgeting evolved?").
# Word boundaries matter: \bbudget\b does not match 'budgeting'.
_QUANTITY_SIGNALS = re.compile(
    r"\b(?:spending|spend|spent|costs?|donated|donations?|budget|"
    r"expenses?|savings?|saved?|mileage|buying|totals?|amount)\b",
    re.IGNORECASE,
)
# Explicit scalar frames stay with plain gaṇita even when an evolution
# word appears elsewhere.
_SCALAR_FRAMES = re.compile(
    r"\b(?:in\s+total|altogether|overall|all\s+time)\b", re.IGNORECASE,
)


@dataclass(frozen=True)
class CompositionIntent:
    op: str            # "sum" | "count" | "avg"
    granularity: str   # "month" | "year"


def detect_composition(question: str) -> CompositionIntent | None:
    """Composition = an aggregation signal AND a per-period/evolution
    marker, minus explicit scalar frames. Conservative: when this
    declines, the question falls to the plain gaṇita gate unchanged."""
    if _SCALAR_FRAMES.search(question):
        return None
    period = _PERIOD_MARKERS.search(question) is not None
    evolution = (
        _EVOLUTION_MARKERS.search(question) is not None
        and _MIND_IDIOM.search(question) is None
    )
    if not (period or evolution):
        return None
    agg = detect_aggregation(question)
    op = None
    if agg is not None:
        op = getattr(agg, "value", None) or str(agg)
        op = op.lower()
    if op not in ("sum", "count", "avg", "average"):
        op = "sum" if _QUANTITY_SIGNALS.search(question) else None
    if op == "average":
        op = "avg"
    if op is None:
        return None
    granularity = "year" if _YEAR_MARKERS.search(question) else "month"
    return CompositionIntent(op=op, granularity=granularity)


# ─── Frozen trend rule (mirrors eval/composition_eval.py v1) ────────


def classify_trend(values: list[float]) -> str | None:
    """Trend vocabulary over bucket values in period order. MIRRORS the
    frozen v1 rule in eval/composition_eval.py — the eval owns the
    frozen copy, this is the production twin, and a cross-test asserts
    they never drift. Order: spike, flat, rising, falling; None on <2
    buckets or an ambiguous series."""
    n = len(values)
    if n < 2:
        return None
    if n >= 3:
        peak = max(values)
        rest = sorted(values)[:-1]
        rest_max, rest_min = max(rest), min(rest)
        if (
            rest_max > 0
            and peak > 2.5 * rest_max
            and (rest_max - rest_min) <= 0.5 * rest_max
        ):
            return "spike"
    hi, lo = max(values), min(values)
    if hi > 0 and (hi - lo) <= 0.10 * hi:
        return "flat"
    nondecreasing = all(values[i + 1] >= values[i] for i in range(n - 1))
    nonincreasing = all(values[i + 1] <= values[i] for i in range(n - 1))
    if nondecreasing and values[-1] > values[0]:
        return "rising"
    if nonincreasing and values[-1] < values[0]:
        return "falling"
    return None


# ─── The primitive ──────────────────────────────────────────────────


@dataclass
class CompositionBucket:
    period: str                # "2026-03" (month) or "2026" (year)
    value: float
    unit: str
    contributing_belief_ids: list[str] = field(default_factory=list)


@dataclass
class CompositionResult:
    op: str
    granularity: str
    entity_hints: list[str]
    buckets: list[CompositionBucket]
    trend: str | None
    unbucketed: int            # matching tuples without a usable time

    def render(self) -> str:
        """Deterministic through-line — same zero-LLM contract as the
        narrative renderer."""
        parts = [
            f"{b.period}: {b.value:g} {b.unit}"
            f" ({len(b.contributing_belief_ids)} fact(s))"
            for b in self.buckets
        ]
        line = (
            f"Computed via gaṇita arithmetic per {self.granularity} "
            f"(no LLM): {self.op} series — " + "; ".join(parts) + "."
        )
        if self.trend:
            line += f" Trend: {self.trend}."
        if self.unbucketed:
            line += (
                f" ({self.unbucketed} matching fact(s) carried no date "
                f"and are excluded — receipts only.)"
            )
        return line


def _period_of(iso_time: str, granularity: str) -> str | None:
    m = re.match(r"(\d{4})(?:-(\d{2}))?", iso_time or "")
    if m is None:
        return None
    if granularity == "year":
        return m.group(1)
    if m.group(2) is None:
        return None
    return f"{m.group(1)}-{m.group(2)}"


_HINT_STOP = frozenset({
    "spending", "spend", "spent", "changed", "changing", "evolved",
    "over", "month", "months", "year", "years", "many", "much", "each",
    "per", "time", "times",
})


def _select_candidates(question: str, intent: CompositionIntent, index):
    """Mirror answer_aggregation_question's selection: entity hints →
    index.all_for → attribute filter → (sum/avg) USD-only."""
    hints = extract_entity_hints(question)
    if not hints:
        return [], hints
    candidates = []
    seen: set[int] = set()
    for h in hints:
        for c in index.all_for(h):
            if id(c) in seen:
                continue
            seen.add(id(c))
            candidates.append(c)
    hinted_attr = _detect_attribute(question)
    if candidates and hinted_attr != "value":
        filtered = [c for c in candidates if c.attribute == hinted_attr]
        if filtered:
            candidates = filtered
    if intent.op in ("sum", "avg"):
        candidates = [
            c for c in candidates
            if (c.unit or "").lower() in _USD_UNITS
        ]
    return candidates, hints


def _count_events_from_store(question: str, hints: list[str], store):
    """Count-op fallback: countable events ("went to a yoga class",
    "sent a job application") carry no number, so the extractor emits
    no tuple. Counting EVENTS is counting ASSERTIONS: match current
    beliefs whose proposition mentions a non-generic hint, timestamped
    by asserted_at. Receipts are belief ids — same contract."""
    if store is None:
        return []
    content_hints = [h for h in hints if h.lower() not in _HINT_STOP]
    if not content_hints:
        return []
    matched = []
    for b in store.current():
        text = b.proposition.lower()
        if any(re.search(rf"\b{re.escape(h.lower())}", text)
               for h in content_hints):
            matched.append(b)
    return matched


def compose_series(
    question: str,
    intent: CompositionIntent,
    index,
    store=None,
) -> CompositionResult | None:
    """Group the question's matching tuples by period and run the op
    per bucket. Returns None when fewer than 2 non-empty buckets exist
    (degradation contract — the caller falls back to a scalar over the
    same candidates, see compose_scalar).

    Candidate selection mirrors answer_aggregation_question: entity
    hints → index.all_for → attribute filter. The currency rule then
    keeps sum/avg buckets USD-only. For op=count with no numeric
    tuples, events are counted as assertions from the store."""
    candidates, hints = _select_candidates(question, intent, index)

    buckets: dict[str, list] = {}
    unbucketed = 0
    count_by_beliefs = False
    if intent.op == "count" and not candidates:
        count_by_beliefs = True
        for b in _count_events_from_store(question, hints, store):
            period = _period_of(
                b.asserted_at.isoformat() if b.asserted_at else "",
                intent.granularity,
            )
            if period is None:
                unbucketed += 1
                continue
            buckets.setdefault(period, []).append(b)
    else:
        if not candidates:
            return None
        for c in candidates:
            period = _period_of(
                getattr(c, "time", None) or "", intent.granularity,
            )
            if period is None:
                unbucketed += 1
                continue
            buckets.setdefault(period, []).append(c)
    if len(buckets) < 2:
        return None
    if count_by_beliefs:
        out = [
            CompositionBucket(
                period=p, value=float(len(bs)), unit="item",
                contributing_belief_ids=sorted({b.id for b in bs}),
            )
            for p, bs in sorted(buckets.items())
        ]
        return CompositionResult(
            op=intent.op, granularity=intent.granularity,
            entity_hints=list(hints), buckets=out,
            trend=classify_trend([b.value for b in out]),
            unbucketed=unbucketed,
        )

    out: list[CompositionBucket] = []
    for period in sorted(buckets):
        cs = buckets[period]
        if intent.op == "count":
            value, unit = float(len(cs)), "item"
        elif intent.op == "avg":
            value = sum(float(c.value) for c in cs) / len(cs)
            unit = "USD"
        else:
            value, unit = sum(float(c.value) for c in cs), "USD"
        out.append(CompositionBucket(
            period=period,
            value=value,
            unit=unit,
            contributing_belief_ids=sorted({c.belief_id for c in cs}),
        ))
    return CompositionResult(
        op=intent.op,
        granularity=intent.granularity,
        entity_hints=list(hints),
        buckets=out,
        trend=classify_trend([b.value for b in out]),
        unbucketed=unbucketed,
    )


def compose_scalar(question: str, intent: CompositionIntent, index):
    """Degradation fallback: the composition phrasing fired but fewer
    than 2 period buckets exist — a single month of data IS a scalar
    question, so answer it as plain gaṇita would (same candidates,
    one number). Returns a GanitaResult or None.

    Needed because the plain gaṇita gate's detect_aggregation never
    fires on "how has my spending evolved?" phrasings — without this,
    degradation would fall through to narrative/retrieval and lose the
    arithmetic answer entirely."""
    from patha.belief.ganita import GanitaResult

    candidates, _hints = _select_candidates(question, intent, index)
    if not candidates:
        return None
    values = [float(c.value) for c in candidates]
    if intent.op == "count":
        value, unit = float(len(candidates)), "item"
    elif intent.op == "avg":
        value, unit = sum(values) / len(values), "USD"
    else:
        value, unit = sum(values), "USD"
    # AggOp is a typing.Literal — the operator IS the string
    return GanitaResult(
        operator=intent.op,
        value=value,
        unit=unit,
        contributing_belief_ids=sorted({c.belief_id for c in candidates}),
        explanation=(
            f"composition degraded to scalar: {len(candidates)} matching "
            f"fact(s) span <2 {intent.granularity} buckets"
        ),
    )
