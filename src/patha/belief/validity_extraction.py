"""Validity window extraction from proposition text.

Phase 2 v0.1 scope (D4 decision): explicit markers only. Given a
proposition, try to identify an explicit temporal scope — "until June",
"for three weeks", "through Friday" — and return a Validity object.
If no marker is recognised, return None; the caller should fall back to
the default (permanent) or assign a decay validity.

Deferred to v0.2:
  - LLM-based inference of implicit durations ("training for a marathon")
  - HeidelTime/SUTime integration for richer temporal expressions
  - Complex compositional forms ("from X until Y if Z")

The module is deliberately small and rule-based. The grammar is a tiny
set of patterns we can enumerate, test, and debug. When it doesn't
fire, that's a feature — v0.1 prefers silence to a wrong answer.

Note on the `dateparser` dependency: it is already required by the
project (see pyproject.toml). It handles a wide range of date formats
including relative ones ("next week", "3 months from now"), which we
need for realistic conversational propositions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable

from patha.belief.types import Validity


# ─── Pattern library ─────────────────────────────────────────────────

@dataclass
class _Pattern:
    """One compiled pattern with a handler that builds a Validity."""

    regex: re.Pattern
    handler: Callable[[re.Match, datetime], Validity | None]
    name: str  # for debugging + tests


# ─── Handlers (defined before _build_patterns references them) ──────

def _handle_until(match: re.Match, start: datetime) -> Validity | None:
    # Lazy import to keep module import cheap
    import dateparser

    date_text = match.group(1).strip()
    # Strip trailing connector words dateparser sometimes struggles with
    date_text = re.sub(r"\s+(?:i|he|she|they|we).*$", "", date_text)
    end = dateparser.parse(
        date_text,
        settings={"RELATIVE_BASE": start, "PREFER_DATES_FROM": "future"},
    )
    if end is None:
        return None
    if end <= start:
        return None  # "until yesterday" doesn't make sense as a validity
    return Validity(
        mode="dated_range",
        start=start,
        end=end,
        source="explicit",
    )


_NUMBER_WORDS: dict[str, int] = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def _handle_for_duration(match: re.Match, start: datetime) -> Validity | None:
    n_raw = match.group(1).lower()
    unit = match.group(2).lower().rstrip("s")  # normalise to singular
    return _build_duration_validity(n_raw, unit, start)


def _handle_next_duration(match: re.Match, start: datetime) -> Validity | None:
    # For "the next <unit>" without a number, treat as 1.
    n_raw = match.group(1)
    if n_raw is None or n_raw.strip() == "":
        n_raw = "1"
    n_raw = n_raw.lower()
    unit = match.group(2).lower().rstrip("s")
    return _build_duration_validity(n_raw, unit, start)


def _build_duration_validity(
    n_raw: str, unit: str, start: datetime
) -> Validity | None:
    try:
        n = int(n_raw)
    except ValueError:
        n = _NUMBER_WORDS.get(n_raw)
        if n is None:
            return None

    days_per_unit = {
        "day": 1,
        "week": 7,
        "month": 30,   # approximate — good enough for v0.1
        "year": 365,
    }
    days = n * days_per_unit[unit]
    return Validity(
        mode="dated_range",
        start=start,
        end=start + timedelta(days=days),
        source="explicit",
    )


# ─── Pattern registry (populated after handlers are defined) ────────

def _build_patterns() -> list[_Pattern]:
    patterns: list[_Pattern] = []

    # "until <date>" / "through <date>" / "till <date>"
    patterns.append(
        _Pattern(
            regex=re.compile(
                r"\b(?:until|through|till|til)\s+([A-Za-z0-9,\s\-/]+?)"
                r"(?=[\.\!\?,;]|$)",
                re.IGNORECASE,
            ),
            handler=_handle_until,
            name="until_X",
        )
    )

    # "for <N> <unit>" where unit in {day, days, week, weeks, month, months, year, years}
    patterns.append(
        _Pattern(
            regex=re.compile(
                r"\bfor\s+(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
                r"(day|days|week|weeks|month|months|year|years)\b",
                re.IGNORECASE,
            ),
            handler=_handle_for_duration,
            name="for_N_units",
        )
    )

    # "the next <N> <unit>" — e.g., "for the next two weeks" or
    # "for the next month" (N implicit = 1)
    patterns.append(
        _Pattern(
            regex=re.compile(
                r"\b(?:for\s+)?the\s+next\s+"
                r"(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)?"
                r"\s*"
                r"(day|days|week|weeks|month|months|year|years)\b",
                re.IGNORECASE,
            ),
            handler=_handle_next_duration,
            name="next_N_units",
        )
    )

    return patterns


_PATTERNS = _build_patterns()


# ─── Public API ──────────────────────────────────────────────────────

def extract_validity(
    proposition: str,
    *,
    asserted_at: datetime,
) -> Validity | None:
    """Try to extract an explicit validity window from a proposition.

    Parameters
    ----------
    proposition
        The text to scan.
    asserted_at
        The reference time for resolving relative expressions like
        "until next week" or "for three months". Required — validity
        windows are anchored to assertion time.

    Returns
    -------
    Validity | None
        A Validity with mode="dated_range" and source="explicit" if a
        marker fired, otherwise None. Callers should treat None as
        "fall back to default permanent validity".
    """
    for pattern in _PATTERNS:
        match = pattern.regex.search(proposition)
        if match is None:
            continue
        validity = pattern.handler(match, asserted_at)
        if validity is not None:
            return validity
    return None


def list_supported_patterns() -> list[str]:
    """Expose the pattern names for docs, debugging, and tests."""
    return [p.name for p in _PATTERNS]
