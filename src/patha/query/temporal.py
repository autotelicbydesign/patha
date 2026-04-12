"""Temporal expression extraction and soft-penalty builder.

Extracts date/time expressions from query and passage text to support:
1. Temporal-anchored view (v7) enrichment at ingest time.
2. Soft temporal penalties at retrieval time — candidates outside the
   query's time window get a score penalty, but are never hard-filtered
   (to protect abstention recall).

Uses ``dateparser`` for robust multi-format date parsing.

Usage::

    extractor = TemporalExtractor()
    dates = extractor.extract("I started working there in January 2026.")
    # => [datetime(2026, 1, 1, ...)]

    # Build a soft penalty function for retrieval:
    penalty = build_temporal_penalty(query_dates, candidate_dates)
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Sequence

import dateparser


# Common temporal signal patterns that dateparser might miss or
# over-interpret (e.g., "today", "yesterday" are relative and ambiguous
# in a memory retrieval context)
_RELATIVE_PATTERNS = re.compile(
    r"\b(before|after|when|while|during|until|since|earlier|later|"
    r"previously|recently|formerly|originally|initially)\b",
    re.IGNORECASE,
)


class TemporalExtractor:
    """Extract date expressions from text.

    Parameters
    ----------
    prefer_dates_from
        ``dateparser`` setting: ``"past"`` or ``"future"``. Default
        ``"past"`` since memory retrieval is about past events.
    """

    def __init__(self, prefer_dates_from: str = "past") -> None:
        self._settings = {
            "PREFER_DATES_FROM": prefer_dates_from,
            "RETURN_AS_TIMEZONE_AWARE": False,
        }

    def extract(self, text: str) -> list[datetime]:
        """Extract parseable dates from text.

        Returns a list of datetime objects, sorted chronologically.
        Deduplicates by date (ignoring time).
        """
        # Try to parse the whole text as a date (for simple cases like "2026-01-15")
        result = dateparser.parse(text, settings=self._settings)
        if result is not None:
            return [result]

        # For longer text, try to find date-like substrings
        dates: list[datetime] = []
        seen_dates: set[str] = set()

        # Try common date patterns
        patterns = [
            r"\b\d{4}-\d{2}-\d{2}\b",  # ISO dates
            r"\b(?:January|February|March|April|May|June|July|August|"
            r"September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
            r"\s+\d{1,2},?\s+\d{4}\b",
            r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",  # US date format
            r"\b(?:in|on|around|before|after|since)\s+"
            r"(?:January|February|March|April|May|June|July|August|"
            r"September|October|November|December)\s+\d{4}\b",
        ]

        for pat in patterns:
            for match in re.finditer(pat, text, re.IGNORECASE):
                parsed = dateparser.parse(match.group(), settings=self._settings)
                if parsed is not None:
                    date_key = parsed.strftime("%Y-%m-%d")
                    if date_key not in seen_dates:
                        seen_dates.add(date_key)
                        dates.append(parsed)

        dates.sort()
        return dates

    def has_temporal_signal(self, text: str) -> bool:
        """Check if text contains temporal reasoning signals.

        Returns True if the text contains words like "before", "after",
        "when", etc. that suggest temporal reasoning is needed.
        """
        return bool(_RELATIVE_PATTERNS.search(text))


def build_temporal_penalty(
    query_dates: Sequence[datetime],
    candidate_date: datetime | str | None,
    penalty: float = 2.0,
) -> float:
    """Compute a soft temporal penalty for a candidate.

    Returns 0.0 (no penalty) if the candidate is within the query's
    time window, or ``-penalty`` if it's outside. Never returns a hard
    filter — always a soft score adjustment.

    Parameters
    ----------
    query_dates
        Dates extracted from the query. Defines the time window.
    candidate_date
        Date associated with the candidate (session date).
    penalty
        Score penalty for out-of-window candidates. Default 2.0.

    Returns
    -------
    float
        0.0 or -penalty.
    """
    if not query_dates or candidate_date is None:
        return 0.0

    if isinstance(candidate_date, str):
        parsed = dateparser.parse(candidate_date)
        if parsed is None:
            return 0.0
        candidate_date = parsed

    # Simple heuristic: penalty if candidate is after all query dates
    # (query asks about the past, candidate is from after)
    latest_query = max(query_dates)
    if candidate_date > latest_query:
        return -penalty

    return 0.0
