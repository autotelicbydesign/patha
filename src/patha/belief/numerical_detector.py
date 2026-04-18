"""Numerical-change contradiction detector.

NLI misses contradictions where the surface structure is the same but
one numeric value differs from another: 'rent 1500' vs 'rent 1800',
'commute 45 minutes' vs 'commute 15 minutes', 'closes at 8pm' vs
'closes at 9pm'. These are extremely common in real memory
(pricing, durations, counts, ages, quantities, times).

This detector is a protocol-conforming wrapper. For each pair, it:
  1. Extracts numeric tokens with their surrounding subject (via
     simple noun+number adjacency).
  2. If the two propositions share a subject phrase but carry
     different numbers, returns CONTRADICTS with high confidence.
  3. Otherwise delegates to the inner detector.

Combine with AdhyasaAwareDetector for a full v0.6-shaped stack:
    NumericalAwareDetector(
        inner=AdhyasaAwareDetector(
            inner=NLIContradictionDetector()
        )
    )
"""

from __future__ import annotations

import re

from patha.belief.contradiction import ContradictionDetector
from patha.belief.types import ContradictionLabel, ContradictionResult


# Patterns that extract (subject, number) pairs from text.
# Conservative: only fires on patterns where a number clearly
# belongs to a subject.
_NUM_SUBJECT_PATTERNS: list[re.Pattern] = [
    # "rent 1500", "charges 1500 for rent", "price is 29"
    re.compile(
        r"\b(?P<subj>rent|price|salary|income|cost|fee|rate)\s*(?:is\s+|went\s+up\s+to\s+|charges?\s+)?(?P<num>\d+(?:\.\d+)?)\b",
        re.IGNORECASE,
    ),
    # "commute is 45 minutes", "runs for 30 minutes", "routine is 30 minutes"
    re.compile(
        r"\b(?P<subj>commute|routine|workout|session|meeting|class)\s+(?:is\s+|lasts\s+|runs\s+)?(?P<num>\d+)\s*(?:minute|min|hour|hr|second|sec)",
        re.IGNORECASE,
    ),
    # "weigh 85 kilograms" / "weigh 85 kg"
    re.compile(
        r"\b(?P<subj>weigh|height|weight)\s+(?P<num>\d+(?:\.\d+)?)\s*(?:kilograms?|kg|pounds?|lbs|cm|m)\b",
        re.IGNORECASE,
    ),
    # "X years old", "turned four", "daughter is three years old"
    re.compile(
        r"\b(?P<subj>daughter|son|child|partner|parent)\s+(?:is\s+|just\s+turned\s+)?(?P<num>\d+|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen)\b",
        re.IGNORECASE,
    ),
    # "store closes at 8pm" / "they close at 9pm" / "starts at 6am"
    re.compile(
        r"\b(?P<subj>close|closes|open|opens|start|starts|end|ends|begin|begins)\s+at\s+(?P<num>\d{1,2})\s*(?:am|pm|AM|PM)?",
        re.IGNORECASE,
    ),
    # "I speak three languages" / "I speak 3 languages"
    re.compile(
        r"\b(?:speak|own|have)\s+(?P<num>\d+|two|three|four|five|six|seven|eight|nine|ten)\s+(?P<subj>languages?|cars?|pets?|dogs?|cats?|bedrooms?|children|siblings?)",
        re.IGNORECASE,
    ),
    # "N bedrooms" directly
    re.compile(
        r"\b(?P<num>\d+|two|three|four|five|six|seven|eight|nine|ten)\s+(?P<subj>bedrooms?|bathrooms?|engineers?|users?|subscribers?|minutes?|hours?)\b",
        re.IGNORECASE,
    ),
    # Percentage: "30 percent stocks" / "3.5 percent"
    re.compile(
        r"\b(?P<num>\d+(?:\.\d+)?)\s*(?:percent|%)\s*(?P<subj>stocks?|bonds?|mortgage|rate|interest)?",
        re.IGNORECASE,
    ),
    # "N-minute routine" / "60 minutes" in cardio context
    re.compile(
        r"\b(?P<num>\d+)[-\s]?(?P<subj>minute|hour|second|week|day|month|year)",
        re.IGNORECASE,
    ),
    # "charges N" / "N for rent" → subject = rent
    re.compile(
        r"\bcharges?\s+(?P<num>\d+(?:\.\d+)?)\s+for\s+(?P<subj>rent|room|board|parking)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?P<subj>rent|fee|charge)\s+(?:went\s+up|increased|raised|is\s+now)\s+(?:to\s+)?(?P<num>\d+(?:\.\d+)?)",
        re.IGNORECASE,
    ),
]


# Word-number canonicalisation
_WORD_TO_INT: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15,
}


def _canonical_num(raw: str) -> str:
    """Normalise number representation so '3' and 'three' compare equal."""
    lower = raw.lower().strip()
    if lower in _WORD_TO_INT:
        return str(_WORD_TO_INT[lower])
    try:
        f = float(lower)
        if f.is_integer():
            return str(int(f))
        return str(f)
    except ValueError:
        return lower


def _normalise_subject(subj: str) -> str:
    """Crude lemmatisation — strip trailing 's' (not 'es') so 'closes'
    and 'close', 'minutes' and 'minute', 'cars' and 'car' match.

    Keep it simple: just strip the final 's' when the word is long
    enough and doesn't end in 'ss'. 'closes' → 'close', 'cars' → 'car'.
    Leaves 'close' unchanged (no trailing s).
    """
    s = subj.lower().strip()
    if len(s) > 3 and s.endswith("s") and not s.endswith("ss"):
        return s[:-1]
    return s


def _extract_subject_number_pairs(text: str) -> list[tuple[str, str]]:
    """Return list of (normalised_subject, canonical_number) extracted from text."""
    pairs: list[tuple[str, str]] = []
    for pat in _NUM_SUBJECT_PATTERNS:
        for m in pat.finditer(text):
            subj_raw = m.group("subj") or ""
            subj = _normalise_subject(subj_raw)
            num = _canonical_num(m.group("num"))
            if subj and num:
                pairs.append((subj, num))
    return pairs


def check_numerical_contradiction(p1: str, p2: str) -> bool:
    """Return True if p1 and p2 share a subject but disagree on its number."""
    pairs1 = _extract_subject_number_pairs(p1)
    pairs2 = _extract_subject_number_pairs(p2)
    if not pairs1 or not pairs2:
        return False
    nums_by_subject_1: dict[str, set[str]] = {}
    for subj, num in pairs1:
        nums_by_subject_1.setdefault(subj, set()).add(num)
    for subj, num in pairs2:
        if subj in nums_by_subject_1 and num not in nums_by_subject_1[subj]:
            return True
    return False


# ─── Shared-subject value detector ──────────────────────────────────
# Catches 'my email is X' vs 'my new email is Y' (or 'my laptop is X'
# vs 'my laptop is Y'). Different from numerical: the values are
# text strings, not numbers, but the subject is shared.

_VALUE_PROPERTIES: frozenset[str] = frozenset({
    "email", "phone", "address", "number",
    "laptop", "computer", "phone number",
    "company", "employer", "job", "role", "title",
    "boss", "manager",
    "landlord", "partner", "spouse",
    "doctor", "dentist", "therapist", "accountant",
})


_VALUE_PATTERN = re.compile(
    # "my [new] <property> is <value>" — capture property and value.
    # Value is everything up to end of sentence / period / comma.
    r"\bmy\s+(?:new\s+|old\s+|current\s+|former\s+|previous\s+)?"
    r"(?P<prop>\w+(?:\s+\w+)?)\s+is\s+"
    r"(?P<val>[^\.\,;!?]+?)(?=[\.\,;!?]|$)",
    re.IGNORECASE,
)


def _extract_subject_value_pairs(text: str) -> list[tuple[str, str]]:
    """Extract (property, value) pairs from 'my X is Y' constructions.

    Only returns properties from _VALUE_PROPERTIES so we don't match
    every 'my X is Y' (e.g., 'my dog is sweet' shouldn't count).
    """
    pairs: list[tuple[str, str]] = []
    for m in _VALUE_PATTERN.finditer(text):
        prop = m.group("prop").lower().strip()
        val = m.group("val").lower().strip()
        # Only count if the property is in our canonical list
        prop_singular = prop.rstrip("s") if len(prop) > 3 else prop
        if prop in _VALUE_PROPERTIES or prop_singular in _VALUE_PROPERTIES:
            pairs.append((prop_singular, val))
    return pairs


def check_value_replacement(p1: str, p2: str) -> bool:
    """Return True if p1 and p2 share a 'my X is Y' property but assign
    different values to X. Catches email / phone / landlord / boss
    replacements that NLI misses."""
    pairs1 = _extract_subject_value_pairs(p1)
    pairs2 = _extract_subject_value_pairs(p2)
    if not pairs1 or not pairs2:
        return False
    vals_by_prop_1: dict[str, set[str]] = {}
    for prop, val in pairs1:
        vals_by_prop_1.setdefault(prop, set()).add(val)
    for prop, val in pairs2:
        if prop in vals_by_prop_1 and val not in vals_by_prop_1[prop]:
            return True
    return False


class NumericalAwareDetector:
    """Delegates to an inner detector, but overrides with CONTRADICTS
    when a numerical-change heuristic fires.

    When both propositions reference the same subject phrase with
    different numbers, we return CONTRADICTS at confidence 0.9 —
    high enough to trigger supersession in BeliefLayer at default
    threshold 0.7. Delegation otherwise.
    """

    def __init__(self, inner: ContradictionDetector) -> None:
        self._inner = inner
        self.numerical_overrides = 0
        self.inner_calls = 0

    def detect(self, p1: str, p2: str) -> ContradictionResult:
        return self.detect_batch([(p1, p2)])[0]

    def detect_batch(
        self, pairs: list[tuple[str, str]]
    ) -> list[ContradictionResult]:
        if not pairs:
            return []

        # Route: numerical-change or shared-value-replacement pairs
        # short-circuit; others go to inner.
        numerical_verdicts: dict[int, ContradictionResult] = {}
        inner_pairs: list[tuple[int, tuple[str, str]]] = []
        for idx, (p1, p2) in enumerate(pairs):
            if check_numerical_contradiction(p1, p2):
                numerical_verdicts[idx] = ContradictionResult(
                    label=ContradictionLabel.CONTRADICTS,
                    confidence=0.9,
                    rationale="numerical: shared subject, differing numbers",
                )
                self.numerical_overrides += 1
            elif check_value_replacement(p1, p2):
                numerical_verdicts[idx] = ContradictionResult(
                    label=ContradictionLabel.CONTRADICTS,
                    confidence=0.9,
                    rationale="value-replacement: shared property, differing values",
                )
                self.numerical_overrides += 1
            else:
                inner_pairs.append((idx, (p1, p2)))

        # Batch the remaining pairs through the inner detector
        if inner_pairs:
            batch = [pair for _, pair in inner_pairs]
            self.inner_calls += len(batch)
            inner_results = self._inner.detect_batch(batch)
            inner_results_by_idx = {
                inner_pairs[i][0]: inner_results[i]
                for i in range(len(inner_pairs))
            }
        else:
            inner_results_by_idx = {}

        # Reassemble in original order
        return [
            numerical_verdicts.get(i, inner_results_by_idx.get(i))
            for i in range(len(pairs))
        ]
