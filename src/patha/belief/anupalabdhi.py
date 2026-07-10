"""Anupalabdhi — absence queries, the fifth question class.

The pramāṇa where NON-perception is itself the evidence: "have I ever
been to Japan?", "have we decided on a venue yet?", "do I still go to
the Italian class?", "am I a homeowner?". No top-K retrieval can answer
these — surfacing similar beliefs is not the same as certifying that no
belief settles the question.

The epistemics are the design (docs/roadmap.md §4): *anupalabdhi asserts
absence only after qualified search.* Concretely, `answer_absence`
scans the ENTIRE current belief set (the same exhaustiveness guarantee
gaṇita makes — never a retrieval-scoped subset), classifies candidate
mentions with the four-fold abhāva taxonomy, applies the question's
temporal scope, and only then returns an "absent" verdict — carrying
the search receipt (how many beliefs were scanned) and the nearest
*present* beliefs as contrast, so the absence claim is auditable.

Temporal scoping by kind:
  - "ever …"        → ATYANTABHAVA: any positive mention at any time
                       settles it as present.
  - "… yet?"        → PRAGABHAVA: the latest relevant assertion decides;
                       a settling positive → present.
  - "still …?"      → PRADHVAMSABHAVA: the latest relevant assertion
                       decides; a destruction-negation ("dropped the
                       class") → absent even though older positives
                       exist.
  - "am I a …?"     → ANYONYABHAVA: identity — present iff a current
                       non-negated belief co-mentions subject and role.

Deviation from the roadmap sketch, documented: no `.abhava.jsonl`
sidecar index in v1 — `classify_abhava` runs inline during the scan
(with an in-memory cache). Semantics are identical (the exhaustive scan
IS the guarantee); the sidecar is a performance artifact that becomes
worthwhile at store sizes where the scan hurts. Revisit then.

Instrument: eval/absence_eval.py (AbsenceEval). The catastrophic
failure it exists to prevent: a false "you never…" when a positive
belief exists (false_absence). When in doubt, this module prefers
"present with weak evidence" over "absent".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from patha.belief.abhava import AbhavaKind, classify_abhava
from patha.belief.ganita import _canonicalize_entity


# ─── Question detection ─────────────────────────────────────────────


@dataclass(frozen=True)
class AbsenceQuestion:
    """A recognised absence question, scoped and located."""

    scope: str                # "ever" | "yet" | "still" | "identity" | "open"
    kind: AbhavaKind          # the taxonomy kind the scope implies
    locus: str | None         # the entity whose absence/presence is asked
    subject: str | None = None  # identity questions: "is Maya my …" → "maya"
    role: str | None = None     # identity questions: the role/category word
    cue: str = ""             # matched phrase, for debugging


_SCOPE_KIND = {
    "ever": AbhavaKind.ATYANTABHAVA,
    "yet": AbhavaKind.PRAGABHAVA,
    "still": AbhavaKind.PRADHVAMSABHAVA,
    "identity": AbhavaKind.ANYONYABHAVA,
    "open": AbhavaKind.PRAGABHAVA,  # "what have I not decided…" = not-yet
}

# Aggregation frames own these phrasings even when absence cues co-occur:
# "have I ever spent more than 100…" is a synthesis question; "how much
# of the loan haven't I paid off yet?" is arithmetic. Declining here is
# what keeps the route from stealing gaṇita's questions (RouterEval's
# boundary cases rt-bnd-01/-13 and AbsenceEval's routing controls).
_AGGREGATION_GUARDS = [
    re.compile(r"\bhow\s+(?:much|many)\b", re.IGNORECASE),
    re.compile(r"\b(?:more|less|fewer)\s+than\s+\$?\d", re.IGNORECASE),
    re.compile(r"\b(?:over|under|at\s+least)\s+\$?\d", re.IGNORECASE),
    re.compile(r"\bin\s+total\b", re.IGNORECASE),
]

# (scope, pattern) — ordered, most specific first. Each pattern's last
# group captures the object phrase the locus is extracted from.
_QUESTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("ever", re.compile(
        r"\b(?:have|did|was|has)\s+(?:i|we)\s+ever\s+(.+?)\??$",
        re.IGNORECASE)),
    ("yet", re.compile(
        r"\b(?:have|haven'?t|has|hasn'?t|did)\s+(?:i|we)\s+(.+?)\s+yet\b",
        re.IGNORECASE)),
    ("still", re.compile(
        r"\b(?:do|does|am|is|are)\s+(?:i|we)\s+still\s+(.+?)\??$",
        re.IGNORECASE)),
    ("open", re.compile(
        r"\bwhat\s+have(?:n'?t)?\s+(?:i|we)\s+(?:not\s+)?(.+?)\??$",
        re.IGNORECASE)),
    ("open", re.compile(
        r"\b(?:is\s+there\s+)?anything\s+(?:i|we)\s+(?:still\s+)?haven'?t\s+(.+?)\??$",
        re.IGNORECASE)),
    # enumerative never-absence: "what topics have I never brought up
    # with my manager?", "what did I never get around to reading?",
    # "is there any part of the business plan I never finished?"
    ("ever", re.compile(
        r"\b(?:what|which|any\s+part\s+of|anything)\b.{0,40}?"
        r"\b(?:have|has|did)\s+(?:i|we)\s+never\s+(.+?)\??$",
        re.IGNORECASE)),
    ("ever", re.compile(
        r"\bany\s+part\s+of\s+(.+?)\s+(?:i|we)\s+never\b",
        re.IGNORECASE)),
    # enumerative still-absence: "what's still on the list I haven't
    # bought for the nursery?"
    ("open", re.compile(
        r"\bwhat'?s\s+still\b.{0,40}?\b(?:i|we)\s+haven'?t\s+(.+?)\??$",
        re.IGNORECASE)),
    # existence question: "have I spent anything on the bike?" —
    # absence of any instance (secondary synthesis; the aggregation
    # guards above still win on explicit totals/thresholds)
    ("ever", re.compile(
        r"\bhave\s+(?:i|we)\s+\w+\s+anything\s+(?:on|about|for|with)\s+(.+?)\??$",
        re.IGNORECASE)),
    ("identity", re.compile(
        r"\bam\s+i\s+(?:a|an|the)\s+(.+?)\??$", re.IGNORECASE)),
    ("identity", re.compile(
        r"\bis\s+([A-Z]\w+)\s+my\s+(.+?)\??$", re.IGNORECASE)),
]

# "what have I …" only reads as absence with a not/haven't in it —
# "what have I said about X" is retrieval.
_OPEN_REQUIRES_NEGATION = re.compile(
    r"\b(?:not|haven'?t|hasn'?t)\b", re.IGNORECASE
)


def detect_absence_question(question: str) -> AbsenceQuestion | None:
    """Recognise an absence question, or None (→ other routes).

    Conservative by construction: aggregation frames decline, "what
    have I…" needs an explicit negation, and no bare-'never' cue is
    used ("I can never remember — what's my physio's name?" is
    retrieval)."""
    q = question.strip()
    for guard in _AGGREGATION_GUARDS:
        if guard.search(q):
            return None
    for scope, pat in _QUESTION_PATTERNS:
        m = pat.search(q)
        if m is None:
            continue
        if scope == "open" and not _OPEN_REQUIRES_NEGATION.search(q):
            continue
        if scope == "identity" and len(m.groups()) == 2:
            subject, role_phrase = m.group(1), m.group(2)
            return AbsenceQuestion(
                scope=scope, kind=_SCOPE_KIND[scope],
                locus=subject.lower(),
                subject=subject.lower(),
                role=_extract_locus(role_phrase),
                cue=m.group(0),
            )
        obj = m.group(m.lastindex or 1)
        locus = _extract_locus(obj)
        if not locus:
            return None
        role = None
        if scope == "identity":
            # "am I a homeowner" — the category IS the locus
            role = locus
        return AbsenceQuestion(
            scope=scope, kind=_SCOPE_KIND[scope], locus=locus,
            subject=None, role=role, cue=m.group(0),
        )
    return None


# ─── Locus extraction ───────────────────────────────────────────────

_FRAME_VERBS = (
    "been to|gone to|tried|done|ridden|visited|eaten|owned|had|taken|"
    "used|played|watched|read|met|seen|heard|lived in|lived|worked at|"
    "worked|decided on|decided about|decided|picked|chosen|settled on|"
    "submitted|sent|booked|started|begun|finished|signed|bought|joined|"
    "go to|get|take|have|attend|member of"
)
_FRAME_RE = re.compile(rf"^(?:{_FRAME_VERBS})\s+", re.IGNORECASE)
_ARTICLES = {"a", "an", "the", "my", "our", "his", "her", "their", "some"}
_QUALIFIER_ADJS = {"certified", "official", "proper", "new", "old"}
# single nouns too generic to stand as a locus alone; qualified by a
# following "for the X" when present ("a name for the puppy" → puppy name)
_GENERIC_HEADS = {"name", "one", "date"}


def _extract_locus(phrase: str) -> str | None:
    """Head noun-phrase of the question's object, in the conventions
    the gold sets use: last two content tokens of the NP ("basketball
    season ticket" → "season ticket"), of/for-qualifiers dropped
    ("builder for the kitchen" → "builder") unless the head is generic
    ("name for the puppy" → "puppy name"). Plural harmonisation is the
    scorer's job (canonicalize_locus strips a trailing s)."""
    p = phrase.strip().strip("?.!,")
    p = _FRAME_RE.sub("", p).strip()
    # "team lead of the platform group" / "builder for the kitchen"
    head, qualifier = p, None
    m = re.split(r"\s+(?:of|for|to|at|in|with|on)\s+", p, maxsplit=1)
    if len(m) == 2:
        head, qualifier = m[0], m[1]
    toks = [t for t in re.findall(r"[A-Za-z][\w'-]*", head)
            if t.lower() not in _ARTICLES]
    toks = [t for t in toks if t.lower() not in _QUALIFIER_ADJS] or toks
    if not toks:
        return None
    if len(toks) == 1 and toks[0].lower() in _GENERIC_HEADS and qualifier:
        q_toks = [t for t in re.findall(r"[A-Za-z][\w'-]*", qualifier)
                  if t.lower() not in _ARTICLES]
        if q_toks:
            return f"{q_toks[-1]} {toks[0]}".lower()
    return " ".join(toks[-2:]).lower()


# ─── Qualified search + verdict ─────────────────────────────────────


@dataclass
class AbsenceResult:
    """Outcome of the qualified search. `verdict` is "present" or
    "absent"; evidence/contrast are belief ids so callers can render or
    map them; `searched_n` is the exhaustiveness receipt."""

    verdict: str
    kind: AbhavaKind
    locus: str
    scope: str
    evidence_ids: list[str] = field(default_factory=list)
    contrast_ids: list[str] = field(default_factory=list)
    searched_n: int = 0

    def render(self) -> str:
        if self.verdict == "present":
            return (
                f"Present: a belief settles '{self.locus}' "
                f"({len(self.evidence_ids)} evidence belief(s); "
                f"{self.searched_n} current beliefs searched)."
            )
        return (
            f"Absent ({self.kind.value}): no current belief settles "
            f"'{self.locus}' — verified by exhaustive scan of "
            f"{self.searched_n} current beliefs. Nearest present beliefs "
            f"cited for contrast: {len(self.contrast_ids)}."
        )


_SUFFIXES = ("ing", "ed", "er", "es", "s")


def _stem(token: str) -> str:
    """Iterated suffix stripping to a fixpoint, so both sides of a
    comparison converge on the same root ('builders'→'builder'→'build'
    and 'builder'→'build' — a single pass would leave them unequal)."""
    t = token.lower()
    changed = True
    while changed:
        changed = False
        for suf in _SUFFIXES:
            if len(t) > 3 + len(suf) - 1 and t.endswith(suf):
                t = t[: -len(suf)]
                changed = True
                break
    return t


def _token_set(text: str) -> set[str]:
    return {_stem(t) for t in re.findall(r"[A-Za-z][\w'-]*", text)
            if len(t) >= 3}


def _mentions(text: str, locus: str) -> bool:
    """Every content token of the locus appears (stemmed) in the text —
    'first aider' matches 'first aid certificate', 'violin lesson'
    matches 'violin lessons on Thursdays'."""
    locus_toks = {_stem(t) for t in re.findall(r"[A-Za-z][\w'-]*", locus)}
    if not locus_toks:
        return False
    return locus_toks <= _token_set(text)


def _negates_locus(proposition: str, locus: str) -> bool:
    """Does this proposition itself assert the ABSENCE of the locus
    state (rather than its presence)? classify_abhava + a check that
    the negation is about the locus and not an unrelated clause."""
    inf = classify_abhava(proposition)
    if inf.kind in (AbhavaKind.NONE,):
        return False
    if inf.referenced_state and _mentions(inf.referenced_state, locus):
        return True
    # negation present and the locus sits in the proposition: treat the
    # mention as non-positive (conservative — prevents "I never did a
    # triathlon" from counting as presence evidence).
    return inf.kind != AbhavaKind.NONE


def _stems_of(text: str) -> set[str]:
    return {_stem(t) for t in re.findall(r"[A-Za-z][\w'-]*", text)}


def _stem_set(words: str) -> frozenset[str]:
    """Marker sets are stored PRE-STEMMED with the same stemmer the text
    side uses, so both converge on the same roots ('wondering'→'wond'
    must meet 'wonder'→'wond', not a hand-written 'wonder')."""
    return frozenset(_stem(w) for w in words.split())


# "have we decided on X yet?" — a mention of X is not a decision.
# Settling evidence for the not-yet scope needs a commitment verb.
_COMMIT_STEMS = _stem_set(
    "decided chose chosen picked settled signed submitted sent confirmed "
    "paid hired ordered booked applied registered"
)

# "have I ever ridden a motorcycle?" — a booked taster session is a
# plan, not an actuality. Irrealis/future markers veto 'ever' evidence.
_IRREALIS_STEMS = _stem_set(
    "booked scheduling planning will upcoming whether wondering might "
    "hoping someday considering eyeing"
)

# Presupposition triggers project through irrealis operators: "been
# wondering whether I'd move abroad AGAIN" presupposes a prior stint
# abroad even though the wondering itself is irrealis. For the 'ever'
# scope such a mention IS actuality evidence.
_PRESUP_STEMS = _stem_set("again back returned")

# third-person assignment verbs: "<Name> was ANNOUNCED as team lead"
_ASSIGNMENT_STEMS = _stem_set(
    "was is became announced named promoted appointed made elected"
)


def _third_person_owns(text: str, role: str) -> bool:
    """Identity guard: 'Dario was announced as team lead' must not make
    'am I the team lead?' present. True when the tokens just before the
    role mention contain BOTH a capitalized non-'I' name and an
    assignment verb — the conjunction is what distinguishes 'Dario was
    announced as team lead' from 'renewed my film society membership'
    (sentence-initial capitals alone are not names)."""
    role_head = role.split()[0] if role else ""
    if not role_head:
        return False
    toks = re.findall(r"[A-Za-z][\w'-]*", text)
    stems = [_stem(t) for t in toks]
    target = _stem(role_head)
    for i, s in enumerate(stems):
        if s != target:
            continue
        window_toks = toks[max(0, i - 5): i]
        window_stems = stems[max(0, i - 5): i]
        has_name = any(
            t[0].isupper() and t.lower() != "i" for t in window_toks
        )
        has_assignment = any(s in _ASSIGNMENT_STEMS for s in window_stems)
        if has_name and has_assignment:
            return True
    return False


def answer_absence(
    q: AbsenceQuestion, *, store, similarity_fn=None,
) -> AbsenceResult:
    """The qualified search. Scans ALL current beliefs (exhaustive — the
    gaṇita guarantee), splits locus mentions into positive evidence vs
    absence assertions, applies the temporal scope with its evidence
    bar (actuality for 'ever', commitment for 'yet', latest-assertion
    for 'still'), and prefers "present" whenever a defensible settling
    belief exists (a false "you never…" is the catastrophic failure; a
    weak "present" is merely unhelpful).

    `similarity_fn(question, texts) -> list[float]`, when provided,
    ranks semantic neighbours for the contrast set (the "nearest
    present beliefs" a purely lexical scan can't find — Korea trips as
    contrast for a Japan absence). It is never used to flip a verdict.
    """
    current = store.current()
    positives: list = []
    negations: list = []
    for b in current:
        text = b.proposition
        if q.scope == "identity" and q.subject:
            hit = _mentions(text, q.subject)
        else:
            hit = _mentions(text, q.locus or "")
        if not hit:
            continue
        if _negates_locus(text, q.locus or ""):
            negations.append(b)
        else:
            positives.append(b)

    verdict = "absent"
    evidence: list = []

    if q.scope == "identity":
        role = q.role or q.locus or ""
        for b in positives:
            if _mentions(b.proposition, role) and \
                    not _third_person_owns(b.proposition, role):
                verdict, evidence = "present", [b]
                break
    elif q.scope == "ever":
        actual = [
            b for b in positives
            if not (_stems_of(b.proposition) & _IRREALIS_STEMS)
            or (_stems_of(b.proposition) & _PRESUP_STEMS)
        ]
        if actual:
            verdict, evidence = "present", actual[:1]  # minimal evidence
    elif q.scope == "yet" or q.scope == "open":
        settling = [b for b in positives
                    if _stems_of(b.proposition) & _COMMIT_STEMS]
        if settling:
            latest = max(settling, key=lambda b: b.asserted_at)
            verdict, evidence = "present", [latest]
    else:  # still — the LATEST relevant assertion decides
        relevant = sorted(
            [(b, False) for b in positives] + [(b, True) for b in negations],
            key=lambda pair: pair[0].asserted_at,
        )
        if relevant:
            latest, is_negation = relevant[-1]
            if not is_negation:
                verdict, evidence = "present", [latest]

    # Cited beliefs. Present: the settling evidence. Absent: the locus
    # LINEAGE (every belief that touches the locus, either polarity —
    # "rented the unit" + "cancelled the unit" together constitute a
    # destroyed-state absence), topped up with semantic neighbours when
    # the lineage is thin (a Japan absence cites the Korea trips).
    if verdict == "absent":
        lineage = sorted(
            positives + negations, key=lambda b: b.asserted_at, reverse=True,
        )[:3]
        if lineage:
            cited = list(lineage)
        elif similarity_fn is not None:
            pool = [b for b in current
                    if not _negates_locus(b.proposition, q.locus or "")]
            sims = similarity_fn(q.cue, [b.proposition for b in pool]) \
                if pool else []
            ranked = sorted(zip(sims, pool), key=lambda x: -x[0])
            # floor keeps junk neighbours out of the citation — an
            # absence claim contrasted with the landlord's stairwell
            # repaint is worse than one with no contrast at all
            cited = [b for s, b in ranked[:3] if s >= 0.3]
        else:
            cited = []
        evidence = negations[:2]
        contrast_ids = [b.id for b in cited]
    else:
        contrast_ids = [b.id for b in evidence]

    return AbsenceResult(
        verdict=verdict,
        kind=q.kind,
        locus=q.locus or "",
        scope=q.scope,
        evidence_ids=[b.id for b in evidence],
        contrast_ids=contrast_ids,
        searched_n=len(current),
    )
