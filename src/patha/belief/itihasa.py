"""Itihāsa — narrative-synthesis intent detection and result types.

The third question class.

Patha already distinguishes two question classes and routes them
through different recall paths:

  - RETRIEVAL  ("what did I say about the saddle?")  → pratyakṣa
        Phase 1 top-K → belief filter → summary. Point lookup.
  - SYNTHESIS  ("how much have I spent on bikes total?") → anumāna
        gaṇita exhaustive tuple arithmetic, zero LLM tokens at recall.
        Collapses many facts into one scalar.

This module adds the detector and result types for the THIRD class:

  - NARRATIVE  ("how has my thinking on agency evolved this year?")
        A temporally-ordered thematic walk. Preserves and *orders*
        many facts into a sequence rather than collapsing them.

Top-K is the wrong primitive here for the same reason it's wrong for
synthesis: a narrative question has no single "right" session — it
wants the ordered sequence of a theme's mentions across time, with the
supersession structure made visible. The songline graph already encodes
the edges (shared entity / time / topic) that connect a theme's mentions
across sessions; the walker (``retrieval/narrative_walk.py``) traverses
them and emits the path. This module is the intent gate + the typed
result those two halves share — exactly the role ``ganita.py`` plays
for the synthesis path.

On the name: *itihāsa* (इतिहास, "thus indeed it was") is the Sanskrit
genre of narrative history — the temporally-ordered emplotment of
remembered events around a theme (the Mahābhārata, the Rāmāyaṇa). That
is precisely what the walker produces: ordered beats + a through-line.
Strictly, itihāsa is a *genre*, not one of the six Nyāya pramāṇa; the
narrative path's epistemic authority is **śabda** (the user's own
recorded word over time), *organized as* itihāsa. So: pramāṇa = śabda,
form = itihāsa. (This is the rare case where the Sanskrit fits more
precisely than the English — "narrative" undersells the
ordered-arc-around-a-theme connotation itihāsa carries natively.)

Distinct from ``include_history``: history returns a single belief's
supersession lineage (A superseded by B superseded by C). Narrative
walks across *related but not necessarily superseding* beliefs — the
agency-belief from March and the one from September may both be
``current`` with no supersedes edge between them; only the shared
entity/topic songline edges connect them. Narrative subsumes history
by also rendering supersession status inside the walk.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Literal

from patha.belief.ganita import extract_entity_hints

NarrativeOp = Literal["evolution", "origin", "throughline"]


# ─── Intent detection ───────────────────────────────────────────────

# Order matters: the more specific "origin" forms are tried before the
# broad "evolution" forms so "when did I first start thinking about X"
# routes to `origin`, not `evolution`. Mirrors ganita's _AGG_PATTERNS
# ordering discipline.
_NARRATIVE_PATTERNS: list[tuple[NarrativeOp, re.Pattern]] = [
    ("origin", re.compile(
        r"\bwhen\s+did\s+i\s+(?:first|start|begin)\b"
        r"|\b(?:the\s+)?first\s+time\s+i\b"
        r"|\bhow\s+did\s+i\s+(?:get\s+into|come\s+to|start|begin|first)\b"
        r"|\bwhat\s+(?:got|made|led)\s+me\s+(?:into|to)\b"
        r"|\bwhere\s+did\s+(?:my|this)\b.{0,40}?\bcome\s+from\b",
        re.I)),
    ("evolution", re.compile(
        r"\bhow\s+(?:has|have|did|'?s)\b.{0,40}?"
        r"(?:evolv|chang|shift|develop|progress|grow|grew|matur|"
        r"come\s+around)"
        r"|\bevolv(?:ing|ed|ution)\s+"
        r"(?:view|thinking|stance|position|opinion|thoughts?|take)"
        r"|\bmy\s+evolving\b"
        r"|\bhow\s+(?:my|i)\b.{0,30}?(?:changed|evolved)\s+over\b",
        re.I)),
    ("throughline", re.compile(
        r"\bthrough[\s-]?line\b"
        r"|\btrace\s+(?:my|the)\s+"
        r"(?:thinking|thoughts|reasoning|views?|journey|arc)\b"
        r"|\bwhat'?s\s+the\s+"
        r"(?:arc|story|trajectory|narrative|thread|throughline)\b"
        r"|\bpatterns?\s+in\s+how\s+i\s+"
        r"(?:think|write|feel|approach|talk|reason)\b"
        r"|\bthe\s+thread\s+(?:running\s+)?through\b"
        r"|\bhow\s+do\s+my\b.{0,40}?\b(?:connect|relate|fit\s+together)\b",
        re.I)),
]


def detect_narrative(question: str) -> NarrativeOp | None:
    """Return the narrative operator implied by the question, if any.

    Returns one of "origin" | "evolution" | "throughline", or None when
    the question is not a narrative query.

    Precedence (handled by routing, not here): the synthesis-intent
    gate (gaṇita) runs FIRST and wins only if it actually finds matching
    tuples — so "how has my spending changed" goes to gaṇita iff there
    are spend tuples, else falls through to narrative. This detector is
    only consulted on that fall-through, the lowest-risk insertion.
    """
    for op, pat in _NARRATIVE_PATTERNS:
        if pat.search(question):
            return op
    return None


# Narrative-specific stopwords on top of gaṇita's, so theme extraction
# doesn't surface the narrative scaffolding words ("evolved", "thinking",
# "trace", ...) as if they were the theme.
_NARRATIVE_STOPWORDS = {
    "evolved", "evolve", "evolving", "evolution", "changed", "change",
    "changing", "shifted", "shift", "develop", "developed", "progress",
    "progressed", "grow", "grown", "grew", "matured", "trace", "tracing",
    "thinking", "thoughts", "thought", "view", "views", "opinion",
    "opinions", "stance", "position", "arc", "story", "trajectory",
    "narrative", "thread", "throughline", "through", "line", "pattern",
    "patterns", "first", "begin", "began", "begun", "start", "started",
    "come", "came", "around", "over", "last", "past", "recent",
    "recently", "lately", "currently", "now", "today", "feel", "feeling",
    "approach", "reason", "reasoning", "journey", "connect", "relate",
    "fit", "together", "got", "made", "led", "into", "time", "long",
    "about",
    # Mental-verb gerunds/inflections — scaffolding, never the theme.
    # (Found via dogfooding: "when did I first start doubting the
    # benchmark numbers?" resolved theme='doubting' instead of
    # 'benchmark'.)
    "doubt", "doubting", "doubted", "wonder", "wondering", "wondered",
    "questioning", "questioned", "realize", "realizing", "realized",
    "realise", "realising", "realised", "notice", "noticing", "noticed",
    "believe", "believing", "believed", "belief", "beliefs",
    "mention", "mentioning", "mentioned", "consider", "considering",
    "considered", "understand", "understanding", "understood",
}


def extract_theme(question: str) -> str | None:
    """Resolve the theme of a narrative question (the topical noun).

    "how has my thinking on agency evolved?" → "agency"
    "trace my views on the productivity stack" → "productivity" (first
        content token; theme resolution is heuristic, inheriting
        gaṇita's known entity-resolution limits).

    Returns the primary theme token (canonicalized) or None if the
    question carries no content noun beyond scaffolding.
    """
    hints = [
        h for h in extract_entity_hints(question)
        if h not in _NARRATIVE_STOPWORDS
    ]
    return hints[0] if hints else None


def extract_themes(question: str, *, limit: int = 3) -> list[str]:
    """All candidate theme tokens, in question order (primary first).

    The walker may try secondary themes if the primary resolves to too
    few beliefs — multi-word themes ("productivity stack") otherwise
    lose the discriminating second token.
    """
    hints = [
        h for h in extract_entity_hints(question)
        if h not in _NARRATIVE_STOPWORDS
    ]
    return hints[:limit]


# ─── Result types ───────────────────────────────────────────────────

SupersessionStatus = Literal["current", "superseded", "revised-from", "origin"]


@dataclass(frozen=True)
class NarrativeBeat:
    """One point on the narrative timeline.

    A beat is a single belief surfaced by the walk, tagged with its
    place in the theme's evolution. Beats are ordered by ``asserted_at``
    in the containing NarrativeResult.
    """

    belief_id: str
    proposition: str
    asserted_at: datetime | None
    supersession_status: SupersessionStatus
    superseded_by: list[str] = field(default_factory=list)
    channels_to_prev: list[str] = field(default_factory=list)
    walk_score: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["asserted_at"] = (
            self.asserted_at.isoformat() if self.asserted_at else None
        )
        return d


@dataclass(frozen=True)
class NarrativeResult:
    """Result of a successful narrative walk.

    Parallels GanitaResult: a typed payload the recall path surfaces on
    ``Recall.narrative``. Where GanitaResult carries one scalar, this
    carries an ordered sequence + a deterministic through-line.
    """

    operator: NarrativeOp
    theme: str
    beats: list[NarrativeBeat]  # temporally ordered (earliest first)
    through_line: str  # one-line deterministic synthesis, no LLM
    contributing_belief_ids: list[str]
    anchors: list[str] = field(default_factory=list)

    @property
    def beat_count(self) -> int:
        return len(self.beats)

    def to_dict(self) -> dict:
        return {
            "operator": self.operator,
            "theme": self.theme,
            "through_line": self.through_line,
            "beats": [b.to_dict() for b in self.beats],
            "contributing_belief_ids": list(self.contributing_belief_ids),
            "anchors": list(self.anchors),
        }

    def as_timeline(self) -> str:
        """Render the beats as a prompt-ready timeline string.

        This is the ``Recall.summary`` payload for the narrative path —
        an ordered, supersession-tagged list a downstream LLM can
        prose-ify. The synthesis work (selection, ordering,
        supersession-tagging) is already done here; the LLM only
        verbalizes a pre-structured timeline instead of reconstructing
        it from an unordered top-K bag.
        """
        lines = [f"Theme: {self.theme} — {self.through_line}", ""]
        for b in self.beats:
            date = b.asserted_at.strftime("%Y-%m-%d") if b.asserted_at else "-"
            if b.supersession_status == "revised-from":
                marker = "~ (later revised)"
            elif b.supersession_status == "origin":
                marker = "● (origin)"
            elif b.supersession_status == "superseded":
                marker = "~ (superseded)"
            else:
                marker = "→ (current)"
            lines.append(f"- [{date}] {marker} {b.proposition}")
        return "\n".join(lines)


# ─── Deterministic through-line rendering (no LLM) ──────────────────


def render_through_line(
    op: NarrativeOp, theme: str, beats: list[NarrativeBeat]
) -> str:
    """Produce a one-line synthesis from the beats' STRUCTURE.

    Deterministic and template-based — it reads the beats' dates and
    supersession tags, never their semantics. The actual narrative prose
    is left to a downstream LLM (fed the ordered beats); this line is
    the headline a human or model gets for free, at zero LLM cost.
    """
    if not beats:
        return f"No recorded reflections on {theme}."

    first = beats[0]
    last = beats[-1]
    first_date = (
        first.asserted_at.strftime("%Y-%m-%d") if first.asserted_at else "an earlier date"
    )
    last_date = (
        last.asserted_at.strftime("%Y-%m-%d") if last.asserted_at else "now"
    )
    n = len(beats)
    revised = [b for b in beats if b.supersession_status in ("revised-from", "superseded")]

    if op == "origin":
        snippet = first.proposition.strip()
        if len(snippet) > 90:
            snippet = snippet[:87] + "..."
        tail = (
            f" {n - 1} related reflection{'s' if n - 1 != 1 else ''} since."
            if n > 1 else ""
        )
        return f'First engaged with {theme} on {first_date}: "{snippet}".{tail}'

    if op == "evolution":
        if revised:
            return (
                f"View on {theme} shifted over {n} reflections "
                f"({first_date} → {last_date}): {len(revised)} revision"
                f"{'s' if len(revised) != 1 else ''} recorded."
            )
        return (
            f"{theme}: {n} reflections from {first_date} to {last_date}, "
            f"no recorded reversals — a continuous line."
        )

    # throughline
    return (
        f"Thread on {theme} spans {first_date} → {last_date} "
        f"across {n} reflection{'s' if n != 1 else ''}."
    )


__all__ = [
    "NarrativeOp",
    "detect_narrative",
    "extract_theme",
    "extract_themes",
    "NarrativeBeat",
    "NarrativeResult",
    "SupersessionStatus",
    "render_through_line",
]
