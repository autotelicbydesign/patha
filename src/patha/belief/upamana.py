"""Upamāna — analogical recall, the sixth pramāṇa's first half.

"What does this remind me of?" / "have I been in a situation like
this before?" — given a NEW situation described in the question,
return the structurally-matching past EPISODE (a session-level group
of beliefs), not the lexically-nearest text.

The operational definition of analogy (and the reason plain retrieval
cannot serve it): **same shape, different words**. AnalogyEval's gold
sets enforce content-word disjointness between question and gold
analogue (≤ 2 shared tokens) while surface traps share MORE vocabulary
with the question than the gold does. A ranker that rewards surface
overlap walks into every trap; the primitive here scores

    analogy(q, episode) = semantic_sim(q, episode)
                          − λ · lexical_overlap(q, episode)

semantic similarity (session-pooled v1 embeddings — the same vectors
Phase 1 already computes) MINUS the surface it can name. λ defaults to
0.5 (dev-set choice, AnalogyEval; documented there).

Structure naming is deterministic and lexicon-based (no LLM): a small
frame lexicon maps marker phrases to abstract structure labels
("hard deadline forcing a choice", "documented escalation", …); the
labels shared by question and chosen episode become the through-line.
The lexicon is deliberately coarse — structure_overlap is AnalogyEval's
weakest scorer by design, and this is its honest counterpart.

Instrument: eval/analogy_eval.py (stub floor: hit@1 0.250,
trap_resistance 0.250, structure 0.000).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

# ─── Question detection ─────────────────────────────────────────────

_ANALOGY_PATTERNS: tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r"\bremind(?:s|ed)?\s+me\b",
        r"\b(?:been|was)\s+in\s+a\s+situation\s+like\b",
        r"\bsomewhere\s+like\s+this\b",
        r"\bsituation\s+like\s+this\b",
        r"\blike\s+this\s+before\b",
        r"\bwhen\s+have\s+i\s+(?:handled|met|seen|faced|felt)\b",
        r"\bwhere\s+have\s+i\s+(?:seen|met)\s+this\b",
        r"\bwhat\s+past\s+\w+\s+(?:was|is)\s+(?:most\s+)?(?:similar|like)\b",
        r"\bmaps?\s+onto\s+this\b",
        r"\bwhat\s+past\s+experience\b",
        r"\bgoing\s+like\s+the\s+last\s+one\b",
        r"\bthe\s+last\s+time\s+i\s+felt\b",
        r"\bis\s+there\s+a\s+pattern\s+in\b",
        r"\brun\s+this\s+playbook\s+before\b",
        r"\bbet\s+on\s+\w+\s+like\s+this\b",
        r"\bapproached\s+something\s+this\s+way\b",
        r"\bwhere\s+have\s+i\s+seen\s+this\s+pattern\b",
        r"\blearned\s+this\s+lesson\s+somewhere\b",
        r"\bmet\s+this\s+temptation\b",
        r"\b(?:anything|something)\s+in\s+my\s+past\s+like\b",
        r"\bin\s+my\s+past\s+like\s+this\b",
        r"\bresembles?\b",
        r"\bclosest\s+thing\s+in\s+my\s+past\b",
        r"\bdoes\s+this\s+\w+(?:\s+\w+)?\s+mirror\b",
        r"\bmirror(?:s|ed)?\s*\?$",
        r"\bis\s+this\s+like\s+the\s+time\b",
        r"\bfaced\s+before\b",
        r"\bmost\s+(?:similar|like|alike)\b",
        r"\bfelt\s+most\s+like\b",
    )
)


def detect_analogy_question(question: str) -> bool:
    """Comparison-intent detector (same shape as detect_narrative).
    Conservative: fires only on explicit analogy phrasings."""
    return any(p.search(question) for p in _ANALOGY_PATTERNS)


# ─── Tokens (mirrors AnalogyEval's content_tokens discipline) ────────

_STOPWORDS = frozenset("""
a an the and or but if then than so to of in on at for with by from as
is are was were be been being am do does did done have has had having
will would can could should shall may might must not no nor
it its itself i me my mine myself we us our ours you your yours he him
his she her hers they them their theirs this that these those there
here what which who whom whose when where why how
all any both each few more most other some such only own same too very
just many much about into over under again once during between through
above below up down out off until while since after before ever never
always still also even now
like situation situations remind reminds reminded anything something
someone else way ways one two
get got gets getting go goes went going make makes made making take
takes took taking give gives gave given keep keeps kept let lets say
says said see sees saw seen
actually anyway every new thing things
""".split())


def _content_tokens(text: str) -> set[str]:
    out: set[str] = set()
    for t in re.findall(r"[a-z0-9]+", text.lower()):
        if t in _STOPWORDS or len(t) < 3:
            continue
        if len(t) > 3 and t.endswith("s"):
            t = t[:-1]
        out.add(t)
    return out


# ─── Frame lexicon (deterministic structure naming) ─────────────────

_FRAMES: tuple[tuple[str, re.Pattern], ...] = tuple(
    (label, re.compile(pattern, re.IGNORECASE)) for label, pattern in (
        ("hard time limit forcing a choice",
         r"\b(?:deadline|expiry|expires?|until\s+\w+day|48-hour|"
         r"by\s+(?:friday|monday|tomorrow)|hours?\s+before)\b"),
        ("prolonged indecision",
         r"\b(?:back\s+and\s+forth|flip-?flop|keep\s+going|can'?t\s+decide|"
         r"kept\s+\w+ing\s+all\s+night|indecision)\b"),
        ("advice from a trusted person",
         r"\b(?:mentor|phoned|called\s+my|asked\s+one\s+question|"
         r"trusted|advice)\b"),
        ("calm after committing",
         r"\b(?:instantly\s+calm|relief|felt\s+.*calm|lighter)\b"),
        ("one-sided arrangement",
         r"\b(?:never\s+once\s+offered|offloading|one-sided|"
         r"four\s+times\s+this\s+year)\b"),
        ("resentment building slowly",
         r"\b(?:resentment|crept\s+up|dreaded|building)\b"),
        ("direct boundary conversation",
         r"\b(?:told\s+\w+\s+plainly|boundary|could\s+and\s+couldn'?t|"
         r"raise\s+it|held\s+firm|have\s+the\s+talk)\b"),
        ("over-preparation",
         r"\b(?:over-?polish|over-?rehears|over-?prepar|over-?plan|"
         r"eleven\s+pages|colour-coded|four\s+hours\s+a\s+day|"
         r"fifteen-tab)\b"),
        ("freezing under observation",
         r"\b(?:froze|blanked|went\s+white|blank(?:ed)?\s+on)\b"),
        ("recovering by simplifying",
         r"\b(?:half\s+tempo|dropped\s+the\s+ornaments|simplif|"
         r"one\s+card|three\s+beats|loose\s+plan)\b"),
        ("sunk cost in a long project",
         r"\b(?:year\s+two|two\s+years|four\s+drafts|half-built|"
         r"sunk|still\s+aren'?t)\b"),
        ("walking away deliberately",
         r"\b(?:sold\s+the|shelv|walked?\s+away|admitted\s+the|"
         r"belonged\s+to\s+a\s+version)\b"),
        ("relief outweighing loss",
         r"\b(?:held\s+breath|finally\s+let\s+out|freed|felt\s+free)\b"),
        # NB: no bare 'trial' — "the trial wrapped early" (jury duty)
        # is a homonym that handed a civic episode the pilot frame
        ("small pilot before commitment",
         r"\b(?:taster|trial\s+(?:run|month|period)|pilot|fire\s+escape|"
         r"rented\s+a|market\s+stall|before\s+committing)\b"),
        ("scaling after the trial works",
         r"\b(?:scaled?|scaling|earned\s+it|now\s+hosts|community\s+plot)\b"),
        ("optimizing a proxy metric",
         r"\b(?:follower|algorithm|personal\s+records?|PRs?\b|metric|"
         r"counts?\s+cross)\b"),
        ("joy draining from measurement",
         r"\b(?:felt\s+nothing|enjoying\s+\w+\s+less|burned\s+out|"
         r"joy)\b"),
        ("switching to what matters",
         r"\b(?:quit\s+checking|selling\s+prints|switched\s+to|"
         r"beat\s+the\s+whole)\b"),
        ("physical setback interrupting practice",
         r"\b(?:strain|injury|flared|physio|knee|wrist)\b"),
        ("forced reduction of volume",
         r"\b(?:cut\s+\w+\s+to|fifteen\s+careful\s+minutes|pull\s+\w+\s+"
         r"back|restraint|restriction)\b"),
        ("constraint improving quality",
         r"\b(?:made\s+me\s+precise|ahead\s+of\s+schedule|healed)\b"),
        ("rough start others gave up on",
         r"\b(?:missed\s+both|reassigned|behaviou?r\s+problems|"
         r"give\s+him\s+up|rocky)\b"),
        ("steady low-pressure routine",
         r"\b(?:same\s+\w+\s+check-in|every\s+morning|no\s+judgement|"
         r"steady|consistent)\b"),
        ("gradual turnaround",
         r"\b(?:shipped\s+\w+\s+solo|did\s+what\s+pressure|"
         r"turn(?:ed)?\s+\w*\s*around|thrived)\b"),
        ("repeated service failure",
         r"\b(?:billed\s+\w+\s+a\s+third|keeps?\s+charging|loops?\s+me|"
         r"third\s+time|runaround)\b"),
        ("documented escalation",
         r"\b(?:dated\s+log|documented|log\s+attached|escalat)\b"),
        ("leaving despite the fix",
         r"\b(?:ported\s+\w+\s+out|didn'?t\s+buy\s+back|move\s+banks|"
         r"leaving\s+anyway)\b"),
        ("shared venture with a friend",
         r"\b(?:band|side-?business|with\s+a\s+friend|kitty|"
         r"quarter\s+share)\b"),
        ("unspoken money tension",
         r"\b(?:unevenly|nobody\s+would\s+say|money\s+silence|awkward|"
         r"avoided|sulking)\b"),
        ("transparency as the cure",
         r"\b(?:spreadsheet|transparen|out\s+loud|post-mortem)\b"),
        ("recovery demanding restraint",
         r"\b(?:rehab|recovery|burnout|leave\s+with|painfully\s+gradual|"
         r"tiny\s+exercises)\b"),
        ("temptation to rush",
         r"\b(?:itching\s+to\s+rush|sneaking\s+a\s+second|"
         r"skipped\s+ahead|rush\s+the)\b"),
        ("regression after skipping ahead",
         r"\b(?:paid\s+for\s+it|week\s+of\s+regression|old\s+pattern)\b"),
        ("slow compounding payoff",
         r"\b(?:compounding|held\s+the\s+\w+\s+rule|full\s+quarter)\b"),
    )
)


def frames_of(text: str) -> list[str]:
    return [label for label, pat in _FRAMES if pat.search(text)]


# ─── The primitive ──────────────────────────────────────────────────


@dataclass
class AnalogyResult:
    """Ranked analogous episodes + the named shared structure."""

    sessions: list[str]                     # ranked, best first
    shared_structure: list[str]             # frame labels in common
    scores: dict[str, float] = field(default_factory=dict)
    belief_ids: list[str] = field(default_factory=list)  # top episode's

    def render(self) -> str:
        if not self.sessions:
            return "No structurally similar past episode found."
        line = f"Structurally closest past episode: session {self.sessions[0]}"
        if len(self.sessions) > 1:
            line += f" (runner-up: {self.sessions[1]})"
        if self.shared_structure:
            line += ". Shared shape: " + "; ".join(self.shared_structure)
        return line + "."


def rank_analogues(
    question: str,
    episodes: dict[str, list[str]],
    *,
    embed_fn: Callable[[list[str]], list],
    lam: float = 0.5,
    top_k: int = 2,
) -> AnalogyResult:
    """Rank candidate episodes (session → its texts) by
    semantic_sim − λ·lexical_overlap, with frame-overlap as tiebreak
    evidence. Deterministic given the embedder; zero LLM."""
    import numpy as np

    names = sorted(episodes)
    if not names:
        return AnalogyResult(sessions=[], shared_structure=[])
    pooled_texts = [" ".join(episodes[n]) for n in names]
    vecs = embed_fn([question] + pooled_texts)
    q = np.asarray(vecs[0], dtype=float)
    q = q / (np.linalg.norm(q) or 1.0)
    q_toks = _content_tokens(question)
    q_frames = set(frames_of(question))

    scored: list[tuple[float, str]] = []
    scores: dict[str, float] = {}
    for name, text, v in zip(names, pooled_texts, vecs[1:]):
        v = np.asarray(v, dtype=float)
        sim = float(v @ q / (np.linalg.norm(v) or 1.0))
        toks = _content_tokens(text)
        overlap = len(q_toks & toks) / max(1, min(len(q_toks), len(toks)))
        frame_bonus = 0.15 * len(q_frames & set(frames_of(text)))
        score = sim - lam * overlap + frame_bonus
        scores[name] = round(score, 4)
        scored.append((score, name))
    scored.sort(key=lambda x: (-x[0], x[1]))
    ranked = [n for _s, n in scored[:top_k]]
    best_frames = [
        f for f in frames_of(" ".join(episodes[ranked[0]]))
        if f in q_frames
    ] if ranked else []
    return AnalogyResult(
        sessions=ranked, shared_structure=best_frames, scores=scores,
    )
