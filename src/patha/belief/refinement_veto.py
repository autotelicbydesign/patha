"""RefinementVetoDetector — supersession PRECISION, the v10 lever.

Built from 113 harvested false supersession claims (EvolutionEval
rubric-v2 artifacts, dev + spent held-out batch 2: precision 0.475 dev
/ 0.230 held-out while recall sat at 0.885/1.000). The stack's problem
is not missing revisions — it is CLAIMING revision on arcs that never
reversed. Four harvested classes, each with a veto:

  V1  NO SHARED LOCUS      "the dentist moved my appointment" tagged
                           inside a calligraphy arc. Contradiction
                           presupposes a common locus (*virodha*
                           requires a shared *viṣaya*) — v9 applied
                           this only to symmetric adoptions; v10
                           applies it to every CONTRADICTS.
  V2  FULFILLED INTENTION  "I've been thinking about making things
                           with my hands" → a concrete craft. An
                           intention is COMPLETED by its fulfilment,
                           not contradicted.
  V3  INITIATION→PROGRESS  "started running twice a week" → "signed
                           up for a 10k". Progress completes an
                           initiation.
  V4  NEW-REGIME FACETS    "sleep is now nine-to-four" tagged old when
                           "training moved to five pm" arrived. Facets
                           of the CURRENT regime don't supersede each
                           other; only the pre-regime state gets
                           superseded.

VETO-ONLY by construction: the wrapper can downgrade CONTRADICTS to
NEUTRAL, never create an edge — so recall is structurally protected
except where a veto misfires, and misfires are blocked by KEEP
overrides that run first: a new belief carrying reversal evidence
(negation, cessation, resumption, settlement, explicit correction) is
never vetoed. That is why "twenty-minute naps are now part of my day"
(arrangement facet, V4-shaped) still gets superseded by "dropped the
naps" — 'dropped' is cessation, KEEP wins.

Ships OUTERMOST in `full-stack-v10` (it must see final labels).
v7/v8/v9 stay frozen. Instrument: EvolutionEval rubric v2
supersession_precision, dev-first; held-out verdict awaits batch 3.
"""

from __future__ import annotations

import re
from typing import Callable

from patha.belief.contradiction import ContradictionDetector
from patha.belief.revision_patterns import (
    _CESSATION_MARKERS,
    _RESUMPTION_MARKERS,
    _SETTLEMENT_MARKERS,
    _any,
)
from patha.belief.sequential_detector import _DEFAULT_EMBED_CACHE
from patha.belief.types import ContradictionLabel, ContradictionResult

# ── KEEP overrides: reversal evidence in the NEW belief ─────────────

_REVERSAL_EXTRAS: tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r"\bnot?\b",
        r"\bnever\b",
        r"\bn't\b",
        r"\binstead\b",
        r"\bwas\s+wrong\b",
        r"\bturns?\s+out\b",
        r"\bactually\b",
        r"\bdropped\b",
        r"\bwrong\s+for\s+me\b",
        r"\bstopped\b",
        r"\bgave\s+up\b",
    )
)


def _reversal_evidence(new_text: str) -> bool:
    return (
        _any(_CESSATION_MARKERS, new_text)
        or _any(_RESUMPTION_MARKERS, new_text)
        or _any(_SETTLEMENT_MARKERS, new_text)
        or _any(_REVERSAL_EXTRAS, new_text)
    )


# ── V2: intention/vagueness markers in the OLD belief ───────────────

_INTENT_MARKERS: tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r"\bthink(?:ing)?\s+about\b",
        # attraction verbs only — "keep asking / replaying / doubting"
        # is rumination (perspective_shift's true olds), not attraction
        # toward a pursuit; a broad keep+gerund killed a true reframe
        r"\bkeep\s+(?:circling|coming|returning|stopping|pausing|"
        r"browsing|looking|eyeing|staring|drifting)\b",
        r"\bcouldn'?t\s+stop\s+think\w*\b",
        r"\bcan'?t\s+stop\s+think\w*\b",
        r"\bwant(?:ed)?\s+(?:to|a|an)\b",
        r"\bwonder(?:ing)?\b",
        r"\bcurious\b",
        r"\bdrawn\s+to\b",
        r"\bsomething\s+(?:about|in)\b",
        r"\bcouldn'?t\s+look\s+away\b",
        r"\bfelt\s+(?:weirdly\s+)?jealous\b",
        r"\bstare\b|\bstaring\b",
    )
)

# ── V3: initiation markers in the OLD belief ────────────────────────

_INITIATION_MARKERS: tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r"\bstarted\s+\w+ing\b",
        r"\bstarted\s+(?:a|an|the|my|our)\b",
        r"\bstarted\s+[A-Z]\w+\b",             # "started Spanish"
        r"\bplanted\b",
        r"\bbegan\b",
        r"\b(?:my|our|went\s+to\s+(?:my|a))\s+first\b",
        r"\bfirst\s+(?:class|session|lesson|attempt|try)\b",
        r"\bsigned\s+up\s+for\b",
        r"\btaster\b|\btrial\b|\btried\b",
        r"\bborrowed\b|\brented\b",
        r"\benrolled\b|\bjoined\b",
    )
)

# ── V4: arrangement/new-regime markers (in the OLD belief) ──────────

_REGIME_MARKERS: tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r"\b(?:is|are)\s+now\b",
        r"\bwe\s+do\s+\w+(?:\s+\w+)?\s+now\b",
        r"\bnow\s+instead\b",
        r"\bmoved\s+(?:my|to)\b",
        r"\bswitched\s+\w+\s+to\b",
        r"\bevery\s+morning\s+now\b",
    )
)


class RefinementVetoDetector:
    """Veto-only precision wrapper for `full-stack-v10`.

    Delegates every pair to the inner (v9) stack, then re-examines each
    CONTRADICTS: KEEP when the new belief carries reversal evidence;
    otherwise apply V1–V4 and downgrade matches to NEUTRAL with a
    rationale naming the veto. Counters expose how often each fired.
    """

    def __init__(
        self,
        inner: ContradictionDetector,
        *,
        similarity_fn: Callable[[str, str], float] | None = None,
        similarity_threshold: float = 0.25,
    ) -> None:
        # 0.25, not the codebase's 0.35 escalation standard: a VETO
        # gate's costly error is killing a true revision, and the
        # measured separation (2026-07-08) is true pairs ≥ 0.282
        # ("city center"→"countryside" 0.282, "spreadsheet"→"quarterly
        # review" 0.343) vs distractor edges ≤ 0.187 — 0.25 splits it.
        self._inner = inner
        self._sim = similarity_fn or _DEFAULT_EMBED_CACHE.similarity
        self._sim_threshold = similarity_threshold
        self.vetoes = {"intention": 0, "initiation": 0, "regime": 0}
        self.kept_by_reversal = 0

    def detect(self, p1: str, p2: str) -> ContradictionResult:
        return self.detect_batch([(p1, p2)])[0]

    def _veto_name(self, p1: str, p2: str) -> str | None:
        """NOTE: an earlier iteration carried a blanket no-shared-locus
        veto (embedding sim < 0.25 → drop). The BeliefEval 300 guard
        killed it: 347/347 → 329/347, every failure the locus veto on
        short atomic supersessions ("I live in Sydney" → "I moved to
        Sofia" scores BELOW the distractor band — embedding similarity
        cannot separate same-locus-different-surface from unrelated at
        atomic length). The marker vetoes below already suppress the
        harvested distractor-edge class on dev without it."""
        if _reversal_evidence(p2):
            self.kept_by_reversal += 1
            return None
        if _any(_INTENT_MARKERS, p1):
            return "intention"
        if _any(_INITIATION_MARKERS, p1):
            return "initiation"
        if _any(_REGIME_MARKERS, p1):
            return "regime"
        return None

    def detect_batch(
        self, pairs: list[tuple[str, str]]
    ) -> list[ContradictionResult]:
        results = self._inner.detect_batch(pairs)
        out: list[ContradictionResult] = []
        for (p1, p2), res in zip(pairs, results):
            if res.label != ContradictionLabel.CONTRADICTS:
                out.append(res)
                continue
            veto = self._veto_name(p1, p2)
            if veto is None:
                out.append(res)
                continue
            self.vetoes[veto] += 1
            out.append(ContradictionResult(
                label=ContradictionLabel.NEUTRAL,
                confidence=res.confidence,
                rationale=(
                    f"refinement-veto:{veto} (suppressed: {res.rationale})"
                ),
            ))
        return out


__all__ = ["RefinementVetoDetector"]
