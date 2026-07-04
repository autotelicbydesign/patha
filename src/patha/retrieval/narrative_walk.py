"""Narrative walk — the songline-traversal recall strategy (itihāsa path).

This is the *first* recall strategy that makes graph traversal the
primitive. Phase 1's `songline_walk` only adds a score bonus to reranked
candidates; this walker produces an *ordered narrative* — the temporally
sequenced beats of a theme across time, with supersession structure made
visible — which is what a narrative question ("how has my thinking on X
evolved?") actually wants and which top-K cannot produce.

It is a pure function over explicit dependencies (the songline graph, the
chunk→proposition id map, the belief store, and an optional Phase 1
retriever for semantic anchors), mirroring how `ganita.answer_aggregation_question`
takes its index explicitly. That keeps it unit-testable with a synthetic
graph + store and decouples it from `Memory.recall()` wiring, which is the
next integration step.

Design (see the Phase 4 plan): anchor on theme, walk theme-relevant edges,
fold in supersession lineage, order by time, render. The walk *prefers
on-theme edges* (entity / topic) and uses temporal edges only to reach the
same theme at other times — the inverse of `songline_walk`, which prefers
channel diversity to escape a cluster. Narrative wants to *stay on the
theme* and span it across time.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Callable, Optional

from patha.belief.ganita import _canonicalize_entity
from patha.belief.itihasa import (
    NarrativeBeat,
    NarrativeOp,
    NarrativeResult,
    render_through_line,
)

# Channel preference weights — entity/topic keep us on-theme; temporal
# spans the theme across time; session/speaker drift off-theme and are
# only followed to theme-relevant endpoints.
_CHANNEL_PREF: dict[str, float] = {
    "entity": 0.6,
    "topic": 0.5,
    "temporal": 0.3,
    "session": 0.1,
    "speaker": 0.1,
}

_DATETIME_MIN = datetime.min


def _resolve_belief(chunk_id: str, id_map: dict[str, str], store: Any):
    """chunk_id → source_proposition_id → Belief (or None)."""
    pid = id_map.get(chunk_id)
    if pid is None:
        return None
    return store.by_proposition(pid)


def _belief_mentions_theme(belief: Any, theme: str) -> bool:
    """Cheap on-theme check: theme token appears in the proposition."""
    if belief is None:
        return False
    return theme in belief.proposition.lower()


def _status_for(belief: Any) -> str:
    """Map a belief's store status to a narrative supersession status."""
    # Beliefs with successors are "revised-from"; those without are current.
    if getattr(belief, "superseded_by", None):
        return "revised-from"
    status = getattr(belief, "status", None)
    val = getattr(status, "value", status)
    if val in ("superseded", "archived", "badhita"):
        return "superseded"
    return "current"


def narrative_walk(
    question: str,
    op: NarrativeOp,
    theme: str,
    *,
    graph: Any,
    id_map: dict[str, str],
    store: Any,
    phase1_retrieve: Optional[Callable[[str, int], list[str]]] = None,
    max_anchors: int = 4,
    hops: int = 4,
    max_branch: int | None = None,
    max_beats: int = 24,
    recency_window: Optional[timedelta] = None,
    now: Optional[datetime] = None,
) -> Optional[NarrativeResult]:
    """Walk the songline graph for a theme and return an ordered narrative.

    Returns None when the theme yields fewer than two on-theme beats —
    the caller falls through to the ordinary retrieval path (the same
    "detector fired but the path can't deliver, degrade gracefully"
    contract gaṇita uses). This guarantees zero regression on any query
    the walker can't serve.

    Parameters mirror the Phase 4 plan; defaults are tuned for personal
    stores (hundreds–low-thousands of beliefs).
    """
    theme = _canonicalize_entity(theme)
    if not theme:
        return None

    # Frontier budget: hops × max_branch bounds total walk visits. A
    # fixed max_branch=8 starved real corpora (4×8 = 32 visits against a
    # 76-node dogfood graph — true on-theme beats were displaced at the
    # frontier once the topic gate widened admission). Scale with graph
    # size, floored at 8; final tuning belongs to the evolution-benchmark
    # sweep, not per-corpus hand-fitting.
    if max_branch is None:
        max_branch = max(8, graph.node_count() // 4)

    # ── A. ANCHOR SELECTION ─────────────────────────────────────────
    # Union of (Phase 1 semantic top-K) and (direct entity-channel
    # members for the theme). The union is the gaṇita lesson applied:
    # don't let top-K alone bound a synthesis query.
    anchor_chunks: set[str] = set()
    if phase1_retrieve is not None:
        try:
            seed_pids = phase1_retrieve(question, max_anchors * 3)
            # phase1 returns proposition_ids; invert id_map to chunk ids
            pid_to_chunk = {pid: cid for cid, pid in id_map.items()}
            for pid in seed_pids:
                cid = pid_to_chunk.get(pid)
                if cid is not None:
                    anchor_chunks.add(cid)
        except Exception:
            pass  # anchors fall back to the entity channel below
    entity_index = graph._channel_index.get("entity", {})
    anchor_chunks |= set(entity_index.get(theme, set()))

    if not anchor_chunks:
        return None

    # Rank anchors: theme-in-text first, then graph degree (well-connected
    # nodes reach more of the theme's landscape). Keep top max_anchors.
    def _degree(cid: str) -> int:
        return len(graph.neighbors(cid))

    def _on_theme(cid: str) -> int:
        b = _resolve_belief(cid, id_map, store)
        return 1 if _belief_mentions_theme(b, theme) else 0

    # chunk_id as final tiebreak: anchor_chunks is a set, and without a
    # total order, ties fall back to set-iteration order — which is
    # hash-seed dependent and made benchmark runs non-reproducible
    # (EvolutionEval caught this on day one: progressive_revelation
    # scores shifted between identical runs).
    ranked_anchors = sorted(
        anchor_chunks,
        key=lambda c: (-_on_theme(c), -_degree(c), c),
    )[:max_anchors]

    # Topic clusters represented among the anchors. Membership in one
    # of these makes a node on-theme even when the theme token is
    # paraphrased away (dogfood finding F4/F5: abstract themes are
    # never in the entity channel, and substring gating misses
    # paraphrases). Defensive getattr: hand-built/mock graphs may
    # predate topic_of. Degenerate clusters spanning >50% of the graph
    # are ignored — a chained mega-cluster must not blow the gate open.
    _topic_of = getattr(graph, "topic_of", None) or (lambda cid: None)
    n_nodes = max(1, graph.node_count())
    topic_index = graph._channel_index.get("topic", {})
    anchor_topics: set = set()
    for c in ranked_anchors:
        t = _topic_of(c)
        if t is None:
            continue
        if len(topic_index.get(t, ())) > 0.5 * n_nodes:
            continue
        anchor_topics.add(t)

    def _on_theme_node(cid: str, belief) -> bool:
        """On-theme := theme substring in text OR shares a topic cluster
        with an anchor."""
        if _belief_mentions_theme(belief, theme):
            return True
        t = _topic_of(cid)
        return t is not None and t in anchor_topics

    # ── B. THEME-CONSTRAINED MULTI-HOP WALK ─────────────────────────
    visited: set[str] = set(ranked_anchors)
    # discovered: chunk -> (best cumulative weight, channel path)
    discovered: dict[str, tuple[float, list[str]]] = {
        c: (1.0, []) for c in ranked_anchors
    }
    frontier: list[tuple[str, float, list[str]]] = [
        (c, 1.0, []) for c in ranked_anchors
    ]

    for _hop in range(hops):
        scored_next: list[tuple[str, float, list[str]]] = []
        for cur, cum_w, chpath in frontier:
            for nbr, w, channel in graph.neighbors(cur):
                if nbr in visited:
                    continue
                nbr_belief = _resolve_belief(nbr, id_map, store)
                on_theme = _on_theme_node(nbr, nbr_belief)
                # Gate: entity/topic edges always allowed; temporal +
                # session/speaker only if the endpoint is on-theme
                # (substring OR shares a topic cluster with an anchor,
                # so paraphrased beliefs pass). This is what keeps the
                # walk from drifting into whatever else happened to be
                # in a shared session.
                if channel in ("temporal", "session", "speaker") and not on_theme:
                    continue
                step = w + _CHANNEL_PREF.get(channel, 0.1)
                new_w = cum_w + step
                scored_next.append((nbr, new_w, chpath + [channel]))
        # bound blow-up: keep the strongest max_branch of this hop
        # (chunk_id tiebreak for run-to-run determinism)
        scored_next.sort(key=lambda x: (-x[1], x[0]))
        scored_next = scored_next[:max_branch]
        frontier = []
        for nbr, new_w, path in scored_next:
            if nbr in visited:
                continue
            visited.add(nbr)
            prev = discovered.get(nbr)
            if prev is None or new_w > prev[0]:
                discovered[nbr] = (new_w, path)
            frontier.append((nbr, new_w, path))
        if not frontier:
            break

    # ── C. MAP CHUNKS → BELIEFS, DEDUP ──────────────────────────────
    beats_by_belief: dict[str, NarrativeBeat] = {}
    for chunk, (cum_w, chpath) in discovered.items():
        b = _resolve_belief(chunk, id_map, store)
        if b is None:
            continue
        existing = beats_by_belief.get(b.id)
        if existing is not None and existing.walk_score >= cum_w:
            continue
        beats_by_belief[b.id] = NarrativeBeat(
            belief_id=b.id,
            proposition=b.proposition,
            asserted_at=b.asserted_at,
            supersession_status=_status_for(b),
            superseded_by=list(getattr(b, "superseded_by", [])),
            channels_to_prev=chpath,
            walk_score=cum_w,
        )

    # ── D. SUPERSESSION FOLD ────────────────────────────────────────
    # The walk surfaces current beliefs; pull each one's superseded
    # ancestors via lineage — those ARE the "used to think X, now Y"
    # beats that make evolution visible.
    for bid in list(beats_by_belief):
        for ancestor in store.lineage(bid):
            if ancestor.id == bid:
                continue
            if ancestor.id not in beats_by_belief:
                beats_by_belief[ancestor.id] = NarrativeBeat(
                    belief_id=ancestor.id,
                    proposition=ancestor.proposition,
                    asserted_at=ancestor.asserted_at,
                    supersession_status="revised-from",
                    superseded_by=list(getattr(ancestor, "superseded_by", [])),
                    channels_to_prev=["supersession"],
                    walk_score=0.0,
                )

    # ── E. TEMPORAL ORDERING + RECENCY WINDOW ───────────────────────
    beats = sorted(
        beats_by_belief.values(),
        key=lambda x: (x.asserted_at or _DATETIME_MIN),
    )
    if recency_window is not None and now is not None:
        cutoff = now - recency_window
        windowed = [b for b in beats if (b.asserted_at or _DATETIME_MIN) >= cutoff]
        # keep the window, but never drop below 2 beats if the theme has more
        if len(windowed) >= 2:
            beats = windowed

    # ── F. TERMINATION ──────────────────────────────────────────────
    if len(beats) < 2:
        return None  # not a narrative; degrade to retrieval

    # ── G. TRUNCATION (preserve the arc, thin the middle) ───────────
    if len(beats) > max_beats:
        beats = _temporal_thin(beats, max_beats)

    # ── H. ORIGIN TAG + THROUGH-LINE ────────────────────────────────
    # Mark the earliest beat as origin unless it's already a revision.
    beats = list(beats)
    if beats[0].supersession_status == "current":
        first = beats[0]
        beats[0] = NarrativeBeat(
            belief_id=first.belief_id,
            proposition=first.proposition,
            asserted_at=first.asserted_at,
            supersession_status="origin",
            superseded_by=first.superseded_by,
            channels_to_prev=first.channels_to_prev,
            walk_score=first.walk_score,
        )

    through_line = render_through_line(op, theme, beats)
    contributing = [b.belief_id for b in beats]
    return NarrativeResult(
        operator=op,
        theme=theme,
        beats=beats,
        through_line=through_line,
        contributing_belief_ids=contributing,
        anchors=[
            _resolve_belief(c, id_map, store).id
            for c in ranked_anchors
            if _resolve_belief(c, id_map, store) is not None
        ],
    )


def _temporal_thin(beats: list, max_beats: int) -> list:
    """Down-sample to max_beats while preserving the arc.

    Always keep the first (origin) and last (current head) beat plus any
    supersession/revision beats (those carry the evolution signal). The
    remaining middle is kept by descending ``walk_score`` — the walk's
    own connectivity signal, which encodes on-theme edge strength — then
    re-sorted by time. (Dogfood step-4 A/B finding: an even-stride
    middle is relevance-blind — under beat-cap saturation it kept
    weakly-related beats while dropping strongly-connected true ones,
    e.g. N3 losing "the ablations humbled me" when the topic gate
    widened the candidate pool.)
    """
    if len(beats) <= max_beats:
        return beats
    must_keep_idx = {0, len(beats) - 1}
    for i, b in enumerate(beats):
        if b.supersession_status in ("revised-from", "superseded"):
            must_keep_idx.add(i)
    must_keep = [beats[i] for i in sorted(must_keep_idx)]
    if len(must_keep) >= max_beats:
        # Even the must-keeps exceed budget: stride over them (they're
        # temporally sorted, so a stride preserves the arc's shape).
        stride = max(1, len(must_keep) // max_beats)
        return must_keep[::stride][:max_beats]
    # Fill the remaining budget with the strongest-connected middle
    # beats (walk_score desc; date as deterministic tiebreak).
    remaining_budget = max_beats - len(must_keep)
    middle = [beats[i] for i in range(len(beats)) if i not in must_keep_idx]
    middle.sort(
        key=lambda x: (-x.walk_score, x.asserted_at or _DATETIME_MIN),
    )
    sampled = middle[:remaining_budget]
    combined = must_keep + sampled
    combined.sort(key=lambda x: (x.asserted_at or _DATETIME_MIN))
    return combined


__all__ = ["narrative_walk"]
