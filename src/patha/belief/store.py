"""Belief store — persistent, non-destructive storage for the belief layer.

The store is the authority on what beliefs exist and what their current
status is (current vs. superseded vs. reinforced). It has four core
operations:

- add(belief)           — record a new belief
- supersede(old, new)   — mark old as superseded by new (bidirectional)
- reinforce(old, new)   — mark old as reinforced by new (confidence bump)
- query(...)            — retrieve beliefs matching criteria

Supersession is non-destructive (AGM's Preservation postulate). Nothing
is ever deleted — only marked. The `supersedes` relation forms a DAG,
not a linked list: a new belief can supersede multiple older beliefs
simultaneously (e.g., "I'm vegetarian now" supersedes both "I love
sushi" and "I love steak").

Storage:
- In-memory dict for fast lookup.
- Optional JSONL persistence: every mutation is appended as an event
  record, replay-able on reload. Simple, human-readable, append-only.

Deferred to v0.2+:
- LanceDB or other vector-aware storage
- Decay-over-time application to confidence scores
- Compaction/archival of heavily-superseded lineages
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterator

from patha.belief.types import (
    PRAMANA_STRENGTH,
    Belief,
    BeliefId,
    Pramana,
    PropositionId,
    ResolutionStatus,
    Validity,
)


# ─── Persistence: event log records ──────────────────────────────────

# Rather than serialise entire Belief objects on every mutation, we append
# small event records to the JSONL log. This keeps writes cheap and gives
# us a durable audit trail for free.

EVENT_ADD = "add"
EVENT_SUPERSEDE = "supersede"
EVENT_REINFORCE = "reinforce"
EVENT_CONFIDENCE_SET = "confidence_set"
EVENT_COEXIST = "coexist"
EVENT_DISPUTE = "dispute"
EVENT_RESOLVE_DISPUTE = "resolve_dispute"
EVENT_STATUS_SET = "status_set"
EVENT_SUBLATE = "sublate"
EVENT_RELIABILITY_UPDATE = "reliability_update"


# ─── Source reliability tracking ─────────────────────────────────────

@dataclass
class SourceReliability:
    """Running reliability score for an information source.

    The score tracks, for a given source_id, how often that source's
    testimony has been contradicted by stronger-pramāṇa evidence (via
    sublation) vs. how often it has held up or been reinforced.

    Attributes
    ----------
    corroborated
        Count of the source's claims that were reinforced or remained
        uncontradicted (proxy for "held up").
    sublated
        Count of the source's claims that were contradicted by stronger
        pramāṇa (sublation via BADHITA status).
    score
        Derived reliability in [0.3, 1.0]. Computed as:
            score = max(0.3, 1.0 - sublated / max(1, corroborated + sublated))
        Floor at 0.3 so a source is never completely dismissed — an
        unreliable witness can still carry information worth checking.
        With zero history, score defaults to 1.0 (give new sources the
        benefit of the doubt).
    """

    corroborated: int = 0
    sublated: int = 0

    @property
    def score(self) -> float:
        total = self.corroborated + self.sublated
        if total == 0:
            return 1.0
        raw = 1.0 - (self.sublated / total)
        return max(0.3, raw)


# ─── Store ───────────────────────────────────────────────────────────

class BeliefStore:
    """In-memory belief store with optional JSONL persistence.

    Parameters
    ----------
    persistence_path
        If provided, every mutation is appended to this file as a JSON
        event record. On construction, if the file exists, events are
        replayed to rebuild state. None disables persistence entirely.
    """

    def __init__(self, persistence_path: str | Path | None = None) -> None:
        self._beliefs: dict[BeliefId, Belief] = {}
        self._by_session: dict[str, list[BeliefId]] = {}
        self._by_proposition: dict[PropositionId, BeliefId] = {}
        # Source reliability: source_id -> SourceReliability
        self._source_reliability: dict[str, SourceReliability] = {}
        self._persistence_path: Path | None = (
            Path(persistence_path) if persistence_path is not None else None
        )

        if self._persistence_path is not None and self._persistence_path.exists():
            self._replay()

    # ── core mutations ──────────────────────────────────────────────

    def add(
        self,
        proposition: str,
        *,
        asserted_at: datetime,
        asserted_in_session: str,
        source_proposition_id: PropositionId,
        validity: Validity | None = None,
        confidence: float | None = None,
        belief_id: BeliefId | None = None,
        pramana: Pramana | None = None,
        source_id: str | None = None,
        context: str | None = None,
    ) -> Belief:
        """Record a new belief. Returns the created Belief.

        If ``belief_id`` is None, a UUID is generated. Raises if the id
        already exists.

        ``pramana`` is the source-of-valid-knowledge tag (Pramana enum).
        Defaults to Pramana.UNKNOWN when not provided.

        ``confidence`` is auto-set based on pramāṇa strength and (for
        SHABDA) source reliability if not explicitly provided:

          - PRATYAKṢA / UNKNOWN  → 1.0 (direct assertion by user)
          - ANUMANA              → 0.80
          - ARTHAPATTI           → 0.70
          - ANUPALABDHI          → 0.65
          - SHABDA               → 0.60 × source_reliability_score
                                    (unreliable source discounts further)
          - UPAMANA              → 0.55

        The user's own direct assertions ("I have X") remain at high
        confidence. Reported testimony ("my doctor said I have X")
        starts lower so future direct-perception contradictions can
        sublate it cleanly via pramāṇa-hierarchy.
        """
        bid = belief_id if belief_id is not None else str(uuid.uuid4())
        if bid in self._beliefs:
            raise ValueError(f"belief id {bid!r} already exists")

        effective_pramana = pramana if pramana is not None else Pramana.UNKNOWN
        effective_source_id = source_id if source_id is not None else asserted_in_session

        # Compute default confidence if not specified.
        if confidence is None:
            base_strength = PRAMANA_STRENGTH.get(effective_pramana.value, 0.7)
            # For SHABDA specifically, discount by source reliability
            if effective_pramana == Pramana.SHABDA:
                rel = self._source_reliability.get(
                    effective_source_id, SourceReliability()
                ).score
                confidence = base_strength * rel
            else:
                confidence = base_strength

        belief = Belief(
            id=bid,
            proposition=proposition,
            asserted_at=asserted_at,
            asserted_in_session=asserted_in_session,
            source_proposition_id=source_proposition_id,
            confidence=confidence,
            validity=validity if validity is not None else Validity(),
            pramana=effective_pramana,
            source_id=effective_source_id,
            context=context,
        )
        self._beliefs[bid] = belief
        self._by_session.setdefault(asserted_in_session, []).append(bid)
        self._by_proposition[source_proposition_id] = bid
        self._append_event(EVENT_ADD, belief=belief)
        return belief

    def supersede(self, old_id: BeliefId, new_id: BeliefId) -> None:
        """Record that ``new_id`` supersedes ``old_id``.

        Bidirectional: old.superseded_by += [new], new.supersedes += [old].
        Non-destructive: both beliefs remain queryable; old becomes
        is_superseded. Safe to call multiple times; relations are deduped.
        """
        if old_id == new_id:
            raise ValueError("a belief cannot supersede itself")
        old = self._require(old_id)
        new = self._require(new_id)
        if new_id not in old.superseded_by:
            old.superseded_by.append(new_id)
        if old_id not in new.supersedes:
            new.supersedes.append(old_id)
        old.status = ResolutionStatus.SUPERSEDED
        # Keep new's status as CURRENT unless a dispute/coexist edge exists.
        if (
            new.status in (ResolutionStatus.DISPUTED, ResolutionStatus.AMBIGUOUS)
            and not new.disputed_with
        ):
            new.status = ResolutionStatus.CURRENT
        self._append_event(EVENT_SUPERSEDE, old=old_id, new=new_id)

    def reinforce(self, existing_id: BeliefId, new_id: BeliefId) -> None:
        """Record that ``new_id`` reinforces ``existing_id``.

        Effect:
        - existing.reinforced_by += [new_id]
        - existing.reinforcement_sources += [new.source_id] if distinct
        - existing.reinforcement_pramanas += [new.pramana] if distinct
        - existing.confidence bumped toward 1.0 (capped)

        Two dimensions of epistemic diversity contribute to the bump:

        * Source independence — distinct source_id (i.e., different
          session cluster) means we're not just hearing the same
          utterance echoed.
        * Pramāṇa diversity — distinct pramāṇa means the reinforcement
          comes from a different kind of knowledge (e.g., perception
          confirming what was earlier heard via testimony). A belief
          held via perception AND testimony is more robust than one
          held via repeated perception alone.

        Bump rate (closes this fraction of the gap to 1.0):

          both source and pramāṇa distinct:    0.40  (full cross-corroboration)
          source distinct, pramāṇa same:       0.30  (v0.2 default)
          pramāṇa distinct, source same:       0.20  (same person, different mode of knowing)
          both same:                           0.10  (pure repetition)

        This makes "a doctor told me I have diabetes" + "I checked my
        own blood sugar" a stronger joint signal than two echoes of the
        doctor's statement, which is the point.
        """
        if existing_id == new_id:
            raise ValueError("a belief cannot reinforce itself")
        existing = self._require(existing_id)
        new = self._require(new_id)
        if new_id not in existing.reinforced_by:
            existing.reinforced_by.append(new_id)

        # Source-independence tracking
        new_source = new.source_id or new.asserted_in_session
        existing_sources = set(existing.reinforcement_sources)
        existing_sources.add(existing.source_id or existing.asserted_in_session)
        distinct_source = new_source not in existing_sources
        if distinct_source and new_source not in existing.reinforcement_sources:
            existing.reinforcement_sources.append(new_source)

        # Pramāṇa-diversity tracking
        new_pramana = new.pramana.value if new.pramana else Pramana.UNKNOWN.value
        existing_pramanas = set(existing.reinforcement_pramanas)
        if existing.pramana:
            existing_pramanas.add(existing.pramana.value)
        distinct_pramana = (
            new_pramana != Pramana.UNKNOWN.value
            and new_pramana not in existing_pramanas
        )
        if distinct_pramana and new_pramana not in existing.reinforcement_pramanas:
            existing.reinforcement_pramanas.append(new_pramana)

        # Bump rate: combine both dimensions of diversity.
        if distinct_source and distinct_pramana:
            rate = 0.40
        elif distinct_source:
            rate = 0.30
        elif distinct_pramana:
            rate = 0.20
        else:
            rate = 0.10

        gap = 1.0 - existing.confidence
        existing.confidence = min(1.0, existing.confidence + rate * gap)

        # Saṁskāra → Vāsanā (v0.4): track reinforcement count and
        # crystallise deep confidence once the chain is long enough.
        # A distinct-source or distinct-pramāṇa reinforcement counts
        # as a real saṁskāra. Same-source same-pramāṇa pure echoes
        # do NOT deepen the vāsanā (matches the Yoga tradition —
        # habit formation requires repetition across occasions, not
        # rapid restatement).
        if distinct_source or distinct_pramana:
            existing.samskara_count += 1

        # Establishment threshold: once a belief has been reinforced
        # 5 times from distinct samskaras, crystallise deep confidence
        # to the current surface confidence. Deep confidence then
        # decays 10x more slowly than surface in LongTermDepression.
        SAMSKARA_THRESHOLD = 5
        if (
            existing.deep_confidence is None
            and existing.samskara_count >= SAMSKARA_THRESHOLD
        ):
            existing.deep_confidence = existing.confidence
        # Once vāsanā is established, a new reinforcement also
        # pulls deep confidence upward (at 1/5 the rate of surface).
        elif existing.deep_confidence is not None:
            deep_gap = 1.0 - existing.deep_confidence
            existing.deep_confidence = min(
                1.0, existing.deep_confidence + (rate / 5.0) * deep_gap
            )

        self._append_event(
            EVENT_REINFORCE,
            existing=existing_id,
            new=new_id,
            distinct_source=distinct_source,
            distinct_pramana=distinct_pramana,
        )

    def sublate(self, weaker_id: BeliefId, stronger_id: BeliefId) -> None:
        """Record that ``stronger_id`` sublates (bādhita) ``weaker_id``.

        Sublation is the Nyāya operation where a stronger pramāṇa
        (typically direct perception) contradicts a weaker one
        (typically testimony). Semantically distinct from supersession:

          supersede(old, new)    → 'you changed your mind over time'
          sublate(weaker, strong) → 'stronger evidence arrived; the
                                     weaker claim was right-to-believe-
                                     at-the-time but is now demoted'

        Effect:
          - weaker.status = BADHITA
          - Supersession edges added (weaker in stronger.supersedes)
          - If weaker.source_id has a reliability record, mark this as
            a 'sublated' event for that source (drives future
            source-reliability discount on SHABDA from that source).

        Non-destructive: weaker remains queryable, lineage preserved.
        """
        if weaker_id == stronger_id:
            raise ValueError("a belief cannot sublate itself")
        weaker = self._require(weaker_id)
        stronger = self._require(stronger_id)
        if stronger_id not in weaker.superseded_by:
            weaker.superseded_by.append(stronger_id)
        if weaker_id not in stronger.supersedes:
            stronger.supersedes.append(weaker_id)
        weaker.status = ResolutionStatus.BADHITA
        # Restore stronger to CURRENT if dispute cleared
        if (
            stronger.status in (ResolutionStatus.DISPUTED, ResolutionStatus.AMBIGUOUS)
            and not stronger.disputed_with
        ):
            stronger.status = ResolutionStatus.CURRENT

        # Update source reliability: this source's claim just got sublated.
        source_id = weaker.source_id or weaker.asserted_in_session
        rel = self._source_reliability.setdefault(source_id, SourceReliability())
        rel.sublated += 1

        self._append_event(
            EVENT_SUBLATE, weaker=weaker_id, stronger=stronger_id
        )
        self._append_event(
            EVENT_RELIABILITY_UPDATE,
            source_id=source_id,
            corroborated=rel.corroborated,
            sublated=rel.sublated,
        )

    def resolve_contradiction(
        self, old_id: BeliefId, new_id: BeliefId,
        *,
        confidence_weighted: bool = False,
        confidence_margin: float = 0.2,
    ) -> ResolutionStatus:
        """Resolve a contradiction between two beliefs.

        Resolution proceeds in three tiers:

          1. Pramāṇa hierarchy (always applied first). If one belief's
             pramāṇa is strictly stronger than the other's (after
             source-reliability weighting for SHABDA), the weaker one
             becomes BADHITA.

          2. Confidence-weighted (D3 advanced; opt-in via confidence_
             weighted=True). When pramāṇas tie, if one belief's
             confidence exceeds the other's by more than
             confidence_margin, the higher-confidence belief wins (the
             lower one is sublated). This captures the 'I'm quite sure
             I saw X vs. I might have seen Y' case: both PRATYAKṢA but
             one held with more certainty.

          3. Temporal supersession (default). Pramāṇas tied, confidences
             close → newer belief supersedes older.

        Returns the resolution outcome as a ResolutionStatus:
          - ResolutionStatus.BADHITA if a sublation happened
          - ResolutionStatus.SUPERSEDED if temporal supersession happened

        Parameters
        ----------
        confidence_weighted
            Opt in to D3 advanced (Hansson's non-prioritised revision).
            Default False preserves pure temporal-order behaviour.
        confidence_margin
            Minimum confidence delta required to trigger confidence-
            weighted winner selection. Default 0.2 — prevents flip-
            flopping from noise in confidence updates. Only used when
            confidence_weighted=True.
        """
        old = self._require(old_id)
        new = self._require(new_id)
        old_strength = PRAMANA_STRENGTH.get(old.pramana.value, 0.7)
        new_strength = PRAMANA_STRENGTH.get(new.pramana.value, 0.7)

        # Apply source-reliability multiplier for SHABDA (unreliable
        # testimony is weaker than reliable testimony even within SHABDA).
        if old.pramana == Pramana.SHABDA:
            rel = self._source_reliability.get(
                old.source_id or old.asserted_in_session, SourceReliability()
            )
            old_strength *= rel.score
        if new.pramana == Pramana.SHABDA:
            rel = self._source_reliability.get(
                new.source_id or new.asserted_in_session, SourceReliability()
            )
            new_strength *= rel.score

        epsilon = 0.05  # Treat near-equal strengths as equal; avoids
                        # flip-flopping on tiny reliability-score shifts.

        # Tier 1: pramāṇa hierarchy
        if new_strength > old_strength + epsilon:
            self.sublate(old_id, new_id)
            return ResolutionStatus.BADHITA
        if old_strength > new_strength + epsilon:
            self.sublate(new_id, old_id)
            return ResolutionStatus.BADHITA

        # Tier 2: confidence-weighted (D3 advanced, opt-in)
        if confidence_weighted:
            delta = new.confidence - old.confidence
            if delta > confidence_margin:
                self.sublate(old_id, new_id)
                return ResolutionStatus.BADHITA
            if delta < -confidence_margin:
                self.sublate(new_id, old_id)
                return ResolutionStatus.BADHITA

        # Tier 3: temporal supersession (default)
        self.supersede(old_id, new_id)
        return ResolutionStatus.SUPERSEDED

    # ── source reliability (read-only for external callers) ────────

    def source_reliability(self, source_id: str) -> SourceReliability:
        """Return the reliability record for a source.

        Creates a default record (corroborated=0, sublated=0, score=1.0)
        if the source has no history.
        """
        return self._source_reliability.get(source_id, SourceReliability())

    def coexist(self, a_id: BeliefId, b_id: BeliefId) -> None:
        """Record that two beliefs hold simultaneously (no contradiction).

        Symmetric: a.coexists_with += [b], b.coexists_with += [a].
        Both beliefs stay CURRENT; their status is decorated with
        coexistence rather than replaced. Use this when a new proposition
        is related to an existing one but explicitly non-contradictory
        (e.g., 'I like sushi' + 'I also like steak').
        """
        if a_id == b_id:
            raise ValueError("a belief cannot coexist with itself")
        a = self._require(a_id)
        b = self._require(b_id)
        if b_id not in a.coexists_with:
            a.coexists_with.append(b_id)
        if a_id not in b.coexists_with:
            b.coexists_with.append(a_id)
        # Promote CURRENT-labelled beliefs to COEXISTS so downstream
        # code can surface the relationship.
        for belief in (a, b):
            if belief.status == ResolutionStatus.CURRENT:
                belief.status = ResolutionStatus.COEXISTS
        self._append_event(EVENT_COEXIST, a=a_id, b=b_id)

    def dispute(
        self,
        a_id: BeliefId,
        b_id: BeliefId,
        *,
        ambiguous: bool = False,
    ) -> None:
        """Record an unresolved contradiction between two beliefs.

        Unlike supersession, neither belief wins. Both remain current
        (not superseded) but carry a DISPUTED (or AMBIGUOUS) status so
        callers know to surface them with a caveat.

        Parameters
        ----------
        ambiguous
            If True, record as AMBIGUOUS (low-confidence contradiction
            signal, flagged for review). If False (default), DISPUTED
            (clear conflict, resolution pending).
        """
        if a_id == b_id:
            raise ValueError("a belief cannot dispute itself")
        a = self._require(a_id)
        b = self._require(b_id)
        if b_id not in a.disputed_with:
            a.disputed_with.append(b_id)
        if a_id not in b.disputed_with:
            b.disputed_with.append(a_id)
        target_status = (
            ResolutionStatus.AMBIGUOUS if ambiguous else ResolutionStatus.DISPUTED
        )
        for belief in (a, b):
            # Don't downgrade superseded beliefs; their history status wins.
            if belief.status in (
                ResolutionStatus.CURRENT,
                ResolutionStatus.COEXISTS,
            ):
                belief.status = target_status
        self._append_event(EVENT_DISPUTE, a=a_id, b=b_id, ambiguous=ambiguous)

    def resolve_dispute(
        self,
        winner_id: BeliefId,
        loser_id: BeliefId,
    ) -> None:
        """Resolve a previously-disputed pair: winner supersedes loser.

        Moves loser from DISPUTED to SUPERSEDED, promotes winner back to
        CURRENT, and drops the disputed_with edge between them.
        """
        winner = self._require(winner_id)
        loser = self._require(loser_id)
        if winner_id not in loser.disputed_with:
            raise ValueError(
                f"{loser_id} is not disputed with {winner_id}"
            )
        # Remove the symmetric dispute edge
        loser.disputed_with = [x for x in loser.disputed_with if x != winner_id]
        winner.disputed_with = [x for x in winner.disputed_with if x != loser_id]
        # Apply the supersession
        self.supersede(loser_id, winner_id)
        # Restore winner's status if no other disputes remain
        if not winner.disputed_with:
            winner.status = (
                ResolutionStatus.COEXISTS
                if winner.coexists_with
                else ResolutionStatus.CURRENT
            )
        self._append_event(
            EVENT_RESOLVE_DISPUTE, winner=winner_id, loser=loser_id
        )

    def archive(self, belief_id: BeliefId) -> None:
        """Mark a belief as archived (pruned from default surfaces).

        Non-destructive: still stored, still queryable through explicit
        archive queries. Used by SynapticPruning and for manual cleanup.
        """
        b = self._require(belief_id)
        b.status = ResolutionStatus.ARCHIVED
        self._append_event(EVENT_STATUS_SET, belief=belief_id, status="archived")

    def set_status(self, belief_id: BeliefId, status: ResolutionStatus) -> None:
        """Explicit status override. For advanced callers."""
        b = self._require(belief_id)
        b.status = status
        self._append_event(EVENT_STATUS_SET, belief=belief_id, status=status.value)

    def set_confidence(self, belief_id: BeliefId, confidence: float) -> None:
        """Explicitly set a belief's confidence. Used by decay and external callers."""
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1]; got {confidence}")
        b = self._require(belief_id)
        b.confidence = confidence
        self._append_event(EVENT_CONFIDENCE_SET, belief=belief_id, confidence=confidence)

    # ── queries ─────────────────────────────────────────────────────

    def get(self, belief_id: BeliefId) -> Belief | None:
        return self._beliefs.get(belief_id)

    def all(self) -> list[Belief]:
        """All beliefs, current and historical."""
        return list(self._beliefs.values())

    def current(self) -> list[Belief]:
        """Only current beliefs (not superseded)."""
        return [b for b in self._beliefs.values() if b.is_current]

    def superseded(self) -> list[Belief]:
        """Only superseded beliefs."""
        return [b for b in self._beliefs.values() if b.is_superseded]

    def disputed(self) -> list[Belief]:
        """Beliefs currently in a DISPUTED or AMBIGUOUS state."""
        return [
            b for b in self._beliefs.values()
            if b.status in (ResolutionStatus.DISPUTED, ResolutionStatus.AMBIGUOUS)
        ]

    def coexisting(self) -> list[Belief]:
        """Beliefs that have at least one coexists_with edge."""
        return [b for b in self._beliefs.values() if b.coexists_with]

    def archived(self) -> list[Belief]:
        """Archived beliefs (pruned from default surfaces)."""
        return [
            b for b in self._beliefs.values()
            if b.status == ResolutionStatus.ARCHIVED
        ]

    def by_status(self, status: ResolutionStatus) -> list[Belief]:
        """All beliefs with the exact given status."""
        return [b for b in self._beliefs.values() if b.status == status]

    def by_context(self, context: str | None) -> list[Belief]:
        """All beliefs tagged with the given context.

        Pass None to get only context-independent beliefs (those
        never tagged with a specific context).
        """
        return [b for b in self._beliefs.values() if b.context == context]

    def lineage(self, belief_id: BeliefId) -> list[Belief]:
        """Walk from a belief back through its supersession ancestors.

        Returns the belief itself first, then beliefs it supersedes
        (and beliefs they supersede, transitively), breadth-first.
        Useful for answering "what was the previous view on X?" queries.

        Cycles in the supersedes graph would cause infinite traversal;
        the implementation guards against this with a visited set.
        """
        root = self.get(belief_id)
        if root is None:
            return []
        result: list[Belief] = [root]
        visited: set[BeliefId] = {belief_id}
        frontier: list[BeliefId] = list(root.supersedes)
        while frontier:
            next_frontier: list[BeliefId] = []
            for bid in frontier:
                if bid in visited:
                    continue
                visited.add(bid)
                b = self.get(bid)
                if b is None:
                    continue
                result.append(b)
                next_frontier.extend(b.supersedes)
            frontier = next_frontier
        return result

    def by_session(self, session_id: str) -> list[Belief]:
        ids = self._by_session.get(session_id, [])
        return [self._beliefs[b] for b in ids if b in self._beliefs]

    def by_proposition(self, proposition_id: PropositionId) -> Belief | None:
        bid = self._by_proposition.get(proposition_id)
        return self._beliefs.get(bid) if bid is not None else None

    def __len__(self) -> int:
        return len(self._beliefs)

    def __iter__(self) -> Iterator[Belief]:
        return iter(self._beliefs.values())

    def __contains__(self, belief_id: object) -> bool:
        return isinstance(belief_id, str) and belief_id in self._beliefs

    # ── persistence internals ───────────────────────────────────────

    def _append_event(self, event_type: str, **payload) -> None:
        if self._persistence_path is None:
            return
        self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"type": event_type, **self._serialise_payload(payload)}
        # Append-only. Crash safety comes from the append being atomic
        # at the filesystem level for small records (<PIPE_BUF bytes);
        # for long-term durability we'd fsync here — deferred to v0.2.
        with open(self._persistence_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    @staticmethod
    def _serialise_payload(payload: dict) -> dict:
        out: dict = {}
        for k, v in payload.items():
            if isinstance(v, Belief):
                out[k] = _belief_to_dict(v)
            elif isinstance(v, datetime):
                out[k] = v.isoformat()
            else:
                out[k] = v
        return out

    def _replay(self) -> None:
        assert self._persistence_path is not None
        with open(self._persistence_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                self._apply_event(record)

    def _apply_event(self, record: dict) -> None:
        t = record["type"]
        if t == EVENT_ADD:
            belief = _belief_from_dict(record["belief"])
            self._beliefs[belief.id] = belief
            self._by_session.setdefault(
                belief.asserted_in_session, []
            ).append(belief.id)
            self._by_proposition[belief.source_proposition_id] = belief.id
        elif t == EVENT_SUPERSEDE:
            old = self._beliefs.get(record["old"])
            new = self._beliefs.get(record["new"])
            if old is not None and record["new"] not in old.superseded_by:
                old.superseded_by.append(record["new"])
                old.status = ResolutionStatus.SUPERSEDED
            if new is not None and record["old"] not in new.supersedes:
                new.supersedes.append(record["old"])
        elif t == EVENT_REINFORCE:
            # Replay the full reinforce() mutation from the event record,
            # including the source/pramāṇa diversity bump rate and the
            # saṁskāra → vāsanā accumulation. Re-derived here rather
            # than calling reinforce() because that would re-emit the
            # event and double-log it.
            existing = self._beliefs.get(record["existing"])
            new_b = self._beliefs.get(record["new"])
            if existing is None or new_b is None:
                return
            if record["new"] not in existing.reinforced_by:
                existing.reinforced_by.append(record["new"])

            distinct_source = bool(record.get("distinct_source", False))
            distinct_pramana = bool(record.get("distinct_pramana", False))
            if distinct_source and distinct_pramana:
                rate = 0.40
            elif distinct_source:
                rate = 0.30
            elif distinct_pramana:
                rate = 0.20
            else:
                rate = 0.10
            gap = 1.0 - existing.confidence
            existing.confidence = min(1.0, existing.confidence + rate * gap)

            # Saṁskāra → Vāsanā
            if distinct_source or distinct_pramana:
                existing.samskara_count += 1
            SAMSKARA_THRESHOLD = 5
            if (
                existing.deep_confidence is None
                and existing.samskara_count >= SAMSKARA_THRESHOLD
            ):
                existing.deep_confidence = existing.confidence
            elif existing.deep_confidence is not None:
                deep_gap = 1.0 - existing.deep_confidence
                existing.deep_confidence = min(
                    1.0, existing.deep_confidence + (rate / 5.0) * deep_gap
                )
        elif t == EVENT_CONFIDENCE_SET:
            b = self._beliefs.get(record["belief"])
            if b is not None:
                b.confidence = float(record["confidence"])
        elif t == EVENT_COEXIST:
            a = self._beliefs.get(record["a"])
            b = self._beliefs.get(record["b"])
            if a is not None and record["b"] not in a.coexists_with:
                a.coexists_with.append(record["b"])
            if b is not None and record["a"] not in b.coexists_with:
                b.coexists_with.append(record["a"])
            for belief in (a, b):
                if belief is not None and belief.status == ResolutionStatus.CURRENT:
                    belief.status = ResolutionStatus.COEXISTS
        elif t == EVENT_DISPUTE:
            a = self._beliefs.get(record["a"])
            b = self._beliefs.get(record["b"])
            if a is not None and record["b"] not in a.disputed_with:
                a.disputed_with.append(record["b"])
            if b is not None and record["a"] not in b.disputed_with:
                b.disputed_with.append(record["a"])
            target = (
                ResolutionStatus.AMBIGUOUS
                if record.get("ambiguous")
                else ResolutionStatus.DISPUTED
            )
            for belief in (a, b):
                if belief is not None and belief.status in (
                    ResolutionStatus.CURRENT,
                    ResolutionStatus.COEXISTS,
                ):
                    belief.status = target
        elif t == EVENT_RESOLVE_DISPUTE:
            winner = self._beliefs.get(record["winner"])
            loser = self._beliefs.get(record["loser"])
            if loser is not None:
                loser.disputed_with = [
                    x for x in loser.disputed_with if x != record["winner"]
                ]
            if winner is not None:
                winner.disputed_with = [
                    x for x in winner.disputed_with if x != record["loser"]
                ]
            # The dispute-resolving also emits a supersede event; don't
            # double-apply here.
        elif t == EVENT_STATUS_SET:
            b = self._beliefs.get(record["belief"])
            if b is not None:
                try:
                    b.status = ResolutionStatus(record["status"])
                except ValueError:
                    pass
        elif t == EVENT_SUBLATE:
            weaker = self._beliefs.get(record["weaker"])
            stronger = self._beliefs.get(record["stronger"])
            if weaker is not None and record["stronger"] not in weaker.superseded_by:
                weaker.superseded_by.append(record["stronger"])
                weaker.status = ResolutionStatus.BADHITA
            if stronger is not None and record["weaker"] not in stronger.supersedes:
                stronger.supersedes.append(record["weaker"])
        elif t == EVENT_RELIABILITY_UPDATE:
            sid = record["source_id"]
            self._source_reliability[sid] = SourceReliability(
                corroborated=int(record.get("corroborated", 0)),
                sublated=int(record.get("sublated", 0)),
            )

    # ── utility ─────────────────────────────────────────────────────

    def _require(self, belief_id: BeliefId) -> Belief:
        b = self._beliefs.get(belief_id)
        if b is None:
            raise KeyError(f"belief id {belief_id!r} not found")
        return b


# ─── (de)serialisation helpers ───────────────────────────────────────

def _belief_to_dict(b: Belief) -> dict:
    d = asdict(b)
    d["asserted_at"] = b.asserted_at.isoformat()
    if b.observed_at is not None:
        d["observed_at"] = b.observed_at.isoformat()
    d["status"] = b.status.value
    d["pramana"] = b.pramana.value
    if b.validity.start is not None:
        d["validity"]["start"] = b.validity.start.isoformat()
    if b.validity.end is not None:
        d["validity"]["end"] = b.validity.end.isoformat()
    return d


def _belief_from_dict(d: dict) -> Belief:
    val = d["validity"]
    validity = Validity(
        mode=val["mode"],
        start=datetime.fromisoformat(val["start"]) if val.get("start") else None,
        end=datetime.fromisoformat(val["end"]) if val.get("end") else None,
        half_life_days=val.get("half_life_days"),
        source=val.get("source", "default"),
    )
    try:
        status = ResolutionStatus(d.get("status", "current"))
    except ValueError:
        status = ResolutionStatus.CURRENT
    try:
        pramana = Pramana(d.get("pramana", "unknown"))
    except ValueError:
        pramana = Pramana.UNKNOWN
    return Belief(
        id=d["id"],
        proposition=d["proposition"],
        asserted_at=datetime.fromisoformat(d["asserted_at"]),
        asserted_in_session=d["asserted_in_session"],
        source_proposition_id=d["source_proposition_id"],
        confidence=d.get("confidence", 1.0),
        status=status,
        validity=validity,
        pramana=pramana,
        observed_at=(
            datetime.fromisoformat(d["observed_at"])
            if d.get("observed_at")
            else None
        ),
        source_id=d.get("source_id"),
        context=d.get("context"),
        samskara_count=int(d.get("samskara_count", 0)),
        deep_confidence=d.get("deep_confidence"),
        supersedes=list(d.get("supersedes", [])),
        superseded_by=list(d.get("superseded_by", [])),
        coexists_with=list(d.get("coexists_with", [])),
        disputed_with=list(d.get("disputed_with", [])),
        reinforced_by=list(d.get("reinforced_by", [])),
        reinforcement_sources=list(d.get("reinforcement_sources", [])),
        reinforcement_pramanas=list(d.get("reinforcement_pramanas", [])),
    )
