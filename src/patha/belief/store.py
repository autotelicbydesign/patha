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
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Iterator

from patha.belief.types import (
    Belief,
    BeliefId,
    PropositionId,
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
        confidence: float = 1.0,
        belief_id: BeliefId | None = None,
    ) -> Belief:
        """Record a new belief. Returns the created Belief.

        If ``belief_id`` is None, a UUID is generated. Raises if the id
        already exists.
        """
        bid = belief_id if belief_id is not None else str(uuid.uuid4())
        if bid in self._beliefs:
            raise ValueError(f"belief id {bid!r} already exists")
        belief = Belief(
            id=bid,
            proposition=proposition,
            asserted_at=asserted_at,
            asserted_in_session=asserted_in_session,
            source_proposition_id=source_proposition_id,
            confidence=confidence,
            validity=validity if validity is not None else Validity(),
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
        self._append_event(EVENT_SUPERSEDE, old=old_id, new=new_id)

    def reinforce(self, existing_id: BeliefId, new_id: BeliefId) -> None:
        """Record that ``new_id`` reinforces ``existing_id``.

        Reinforcement means the user restated a belief consistent with an
        existing one — no contradiction, no supersession. Effect:
        - existing.reinforced_by += [new_id]
        - existing.confidence bumped toward 1.0 (capped)

        The new belief is stored as its own record, linked back to
        existing via the reinforced_by edge.
        """
        if existing_id == new_id:
            raise ValueError("a belief cannot reinforce itself")
        existing = self._require(existing_id)
        _ = self._require(new_id)  # validate
        if new_id not in existing.reinforced_by:
            existing.reinforced_by.append(new_id)
        # Confidence bump: simple multiplicative decay toward 1.0.
        # Each reinforcement closes 30% of the gap to 1.0.
        gap = 1.0 - existing.confidence
        existing.confidence = min(1.0, existing.confidence + 0.3 * gap)
        self._append_event(EVENT_REINFORCE, existing=existing_id, new=new_id)

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
            if new is not None and record["old"] not in new.supersedes:
                new.supersedes.append(record["old"])
        elif t == EVENT_REINFORCE:
            existing = self._beliefs.get(record["existing"])
            if existing is not None and record["new"] not in existing.reinforced_by:
                existing.reinforced_by.append(record["new"])
                gap = 1.0 - existing.confidence
                existing.confidence = min(1.0, existing.confidence + 0.3 * gap)
        elif t == EVENT_CONFIDENCE_SET:
            b = self._beliefs.get(record["belief"])
            if b is not None:
                b.confidence = float(record["confidence"])

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
    return Belief(
        id=d["id"],
        proposition=d["proposition"],
        asserted_at=datetime.fromisoformat(d["asserted_at"]),
        asserted_in_session=d["asserted_in_session"],
        source_proposition_id=d["source_proposition_id"],
        confidence=d.get("confidence", 1.0),
        validity=validity,
        supersedes=list(d.get("supersedes", [])),
        superseded_by=list(d.get("superseded_by", [])),
        reinforced_by=list(d.get("reinforced_by", [])),
    )
