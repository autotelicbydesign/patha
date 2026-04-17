"""Raw Archive Layer — immutable provenance substrate.

Stores the original turns/messages that propositions and beliefs were
derived from, with full metadata and stable IDs. Every proposition ID
and belief ID in the rest of Patha can be traced back to its raw
evidence through this layer.

Why this exists:
  - Audit trail: 'where did this belief come from?' → raw_archive.get(id)
    returns the original message, speaker, timestamp, session, source
  - Reversibility: if a belief is sublated incorrectly, we can inspect
    the raw source of both sides of the contradiction and override
  - Ground truth: propositions and beliefs are derived (they summarise,
    paraphrase, split). The raw archive is the source of truth that
    never changes

Design principles (per the v0.2/v0.3 discussion):
  - Append-only. RawTurns are never modified, only added.
  - Content-addressable IDs optional (sha256 of (session_id, turn_index,
    speaker, content)); explicit caller-supplied IDs also accepted.
  - JSONL persistence (same pattern as BeliefStore).
  - Bi-directional links: each raw_turn_id maps to a set of derived
    proposition_ids; each proposition_id maps back to a single
    raw_turn_id. (One turn can yield many propositions — e.g., a long
    utterance gets split into atomic claims.)

This is NOT a retrieval layer. Raw turns are not indexed for search —
that's Phase 1's job. The archive is a reference store for provenance.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path


# ─── Types ──────────────────────────────────────────────────────────

RawTurnId = str
PropositionId = str


@dataclass
class RawTurn:
    """An original conversational turn, verbatim.

    Attributes
    ----------
    id
        Stable unique id. Content-addressable by default (sha256-based
        prefix) so two different stores ingesting the same turn
        produce the same id, which makes cross-instance reference
        possible.
    session_id
        Opaque session identifier.
    turn_index
        Position within the session (0-indexed).
    speaker
        Who said it ('user', 'assistant', or any domain-specific label).
    content
        The verbatim text of the turn.
    timestamp
        When the turn happened (ingested or, if known, original).
    source_name
        Where the turn came from (e.g., 'slack-dm', 'journal-entry',
        'voice-memo-123'). Distinct from session_id — source_name is
        the channel/system; session_id groups consecutive turns.
    metadata
        Arbitrary extra provenance (URL, file path, attachment ids).
        Never used by the belief layer — purely for audit.
    derived_proposition_ids
        Proposition ids that were extracted from this turn. Updated as
        propositions are ingested.
    """

    id: RawTurnId
    session_id: str
    turn_index: int
    speaker: str
    content: str
    timestamp: datetime
    source_name: str = "unknown"
    metadata: dict[str, str] = field(default_factory=dict)
    derived_proposition_ids: list[PropositionId] = field(default_factory=list)


# ─── Store ──────────────────────────────────────────────────────────

_EVENT_ADD_TURN = "add_turn"
_EVENT_LINK_PROPOSITION = "link_proposition"


class RawArchive:
    """Immutable store of original turns with provenance links.

    Optional JSONL persistence replays cleanly from the event log.

    Content-addressable id generation: by default, the id is a
    sha256-prefix of (session_id, turn_index, speaker, content). This
    gives stable, collision-resistant IDs and supports cross-instance
    references.
    """

    def __init__(self, persistence_path: str | Path | None = None) -> None:
        self._turns: dict[RawTurnId, RawTurn] = {}
        self._by_session: dict[str, list[RawTurnId]] = {}
        self._proposition_to_turn: dict[PropositionId, RawTurnId] = {}
        self._persistence_path: Path | None = (
            Path(persistence_path) if persistence_path is not None else None
        )
        if self._persistence_path is not None and self._persistence_path.exists():
            self._replay()

    # ── mutations ──────────────────────────────────────────────────

    def add_turn(
        self,
        *,
        session_id: str,
        turn_index: int,
        speaker: str,
        content: str,
        timestamp: datetime,
        source_name: str = "unknown",
        metadata: dict[str, str] | None = None,
        raw_turn_id: RawTurnId | None = None,
    ) -> RawTurn:
        """Append a raw turn.

        ``raw_turn_id`` is auto-generated (content-addressable) unless
        provided. Attempting to add a turn with an id that already
        exists is idempotent — returns the existing RawTurn.
        """
        tid = raw_turn_id or _content_hash(
            session_id, turn_index, speaker, content
        )
        if tid in self._turns:
            return self._turns[tid]

        turn = RawTurn(
            id=tid,
            session_id=session_id,
            turn_index=turn_index,
            speaker=speaker,
            content=content,
            timestamp=timestamp,
            source_name=source_name,
            metadata=dict(metadata) if metadata else {},
        )
        self._turns[tid] = turn
        self._by_session.setdefault(session_id, []).append(tid)
        self._append_event(_EVENT_ADD_TURN, turn=turn)
        return turn

    def link_proposition(
        self, *, raw_turn_id: RawTurnId, proposition_id: PropositionId
    ) -> None:
        """Record that ``proposition_id`` was derived from ``raw_turn_id``.

        One raw turn can have many propositions (long utterances split
        into atomic claims). Each proposition has exactly one raw turn.

        Raises if raw_turn_id is unknown. Overwriting an existing
        proposition -> raw_turn mapping is rejected to prevent silent
        provenance errors.
        """
        if raw_turn_id not in self._turns:
            raise KeyError(f"unknown raw_turn_id: {raw_turn_id!r}")
        if proposition_id in self._proposition_to_turn:
            existing = self._proposition_to_turn[proposition_id]
            if existing != raw_turn_id:
                raise ValueError(
                    f"proposition {proposition_id!r} already linked to "
                    f"raw_turn {existing!r}; cannot rebind to {raw_turn_id!r}"
                )
            return
        turn = self._turns[raw_turn_id]
        turn.derived_proposition_ids.append(proposition_id)
        self._proposition_to_turn[proposition_id] = raw_turn_id
        self._append_event(
            _EVENT_LINK_PROPOSITION,
            raw_turn_id=raw_turn_id,
            proposition_id=proposition_id,
        )

    # ── queries ────────────────────────────────────────────────────

    def get_turn(self, raw_turn_id: RawTurnId) -> RawTurn | None:
        return self._turns.get(raw_turn_id)

    def turn_for_proposition(
        self, proposition_id: PropositionId
    ) -> RawTurn | None:
        """Return the raw turn that produced a given proposition, or None
        if the link hasn't been recorded."""
        tid = self._proposition_to_turn.get(proposition_id)
        return self._turns.get(tid) if tid else None

    def turns_by_session(self, session_id: str) -> list[RawTurn]:
        ids = self._by_session.get(session_id, [])
        return [self._turns[t] for t in ids if t in self._turns]

    def __len__(self) -> int:
        return len(self._turns)

    def __contains__(self, raw_turn_id: object) -> bool:
        return (
            isinstance(raw_turn_id, str) and raw_turn_id in self._turns
        )

    # ── persistence ────────────────────────────────────────────────

    def _append_event(self, event_type: str, **payload) -> None:
        if self._persistence_path is None:
            return
        self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"type": event_type, **self._serialise_payload(payload)}
        with open(self._persistence_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    @staticmethod
    def _serialise_payload(payload: dict) -> dict:
        out: dict = {}
        for k, v in payload.items():
            if isinstance(v, RawTurn):
                out[k] = {
                    **asdict(v),
                    "timestamp": v.timestamp.isoformat(),
                }
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
                rec = json.loads(line)
                self._apply_event(rec)

    def _apply_event(self, rec: dict) -> None:
        t = rec["type"]
        if t == _EVENT_ADD_TURN:
            data = rec["turn"]
            turn = RawTurn(
                id=data["id"],
                session_id=data["session_id"],
                turn_index=data["turn_index"],
                speaker=data["speaker"],
                content=data["content"],
                timestamp=datetime.fromisoformat(data["timestamp"]),
                source_name=data.get("source_name", "unknown"),
                metadata=dict(data.get("metadata", {})),
                derived_proposition_ids=list(data.get("derived_proposition_ids", [])),
            )
            self._turns[turn.id] = turn
            self._by_session.setdefault(turn.session_id, []).append(turn.id)
            for pid in turn.derived_proposition_ids:
                self._proposition_to_turn[pid] = turn.id
        elif t == _EVENT_LINK_PROPOSITION:
            tid = rec["raw_turn_id"]
            pid = rec["proposition_id"]
            turn = self._turns.get(tid)
            if turn is not None and pid not in turn.derived_proposition_ids:
                turn.derived_proposition_ids.append(pid)
                self._proposition_to_turn[pid] = tid


# ─── helpers ────────────────────────────────────────────────────────

def _content_hash(
    session_id: str, turn_index: int, speaker: str, content: str
) -> RawTurnId:
    """Content-addressable id for a raw turn.

    Uses sha256 of a canonical serialisation. Short prefix (16 chars
    hex = 64 bits) — collision probability vanishingly small for any
    realistic archive.
    """
    m = hashlib.sha256()
    m.update(session_id.encode())
    m.update(b"\x00")
    m.update(str(turn_index).encode())
    m.update(b"\x00")
    m.update(speaker.encode())
    m.update(b"\x00")
    m.update(content.encode())
    return "rt_" + m.hexdigest()[:16]
