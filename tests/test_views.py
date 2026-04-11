"""Tests for the Vedic patha view constructors."""

from __future__ import annotations

import pytest

from patha.chunking.propositionizer import Proposition
from patha.chunking.views import VIEW_NAMES, build_views


def _props(*texts: str) -> list[Proposition]:
    return [
        Proposition(text=t, session_id="s", turn_idx=0, prop_idx=i)
        for i, t in enumerate(texts)
    ]


def test_all_seven_views_present_per_proposition():
    views = build_views(_props("alpha"))
    assert len(views) == 1
    assert set(views[0].keys()) == set(VIEW_NAMES)


def test_single_proposition_all_views_degrade_to_text():
    views = build_views(_props("alpha"))[0]
    for name in VIEW_NAMES:
        assert views[name] == "alpha"


def test_krama_forward_pair():
    views = build_views(_props("alpha", "beta"))
    assert views[0]["v2"] == "alpha beta"
    # no next after the last prop -> degrade to self
    assert views[1]["v2"] == "beta"


def test_reverse_krama():
    views = build_views(_props("alpha", "beta"))
    # no prev before the first prop -> degrade to self
    assert views[0]["v3"] == "alpha"
    assert views[1]["v3"] == "alpha beta"


def test_jata_bidirectional_triple():
    views = build_views(_props("alpha", "beta", "gamma"))
    assert views[0]["v4"] == "alpha beta"  # no prev
    assert views[1]["v4"] == "alpha beta gamma"
    assert views[2]["v4"] == "beta gamma"  # no next


def test_ghana_prepends_entities_to_jata():
    views = build_views(
        _props("alpha", "beta", "gamma"),
        entities=[["Alice"], ["Bob"], []],
    )
    assert "Alice" in views[0]["v5"]
    assert "beta" in views[1]["v5"] and "Bob" in views[1]["v5"]
    # no entities -> degrade to v4
    assert views[2]["v5"] == views[2]["v4"]


def test_reframed_uses_first_listed_entity_as_dominant():
    views = build_views(_props("alpha"), entities=[["Alice", "Bob"]])
    assert views[0]["v6"] == "fact about Alice: alpha"


def test_reframed_degrades_without_entities():
    views = build_views(_props("alpha"))
    assert views[0]["v6"] == "alpha"


def test_temporal_anchoring():
    views = build_views(
        _props("alpha", "beta"),
        timestamps=["2026-04-10", None],
    )
    assert views[0]["v7"] == "2026-04-10: alpha"
    # None timestamp -> degrade to plain proposition
    assert views[1]["v7"] == "beta"


def test_entities_length_mismatch_raises():
    with pytest.raises(ValueError, match="entities length"):
        build_views(_props("a", "b"), entities=[["X"]])


def test_timestamps_length_mismatch_raises():
    with pytest.raises(ValueError, match="timestamps length"):
        build_views(_props("a", "b"), timestamps=["2026-04-10"])


def test_empty_input_yields_empty_output():
    assert build_views([]) == []


def test_views_are_deterministic():
    props = _props("alpha", "beta", "gamma")
    a = build_views(props, entities=[["A"], ["B"], ["C"]], timestamps=["t1", "t2", "t3"])
    b = build_views(props, entities=[["A"], ["B"], ["C"]], timestamps=["t1", "t2", "t3"])
    assert a == b


def test_schema_is_constant_across_enrichment_states():
    """Every proposition must have all 7 keys regardless of what enrichment was supplied."""
    plain = build_views(_props("alpha"))
    with_ents = build_views(_props("alpha"), entities=[["A"]])
    with_ts = build_views(_props("alpha"), timestamps=["t"])
    full = build_views(_props("alpha"), entities=[["A"]], timestamps=["t"])
    for result in (plain, with_ents, with_ts, full):
        assert set(result[0].keys()) == set(VIEW_NAMES)
