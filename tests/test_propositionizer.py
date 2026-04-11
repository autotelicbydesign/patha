"""Tests for the rule-based propositionizer."""

from __future__ import annotations

import pytest

from patha.chunking.propositionizer import propositionize


def test_empty_input_returns_empty():
    assert propositionize("", session_id="s1", turn_idx=0) == []
    assert propositionize("   \n  ", session_id="s1", turn_idx=0) == []


def test_single_sentence_yields_single_prop():
    props = propositionize("Hello world.", session_id="s1", turn_idx=0)
    assert len(props) == 1
    assert props[0].text == "Hello world."
    assert props[0].session_id == "s1"
    assert props[0].turn_idx == 0
    assert props[0].prop_idx == 0


def test_multiple_sentences_split_cleanly():
    props = propositionize("Hello world. Goodbye world.", session_id="s1", turn_idx=0)
    assert [p.text for p in props] == ["Hello world.", "Goodbye world."]
    assert [p.prop_idx for p in props] == [0, 1]


def test_abbreviations_do_not_trigger_sentence_split():
    props = propositionize("Dr. Smith arrived. He was late.", session_id="s", turn_idx=0)
    texts = [p.text for p in props]
    assert texts == ["Dr. Smith arrived.", "He was late."]


def test_decimal_numbers_do_not_trigger_sentence_split():
    props = propositionize(
        "The rate is 3.14 percent. That is accurate.",
        session_id="s",
        turn_idx=0,
    )
    texts = [p.text for p in props]
    assert texts == ["The rate is 3.14 percent.", "That is accurate."]


def test_semicolon_splits_clauses():
    props = propositionize("I went home; she stayed behind.", session_id="s", turn_idx=0)
    texts = [p.text for p in props]
    assert len(texts) == 2
    assert "I went home" in texts[0]
    assert "she stayed behind" in texts[1]


def test_conjunction_split_fires_on_comma():
    props = propositionize(
        "I went to the store, and I bought milk.",
        session_id="s",
        turn_idx=0,
    )
    texts = [p.text for p in props]
    assert len(texts) == 2
    assert texts[0] == "I went to the store"
    assert "bought milk" in texts[1]
    assert not texts[1].lower().startswith("and ")


def test_conjunction_without_comma_does_not_split():
    """No comma = not clearly independent; keep as one proposition."""
    props = propositionize(
        "I went to the store and bought milk.",
        session_id="s",
        turn_idx=0,
    )
    assert len(props) == 1


def test_bullet_list_splits_items_with_lead_in():
    text = "My favorites:\n- apples\n- oranges\n- bananas"
    props = propositionize(text, session_id="s", turn_idx=0)
    texts = [p.text for p in props]
    assert len(texts) == 4
    assert "favorites" in texts[0]
    assert texts[1] == "apples"
    assert texts[2] == "oranges"
    assert texts[3] == "bananas"


def test_numbered_list_splits_items():
    text = "Steps:\n1. First\n2. Second\n3. Third"
    props = propositionize(text, session_id="s", turn_idx=0)
    assert len(props) == 4
    assert props[1].text == "First"
    assert props[2].text == "Second"
    assert props[3].text == "Third"


def test_prop_indices_are_sequential_within_turn():
    props = propositionize("A sentence. B sentence. C sentence.", session_id="s", turn_idx=5)
    assert [p.prop_idx for p in props] == [0, 1, 2]
    assert all(p.turn_idx == 5 for p in props)


def test_output_is_deterministic():
    text = "Hello world. I went to the store, and I bought milk."
    a = propositionize(text, session_id="s", turn_idx=0)
    b = propositionize(text, session_id="s", turn_idx=0)
    assert a == b


def test_speaker_and_timestamp_metadata_propagate():
    props = propositionize(
        "Hi. How are you?",
        session_id="s1",
        turn_idx=3,
        speaker="alice",
        timestamp="2026-04-10T14:00:00Z",
    )
    assert len(props) == 2
    assert all(p.speaker == "alice" for p in props)
    assert all(p.timestamp == "2026-04-10T14:00:00Z" for p in props)


@pytest.mark.parametrize(
    ("text", "expected_count"),
    [
        ("Single.", 1),
        ("Two. Sentences.", 2),
        ("One; two; three.", 3),
        ("Hi!", 1),
        ("Really? Yes.", 2),
    ],
)
def test_parametric_counts(text: str, expected_count: int):
    props = propositionize(text, session_id="s", turn_idx=0)
    assert len(props) == expected_count
