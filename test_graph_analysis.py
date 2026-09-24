"""Tests for the investigation-board data rules in graph_analysis.py: what
counts as an exchange, and when a tone may be asserted. Run: pytest test_graph_analysis.py
"""
from graph_analysis import (
    TENSE_BELOW, TONE_MIN_EXCHANGES, WARM_ABOVE, _adjacent_exchange_texts, tone_for,
)


def _src(*pairs):
    return {"messages": [{"sender": s, "text": t} for s, t in pairs]}


def test_exchange_needs_a_different_sender_replying_right_after():
    sources = [_src(("A", "hi"), ("A", "again"), ("B", "hello"), ("A", "hey"))]
    ex = _adjacent_exchange_texts(sources)
    # A->B ("hello") and B->A ("hey"); A->A is not an exchange
    assert ex == {frozenset({"A", "B"}): ["hello", "hey"]}


def test_a_source_boundary_is_not_a_reply():
    # The last message of chat 1 (A) and the first of chat 2 (C) were never adjacent in time or place.
    chat1 = _src(("A", "one"), ("B", "two"), ("A", "three"))
    chat2 = _src(("C", "four"), ("D", "five"))
    ex = _adjacent_exchange_texts([chat1, chat2])
    assert frozenset({"A", "C"}) not in ex
    assert set(ex) == {frozenset({"A", "B"}), frozenset({"C", "D"})}


def test_thin_evidence_is_never_called_a_tone():
    assert tone_for(-0.9, TONE_MIN_EXCHANGES - 1) == "unclear"
    assert tone_for(0.9, 2) == "unclear"
    assert tone_for(None, 50) == "unclear"


def test_tone_cutoffs_once_there_is_enough_evidence():
    n = TONE_MIN_EXCHANGES
    assert tone_for(WARM_ABOVE + 0.01, n) == "warm"
    assert tone_for(WARM_ABOVE, n) == "neutral"          # boundary is exclusive
    assert tone_for(TENSE_BELOW, n) == "neutral"
    assert tone_for(TENSE_BELOW - 0.01, n) == "tense"
    assert tone_for(0.0, n) == "neutral"
