"""Regression tests for the 'Demo: The Paper Leak' sample case: the answer key
(sample_leak_case_key.json) must only cite lines that really exist in the
sample chats, and the chats themselves must be fully parseable. Run:
pytest test_leak_case.py
"""
import json
from functools import lru_cache

import pytest

from parsers.whatsapp import MSG_RE, parse_whatsapp

KEY = json.load(open("sample_leak_case_key.json", encoding="utf-8"))
SOURCES = KEY["sources"]


def _walk_refs(node):
    """Every {src, at, who, quote} reference anywhere in the key."""
    if isinstance(node, dict):
        if {"src", "at", "who", "quote"} <= node.keys():
            yield node
        for value in node.values():
            yield from _walk_refs(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_refs(item)


@lru_cache(maxsize=None)
def _lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


REFS = list(_walk_refs(KEY))


def test_key_has_a_meaningful_number_of_references():
    ids = {c["id"] for c in KEY["planted_contradictions"]}
    assert len(ids) >= 15 and len(REFS) >= 60


@pytest.mark.parametrize("ref", REFS, ids=lambda r: f"{r['src']}|{r['at']}|{r['who']}")
def test_every_cited_line_exists(ref):
    prefix = f"{ref['at']} - {ref['who']}: "
    matches = [l for l in _lines(SOURCES[ref["src"]]) if l.startswith(prefix)]
    assert matches, f"no line from {ref['who']} at {ref['at']} in {SOURCES[ref['src']]}"
    assert any(ref["quote"] in l for l in matches), f"quote not found: {ref['quote']!r}"


@pytest.mark.parametrize("path", sorted(SOURCES.values()))
def test_sample_file_parses_completely_and_chronologically(path):
    lines = _lines(path)
    notes = []
    messages = parse_whatsapp(path, notes)
    # the parser silently drops non-matching lines, so a typo would vanish without this check
    assert len(messages) == len(lines) == sum(1 for l in lines if MSG_RE.match(l))
    assert notes == [], "dates should be settled by the data, not assumed"
    dates = [m["sent_at"] for m in messages]
    assert all(a <= b for a, b in zip(dates, dates[1:]))


def test_group_chat_is_big_and_has_nine_distinct_voices():
    messages = parse_whatsapp(SOURCES["group"])
    assert len(messages) >= 400
    assert {m["sender"] for m in messages} == set(KEY["cast"])
