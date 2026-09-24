"""BACKLOG E-1: the attribution eval's split must not depend on unrelated senders.
Light (no ML imports). Run: pytest test_eval_split.py
"""
import random
from collections import Counter, defaultdict

import pytest

from eval_split import MIN_MESSAGES_PER_SENDER, SEED, split_by_sender, summarize


def _sources(**counts):
    """One source; each sender gets `n` distinct messages 'name-0', 'name-1', ..."""
    messages = []
    for name, n in counts.items():
        messages.extend({"sender": name, "text": f"{name}-{i}"} for i in range(n))
    return [{"label": "chat", "context": "group", "messages": messages}]


def _held_out(test_examples, sender):
    return sorted(t for t, s in test_examples if s == sender)


def _legacy_split(sources, test_fraction=0.2, min_messages=MIN_MESSAGES_PER_SENDER):
    """The pre-E-1 algorithm, kept here only to show the test below can fail: ONE
    generator shared by every sender, consumed in turn."""
    by_sender = defaultdict(list)
    for src in sources:
        for m in src["messages"]:
            by_sender[m["sender"]].append(m["text"])
    rng = random.Random(SEED)
    test = []
    for sender, texts in by_sender.items():
        if len(texts) < min_messages:
            continue
        shuffled = texts[:]
        rng.shuffle(shuffled)
        n_test = max(1, int(len(shuffled) * test_fraction))
        test.extend((t, sender) for t in shuffled[:n_test])
    return test


def test_a_split_is_reproducible():
    a = split_by_sender(_sources(Anu=30, Bao=25, Cyn=40))
    b = split_by_sender(_sources(Anu=30, Bao=25, Cyn=40))
    assert a == b


@pytest.mark.parametrize("anu_size", [24, 26, 28, 29, 31, 35])
def test_changing_one_sender_never_redraws_another_senders_held_out_set(anu_size):
    before = split_by_sender(_sources(Anu=30, Bao=25, Cyn=40))[1]
    # Anu gains or loses messages and a brand-new sender appears first: nobody else's split may move
    after = split_by_sender(_sources(New=12, Anu=anu_size, Bao=25, Cyn=40))[1]
    assert _held_out(before, "Bao") == _held_out(after, "Bao")
    assert _held_out(before, "Cyn") == _held_out(after, "Cyn")


def test_the_old_shared_generator_did_redraw_other_senders():
    """Shows the property above is real by having the legacy design fail it. A single
    removed message can leave the shared generator at the same position by chance (shuffle
    consumes a variable number of random words), so it is enough that SOME size re-draws."""
    base = _legacy_split(_sources(Anu=30, Bao=25, Cyn=40))
    redrawn = [
        size for size in range(20, 40) if size != 30
        and any(_held_out(base, who) != _held_out(_legacy_split(_sources(Anu=size, Bao=25, Cyn=40)), who)
                for who in ("Bao", "Cyn"))
    ]
    assert redrawn, "the legacy design never re-drew another sender - the E-1 premise would be wrong"
    assert len(redrawn) >= 5                      # and not a fluke: it is the common case


def test_a_different_split_index_gives_a_different_split():
    s0 = split_by_sender(_sources(Anu=40), split_index=0)[1]
    s1 = split_by_sender(_sources(Anu=40), split_index=1)[1]
    assert _held_out(s0, "Anu") != _held_out(s1, "Anu")


def test_train_and_test_partition_each_senders_messages():
    sources = _sources(Anu=30, Bao=25)
    train, test, skipped = split_by_sender(sources)
    assert skipped == []
    for name, n in (("Anu", 30), ("Bao", 25)):
        got = Counter([m["text"] for m in train if m["sender"] == name] + [t for t, s in test if s == name])
        assert got == Counter(f"{name}-{i}" for i in range(n))


def test_about_a_fifth_is_held_out():
    _, test, _ = split_by_sender(_sources(Anu=30, Bao=6))
    assert len(_held_out(test, "Anu")) == 6           # int(30 * 0.2)
    assert len(_held_out(test, "Bao")) == 1           # int(6 * 0.2)


def test_at_least_one_message_is_always_held_out():
    # with a small fraction int(6 * 0.1) would be 0; the floor of one is what keeps the sender evaluable
    _, test, _ = split_by_sender(_sources(Bao=6), test_fraction=0.1)
    assert len(_held_out(test, "Bao")) == 1


def test_a_sender_with_too_few_messages_is_skipped_but_stays_in_training():
    train, test, skipped = split_by_sender(_sources(Anu=30, Tiny=3))
    assert skipped == [("Tiny", 3)]
    assert _held_out(test, "Tiny") == []
    assert sorted(m["text"] for m in train if m["sender"] == "Tiny") == ["Tiny-0", "Tiny-1", "Tiny-2"]


def test_summarize():
    mean_, sd, lo, hi = summarize([0.4, 0.5, 0.6])
    assert mean_ == pytest.approx(0.5) and sd == pytest.approx(0.1) and (lo, hi) == (0.4, 0.6)
    assert summarize([0.5]) == (0.5, 0.0, 0.5, 0.5)     # one split says nothing about its own noise
    with pytest.raises(ValueError):
        summarize([])
