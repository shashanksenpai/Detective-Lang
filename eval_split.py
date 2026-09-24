"""Train/test splitting and summary statistics for the attribution eval
(BACKLOG E-1). Light on purpose: no ML or database imports, so the split rules
are unit-tested in milliseconds (test_eval_split.py).

Why each sender has their own random generator: eval_attribution.py used to
shuffle every sender's messages with ONE shared random.Random(42), in turn. That
made the held-out set of every sender depend on the *length of every earlier
sender's list*, so changing one sender's messages (dropping four media
placeholders did it) silently re-drew everyone else's test set - the paper-leak
top-1 moved from 44.1% to 47.2% (model) and 46.9% to 53.5% (fallback) with no
change to the engine. Now a sender's split depends only on (seed, split index,
that sender's name and messages).
"""
import random
from collections import defaultdict
from statistics import mean, stdev
from typing import Dict, List, Sequence, Tuple

SEED = 42
TEST_FRACTION = 0.2
MIN_MESSAGES_PER_SENDER = 6


def split_by_sender(
    sources,
    split_index: int = 0,
    seed: int = SEED,
    test_fraction: float = TEST_FRACTION,
    min_messages: int = MIN_MESSAGES_PER_SENDER,
) -> Tuple[List[dict], List[Tuple[str, str]], List[Tuple[str, int]]]:
    """Per-sender hold-out. `sources` is engine_cache.load_case_sources' shape.
    Returns (train_messages, test_examples, skipped): train_messages are
    {"sender", "text"} in the order the engine should see them, test_examples
    are (text, true_sender), and skipped lists (sender, count) for senders with
    fewer than `min_messages` (kept in training, never evaluated on).

    Each sender is shuffled with random.Random(f"{seed}:{split_index}:{sender}"),
    so the result is reproducible, independent of every other sender, and a
    different `split_index` gives a different, equally reproducible split.
    """
    by_sender: Dict[str, List[str]] = defaultdict(list)
    for src in sources:
        for m in src["messages"]:
            by_sender[m["sender"]].append(m["text"])

    train_messages: List[dict] = []
    test_examples: List[Tuple[str, str]] = []
    skipped: List[Tuple[str, int]] = []

    for sender, texts in by_sender.items():
        if len(texts) < min_messages:
            skipped.append((sender, len(texts)))
            # Still a candidate sender for others' predictions, just not
            # evaluated on directly - too few messages to hold any out.
            train_messages.extend({"sender": sender, "text": t} for t in texts)
            continue

        shuffled = texts[:]
        random.Random(f"{seed}:{split_index}:{sender}").shuffle(shuffled)
        n_test = max(1, int(len(shuffled) * test_fraction))
        train_messages.extend({"sender": sender, "text": t} for t in shuffled[n_test:])
        test_examples.extend((t, sender) for t in shuffled[:n_test])

    return train_messages, test_examples, skipped


def summarize(values: Sequence[float]) -> Tuple[float, float, float, float]:
    """(mean, sample standard deviation, min, max). One value has sd 0.0 - a
    single split says nothing about its own noise, which callers should say."""
    values = list(values)
    if not values:
        raise ValueError("nothing to summarize")
    sd = stdev(values) if len(values) > 1 else 0.0
    return mean(values), sd, min(values), max(values)
