"""Eval fixture for identity-resolution's soft-match signal - Improvement
Stage (CLAUDE.md). Self-contained: uses only the permanent sample files (not
transient test cases), so this is a stable, rerunnable regression check -
not one-off manual testing that gets thrown away.

True positives: one real person's own messages split in half (same author,
disjoint message sets) - a standard authorship-verification technique.
True negatives: any two different real people, including across completely
unrelated demo cases - this is the exact scenario that exposed the original
bug (topic-embedding similarity scored unrelated people higher than a real
match).

Usage: python eval_identity.py
"""
import itertools
from collections import defaultdict

from identity_resolution import cosine_similarity, stylometric_fingerprint
from detective import embed_texts
from parsers.whatsapp import parse_whatsapp

SOURCES = [
    "sample_chat.txt", "sample_dm_riya_karan.txt",
    "sample_housemates.txt", "sample_dm_meera_dev.txt",
]

MIN_MESSAGES = 8


def _load_people():
    by_sender = defaultdict(list)
    for path in SOURCES:
        for m in parse_whatsapp(path):
            by_sender[m["sender"]].append(m["text"])
    return by_sender


def _embedding_centroid(texts):
    import numpy as np
    return np.mean(embed_texts(texts), axis=0)


def build_pairs(by_sender):
    eligible = {s: t for s, t in by_sender.items() if len(t) >= MIN_MESSAGES}

    true_positives = []  # (label, texts_a, texts_b)
    for sender, texts in eligible.items():
        mid = len(texts) // 2
        true_positives.append((sender, texts[:mid], texts[mid:]))

    true_negatives = []  # (label, texts_a, texts_b)
    for a, b in itertools.combinations(eligible, 2):
        true_negatives.append((f"{a} vs {b}", eligible[a], eligible[b]))

    return true_positives, true_negatives


def report(label_kind, pairs):
    print(f"\n=== {label_kind} ===")
    embedding_scores, stylometric_scores = [], []
    for label, texts_a, texts_b in pairs:
        emb = cosine_similarity(_embedding_centroid(texts_a), _embedding_centroid(texts_b))
        sty = cosine_similarity(stylometric_fingerprint(texts_a), stylometric_fingerprint(texts_b))
        embedding_scores.append(emb)
        stylometric_scores.append(sty)
        print(f"  {label:24s} embedding={emb:.3f}   stylometric={sty:.3f}")
    return embedding_scores, stylometric_scores


def main():
    by_sender = _load_people()
    tp_pairs, tn_pairs = build_pairs(by_sender)

    print(f"people with >= {MIN_MESSAGES} messages: {sorted(s for s in by_sender if len(by_sender[s]) >= MIN_MESSAGES)}")

    tp_emb, tp_sty = report("true positives (same person, split in half)", tp_pairs)
    tn_emb, tn_sty = report("true negatives (different real people)", tn_pairs)

    print("\n=== separation ===")
    print(f"embedding    - true positive range: {min(tp_emb):.3f}-{max(tp_emb):.3f}   "
          f"true negative range: {min(tn_emb):.3f}-{max(tn_emb):.3f}   "
          f"cleanly separated: {min(tp_emb) > max(tn_emb)}")
    print(f"stylometric  - true positive range: {min(tp_sty):.3f}-{max(tp_sty):.3f}   "
          f"true negative range: {min(tn_sty):.3f}-{max(tn_sty):.3f}   "
          f"cleanly separated: {min(tp_sty) > max(tn_sty)}")


if __name__ == "__main__":
    main()
