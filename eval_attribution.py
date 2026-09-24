"""Eval harness for the attribution engine - Improvement Stage (CLAUDE.md).

Per-case, per-sender train/test split: hold out ~20% of each sender's
messages, build a DetectiveEngine on the rest, and measure how well it
attributes the held-out messages back to their true sender. This is the
baseline every future change to signals/weights/thresholds gets compared
against - before this existed, "accuracy" was pure eyeballing.

Usage: python eval_attribution.py [case_id]
No case_id given -> runs every case that has enough ready data.
"""
import random
import sys
from collections import defaultdict

from sqlmodel import Session, select

from db import engine as db_engine
from detective import DetectiveEngine
from engine_cache import load_case_sources
from models import Case

SEED = 42
TEST_FRACTION = 0.2
MIN_MESSAGES_PER_SENDER = 6


def _split_by_sender(sources):
    by_sender = defaultdict(list)
    for src in sources:
        for m in src["messages"]:
            by_sender[m["sender"]].append(m["text"])

    train_messages = []
    test_examples = []  # (text, true_sender)
    skipped = []

    rng = random.Random(SEED)
    for sender, texts in by_sender.items():
        if len(texts) < MIN_MESSAGES_PER_SENDER:
            skipped.append((sender, len(texts)))
            # Still counts as a candidate sender for others' predictions,
            # just not evaluated on directly - too few messages to hold any out.
            train_messages.extend({"sender": sender, "text": t} for t in texts)
            continue

        shuffled = texts[:]
        rng.shuffle(shuffled)
        n_test = max(1, int(len(shuffled) * TEST_FRACTION))
        test_texts = shuffled[:n_test]
        train_texts = shuffled[n_test:]

        train_messages.extend({"sender": sender, "text": t} for t in train_texts)
        test_examples.extend((t, sender) for t in test_texts)

    return train_messages, test_examples, skipped


def evaluate_case(case_id: int, case_name: str):
    with Session(db_engine) as session:
        sources = load_case_sources(session, case_id)

    train_messages, test_examples, skipped = _split_by_sender(sources)

    if not test_examples:
        print(f"[{case_name}] not enough data to evaluate (every sender below {MIN_MESSAGES_PER_SENDER} messages)")
        return None

    engine = DetectiveEngine()
    engine.build(train_messages)

    correct = 0
    correct_when_confident = 0
    confident_count = 0
    per_sender = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0})

    for text, true_sender in test_examples:
        result = engine.investigate(text)
        predicted = result["ranking"][0]["sender"]
        is_correct = predicted == true_sender

        per_sender[true_sender]["support"] += 1
        if is_correct:
            correct += 1
            per_sender[true_sender]["tp"] += 1
        else:
            per_sender[true_sender]["fn"] += 1
            per_sender[predicted]["fp"] += 1

        if not result["uncertain"]:
            confident_count += 1
            if is_correct:
                correct_when_confident += 1

    total = len(test_examples)
    print(f"\n=== {case_name} (case {case_id}) ===")
    print(f"test examples: {total} (from {len({s for _, s in test_examples})} senders)")
    if skipped:
        print(f"skipped (too few messages to hold out): {skipped}")
    print(f"top-1 accuracy (forced choice): {correct}/{total} = {correct / total:.1%}")
    print(f"coverage (engine willing to commit): {confident_count}/{total} = {confident_count / total:.1%}")
    if confident_count:
        print(
            f"precision when confident: {correct_when_confident}/{confident_count} "
            f"= {correct_when_confident / confident_count:.1%}"
        )

    print("per-sender:")
    for sender, stats in sorted(per_sender.items()):
        tp, fp, support = stats["tp"], stats["fp"], stats["support"]
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / support if support else float("nan")
        print(f"  {sender:12s} support={support:3d}  precision={precision:.1%}  recall={recall:.1%}")

    return {"case_id": case_id, "case_name": case_name, "total": total,
            "accuracy": correct / total, "coverage": confident_count / total}


def main():
    with Session(db_engine) as session:
        all_cases = {c.id: c.name for c in session.exec(select(Case)).all()}

    case_ids = [int(sys.argv[1])] if len(sys.argv) > 1 else list(all_cases)

    results = []
    for case_id in case_ids:
        name = all_cases.get(case_id, f"case {case_id}")
        r = evaluate_case(case_id, name)
        if r:
            results.append(r)

    if len(results) > 1:
        print("\n=== summary ===")
        for r in results:
            print(f"  {r['case_name']:20s} accuracy={r['accuracy']:.1%}  coverage={r['coverage']:.1%}")


if __name__ == "__main__":
    main()
