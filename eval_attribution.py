"""Eval harness for the attribution engine - Improvement Stage (CLAUDE.md).

Per-case, per-sender train/test split: hold out ~20% of each sender's
messages, build a DetectiveEngine on the rest, and measure how well it
attributes the held-out messages back to their true sender. This is the
baseline every future change to signals/weights/thresholds gets compared
against - before this existed, "accuracy" was pure eyeballing.

BACKLOG E-1: one split is one draw, and on a small case (18-21 test messages)
a single draw can swing by several points. So the split is repeated (default 5
times, each reproducible) and every figure is reported as mean +- sd over the
splits. Each sender's split depends only on that sender (eval_split.py), so an
unrelated change to the data - dropping a few media placeholders once moved the
paper-leak figure by 3-7 points - no longer re-draws everyone else's test set.

Usage: python eval_attribution.py [case_id] [--splits N]
No case_id given -> runs every case that has enough ready data. `--splits 1` is
a quick single-split run (fast, but with no noise estimate).
"""
import argparse
from collections import defaultdict

from sqlmodel import Session, select

from db import engine as db_engine
from detective import DetectiveEngine
from engine_cache import load_case_sources
from eval_split import MIN_MESSAGES_PER_SENDER, SEED, TEST_FRACTION, split_by_sender, summarize  # noqa: F401
from models import Case

DEFAULT_SPLITS = 5


def _score_split(engine, test_examples, pooled):
    """Runs one split's held-out messages through the engine. Adds each
    sender's tp/fp/fn/support into `pooled` and returns this split's counts."""
    correct = confident = correct_when_confident = 0
    for text, true_sender in test_examples:
        result = engine.investigate(text)
        predicted = result["ranking"][0]["sender"]
        is_correct = predicted == true_sender

        pooled[true_sender]["support"] += 1
        if is_correct:
            correct += 1
            pooled[true_sender]["tp"] += 1
        else:
            pooled[true_sender]["fn"] += 1
            pooled[predicted]["fp"] += 1

        if not result["uncertain"]:
            confident += 1
            if is_correct:
                correct_when_confident += 1
    return {"total": len(test_examples), "correct": correct, "confident": confident,
            "correct_when_confident": correct_when_confident}


def _fmt_spread(values, n_splits):
    """'47.2% +- 3.1 (43.0%-51.4%)', or a plain figure with a warning for one split."""
    mean_, sd, lo, hi = summarize(values)
    if n_splits == 1:
        return f"{mean_:.1%} (single split - no noise estimate)"
    return f"{mean_:.1%} +- {sd * 100:.1f} ({lo:.1%}-{hi:.1%})"


def evaluate_case(case_id: int, case_name: str, n_splits: int = DEFAULT_SPLITS):
    with Session(db_engine) as session:
        sources = load_case_sources(session, case_id)

    runs = []
    pooled = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0})
    skipped = []
    for split_index in range(n_splits):
        train_messages, test_examples, skipped = split_by_sender(sources, split_index)
        if not test_examples:
            print(f"[{case_name}] not enough data to evaluate (every sender below {MIN_MESSAGES_PER_SENDER} messages)")
            return None
        engine = DetectiveEngine()
        engine.build(train_messages)
        runs.append(_score_split(engine, test_examples, pooled))

    total = runs[0]["total"]
    accuracies = [r["correct"] / r["total"] for r in runs]
    coverages = [r["confident"] / r["total"] for r in runs]
    confident = sum(r["confident"] for r in runs)
    correct_when_confident = sum(r["correct_when_confident"] for r in runs)

    print(f"\n=== {case_name} (case {case_id}) ===")
    print(f"test examples per split: {total} (from {len({s for s, p in pooled.items() if p['support']})} senders), "
          f"{n_splits} split{'s' if n_splits != 1 else ''}")
    if skipped:
        print(f"skipped (too few messages to hold out): {skipped}")
    print(f"top-1 accuracy (forced choice): {_fmt_spread(accuracies, n_splits)}")
    print(f"coverage (engine willing to commit): {_fmt_spread(coverages, n_splits)}")
    if confident:
        print(f"precision when confident (pooled over splits): {correct_when_confident}/{confident} "
              f"= {correct_when_confident / confident:.1%}")
    else:
        print("precision when confident: the engine never committed")
    if n_splits > 1:
        print("per-split accuracy: " + "  ".join(f"{a:.1%}" for a in accuracies))

    print("per-sender (pooled over splits):")
    for sender, stats in sorted(pooled.items()):
        tp, fp, support = stats["tp"], stats["fp"], stats["support"]
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / support if support else float("nan")
        print(f"  {sender:12s} support={support:3d}  precision={precision:.1%}  recall={recall:.1%}")

    acc_mean, acc_sd, _, _ = summarize(accuracies)
    cov_mean, cov_sd, _, _ = summarize(coverages)
    return {"case_id": case_id, "case_name": case_name, "total": total, "n_splits": n_splits,
            "accuracy": acc_mean, "accuracy_sd": acc_sd, "coverage": cov_mean, "coverage_sd": cov_sd,
            "confident": confident, "correct_when_confident": correct_when_confident}


def main():
    parser = argparse.ArgumentParser(description="Evaluate the attribution engine on held-out messages.")
    parser.add_argument("case_id", nargs="?", type=int, help="one case (default: every case)")
    parser.add_argument("--splits", type=int, default=DEFAULT_SPLITS,
                        help=f"how many train/test splits to average over (default {DEFAULT_SPLITS}; 1 = quick)")
    args = parser.parse_args()
    if args.splits < 1:
        parser.error("--splits must be at least 1")

    with Session(db_engine) as session:
        all_cases = {c.id: c.name for c in session.exec(select(Case)).all()}

    case_ids = [args.case_id] if args.case_id is not None else list(all_cases)

    results = []
    for case_id in case_ids:
        name = all_cases.get(case_id, f"case {case_id}")
        r = evaluate_case(case_id, name, args.splits)
        if r:
            results.append(r)

    if len(results) > 1:
        print(f"\n=== summary (mean +- sd over {args.splits} split{'s' if args.splits != 1 else ''}) ===")
        for r in results:
            print(f"  {r['case_name']:22s} accuracy={r['accuracy']:.1%} +- {r['accuracy_sd'] * 100:.1f}"
                  f"  coverage={r['coverage']:.1%} +- {r['coverage_sd'] * 100:.1f}  (n={r['total']}/split)")


if __name__ == "__main__":
    main()
