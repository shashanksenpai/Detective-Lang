"""Eval harness for the attribution engines - Improvement Stage (CLAUDE.md), BACKLOG A-1.

Per-case, per-sender train/test split: hold out ~20% of each sender's messages, build an
engine on the rest, and measure how well it attributes the held-out messages back to
their true sender. Every change to an engine gets compared against this - before it
existed, "accuracy" was pure eyeballing.

BACKLOG E-1: one split is one draw, and on a small case (18-21 test messages) a single
draw can swing by several points. So the split is repeated (default 5 times, each
reproducible) and every figure is reported as mean +- sd over the splits. Each sender's
split depends only on that sender (eval_split.py), so an unrelated change to the data no
longer re-draws everyone else's test set.

BACKLOG A-1: it now evaluates either engine (`--engine legacy|learned|both`; `both` runs them
on the very same splits and adds a paired comparison), and reports what an engine that says
"uncertain" also needs to be judged on: top-3, log-loss (how badly a wrong answer is
believed), the "could be" set, and what its probabilities are worth if it only answers above
a cut-off. Two further questions:
  --transfer   train on the group chats only, test on the DM messages: does a person's group
               voice carry into a private chat?
  --curve      cap the training messages per person (5, 10, 20, ...): how much chat does the
               engine need before it is any good?

Usage: python eval_attribution.py [case_id] [--engine ENGINE] [--splits N] [--transfer] [--curve]
No case_id given -> runs every case that has enough ready data. `--splits 1` is a quick
single-split run (fast, but with no noise estimate). Read-only: it never writes the database.
"""
import argparse
import random
from collections import defaultdict

from sqlmodel import Session, select

import engine_cache
from db import engine as db_engine
from engine_cache import load_case_sources
from eval_metrics import abstention_curve, paired_flips, per_sender_recall, record_from_result, score
from eval_split import MIN_MESSAGES_PER_SENDER, SEED, TEST_FRACTION, split_by_sender, summarize  # noqa: F401
from models import Case

DEFAULT_SPLITS = 5
CURVE_SIZES = (5, 10, 20, 40, 80, None)   # messages per person in training; None = all of them
TRANSFER_MIN_TRAIN = 5                    # a person needs this many group messages to be tested on their DMs


def _fmt_spread(values, n_splits):
    """'47.2% +- 3.1 (43.0%-51.4%)', or a plain figure with a warning for one split."""
    mean_, sd, lo, hi = summarize(values)
    if n_splits == 1:
        return f"{mean_:.1%} (single split - no noise estimate)"
    return f"{mean_:.1%} +- {sd * 100:.1f} ({lo:.1%}-{hi:.1%})"


def _fmt_plain(values, n_splits, scale=1.0, digits=2):
    mean_, sd, _, _ = summarize(values)
    return f"{mean_ * scale:.{digits}f}" if n_splits == 1 else f"{mean_ * scale:.{digits}f} +- {sd * scale:.{digits}f}"


def run_split(kind: str, train_messages, test_examples):
    """Builds engine `kind` on the training messages and runs every held-out message through it."""
    engine = engine_cache.new_engine(kind)
    engine.build(train_messages)
    return [record_from_result(true_sender, engine.investigate(text)) for text, true_sender in test_examples]


def _print_engine_block(kind, per_split_records, n_splits):
    """One engine's numbers over all splits (each entry of per_split_records is one split's records)."""
    scores = [score(recs) for recs in per_split_records]
    pooled = [r for recs in per_split_records for r in recs]
    print(f"\n--- engine: {kind} ---")
    print(f"top-1 accuracy (forced choice): {_fmt_spread([s['top1'] for s in scores], n_splits)}")
    print(f"top-3 accuracy: {_fmt_spread([s['top3'] for s in scores], n_splits)}")
    print(f"log-loss (lower is better; guessing uniformly scores ln(candidates)): {_fmt_plain([s['log_loss'] for s in scores], n_splits)}")
    print(f"coverage (engine willing to commit): {_fmt_spread([s['coverage'] for s in scores], n_splits)}")
    committed = [r for r in pooled if not r["uncertain"]]
    if committed:
        right = sum(r["ranking"][0][0] == r["true"] for r in committed)
        print(f"precision when confident (pooled over splits): {right}/{len(committed)} = {right / len(committed):.1%}")
    else:
        print("precision when confident: the engine never committed")
    print(f"'could be' set (fewest people covering 90%): holds the true sender {_fmt_spread([s['set_coverage'] for s in scores], n_splits)}, "
          f"{_fmt_plain([s['set_size'] for s in scores], n_splits, digits=1)} people wide")
    print("if it only answered above a cut-off (its probabilities' worth, pooled):  " + "   ".join(
        f"P>={row['cutoff']:.0%}: answers {row['answered']:.0%}, right {row['precision']:.0%}" if row["precision"] is not None
        else f"P>={row['cutoff']:.0%}: never" for row in abstention_curve(pooled)))
    if n_splits > 1:
        print("per-split top-1: " + "  ".join(f"{s['top1']:.1%}" for s in scores))
    print("per-sender recall (pooled over splits): " + "  ".join(
        f"{name} {right}/{total}" for name, (right, total) in sorted(per_sender_recall(pooled).items())))
    return scores


def evaluate_case(case_id: int, case_name: str, n_splits: int = DEFAULT_SPLITS, kinds=("legacy",)):
    with Session(db_engine) as session:
        sources = load_case_sources(session, case_id)

    runs = {k: [] for k in kinds}
    skipped, total, n_senders = [], 0, 0
    for split_index in range(n_splits):
        train_messages, test_examples, skipped = split_by_sender(sources, split_index)
        if not test_examples:
            print(f"[{case_name}] not enough data to evaluate (every sender below {MIN_MESSAGES_PER_SENDER} messages)")
            return None
        total, n_senders = len(test_examples), len({sender for _, sender in test_examples})
        for k in kinds:
            runs[k].append(run_split(k, train_messages, test_examples))

    print(f"\n=== {case_name} (case {case_id}) ===")
    print(f"test examples per split: {total} (from {n_senders} senders), {n_splits} split{'s' if n_splits != 1 else ''}")
    if skipped:
        print(f"skipped (too few messages to hold out): {skipped}")

    results = {}
    for k in kinds:
        results[k] = _print_engine_block(k, runs[k], n_splits)
    if len(kinds) == 2:
        a, b = kinds
        only_a = only_b = 0
        for ra, rb in zip(runs[a], runs[b]):
            x, y = paired_flips(ra, rb)
            only_a, only_b = only_a + x, only_b + y
        print(f"\npaired ({n_splits * total} messages): only {a} right {only_a}, only {b} right {only_b}"
              f"  (a gap this small in flips is noise; a gap of a few dozen is not)")

    summary = {"case_id": case_id, "case_name": case_name, "total": total, "n_splits": n_splits, "engines": {}}
    for k in kinds:
        sc = results[k]
        acc_mean, acc_sd, _, _ = summarize([s["top1"] for s in sc])
        cov_mean, cov_sd, _, _ = summarize([s["coverage"] for s in sc])
        summary["engines"][k] = {"accuracy": acc_mean, "accuracy_sd": acc_sd, "coverage": cov_mean, "coverage_sd": cov_sd}
    return summary


def transfer_examples(sources):
    """(group training messages, DM test examples) for the cross-context question, or None when the
    case has no group chat or no DM. A DM message is only tested when its sender wrote at least
    TRANSFER_MIN_TRAIN group messages - otherwise there is nothing to have learned."""
    group = [m for s in sources if s["context"] == "group" for m in s["messages"]]
    dms = [m for s in sources if s["context"] == "dm" for m in s["messages"]]
    if not group or not dms:
        return None
    seen = defaultdict(int)
    for m in group:
        seen[m["sender"]] += 1
    tests = [(m["text"], m["sender"]) for m in dms if seen[m["sender"]] >= TRANSFER_MIN_TRAIN]
    return (group, tests) if tests else None


def evaluate_transfer(case_id: int, case_name: str, kinds):
    with Session(db_engine) as session:
        sources = load_case_sources(session, case_id)
    got = transfer_examples(sources)
    if got is None:
        print(f"[{case_name}] transfer test needs both a group chat and DMs - skipped")
        return
    group, tests = got
    print(f"\n=== {case_name} (case {case_id}): train on the group chat ({len(group)} msgs), test on {len(tests)} DM messages ===")
    majority = max(sum(1 for _, s in tests if s == who) for who in {s for _, s in tests}) / len(tests)
    print(f"(always guessing the most frequent DM sender would score {majority:.1%})")
    for k in kinds:
        recs = run_split(k, group, tests)
        s = score(recs)
        cov = f"commits on {s['coverage']:.0%}" + (f", right {s['precision_when_confident']:.0%} of those" if s["precision_when_confident"] is not None else "")
        print(f"  {k:8} top-1 {s['top1']:.1%}  top-3 {s['top3']:.1%}  log-loss {s['log_loss']:.2f}  {cov}")
        print("           per person: " + "  ".join(f"{n} {r}/{t}" for n, (r, t) in sorted(per_sender_recall(recs).items())))


def evaluate_curve(case_id: int, case_name: str, n_splits: int, kinds):
    with Session(db_engine) as session:
        sources = load_case_sources(session, case_id)
    print(f"\n=== {case_name} (case {case_id}): top-1 vs messages per person in training ({n_splits} splits) ===")
    for k in kinds:
        row = []
        for cap in CURVE_SIZES:
            accs = []
            for split_index in range(n_splits):
                train, test, _ = split_by_sender(sources, split_index)
                if not test:
                    return
                if cap is not None:
                    by = defaultdict(list)
                    for m in train:
                        by[m["sender"]].append(m)
                    rng = random.Random(f"curve:{cap}:{split_index}")
                    train = []
                    for sender in sorted(by):
                        msgs = by[sender][:]
                        rng.shuffle(msgs)
                        train.extend(msgs[:cap])
                accs.append(score(run_split(k, train, test))["top1"])
            m_, sd, _, _ = summarize(accs)
            row.append(f"{'all' if cap is None else cap}: {m_:.1%}" + ("" if n_splits == 1 else f" +-{sd * 100:.1f}"))
        print(f"  {k:8} " + "   ".join(row))


def main():
    parser = argparse.ArgumentParser(description="Evaluate the attribution engines on held-out messages.")
    parser.add_argument("case_id", nargs="?", type=int, help="one case (default: every case)")
    parser.add_argument("--engine", choices=list(engine_cache.ENGINE_KINDS) + ["both"], default=None,
                        help=f"which engine (default: the app's, currently {engine_cache.default_engine_kind()}; "
                             "'both' runs them on the same splits and compares)")
    parser.add_argument("--splits", type=int, default=DEFAULT_SPLITS,
                        help=f"how many train/test splits to average over (default {DEFAULT_SPLITS}; 1 = quick)")
    parser.add_argument("--transfer", action="store_true", help="train on group chats, test on DMs")
    parser.add_argument("--curve", action="store_true", help="accuracy vs messages per person in training")
    args = parser.parse_args()
    if args.splits < 1:
        parser.error("--splits must be at least 1")

    engine_arg = args.engine or engine_cache.default_engine_kind()
    kinds = ("legacy", "learned") if engine_arg == "both" else (engine_arg,)

    with Session(db_engine) as session:
        all_cases = {c.id: c.name for c in session.exec(select(Case)).all()}
    case_ids = [args.case_id] if args.case_id is not None else list(all_cases)

    if args.transfer or args.curve:
        for case_id in case_ids:
            name = all_cases.get(case_id, f"case {case_id}")
            if args.transfer:
                evaluate_transfer(case_id, name, kinds)
            if args.curve:
                evaluate_curve(case_id, name, args.splits, kinds)
        return

    results = []
    for case_id in case_ids:
        r = evaluate_case(case_id, all_cases.get(case_id, f"case {case_id}"), args.splits, kinds)
        if r:
            results.append(r)

    if len(results) > 1:
        print(f"\n=== summary (mean +- sd over {args.splits} split{'s' if args.splits != 1 else ''}) ===")
        for r in results:
            for k, e in r["engines"].items():
                print(f"  {r['case_name']:22s} {k:8s} accuracy={e['accuracy']:.1%} +- {e['accuracy_sd'] * 100:.1f}"
                      f"  coverage={e['coverage']:.1%} +- {e['coverage_sd'] * 100:.1f}  (n={r['total']}/split)")


if __name__ == "__main__":
    main()
