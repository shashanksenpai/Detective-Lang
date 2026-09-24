"""Scoring rules for the attribution eval (BACKLOG A-1). Light on purpose - no ML or
database imports - so the arithmetic that decides whether one engine beats another is
unit-tested in milliseconds (test_eval_metrics.py).

A *record* is one held-out message run through an engine:
    {"true": "Yash", "ranking": [("Yash", 0.62), ("Parth", 0.21), ...], "uncertain": False}
`ranking` is best-first with probabilities summing to ~1 (engines report percentages
rounded to 0.1, so they are clipped at EPS before taking a log).
"""
import math
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

EPS = 5e-4                 # half of the 0.1% resolution the engines report
PLAUSIBLE_MASS = 0.90      # same "could be" definition as learned_engine.PLAUSIBLE_MASS


def record_from_result(true_sender: str, result: dict) -> dict:
    """Turns an engine's investigate() result into a record (percentages -> probabilities)."""
    return {
        "true": true_sender,
        "ranking": [(r["sender"], r["confidence"] / 100.0) for r in result["ranking"]],
        "uncertain": bool(result["uncertain"]),
    }


def _plausible(ranking: Sequence[Tuple[str, float]]) -> List[str]:
    """The fewest people, best first, whose probabilities add to PLAUSIBLE_MASS."""
    out, cum = [], 0.0
    for sender, p in ranking:
        out.append(sender)
        cum += p
        if cum >= PLAUSIBLE_MASS - 1e-9:
            break
    return out


def score(records: Sequence[dict]) -> Dict[str, float]:
    """Everything one run is judged on. `precision_when_confident` is None when the engine
    never committed; `log_loss` punishes a confident wrong answer far more than an unsure one."""
    n = len(records)
    if n == 0:
        raise ValueError("no records to score")
    top1 = top3 = committed = committed_right = set_hits = set_size = 0
    nll = 0.0
    for r in records:
        names = [s for s, _ in r["ranking"]]
        right = names[0] == r["true"]
        top1 += right
        top3 += r["true"] in names[:3]
        if not r["uncertain"]:
            committed += 1
            committed_right += right
        plausible = _plausible(r["ranking"])
        set_hits += r["true"] in plausible
        set_size += len(plausible)
        p_true = next((p for s, p in r["ranking"] if s == r["true"]), 0.0)
        nll += -math.log(max(p_true, EPS))
    return {
        "n": n, "top1": top1 / n, "top3": top3 / n, "log_loss": nll / n,
        "coverage": committed / n,
        "precision_when_confident": (committed_right / committed) if committed else None,
        "set_coverage": set_hits / n, "set_size": set_size / n,
    }


def paired_flips(a: Sequence[dict], b: Sequence[dict]) -> Tuple[int, int]:
    """(only A right, only B right) over the same messages in the same order - the paired
    view that says whether a gap between two engines is more than a different draw."""
    if len(a) != len(b):
        raise ValueError("paired comparison needs the same messages")
    only_a = only_b = 0
    for ra, rb in zip(a, b):
        if ra["true"] != rb["true"]:
            raise ValueError("paired comparison needs the same messages in the same order")
        right_a, right_b = ra["ranking"][0][0] == ra["true"], rb["ranking"][0][0] == rb["true"]
        only_a += right_a and not right_b
        only_b += right_b and not right_a
    return only_a, only_b


def per_sender_recall(records: Sequence[dict]) -> Dict[str, Tuple[int, int]]:
    """sender -> (right, total) for the messages that sender really wrote."""
    out: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for r in records:
        out[r["true"]][1] += 1
        out[r["true"]][0] += r["ranking"][0][0] == r["true"]
    return {s: (v[0], v[1]) for s, v in out.items()}


def abstention_curve(records: Sequence[dict], cutoffs=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9)) -> List[dict]:
    """If the engine only answered when its top probability was at least `cutoff`: how much
    of the time would it answer, and how often would it be right? (Independent of the
    engine's own uncertain flag - this is what its probabilities are worth.)"""
    rows = []
    for c in cutoffs:
        answered = [r for r in records if r["ranking"][0][1] >= c]
        right = sum(r["ranking"][0][0] == r["true"] for r in answered)
        rows.append({"cutoff": c, "answered": len(answered) / len(records),
                     "precision": (right / len(answered)) if answered else None})
    return rows
