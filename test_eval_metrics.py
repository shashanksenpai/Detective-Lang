"""BACKLOG A-1: the arithmetic that decides whether one attribution engine beats another.
Light (no ML imports). Run: pytest test_eval_metrics.py
"""
import math

import pytest

from eval_metrics import EPS, abstention_curve, paired_flips, per_sender_recall, record_from_result, score


def rec(true, ranking, uncertain=False):
    return {"true": true, "ranking": ranking, "uncertain": uncertain}


A_FIRST = rec("A", [("A", 0.7), ("B", 0.2), ("C", 0.1)])                 # right, committed
B_SECOND = rec("A", [("B", 0.6), ("A", 0.3), ("C", 0.1)], uncertain=True)  # wrong, unsure, true one is 2nd
C_LAST = rec("A", [("B", 0.5), ("C", 0.3), ("D", 0.15), ("A", 0.05)])      # wrong and committed, true one 4th


def test_record_from_result_converts_percentages_to_probabilities():
    r = record_from_result("A", {"uncertain": 1, "ranking": [{"sender": "A", "confidence": 62.5}, {"sender": "B", "confidence": 37.5}]})
    assert r == {"true": "A", "ranking": [("A", 0.625), ("B", 0.375)], "uncertain": True}


def test_top1_top3_and_coverage():
    s = score([A_FIRST, B_SECOND, C_LAST])
    assert s["n"] == 3
    assert s["top1"] == pytest.approx(1 / 3)
    assert s["top3"] == pytest.approx(2 / 3)          # C_LAST has the truth 4th
    assert s["coverage"] == pytest.approx(2 / 3)      # two committed
    assert s["precision_when_confident"] == pytest.approx(1 / 2)  # of the two committed, one right


def test_precision_when_confident_is_none_if_it_never_commits():
    assert score([B_SECOND])["precision_when_confident"] is None
    assert score([B_SECOND])["coverage"] == 0.0


def test_log_loss_is_the_mean_negative_log_of_the_true_sender_probability():
    s = score([A_FIRST, B_SECOND])
    assert s["log_loss"] == pytest.approx((-math.log(0.7) - math.log(0.3)) / 2)


def test_log_loss_punishes_a_confident_wrong_answer_more_than_an_unsure_one():
    confident_wrong = rec("A", [("B", 0.97), ("A", 0.02), ("C", 0.01)])
    unsure_wrong = rec("A", [("B", 0.4), ("A", 0.35), ("C", 0.25)])
    assert score([confident_wrong])["log_loss"] > score([unsure_wrong])["log_loss"]


def test_log_loss_is_finite_when_the_true_sender_gets_zero_or_is_missing():
    zero = rec("A", [("B", 1.0), ("A", 0.0)])
    missing = rec("A", [("B", 1.0)])
    assert score([zero])["log_loss"] == pytest.approx(-math.log(EPS))
    assert score([missing])["log_loss"] == pytest.approx(-math.log(EPS))


def test_plausible_set_is_the_fewest_people_covering_90_percent():
    r = rec("B", [("A", 0.6), ("B", 0.32), ("C", 0.08)])      # 0.6 + 0.32 = 0.92 >= 0.9
    s = score([r])
    assert s["set_size"] == 2 and s["set_coverage"] == 1.0
    tight = rec("C", [("A", 0.95), ("B", 0.03), ("C", 0.02)])  # one person covers it - and it is not C
    s = score([tight])
    assert s["set_size"] == 1 and s["set_coverage"] == 0.0


def test_score_refuses_nothing_to_score():
    with pytest.raises(ValueError):
        score([])


def test_paired_flips_counts_who_is_alone_in_being_right():
    a = [rec("A", [("A", 0.9), ("B", 0.1)]), rec("A", [("B", 0.9), ("A", 0.1)]), rec("A", [("A", 0.5), ("B", 0.5)])]
    b = [rec("A", [("B", 0.9), ("A", 0.1)]), rec("A", [("B", 0.9), ("A", 0.1)]), rec("A", [("A", 0.6), ("B", 0.4)])]
    assert paired_flips(a, b) == (1, 0)
    assert paired_flips(b, a) == (0, 1)


def test_paired_flips_rejects_different_messages():
    with pytest.raises(ValueError):
        paired_flips([A_FIRST], [A_FIRST, A_FIRST])
    with pytest.raises(ValueError):
        paired_flips([rec("A", [("A", 1.0)])], [rec("B", [("A", 1.0)])])


def test_per_sender_recall():
    got = per_sender_recall([A_FIRST, B_SECOND, rec("B", [("B", 0.8), ("A", 0.2)])])
    assert got == {"A": (1, 2), "B": (1, 1)}


def test_abstention_curve_reports_answered_share_and_precision_per_cutoff():
    rows = abstention_curve([A_FIRST, B_SECOND, C_LAST], cutoffs=(0.5, 0.65, 0.99))
    assert rows[0] == {"cutoff": 0.5, "answered": pytest.approx(1.0), "precision": pytest.approx(1 / 3)}
    assert rows[1] == {"cutoff": 0.65, "answered": pytest.approx(1 / 3), "precision": pytest.approx(1.0)}
    assert rows[2] == {"cutoff": 0.99, "answered": 0.0, "precision": None}
