"""BACKLOG A-1: the learned attribution engine's guarantees - it learns who says what, its
explanation is exact, and it will not commit when it cannot do better than guessing.
Light: scikit-learn only, no sentence-transformer model. Run: pytest test_learned_engine.py
"""
import math
import random

import numpy as np
import pytest

from learned_engine import LearnedEngine, choose_commit_threshold, wilson_lower

SIGNATURES = {
    "Asha": "dont worry guys ill pay treat bill chill covered".split(),
    "Bharat": "notes slides deadline submit rubric syllabus chapter exam".split(),
    "Chitra": "lol omg haha bruh literally vibes cringe meme".split(),
}
SHARED = "the a is to and it we so ok yeah no yes see you me".split()


def _chat(per_sender=40, seed=1, signatures=SIGNATURES):
    """Each person's messages mix their own words with common ones."""
    rng = random.Random(seed)
    out = []
    for sender, vocab in signatures.items():
        for _ in range(per_sender):
            words = rng.sample(vocab, 3) + rng.sample(SHARED, 2)
            rng.shuffle(words)
            out.append({"sender": sender, "text": " ".join(words)})
    rng.shuffle(out)
    return out


@pytest.fixture(scope="module")
def engine():
    return LearnedEngine().build(_chat(per_sender=100))   # 300 messages: enough for it to check itself and commit


def _confidences(result):
    return {r["sender"]: r["confidence"] for r in result["ranking"]}


# ---- the statistics ---------------------------------------------------------------

def test_wilson_lower_bound_is_pessimistic_about_few_checks():
    assert wilson_lower(0, 0) == 0.0
    assert wilson_lower(10, 10) < 0.9                   # ten for ten is not proof of 90%
    assert wilson_lower(400, 400) > 0.99
    assert wilson_lower(8, 10) < wilson_lower(80, 100) < 0.8   # same 80%, more checks, tighter bound
    assert wilson_lower(90, 100) == pytest.approx(0.8548, abs=0.001)   # worked by hand: (0.9082 - 0.0393) / 1.0164


# ---- it learns -------------------------------------------------------------------

def test_names_the_person_whose_words_these_are(engine):
    r = engine.investigate("dont worry guys ill pay the bill")
    assert r["ranking"][0]["sender"] == "Asha"
    assert not r["uncertain"]
    assert r["plausible_senders"] == ["Asha"]
    r = engine.investigate("submit the exam notes by the deadline")
    assert r["ranking"][0]["sender"] == "Bharat"


def test_apostrophes_and_case_do_not_change_who_it_names(engine):
    for text in ("Don't worry guys I'll pay", "dont worry guys ill pay", "DONT WORRY GUYS ILL PAY"):
        conf = _confidences(engine.investigate(text))
        assert max(conf, key=conf.get) == "Asha", text


def test_confidences_are_a_distribution_and_ranking_is_best_first(engine):
    r = engine.investigate("lol omg the meme")
    values = [x["confidence"] for x in r["ranking"]]
    assert values == sorted(values, reverse=True)
    assert sum(values) == pytest.approx(100, abs=0.3)
    assert {x["sender"] for x in r["ranking"]} == set(SIGNATURES)
    assert r["engine"] == "learned"


def test_build_is_deterministic():
    a = LearnedEngine().build(_chat()).investigate("bruh literally the notes")
    b = LearnedEngine().build(_chat()).investigate("bruh literally the notes")
    assert a == b


# ---- the explanation is exact ----------------------------------------------------

@pytest.mark.parametrize("sentence", [
    "dont worry guys ill pay the bill",
    "Don't worry guys, I'll pay!! sooo",
    "lol the deadline",
    "the",
])
def test_evidence_adds_up_to_the_log_odds_the_engine_reports(engine, sentence):
    r = engine.investigate(sentence)
    ev = r["evidence"]
    probs = dict(zip(engine.model.classes, engine.probabilities([sentence])[0]))
    exact = math.log(probs[ev["for"]] / probs[ev["against"]])
    parts = sum(t["log_odds"] for t in ev["terms"]) + ev["other_terms"] + ev["base_rate"]
    assert parts == pytest.approx(exact, abs=0.02)
    assert ev["log_odds"] == pytest.approx(exact, abs=0.02)


def test_evidence_names_the_words_that_pointed_to_the_top_match(engine):
    ev = engine.investigate("dont worry guys ill pay the bill")["evidence"]
    assert ev["for"] == "Asha"
    helpful = {t["text"] for t in ev["terms"] if t["log_odds"] > 0}
    assert helpful & {"pay", "bill", "worry", "dont", "guys"}


# ---- it will not commit when it cannot do better than guessing -------------------

def test_a_message_with_nothing_it_knows_falls_back_to_who_talks_most():
    """Not one of its words or letter patterns is in the chat (a new script, say): a linear model's
    output would be just its intercepts, an artefact - so the answer is the base rate and uncertain."""
    chat = _chat() + [{"sender": "Asha", "text": "dont worry"}] * 20     # Asha now talks the most
    nothing_known = ("", "   ", "नमस्ते", "你好世界", "🧿🧿")
    by_volume = LearnedEngine(class_weight=None).build(chat)
    for text in nothing_known:
        r = by_volume.investigate(text)
        assert r["uncertain"] and r["evidence"] is None
        assert r["ranking"][0]["sender"] == "Asha"
        assert r["ranking"][0]["confidence"] == pytest.approx(100 * 60 / 140, abs=0.1)
        assert len(r["plausible_senders"]) >= 2
        assert any("who talks most" in n for n in r["notes"])
    equal = LearnedEngine().build(chat)      # default: every person equally likely, so volume is no clue
    for text in nothing_known:
        r = equal.investigate(text)
        assert r["uncertain"] and [x["confidence"] for x in r["ranking"]] == [33.3, 33.3, 33.3]
        assert any("nothing to go on" in n for n in r["notes"])


def test_ascii_gibberish_is_never_a_commitment(engine):
    """It shares two-letter fragments with everything, so it is not 'nothing known' - but it must not commit."""
    r = engine.investigate("qzxv wjkl")
    assert r["uncertain"]


def test_a_weak_message_is_below_the_commit_line(engine):
    """'the a is' is all shared filler: whatever the model leans towards, it is not a commitment."""
    r = engine.investigate("the a is")
    assert r["uncertain"]
    assert any("Below the" in n for n in r["notes"])


def test_people_who_sound_alike_never_get_a_commitment():
    """Everyone draws from the same words, so held-out checks are near chance: no confidence level
    reaches the target precision, so the engine must say so instead of committing to a guess."""
    same = {name: SHARED for name in ("Asha", "Bharat", "Chitra")}
    rng = random.Random(3)
    msgs = [{"sender": n, "text": " ".join(rng.choices(SHARED, k=5))} for n in same for _ in range(100)]
    e = LearnedEngine().build(msgs)
    assert e.calibration["calibrated"] is True
    assert e.commit_threshold is None
    for text in ("the a is to and", "yeah no yes see you"):
        r = e.investigate(text)
        assert r["uncertain"]
        assert any("no confidence level" in n for n in r["notes"])


def test_calibration_is_reported(engine):
    c = engine.calibration
    assert c["calibrated"] is True
    assert 0.5 <= c["commit_threshold"] <= 1       # never below "more likely than everyone else together"
    assert 0.5 <= c["temperature"] <= 4.0
    assert c["checked_precision_at_threshold"] >= c["target_precision"] - 1e-9
    assert c["checked_messages"] == 300


# ---- small and odd inputs ---------------------------------------------------------

def test_with_too_little_chat_to_trust_its_confidence_it_ranks_but_never_commits():
    """120 messages is enough to fit and to check itself, but not to trust a confidence cut-off (needs ~250)."""
    e = LearnedEngine().build(_chat(per_sender=40))
    assert e.calibration["calibrated"] is True and e.commit_threshold is None
    r = e.investigate("dont worry guys ill pay the bill")
    assert r["ranking"][0]["sender"] == "Asha" and r["uncertain"]
    assert any("needs about 250" in n for n in r["notes"])


def test_too_few_messages_to_calibrate_says_so_and_still_answers():
    e = LearnedEngine().build(_chat(per_sender=8))
    assert e.calibration["calibrated"] is False
    r = e.investigate("dont worry guys ill pay the bill")
    assert r["ranking"][0]["sender"] == "Asha"
    assert any("not calibrated" in n for n in r["notes"])


def test_two_people_work_and_the_evidence_still_adds_up():
    e = LearnedEngine().build(_chat(signatures={k: SIGNATURES[k] for k in ("Asha", "Bharat")}))
    sentence = "dont worry guys ill pay the bill"
    r = e.investigate(sentence)
    assert r["ranking"][0]["sender"] == "Asha"
    assert sum(x["confidence"] for x in r["ranking"]) == pytest.approx(100, abs=0.2)
    ev = r["evidence"]
    probs = dict(zip(e.model.classes, e.probabilities([sentence])[0]))
    assert ev["log_odds"] == pytest.approx(math.log(probs["Asha"] / probs["Bharat"]), abs=0.02)


def test_one_person_is_reported_as_nothing_to_tell_apart():
    e = LearnedEngine().build([{"sender": "Solo", "text": f"message {i}"} for i in range(20)])
    assert not e.model and e.is_ready
    r = e.investigate("anything")
    assert r["uncertain"] and r["evidence"] is None
    assert any("one person" in n for n in r["notes"])


def test_an_empty_build_is_not_ready():
    assert not LearnedEngine().build([]).is_ready
    assert not LearnedEngine().build([{"sender": "A", "text": "   "}]).is_ready


def test_unusual_text_does_not_break_it(engine):
    for text in ("", "   ", "😂😂😂", "<img src=x onerror=alert(1)>", "a" * 5000, "ｄｏｎｔ　ｗｏｒｒｙ", "हिन्दी me kya"):
        r = engine.investigate(text)
        assert len(r["ranking"]) == 3 and sum(x["confidence"] for x in r["ranking"]) == pytest.approx(100, abs=0.3)


def test_context_is_accepted_but_reported_as_unused(engine):
    plain = engine.investigate("dont worry guys ill pay the bill")
    with_ctx = engine.investigate("dont worry guys ill pay the bill", context=[{"sender": "Bharat", "text": "who pays?"}])
    assert with_ctx["context_used"] == 0
    assert any("were not used" in n for n in with_ctx["notes"])
    assert with_ctx["ranking"] == plain["ranking"]


def test_unweighted_classes_option_runs():
    e = LearnedEngine(class_weight=None).build(_chat())
    assert e.investigate("dont worry guys ill pay the bill")["ranking"][0]["sender"] == "Asha"


def test_senders_with_almost_no_messages_are_still_candidates():
    msgs = _chat() + [{"sender": "Dev", "text": "hello there"}]
    e = LearnedEngine().build(msgs)
    assert "Dev" in {x["sender"] for x in e.investigate("hello there")["ranking"]}


# ---- the commit rule, on its own ---------------------------------------------------

def _checks(rows):
    """rows: [(confidence, right?), ...] -> the two arrays choose_commit_threshold takes."""
    return np.array([c for c, _ in rows]), np.array([bool(h) for _, h in rows])


def test_the_commit_line_admits_the_reliable_checks_and_stops_before_the_unreliable_tail():
    # 200 checks: the 100 most confident (>= 0.8) all right, the next 100 (0.5-0.8) right only 60% of the time.
    # Taken all together that is 80% - and 160 right of 200 cannot rule out a true rate below 80%.
    rows = [(0.99 - 0.0019 * i, True) for i in range(100)] + [(0.79 - 0.0029 * i, i % 5 < 3) for i in range(100)]
    top, hit = _checks(rows)
    threshold, answered, right = choose_commit_threshold(top, hit, target=0.8, min_support=20)
    assert threshold is not None and threshold > top.min()      # it does not commit to everything
    assert 100 <= answered < 200 and right / answered >= 0.8


def test_a_line_proven_on_few_checks_is_not_trusted():
    """25 right out of 30 is 83% - above the 80% target - but 30 checks cannot rule out a true rate under 80%."""
    top, hit = _checks([(0.9, i < 25) for i in range(30)])
    assert choose_commit_threshold(top, hit, target=0.8, min_support=20)[0] is None
    top, hit = _checks([(0.9, i < 290) for i in range(300)])         # the same 97% on 300 checks is
    assert choose_commit_threshold(top, hit, target=0.8, min_support=20)[0] == pytest.approx(0.9)


def test_a_line_reached_by_only_a_handful_of_checks_is_not_trusted():
    top, hit = _checks([(0.95, True)] * 10)
    assert choose_commit_threshold(top, hit, target=0.8, min_support=20)[0] is None


def test_checks_with_the_same_confidence_are_admitted_together_or_not_at_all():
    """Duplicate messages ('ok', 'haha') get identical confidences. A cut-off may not be justified by the first
    few of a tie group while the whole group is what it lets through."""
    rows = [(0.95, True)] * 20 + [(0.6, i % 2 == 0) for i in range(200)]     # the 0.6 group is a coin flip
    top, hit = _checks(rows)
    assert choose_commit_threshold(top, hit, target=0.8, min_support=20) == (0.95, 20, 20)


def test_it_never_commits_below_fifty_percent():
    """Held-out checks that are right at only 45-99% stated confidence still do not license naming one
    person who is less likely than everyone else put together: the line is 0.5 even though 0.45 was reliable."""
    rows = [(0.45 + 0.54 * i / 299, True) for i in range(300)]
    top, hit = _checks(rows)
    threshold, answered, right = choose_commit_threshold(top, hit, target=0.8, min_support=20)
    assert threshold == 0.5 and right == answered and 250 < answered < 300
    top, hit = _checks([(0.4, True)] * 300)                 # nothing reaches 50%: there is no line at all
    assert choose_commit_threshold(top, hit, target=0.8, min_support=20)[0] is None


def test_nothing_reliable_means_no_line():
    top, hit = _checks([(0.6, i % 2 == 0) for i in range(300)])
    assert choose_commit_threshold(top, hit, target=0.8, min_support=20) == (None, 0, 0)
