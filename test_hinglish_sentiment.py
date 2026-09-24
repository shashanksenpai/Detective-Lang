"""Tests for the Hinglish sentiment scorer (hinglish_sentiment.py). Uses only
files in the repo plus the trained model - no downloads. The benchmark comparison
(SentiMix, YouTube, English) lives in eval_sentiment.py. Run: pytest test_hinglish_sentiment.py

The trained model is built locally, not committed (see README). Tests marked
@needs_model check its behaviour and are SKIPPED until you run
`python eval_sentiment.py --retrain`. A model file that exists but loads as the
VADER fallback (built with different features) is still a failure, not a skip.
"""
import json
import os

import pytest

import hinglish_sentiment as hs

A = hs.get_analyzer()

# Skip (don't fail) on a fresh clone where the model was never built.
needs_model = pytest.mark.skipif(
    not os.path.exists(hs.MODEL_PATH),
    reason="hinglish_sentiment_model.joblib not built - run: python eval_sentiment.py --retrain",
)

# Sentences the model is known to get wrong. They are kept (not deleted) so the gap stays visible;
# when a retrain fixes one, pytest reports it as XPASS and the marker should come off. See BACKLOG F-21.
KNOWN_MISS = pytest.mark.xfail(reason="known miss of the Hinglish model", strict=False)


def score(text):
    return A.polarity_scores(text)["compound"]


def label(text):
    """The model's own decision rule (also what eval_sentiment.py uses): its most probable class."""
    s = A.polarity_scores(text)
    return ("negative", "neutral", "positive")[max(range(3), key=lambda i: (s["neg"], s["neu"], s["pos"])[i])]


@needs_model
def test_the_trained_model_is_present_not_the_fallback():
    # Only reached when the file exists (needs_model skips otherwise): a fallback here means it was
    # built with different features than this code expects (hinglish_sentiment.FEATURE_VERSION).
    assert not A.is_fallback, "hinglish_sentiment_model.joblib was built with different features - rebuild: python eval_sentiment.py --retrain"


def test_output_is_vader_shaped():
    s = A.polarity_scores("bahut acha laga yaar")
    assert set(s) == {"neg", "neu", "pos", "compound"}
    assert all(0.0 <= s[k] <= 1.0 for k in ("neg", "neu", "pos"))
    assert abs(s["neg"] + s["neu"] + s["pos"] - 1.0) < 1e-6
    assert abs(s["compound"] - (s["pos"] - s["neg"])) < 1e-9
    assert -1.0 <= s["compound"] <= 1.0


@pytest.mark.parametrize("text", [
    "bahut acha laga yaar",                 # VADER scored all of these exactly 0.0
    "yaar aaj ka din ekdum mast tha",
    "shukriya dost, tere bina yeh possible nahi tha",
    pytest.param("wah kya match tha, jeet gaye", marks=KNOWN_MISS),
    pytest.param("mera din ban gaya tera message dekh ke", marks=KNOWN_MISS),
])
@needs_model
def test_plain_hinglish_positive_is_read_as_positive(text):
    assert label(text) == "positive"
    assert score(text) > 0.1


@pytest.mark.parametrize("text", [
    "yeh bilkul acha nahi hai",             # negation comes AFTER the word in Hindi
    "bakwaas hai sab",
    "mera mood ekdum kharab hai yaar",
    "mujhe bahut gussa aa raha hai",        # was scored +0.06 before the Hinglish lexicon
    "main bahut udaas hoon aaj",
    "main tumse naraaz hoon, samjhe",
])
@needs_model
def test_plain_hinglish_negative_is_read_as_negative(text):
    assert label(text) == "negative"
    assert score(text) < -0.1


@needs_model
def test_the_clearest_cases_are_read_with_conviction():
    assert score("bahut acha laga yaar") > 0.4
    assert score("yeh bilkul acha nahi hai") < -0.3


@needs_model
def test_hindi_style_negation_after_the_word_lowers_the_reading():
    # (plain "yeh acha hai" itself is scored slightly negative today - a known weakness listed in BACKLOG)
    assert score("yeh acha nahi hai") < score("yeh acha hai") - 0.15
    assert score("yeh acha nahi hai") < -0.1


@needs_model
def test_plain_hinglish_information_is_neutral():
    for text in ("kal subah milte hain station pe", "main abhi ghar pahunch gaya", "wo file bhej dena please"):
        assert label(text) == "neutral"


@pytest.mark.parametrize("text,sign", [
    ("i'm so proud of myself, i actually did it", 1),
    ("thanks for saying that, i needed it", 1),
    ("i feel terrible, like i'm completely failing at this", -1),
    ("this is stressing me out so much", -1),
])
@needs_model
def test_english_is_not_lost(text, sign):
    assert score(text) * sign > 0.25


@needs_model
def test_ambiguous_chat_emoji_do_not_decide_the_score():
    # 💀 and 😭 mean laughing or panicking depending on the words; they must not move the score
    base = "bro who signs on a tuesday evening"
    assert score(base + " \U0001F62D\U0001F62D") == pytest.approx(score(base))
    assert score(base + " \U0001F480") == pytest.approx(score(base))


@pytest.mark.parametrize("text,sign", [
    ("😂😂😂", 1),          # a bare laughing reaction was read as -0.61 when tweet-trained n-grams saw the emoji
    ("go get it!! 🔥", 1),                   # VADER itself reads the fire emoji as negative
    ("so proud of you ❤️", 1),
    ("worst day ever 😡", -1),
    ("mera dil toot gaya 😢", -1),
])
@needs_model
def test_unambiguous_chat_emoji_carry_their_chat_meaning(text, sign):
    assert score(text) * sign > 0.4


@needs_model
def test_acknowledgement_emoji_are_not_a_polarity():
    assert abs(score("noted 👍")) < 0.4
    assert label("ok 👍") != "negative"


@needs_model
def test_letter_stretching_is_squashed():
    assert score("yessssss finally") == pytest.approx(score("yess finally"))


def test_batch_matches_one_at_a_time():
    texts = ["thank you so much", "yeh bakwaas hai", "kal 10 baje station pe milte hain", ""]
    batch = A.polarity_scores_batch(texts)
    for t, b in zip(texts, batch):
        assert b["compound"] == pytest.approx(A.polarity_scores(t)["compound"])
    assert A.polarity_scores_batch([]) == []


@needs_model
def test_explain_says_which_pieces_pushed_the_score():
    why = A.explain("bakwaas movie thi, bilkul acha nahi laga")
    assert why["toward_negative"], "expected negative contributors"
    assert all(v < 0 for _, v in why["toward_negative"]) and all(v > 0 for _, v in why["toward_positive"])
    assert all(isinstance(name, str) and name for name, _ in why["toward_negative"])


def test_falls_back_to_plain_vader_if_the_model_file_is_missing(monkeypatch):
    monkeypatch.setattr(hs, "MODEL_PATH", os.path.join(os.path.dirname(__file__), "no_such_model.joblib"))
    monkeypatch.setattr(hs, "_analyzer", None)
    fallback = hs.get_analyzer()
    assert fallback.is_fallback
    assert fallback.polarity_scores("this is great")["compound"] == pytest.approx(hs._vader.polarity_scores("this is great")["compound"])
    assert len(fallback.polarity_scores_batch(["a", "b"])) == 2
    monkeypatch.setattr(hs, "_analyzer", None)   # don't leak the fallback into later tests


@needs_model
def test_beats_vader_on_the_blind_labeled_chat_messages():
    """All 300 hand-labeled chat messages (none of them in the training data)."""
    with open(os.path.join(os.path.dirname(__file__), "sample_hinglish_chat_labels.json"), encoding="utf-8") as f:
        msgs = json.load(f)["messages"]
    texts = [m["text"] for m in msgs]
    truth = [m["label"] for m in msgs]
    model_pred = []
    for s in A.polarity_scores_batch(texts):
        model_pred.append(("negative", "neutral", "positive")[max(range(3), key=lambda i: (s["neg"], s["neu"], s["pos"])[i])])
    vader_pred = []
    for t in texts:
        c = hs._vader.polarity_scores(t)["compound"]
        vader_pred.append("negative" if c <= -0.05 else "positive" if c >= 0.05 else "neutral")
    acc = lambda pred: sum(p == t for p, t in zip(pred, truth)) / len(truth)
    assert acc(model_pred) > acc(vader_pred)
    assert acc(model_pred) > 0.60
