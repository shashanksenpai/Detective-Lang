"""Sentiment scoring that understands Hinglish (Romanized Hindi mixed with
English) - replaces plain VADER as the app's sentiment source.

Why: VADER is an English lexicon. On real Hinglish text it finds no sentiment
word at all in ~44% of messages and catches only about 1 in 4 negative ones
(SentiMix test: 49% accuracy, 47% macro-F1; see CLAUDE.md and eval_sentiment.py).
The tone filters, mood timeline, dossier sentiment and the attribution engine's
emotion signal all read sentiment, so all of them were nearly blind to
negativity in Hinglish.

How: a small, fast, offline model - character and word n-gram TF-IDF (which
copes with Romanized Hindi's free spelling: acha/accha/achha, nahi/nhi) plus
VADER's own scores as extra inputs (so English keeps VADER's knowledge) -
trained by logistic regression on several domains at once (Hindi-English
tweets, Hinglish YouTube comments, English tweets, and a small set of
chat-register lines). Its output is deliberately VADER-shaped:
`polarity_scores(text)` returns neg / neu / pos / compound with compound in
[-1, 1], so every existing caller keeps working unchanged.

Two chat-specific choices, both found through error analysis on real chat
text: emoji that are ambiguous in chat ("😭" and "💀" can mean laughing or
panicking, "🥺" pleading or cute) are removed before scoring so the words
decide instead of the emoji, and letter-stretching ("yessss") is squashed.

If the model file is missing the analyzer falls back to plain VADER (with one
warning) so the app still runs; `analyzer.is_fallback` says which you have.
"""
import functools
import logging
import os
import re
from typing import Dict, List, Sequence

import numpy as np
import scipy.sparse as sp
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import hinglish_lexicon

log = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hinglish_sentiment_model.joblib")
# Bump when the input features change: a model file built with other features must not be loaded.
FEATURE_VERSION = 3   # 1 = n-grams + VADER; 2 = + Hinglish lexicon; 3 = all emoji stripped from text, curated emoji features
LABELS = ("negative", "neutral", "positive")

# Emoji whose polarity in chat depends on the words around them: they get NO polarity anywhere
# (they are simply not in hinglish_lexicon's emoji lists). Kept here for documentation and tests.
AMBIGUOUS_EMOJI = "\U0001F62D\U0001F480\U0001F97A\U0001F605\U0001F921\U0001FAE1\U0001F60C\U0001F64F\U0001F914\U0001F937\U0001F62C"
_AMBIGUOUS_RE = re.compile("[" + AMBIGUOUS_EMOJI + "]")
# Every emoji is removed from the text the n-gram model and VADER see (see hinglish_lexicon.EMOJI_*):
# their meaning in chat differs from tweets, so only a curated list may move the score.
_EMOJI_RE = re.compile("[🌀-🫿☀-➿⭐⭕️‍]")
_MENTION_RE = re.compile(r"@\s*\w+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_STRETCH_RE = re.compile(r"(.)\1{2,}")

_vader = SentimentIntensityAnalyzer()


def prepare_text(text: str) -> str:
    """Normalisation shared by training and scoring - they must never differ."""
    t = _MENTION_RE.sub(" @user ", text)
    t = _URL_RE.sub(" url ", t)
    t = _EMOJI_RE.sub(" ", t)
    t = _STRETCH_RE.sub(r"\1\1", t)
    return t.lower().strip()


def vader_features(texts: Sequence[str]) -> np.ndarray:
    """VADER's compound/pos/neg/neu (emoji removed), scaled x2 - the
    extra inputs that let the model keep VADER's English knowledge."""
    rows = []
    for t in texts:
        s = _vader.polarity_scores(_EMOJI_RE.sub(" ", t))
        rows.append([s["compound"], s["pos"], s["neg"], s["neu"]])
    return np.array(rows, dtype=float) * 2.0


def dense_features(texts: Sequence[str], emoji_ok: Sequence[bool] = None) -> np.ndarray:
    """VADER's scores (emoji stripped) plus the lexicon's word and emoji features, scaled x2.
    `emoji_ok` (training only) says which rows may teach the emoji features: emoji meaning is
    register-specific, so the model learns it from chat-register rows only; the others get 0."""
    lex = hinglish_lexicon.lexicon_features(prepare_text_list(texts), raw_texts=texts) * 2.0
    if emoji_ok is not None:
        lex[:, 2:] *= np.asarray(emoji_ok, dtype=float)[:, None]
    return np.hstack([vader_features(texts), lex])


def prepare_text_list(texts: Sequence[str]) -> List[str]:
    return [prepare_text(t) for t in texts]


def _features(vectorizer, texts: Sequence[str], emoji_ok: Sequence[bool] = None):
    return sp.hstack(
        [vectorizer.transform(prepare_text_list(texts)), sp.csr_matrix(dense_features(texts, emoji_ok))]
    ).tocsr()


def train_model(rows: Sequence[tuple], weights: Sequence[float] = None, emoji_ok: Sequence[bool] = None) -> dict:
    """Fits the vectoriser + classifier on [(text, label)] and returns the
    artifact dict that load/save use. Used by eval_sentiment.py --retrain."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_union

    texts = [t for t, _ in rows]
    vectorizer = make_union(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=120000),
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True,
                        token_pattern=r"(?u)\b\w+\b|[^\w\s]"),
    )
    vectorizer.fit([prepare_text(t) for t in texts])
    X = _features(vectorizer, texts, emoji_ok)
    classifier = LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced")
    classifier.fit(X, [label for _, label in rows], sample_weight=weights)
    return {"vectorizer": vectorizer, "classifier": classifier,
            "meta": {"trained_on": len(rows), "feature_version": FEATURE_VERSION}}


def save_model(artifact: dict, path: str = MODEL_PATH):
    import joblib
    joblib.dump(artifact, path, compress=3)


class HinglishSentiment:
    """VADER-shaped sentiment analyzer backed by the trained hybrid model."""

    is_fallback = False

    def __init__(self, artifact: dict):
        self._vectorizer = artifact["vectorizer"]
        self._classifier = artifact["classifier"]
        classes = list(self._classifier.classes_)
        self._neg, self._neu, self._pos = (classes.index(c) for c in LABELS)

    def polarity_scores_batch(self, texts: Sequence[str]) -> List[Dict[str, float]]:
        """neg/neu/pos are the class probabilities; compound = P(pos) - P(neg),
        so it spans -1..1 like VADER's compound."""
        if not texts:
            return []
        P = self._classifier.predict_proba(_features(self._vectorizer, texts))
        out = []
        for p in P:
            neg, neu, pos = float(p[self._neg]), float(p[self._neu]), float(p[self._pos])
            out.append({"neg": neg, "neu": neu, "pos": pos, "compound": pos - neg})
        return out

    @functools.lru_cache(maxsize=100_000)
    def _cached(self, text: str) -> tuple:
        s = self.polarity_scores_batch([text])[0]
        return (s["neg"], s["neu"], s["pos"], s["compound"])

    def polarity_scores(self, text: str) -> Dict[str, float]:
        neg, neu, pos, compound = self._cached(text)
        return {"neg": neg, "neu": neu, "pos": pos, "compound": compound}

    def explain(self, text: str, top: int = 4) -> Dict[str, list]:
        """Which pieces of the message pushed the score, for the 'why' behind a
        number: the n-grams (and VADER inputs) with the largest contribution
        toward positive and toward negative."""
        x = _features(self._vectorizer, [text])
        coef = self._classifier.coef_
        # contribution of each feature to (positive - negative) log-odds
        delta = coef[self._pos] - coef[self._neg]
        contrib = x.multiply(delta).tocoo()
        names = (list(self._vectorizer.get_feature_names_out())
                 + ["VADER compound", "VADER pos", "VADER neg", "VADER neu"] + hinglish_lexicon.FEATURE_NAMES)
        items = sorted(((float(v), names[j]) for j, v in zip(contrib.col, contrib.data) if abs(v) > 1e-6), reverse=True)
        clean = lambda n: re.sub(r"^tfidfvectorizer-\d+__", "", n)
        return {
            "toward_positive": [(clean(n), round(v, 3)) for v, n in items if v > 0][:top],
            "toward_negative": [(clean(n), round(v, 3)) for v, n in reversed(items) if v < 0][:top],
            # the lexicon words it recognised, with sign after negation ("acha nahi" -> negative)
            "lexicon": [(w, "positive" if v > 0 else "negative") for w, v in hinglish_lexicon.matches(prepare_text(text))],
        }


class _VaderFallback:
    """Plain VADER behind the same interface, used only if the model file is missing."""

    is_fallback = True

    def polarity_scores(self, text: str) -> Dict[str, float]:
        return _vader.polarity_scores(text)

    def polarity_scores_batch(self, texts: Sequence[str]) -> List[Dict[str, float]]:
        return [_vader.polarity_scores(t) for t in texts]

    def explain(self, text: str, top: int = 4) -> Dict[str, list]:
        return {"toward_positive": [], "toward_negative": [], "lexicon": []}


_analyzer = None


def get_analyzer():
    """The process-wide analyzer: the trained model if hinglish_sentiment_model.joblib
    is present, otherwise plain VADER (logged once)."""
    global _analyzer
    if _analyzer is None:
        artifact = None
        if os.path.exists(MODEL_PATH):
            import joblib
            artifact = joblib.load(MODEL_PATH)
            if artifact.get("meta", {}).get("feature_version") != FEATURE_VERSION:
                log.warning("%s was built with different features than this code expects - falling back to plain "
                            "VADER. Rebuild it with: python eval_sentiment.py --retrain", os.path.basename(MODEL_PATH))
                artifact = None
                _analyzer = _VaderFallback()
                return _analyzer
        if artifact is not None:
            _analyzer = HinglishSentiment(artifact)
        else:
            log.warning("%s not found - falling back to plain VADER (English only). "
                        "Build it with: python eval_sentiment.py --retrain", os.path.basename(MODEL_PATH))
            _analyzer = _VaderFallback()
    return _analyzer
