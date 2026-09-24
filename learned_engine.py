"""Learned speaker-attribution engine (BACKLOG A-1).

The legacy engine (detective.py) blends six hand-weighted similarity signals; on
the nine-person paper-leak case it never commits, and it could not name Yash for
"Dont worry guys I'll pay" (his habitual reassure-and-pay line) because "pay" and
"worry" sat in none of its word lists. This engine *learns* each person's texting
behaviour from their own messages instead:

  * features - word 1-2-grams plus character 2-5-grams (so spelling habits, slang and
    emoji count, and "dont" and "don't" share most of their letter patterns);
  * model    - one multinomial logistic regression over those features, classes
    weighted equally so a talkative person does not win ties by volume;
  * honesty  - probabilities are calibrated on held-out (cross-validated)
    predictions of *this case's own messages*, and the engine says "uncertain"
    below the confidence at which those held-out checks were right at least
    `target_precision` of the time (with a statistical margin for how few checks
    there were) - instead of a hand-picked cut-off. With too little chat to check
    itself against, it never commits;
  * evidence - every score is a sum of per-feature contributions, so the
    explanation is exact, not a story: the words and two-word phrases that point to
    the top match over the runner-up (a word's share includes the spelling patterns
    inside it) plus the base rate add up to their log-odds, and the remainder is
    reported, not hidden (test_learned_engine.py checks the sum).

It answers the same `investigate()` question as DetectiveEngine and returns the
same shape (`ranking`, `uncertain`), so the eval harness can compare the two.
What it does not do (yet): read the messages that came just before the query -
`context` is accepted and reported as unused, never silently pretended to matter.

Light on purpose: numpy / scipy / scikit-learn only, no sentence-transformer model.
"""
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize_scalar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

# Hyper-parameters, chosen on the three demo cases with eval_attribution.py (see
# BACKLOG A-1) and deliberately not re-tuned per case: with a few hundred synthetic
# messages, per-case tuning would only fit noise. (Tried and dropped: normalising
# apostrophes/case in the word features - no gain, small loss on the small cases.)
C = 10.0
CLASS_WEIGHT = "balanced"      # every person equally likely a priori; beat unweighted on all three demo cases
WORD_NGRAMS = (1, 2)
CHAR_NGRAMS = (2, 5)
MAX_FEATURES = 200_000
TARGET_PRECISION = 0.80        # abstain below the confidence where held-out checks fall under this
PLAUSIBLE_MASS = 0.90          # the "could be" list holds the fewest people covering this much probability
CV_FOLDS = 5
CV_MIN_PER_SENDER = 3          # a sender needs this many messages to be tested in a fold
CV_MIN_MESSAGES = 60           # fewer testable messages than this and the calibration would be noise
CV_MAX_MESSAGES = 6000         # bound the extra fitting on a very large chat
MIN_CHECKED_TO_COMMIT = 250    # fewer held-out checks than this and it will not commit: on the nine-person demo the calibrated
                               # cut-off met its 80% target from ~270 training messages (80-85%) but not below (74% at 180)
MIN_COMMIT_SUPPORT = 20        # a confidence cut-off must have been reached by at least this many checks...
MIN_COMMIT_SHARE = 0.10        # ...and by at least this share of them (a cut-off "proven" on a handful of checks is luck)
MIN_COMMIT_CONFIDENCE = 0.50   # never name one person unless they are more likely than everyone else put together
WILSON_Z = 1.2816              # one-sided 90%: a cut-off must be reliable even at the pessimistic end of what the checks show
TEMPERATURE_BOUNDS = (0.5, 4.0)  # held-out checks that happen to be perfect would otherwise sharpen without limit
MAX_TERMS = 8

# a word (keeping apostrophes inside it: i'll, don't) or a single punctuation mark / emoji
_WORD_TOKEN = r"(?u)\b\w+(?:['’]\w+)*\b|[^\w\s]"


def wilson_lower(hits: int, n: int, z: float = WILSON_Z) -> float:
    """Lower end of the Wilson score interval for a proportion: how low the true precision could
    plausibly be given `hits` right out of `n` checks. 0.0 for no checks."""
    if n <= 0:
        return 0.0
    p = hits / n
    centre = p + z * z / (2 * n)
    spread = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (centre - spread) / (1 + z * z / n)


def choose_commit_threshold(top: np.ndarray, hit: np.ndarray, target: float, min_support: int):
    """The confidence at which the engine starts committing to one person, from held-out checks
    (`top`: the confidence of each check's best guess, `hit`: whether that guess was right).

    The lowest cut-off such that the checks at or above it were right at least `target` of the time
    even at the pessimistic end of what that many checks show (Wilson lower bound), reached by at least
    `min_support` checks, and never below MIN_COMMIT_CONFIDENCE. Every claim is about the set of checks
    actually at or above the cut-off: checks with the same confidence (duplicate messages give exactly
    that) are admitted together, never half of them. Returns (threshold, answered, right); threshold is
    None when no cut-off qualifies."""
    order = np.argsort(-top, kind="stable")
    ranked_top, cum_hits = top[order], np.cumsum(hit[order])
    for n in range(len(top), min_support - 1, -1):  # consider a cut-off at the n-th most confident check
        threshold = max(float(ranked_top[n - 1]), MIN_COMMIT_CONFIDENCE)
        answered = int(np.sum(ranked_top >= threshold))  # by value, so a tie group is admitted whole
        if answered >= min_support and wilson_lower(int(cum_hits[answered - 1]), answered) >= target:
            return threshold, answered, int(cum_hits[answered - 1])
    return None, 0, 0


class _Model:
    """One fitted feature pipeline + linear model. Holds the weights as a plain
    (classes x features) matrix so probabilities, predictions and explanations are
    all the same arithmetic."""

    def __init__(self, texts: Sequence[str], labels: Sequence[str], class_weight):
        self.word_vec = TfidfVectorizer(
            analyzer="word", ngram_range=WORD_NGRAMS, sublinear_tf=True, token_pattern=_WORD_TOKEN,
            max_features=MAX_FEATURES,
        )
        self.char_vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=CHAR_NGRAMS, sublinear_tf=True, max_features=MAX_FEATURES,
        )
        X = self._features(texts, fit=True)
        clf = LogisticRegression(C=C, max_iter=4000, class_weight=class_weight)
        clf.fit(X, list(labels))
        self.classes: List[str] = [str(c) for c in clf.classes_]
        coef, bias = np.asarray(clf.coef_, dtype=float), np.asarray(clf.intercept_, dtype=float)
        if coef.shape[0] == 1:  # two classes: sklearn stores one sigmoid weight vector; split it into two logits
            coef, bias = np.vstack([-coef[0] / 2, coef[0] / 2]), np.array([-bias[0] / 2, bias[0] / 2])
        self.W, self.b = coef, bias
        self.n_word = len(self.word_vec.vocabulary_)
        self.word_terms = np.array(self.word_vec.get_feature_names_out(), dtype=object)

    def _features(self, texts: Sequence[str], fit: bool = False):
        texts = list(texts)
        if fit:
            Xw, Xc = self.word_vec.fit_transform(texts), self.char_vec.fit_transform(texts)
        else:
            Xw, Xc = self.word_vec.transform(texts), self.char_vec.transform(texts)
        return sp.hstack([Xw, Xc]).tocsr()

    def logits(self, texts: Sequence[str]) -> np.ndarray:
        """(n_texts, n_classes) raw scores before calibration."""
        return np.asarray(self._features(texts) @ self.W.T) + self.b


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class LearnedEngine:
    """Same job as DetectiveEngine, learned instead of hand-blended. See module docstring."""

    name = "learned"

    def __init__(self, class_weight=CLASS_WEIGHT, calibrate: bool = True, target_precision: float = TARGET_PRECISION):
        self.class_weight = class_weight
        self.calibrate = calibrate
        self.target_precision = target_precision
        self.model: Optional[_Model] = None
        self.senders: List[str] = []
        self.message_counts: Dict[str, int] = {}
        self.temperature = 1.0
        self.commit_threshold: Optional[float] = None    # None = never commit
        self.calibration: Dict = {"calibrated": False, "reason": "not built"}

    # ------------------------------------------------------------------ build
    @property
    def is_ready(self) -> bool:
        return bool(self.senders)

    def build(self, messages: Sequence[dict]) -> "LearnedEngine":
        """messages: [{"sender", "text"}] - the case's own messages. Replaces any earlier build."""
        rows = [(str(m["sender"]), str(m["text"])) for m in messages if str(m["text"]).strip()]
        self.message_counts = dict(Counter(s for s, _ in rows))
        self.senders = sorted(self.message_counts)
        self.model, self.temperature, self.commit_threshold = None, 1.0, None
        if len(self.senders) < 2:
            self.calibration = {"calibrated": False, "reason": "only one person in this case - nothing to tell apart"}
            return self
        texts, labels = [t for _, t in rows], [s for s, _ in rows]
        if self.calibrate:
            self._calibrate(texts, labels)
        else:
            self.calibration = {"calibrated": False, "reason": "calibration switched off"}
        self.model = _Model(texts, labels, self.class_weight)
        return self

    def _out_of_fold(self, texts: List[str], labels: List[str]):
        """Cross-validated probabilities for this case's own messages: each one is
        predicted by a model that never saw it. Senders too small to be tested stay
        in every training fold. Returns (true labels, probability rows, classes) or None."""
        counts = Counter(labels)
        testable = [i for i, s in enumerate(labels) if counts[s] >= CV_MIN_PER_SENDER]
        if len(testable) < CV_MIN_MESSAGES:
            return None
        if len(testable) > CV_MAX_MESSAGES:  # deterministic thinning, every sender keeps its share
            rng = np.random.RandomState(0)
            testable = sorted(rng.choice(testable, CV_MAX_MESSAGES, replace=False).tolist())
        always_train = [i for i, s in enumerate(labels) if counts[s] < CV_MIN_PER_SENDER]
        y_test = [labels[i] for i in testable]
        k = min(CV_FOLDS, min(Counter(y_test).values()))
        if k < 2:
            return None
        truth: List[str] = []
        probs: List[np.ndarray] = []
        classes: Optional[List[str]] = None
        for tr, te in StratifiedKFold(k, shuffle=True, random_state=0).split(testable, y_test):
            train_idx = [testable[i] for i in tr] + always_train
            test_idx = [testable[i] for i in te]
            fold = _Model([texts[i] for i in train_idx], [labels[i] for i in train_idx], self.class_weight)
            if classes is None:
                classes = fold.classes
            probs.append(_softmax(fold.logits([texts[i] for i in test_idx])))
            truth.extend(labels[i] for i in test_idx)
        return truth, np.vstack(probs), classes

    def _calibrate(self, texts: List[str], labels: List[str]) -> None:
        """Sets the temperature (so a stated 80% is right about 80% of the time on
        held-out checks) and the confidence below which the engine says "uncertain"."""
        oof = self._out_of_fold(texts, labels)
        if oof is None:
            self.calibration = {"calibrated": False, "reason": "too few messages to check the engine against itself"}
            return
        truth, P, classes = oof
        y = np.array([classes.index(t) for t in truth])
        logp = np.log(np.clip(P, 1e-12, 1.0))

        def nll(t: float) -> float:
            return float(-np.mean(np.log(np.clip(_softmax(logp / t)[np.arange(len(y)), y], 1e-12, 1.0))))

        self.temperature = float(minimize_scalar(nll, bounds=TEMPERATURE_BOUNDS, method="bounded").x)
        Pc = _softmax(logp / self.temperature)
        top, hit = Pc.max(axis=1), (Pc.argmax(axis=1) == y)
        note = None

        threshold, kept, kept_hits = None, 0, 0
        if len(y) < MIN_CHECKED_TO_COMMIT:
            note = (f"Only {len(y)} messages to check against - it needs about {MIN_CHECKED_TO_COMMIT} before its "
                    "confidence can be trusted enough to commit to one person.")
        else:
            min_support = max(MIN_COMMIT_SUPPORT, int(MIN_COMMIT_SHARE * len(y)))
            threshold, kept, kept_hits = choose_commit_threshold(top, hit, self.target_precision, min_support)
            if threshold is None:
                note = "In this case's own held-out checks no confidence level was reliable enough to commit to one person."
        self.commit_threshold = threshold
        self.calibration = {
            "calibrated": True, "temperature": round(self.temperature, 3),
            "target_precision": self.target_precision, "checked_messages": int(len(y)),
            "checked_accuracy": round(float(hit.mean()), 3),
            "commit_threshold": None if threshold is None else round(threshold, 3),
            "checked_precision_at_threshold": None if not kept else round(kept_hits / kept, 3),
            "checked_share_answered": None if not kept else round(kept / len(y), 3),
            "commit_note": note,
        }

    # ------------------------------------------------------------ investigate
    def probabilities(self, texts: Sequence[str]) -> np.ndarray:
        """Calibrated (n_texts, n_senders) probabilities, columns in `self.model.classes` order."""
        return _softmax(self.model.logits(texts) / self.temperature)

    def investigate(self, sentence: str, context=None, **_ignored) -> dict:
        """Ranked candidates with calibrated confidences, an honest uncertain flag,
        the smallest set of people covering PLAUSIBLE_MASS, and the exact evidence.
        `context` is accepted for interface parity and reported as unused."""
        notes = []
        if context:
            notes.append("The messages just before this one were not used - this engine reads only the sentence itself.")
        base = {"engine": self.name, "sentence": sentence, "context_used": 0, "notes": notes}
        if self.model is None:
            notes.append("Only one person in this case, so there is nothing to tell apart." if len(self.senders) == 1
                         else "No messages to learn from.")
            return {**base, "uncertain": True, "plausible_senders": list(self.senders), "evidence": None,
                    "calibration": self.calibration,
                    "ranking": [{"sender": s, "confidence": round(100.0 / max(len(self.senders), 1), 1)} for s in self.senders]}
        classes = self.model.classes

        if self.model._features([sentence]).nnz == 0:
            # Not one word or letter pattern of this message appears anywhere in the chat (a new script,
            # a new emoji): the linear model's output would be just its intercepts, an artefact. All that
            # can honestly be said is the base rate.
            if self.class_weight == "balanced":
                p = np.full(len(classes), 1.0 / len(classes))
                notes.append("None of this message's words or letter patterns appear in the chat, so there is nothing to go on.")
            else:
                total = sum(self.message_counts.values())
                p = np.array([self.message_counts.get(c, 0) / total for c in classes])
                notes.append("None of this message's words or letter patterns appear in the chat, so all it can go on is who talks most.")
            order = np.argsort(-p, kind="stable")
            return {**base, "uncertain": True, "plausible_senders": [classes[i] for i in order],
                    "ranking": [{"sender": classes[i], "confidence": round(float(p[i]) * 100, 1)} for i in order],
                    "evidence": None, "calibration": self.calibration}

        p = self.probabilities([sentence])[0]
        order = np.argsort(-p, kind="stable")
        ranking = [{"sender": classes[i], "confidence": round(float(p[i]) * 100, 1)} for i in order]
        top_p = float(p[order[0]])
        uncertain = self.commit_threshold is None or top_p < self.commit_threshold

        cum = np.cumsum(p[order])
        n_plausible = int(np.searchsorted(cum, PLAUSIBLE_MASS) + 1)
        plausible = [classes[i] for i in order[:n_plausible]]

        if uncertain:
            if not self.calibration.get("calibrated"):
                notes.append(f"Confidence is not calibrated ({self.calibration.get('reason', 'unknown reason')}), "
                             "so it will not commit to one person.")
            elif self.commit_threshold is None:
                notes.append(self.calibration.get("commit_note") or "It will not commit to one person.")
            else:
                notes.append(f"Below the {self.commit_threshold * 100:.0f}% confidence at which this case's held-out checks were right "
                             f"at least {self.target_precision * 100:.0f}% of the time.")

        return {**base, "uncertain": bool(uncertain), "plausible_senders": plausible, "ranking": ranking,
                "evidence": self._evidence(sentence, int(order[0]), int(order[1])),
                "calibration": self.calibration}

    def _evidence(self, sentence: str, top: int, other: int) -> dict:
        """Why `top` over `other`, as an exact decomposition of their calibrated log-odds:
            sum(terms) + other_terms + base_rate == ln(P(top) / P(other)).
        Every feature's contribution (weight difference x the message's value for it, divided by
        the calibration temperature) is assigned to something a person can read: a word (its own
        word feature plus the letter patterns inside it - spelling, slang, emoji) or a two-word
        phrase. `terms` are the ones that moved it most; `other_terms` is everything else, so
        nothing is hidden. `base_rate` is the head start from the model's intercepts (the chat's
        volume, when classes are not weighted equally)."""
        m = self.model
        x = m._features([sentence])
        scale = 1.0 / self.temperature
        row = x.multiply((m.W[top] - m.W[other]) * scale).tocsr()
        val = dict(zip(row.indices.tolist(), row.data.tolist()))

        words = sentence.split()
        weight: Dict[tuple, float] = {}

        def give(kind: str, text: str, amount: float) -> None:
            weight[(kind, text)] = weight.get((kind, text), 0.0) + amount

        # word block: unigram -> the word(s) it came from, bigram -> a phrase
        tokens_of = [re.findall(_WORD_TOKEN, w.lower()) for w in words]
        for i, term in enumerate(m.word_terms):
            if i not in val:
                continue
            if " " in term:
                give("phrase", str(term), val[i])
                continue
            owners = [w for w, toks in zip(words, tokens_of) for t in toks if t == term]
            for w in owners:
                give("word", w.lower(), val[i] / len(owners))

        # character block: n-gram -> the word(s) it sits inside (char_wb never crosses a space)
        analyzer, vocab = m.char_vec.build_analyzer(), m.char_vec.vocabulary_
        owners_of: Dict[int, List[str]] = {}
        for w in words:
            for gram in analyzer(w):
                j = vocab.get(gram)
                if j is not None and (m.n_word + j) in val:
                    owners_of.setdefault(m.n_word + j, []).append(w)
        for j, owners in owners_of.items():
            for w in owners:
                give("word", w.lower(), val[j] / len(owners))

        base = float((m.b[top] - m.b[other]) * scale)
        ranked = sorted(weight.items(), key=lambda kv: -abs(kv[1]))
        terms = [{"text": t, "kind": k, "log_odds": round(v, 3)} for (k, t), v in ranked[:MAX_TERMS]]
        rest = sum(v for _, v in ranked[MAX_TERMS:])
        return {
            "for": m.classes[top], "against": m.classes[other],
            "terms": terms,
            "other_terms": round(rest, 3),
            "base_rate": round(base, 3),
            "log_odds": round(sum(weight.values()) + base, 3),
        }
