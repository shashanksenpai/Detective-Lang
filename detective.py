"""Detective Lang - prototype attribution engine.
Parses a WhatsApp-style export, builds per-sender centroids (embedding +
stylometric), and answers "who said this?" with confidence scores.
"""
import re
import json
from collections import defaultdict
from statistics import mean

import numpy as np
from sentence_transformers import SentenceTransformer
from hinglish_sentiment import get_analyzer

from parsers.whatsapp import parse_whatsapp  # re-exported for backward compat

# Real sentence embeddings for the topic signal - replaces the earlier TF-IDF
# stand-in. Unlike TF-IDF, this gives every sentence a meaningful vector
# regardless of exact vocabulary overlap with training data: "I'm freaking
# out about the test" lands close to "worried about our exam" in embedding
# space even though they share almost no literal words. Loaded once at
# import time (same pattern as the VADER analyzer below).
_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(texts):
    """Public accessor for the shared embedding model - lets other modules
    (identity_resolution.py's person-centroid soft-match tier) reuse the
    same loaded model instead of loading a second copy.
    """
    return _embedding_model.encode(texts)


EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF]"
)

WORD_RE = re.compile(r"[a-zA-Z']+")

# Curated topic-domain lexicons. Unlike TF-IDF (which only recognizes the
# exact tokens seen during training), these match on *meaning families* -
# a message about "quiz" or "syllabus" still reads as "studies" even if the
# training chat only ever said "assignment". This is what lets the topic
# signal generalize to unseen phrasing instead of collapsing to noise.
CATEGORY_LEXICONS = {
    "studies": {
        "exam", "exams", "study", "studying", "studied", "deadline", "deadlines",
        "assignment", "assignments", "homework", "class", "classes", "lecture",
        "lectures", "professor", "project", "projects", "grade", "grades",
        "submit", "thesis", "quiz", "syllabus", "textbook", "essay", "report",
        "methodology", "section", "sections", "draft", "research", "university",
        "college", "school", "revise", "revision", "notes"
    },
    "fun": {
        "party", "parties", "weekend", "movie", "movies", "game", "games",
        "hangout", "chill", "chilling", "fun", "dance", "dancing", "music",
        "trip", "vacation", "lol", "lmao", "haha", "hahaha", "omg", "yay",
        "concert", "festival", "match", "playing", "played"
    },
    "work": {
        "meeting", "meetings", "boss", "client", "clients", "office", "shift",
        "salary", "task", "tasks", "presentation", "email", "emails", "job",
        "interview", "sync", "budget", "invoice"
    },
    "logistics": {
        "pickup", "address", "time", "location", "meet", "schedule", "plan",
        "plans", "arrive", "arriving", "late", "early", "call", "reminder"
    },
    "food": {
        "food", "eat", "eating", "lunch", "dinner", "breakfast", "hungry",
        "starving", "restaurant", "cook", "cooking", "snack", "order",
        "delivery", "recipe"
    },
}


# Hinglish (Romanized Hindi) members of each topic family - additive coverage only.
HINGLISH_CATEGORY_WORDS = {
    "studies": {
        "padhai", "padhna", "padh", "padhta", "padhti", "imtihan", "kitab", "viva", "marks", "result",
        "semester", "sem", "attendance", "practical", "paper", "papers",
    },
    "fun": {
        "maza", "mazaa", "masti", "ghumna", "ghoomna", "picture", "film", "gaana", "gana", "naach",
        "nachna", "cricket", "shaadi", "tyohar", "diwali", "holi", "garba",
    },
    "work": {"kaam", "naukri", "tankhwah"},
    "logistics": {
        "kab", "kahan", "kitne", "baje", "pahunch", "pahuncha", "milte", "milna", "ticket", "station",
        "nikal", "nikalta", "jaldi", "intezaar",
    },
    "food": {
        "khana", "khaana", "chai", "nashta", "bhook", "bhookh", "pizza", "biryani", "sabzi", "roti",
        "dal", "chawal", "paratha", "samosa", "maggi", "coffee", "cake", "mithai", "canteen", "tiffin",
    },
}
for _topic, _words in HINGLISH_CATEGORY_WORDS.items():
    CATEGORY_LEXICONS[_topic] |= _words

def category_features(text):
    """Fraction of a message's category-lexicon hits per domain.
    A message with no lexicon hits gets a uniform (maximally uncertain)
    distribution rather than all-zero, so it doesn't spuriously look
    "similar" to every sender via a shared zero vector.
    """
    words = [w.lower() for w in WORD_RE.findall(text)]
    counts = {cat: 0 for cat in CATEGORY_LEXICONS}
    for w in words:
        for cat, lex in CATEGORY_LEXICONS.items():
            if w in lex:
                counts[cat] += 1
    total = sum(counts.values())
    if total == 0:
        n = len(counts)
        return {cat: 1.0 / n for cat in counts}
    return {cat: c / total for cat, c in counts.items()}


def has_category_signal(text):
    """Whether any of the five lexicons actually matched a word in this
    text. Distinguishes real "no topic leans this way" evidence from a
    coincidental tie - a uniform category vector should never be read as
    "this is about studies" just because 'studies' happens to be first in
    dict order.
    """
    words = {w.lower() for w in WORD_RE.findall(text)}
    return any(words & lex for lex in CATEGORY_LEXICONS.values())


# Rule-based syntax proxies (no POS tagger available in this sandbox - these
# heuristics stand in for it, same swap-later relationship TF-IDF has to
# sentence-transformers). They capture sentence *form* independent of topic
# vocabulary: whether someone habitually asks vs. commands vs. states, how
# clause-heavy their sentences run, and whether they talk about themselves
# or address the other person.
WH_QUESTION_WORDS = {"who", "what", "when", "where", "why", "how", "which"}
# Hinglish (Romanized Hindi) additions - only words that don't collide with an English word.
HINGLISH_WH_QUESTION = {"kya", "kaun", "kaise", "kyun", "kyu", "kab", "kahan", "kitna", "kitne", "kitni", "kidhar", "konsa"}
WH_QUESTION_WORDS |= HINGLISH_WH_QUESTION
AUX_QUESTION_STARTS = {
    "can", "could", "should", "would", "do", "does", "did", "is", "are",
    "will", "have", "has"
}
IMPERATIVE_STARTS = {
    "please", "lets", "let's", "don't", "dont", "send", "share", "check",
    "remind", "block", "take", "go", "finish", "submit", "order", "revise",
    "review", "meet", "call", "stop", "wait", "try", "make", "keep", "bring"
}
HINGLISH_IMPERATIVES = {"bhej", "bhejo", "dekh", "dekho", "sun", "suno", "bol", "bolo", "aao", "chal", "chalo", "ruk", "ruko",
                        "batao", "bata", "karo", "jao", "rakho", "dena", "bhejna", "batana"}
IMPERATIVE_STARTS |= HINGLISH_IMPERATIVES
CONJUNCTIONS = {"and", "but", "because", "so", "also", "since", "although", "however", "though"}
HINGLISH_CONJUNCTIONS = {"aur", "lekin", "magar", "kyunki", "isliye", "toh", "phir", "warna"}
CONJUNCTIONS |= HINGLISH_CONJUNCTIONS
FIRST_PERSON = {"i", "we", "my", "our", "me", "us", "im", "ive", "ill"}
HINGLISH_FIRST_PERSON = {"main", "mai", "hum", "mera", "meri", "mere", "mujhe", "humein", "humara"}
FIRST_PERSON |= HINGLISH_FIRST_PERSON
SECOND_PERSON = {"you", "your", "youre", "youve", "youll"}
HINGLISH_SECOND_PERSON = {"tu", "tum", "aap", "tera", "teri", "tere", "tujhe", "tumhe", "tumhara", "tumhari", "aapka", "aapki"}
SECOND_PERSON |= HINGLISH_SECOND_PERSON


def syntactic_features(text):
    stripped = text.strip()
    words = [w.lower().replace("'", "") for w in WORD_RE.findall(stripped)]
    first_word = words[0] if words else ""

    is_question = 1.0 if (
        stripped.endswith("?") or first_word in WH_QUESTION_WORDS or first_word in AUX_QUESTION_STARTS
    ) else 0.0
    is_exclamatory = 1.0 if stripped.endswith("!") else 0.0
    is_imperative = 1.0 if (not is_question and first_word in IMPERATIVE_STARTS) else 0.0
    is_declarative = 1.0 if not (is_question or is_imperative) else 0.0

    conj_count = sum(1 for w in words if w in CONJUNCTIONS)
    clause_density = conj_count / max(len(words), 1)

    first_person = sum(1 for w in words if w in FIRST_PERSON)
    second_person = sum(1 for w in words if w in SECOND_PERSON)
    total_pronoun = first_person + second_person
    person_ratio = (first_person / total_pronoun) if total_pronoun else 0.5

    return {
        "question": is_question,
        "exclamatory": is_exclamatory,
        "imperative": is_imperative,
        "declarative": is_declarative,
        "clause_density": clause_density,
        "person_ratio": person_ratio,
    }


def dominant_sentence_type(feats):
    for k in ("question", "imperative", "exclamatory"):
        if feats[k] >= 1.0:
            return k
    return "declarative"


def stylometric_features(text):
    words = text.split()
    return {
        "len": len(words),
        "emoji_count": len(EMOJI_RE.findall(text)),
        "exclaim": text.count("!"),
        "question": text.count("?"),
        "caps_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
    }


# Hinglish-aware sentiment (hinglish_sentiment.py), VADER-shaped so callers didn't change.
# Falls back to plain VADER if the trained model file is missing.
_sentiment_analyzer = get_analyzer()


def emotion_features(text):
    """Lexicon-based sentiment -> a per-message emotional fingerprint.
    'compound' = overall valence (-1 very negative .. +1 very positive).
    'neg' = intensity of negative/anxious language specifically - this is
    the dimension that catches worry/stress regardless of topic vocabulary.
    """
    scores = _sentiment_analyzer.polarity_scores(text)
    return {"compound": scores["compound"], "neg": scores["neg"], "pos": scores["pos"]}


def emotion_features_batch(texts):
    """emotion_features for many messages at once - the trained scorer is
    much faster batched, so loops over a whole chat should use this."""
    return [
        {"compound": s["compound"], "neg": s["neg"], "pos": s["pos"]}
        for s in _sentiment_analyzer.polarity_scores_batch(list(texts))
    ]


class DetectiveEngine:
    """Topic signal runs on real sentence embeddings (all-MiniLM-L6-v2) -
    every message gets a meaningful 384-dim vector regardless of exact
    vocabulary overlap with training data, so semantic paraphrases of a
    topic a sender is known for still match them. The rest of the pipeline
    (centroids, blending, softmax, thresholds) is unchanged from the earlier
    TF-IDF version - only how the topic vector itself is produced.
    """

    def __init__(self):
        self.centroids = {}          # sender -> mean embedding vector
        self.style_profiles = {}     # sender -> avg stylometric features
        self.emotion_profiles = {}   # sender -> avg emotional fingerprint
        self.category_profiles = {}  # sender -> avg topic-domain distribution
        self.syntactic_profiles = {}  # sender -> avg sentence-form fingerprint
        self.transition_counts = defaultdict(lambda: defaultdict(int))  # prev_sender -> {next_sender: count}
        self.messages_by_sender = defaultdict(list)

    def build(self, messages):
        for m in messages:
            self.messages_by_sender[m["sender"]].append(m["text"])

        # Turn-taking / adjacency pairs, in original conversation order - the
        # discourse-level signal. Who habitually replies right after whom is
        # information no single message's wording carries on its own.
        for prev, curr in zip(messages, messages[1:]):
            self.transition_counts[prev["sender"]][curr["sender"]] += 1

        for sender, texts in self.messages_by_sender.items():
            vecs = _embedding_model.encode(texts)
            self.centroids[sender] = np.mean(vecs, axis=0)

            style_feats = [stylometric_features(t) for t in texts]
            self.style_profiles[sender] = {
                k: mean(f[k] for f in style_feats) for k in style_feats[0]
            }

            emo_feats = emotion_features_batch(texts)
            self.emotion_profiles[sender] = {
                k: mean(f[k] for f in emo_feats) for k in emo_feats[0]
            }

            cat_feats = [category_features(t) for t in texts]
            self.category_profiles[sender] = {
                k: mean(f[k] for f in cat_feats) for k in cat_feats[0]
            }

            syn_feats = [syntactic_features(t) for t in texts]
            self.syntactic_profiles[sender] = {
                k: mean(f[k] for f in syn_feats) for k in syn_feats[0]
            }

    def _style_score(self, text, sender):
        """Rough similarity between a message's style and a sender's profile."""
        feats = stylometric_features(text)
        profile = self.style_profiles[sender]
        diffs = []
        for k in feats:
            scale = max(profile[k], 1)  # avoid div by zero, normalize by scale
            diffs.append(abs(feats[k] - profile[k]) / scale)
        return 1 / (1 + mean(diffs))  # closer to 1 = more similar

    def _emotion_score(self, text, sender):
        """Does this message's emotional tone match this sender's baseline
        temperament? Someone who habitually runs anxious/negative (e.g. always
        stressed about deadlines) will score high here for a new worried
        message even if none of the words overlap - this is what catches
        'worried about the exam' as matching a chronically anxious speaker.
        """
        feats = emotion_features(text)
        profile = self.emotion_profiles[sender]
        diffs = [abs(feats[k] - profile[k]) for k in feats]  # scores are -1..1 already
        return 1 / (1 + mean(diffs))

    def _syntactic_score(self, text, sender):
        """Does this message's sentence form (question/command/statement,
        clause density, self- vs. other-reference) match how this sender
        typically constructs sentences, independent of what it's about?
        A person who always phrases things as direct commands ('send it',
        'lets sync') will score high here even on a brand-new topic.
        """
        feats = syntactic_features(text)
        profile = self.syntactic_profiles[sender]
        diffs = [abs(feats[k] - profile[k]) for k in feats]  # all features are 0..1 scaled
        return 1 / (1 + mean(diffs))

    def _category_vector(self, feats):
        return np.array([feats[c] for c in CATEGORY_LEXICONS])

    def _category_score(self, text, sender):
        """Cosine similarity between a message's topic-domain distribution
        (studies/fun/work/logistics/food) and a sender's typical distribution.
        Generalizes to unseen wording because it keys off curated meaning
        families, not literal training-corpus tokens.
        """
        f = self._category_vector(category_features(text))
        p = self._category_vector(self.category_profiles[sender])
        denom = (np.linalg.norm(f) * np.linalg.norm(p)) or 1e-9
        return float(np.dot(f, p) / denom)

    def _turn_taking_score(self, sender, prev_sender):
        """Discourse-level signal: given who spoke immediately before this
        message (known from the surrounding conversation, not from the
        message's own wording), how often does `sender` reply right after
        `prev_sender`? This is the one signal that can rescue a message with
        almost no lexical/stylistic content of its own (e.g. a bare "crazy")
        by leaning on conversational turn-taking structure instead of words.
        """
        n = len(self.centroids)
        if prev_sender is None or prev_sender not in self.centroids:
            return 1.0 / n
        transitions = self.transition_counts.get(prev_sender, {})
        total = sum(transitions.values())
        # Laplace smoothing so an unseen (prev, sender) pair isn't a hard 0
        return (transitions.get(sender, 0) + 1) / (total + n)

    @staticmethod
    def _apply_trust(base_weights, trust_by_key):
        """Shrink the weight of any signal whose trust < 1 (it has little or
        nothing real to say for this particular query) and hand the leftover
        to the signals that are always well-defined, keeping weights summing
        to 1. This is the general form of the topic/vocabulary-coverage fix -
        any signal can go quiet on a given input, and the blend should lean
        away from it rather than dilute the result with noise.
        """
        weights = dict(base_weights)
        leftover = 0.0
        for k, trust in trust_by_key.items():
            original = base_weights[k]
            weights[k] = original * trust
            leftover += original * (1 - trust)
        fixed_keys = [k for k in base_weights if k not in trust_by_key]
        fixed_total = sum(base_weights[k] for k in fixed_keys)
        for k in fixed_keys:
            weights[k] += leftover * (base_weights[k] / fixed_total)
        return weights

    def investigate(self, sentence, context=None, unknown_threshold=0.35, margin_threshold=0.08):
        """context: optional list of {"sender": str|None, "text": str} for
        the messages immediately preceding `sentence` (oldest first) - the
        surrounding conversation, if known, even though the sender of
        `sentence` itself is the thing being figured out. Powers the
        discourse-level signals (topical continuity + turn-taking).
        """
        context = context or []
        cats = list(CATEGORY_LEXICONS)

        # Recency-weighted blend of the query with its immediate context for
        # the topic/category signals - lets a near-featureless short message
        # ("crazy") inherit the topical grounding of what was just said,
        # instead of being scored in a vacuum.
        window_texts = [c["text"] for c in context] + [sentence]
        if len(window_texts) == 1:
            recency_weights = np.array([1.0])
        else:
            decay = 0.6
            raw = [decay ** (len(window_texts) - 1 - i) for i in range(len(window_texts))]
            recency_weights = np.array(raw) / sum(raw)

        embed_window = _embedding_model.encode(window_texts)
        query_vec = np.average(embed_window, axis=0, weights=recency_weights)

        cat_window = np.array([self._category_vector(category_features(t)) for t in window_texts])
        query_cat_vec = np.average(cat_window, axis=0, weights=recency_weights)

        # Embeddings give every sentence a meaningful vector regardless of
        # vocabulary overlap, so - unlike the old TF-IDF topic signal - topic
        # doesn't need trust-scaling here. But category still does: when no
        # lexicon term matched, category_features falls back to a uniform
        # vector, and cosine similarity against that isn't neutral - it
        # quietly favors whichever sender's own profile happens to be most
        # topically spread-out, which has nothing to do with this message.
        # Discourse similarly only means something when the caller actually
        # supplied context.
        prev_sender = next(
            (c["sender"] for c in reversed(context) if c.get("sender")), None
        )
        discourse_trust = 1.0 if prev_sender in self.centroids else 0.0
        category_trust = 1.0 if any(has_category_signal(t) for t in window_texts) else 0.0

        base_weights = {
            "topic": 0.25, "style": 0.12, "emotion": 0.13,
            "category": 0.25, "syntactic": 0.10, "discourse": 0.15,
        }
        weights = self._apply_trust(
            base_weights, {"discourse": discourse_trust, "category": category_trust}
        )

        scores = {}
        breakdown = {}
        for sender, centroid in self.centroids.items():
            denom = (np.linalg.norm(query_vec) * np.linalg.norm(centroid)) or 1e-9
            topic_sim = float(np.dot(query_vec, centroid) / denom)

            style_sim = self._style_score(sentence, sender)
            emotion_sim = self._emotion_score(sentence, sender)
            syntactic_sim = self._syntactic_score(sentence, sender)

            profile_cat_vec = self._category_vector(self.category_profiles[sender])
            cdenom = (np.linalg.norm(query_cat_vec) * np.linalg.norm(profile_cat_vec)) or 1e-9
            category_sim = float(np.dot(query_cat_vec, profile_cat_vec) / cdenom)

            discourse_sim = self._turn_taking_score(sender, prev_sender) if discourse_trust else 0.0

            combined = (
                weights["topic"] * topic_sim
                + weights["style"] * style_sim
                + weights["emotion"] * emotion_sim
                + weights["category"] * category_sim
                + weights["syntactic"] * syntactic_sim
                + weights["discourse"] * discourse_sim
            )
            scores[sender] = combined
            breakdown[sender] = {
                "topic": round(topic_sim, 3),
                "style": round(style_sim, 3),
                "emotion": round(emotion_sim, 3),
                "category": round(category_sim, 3),
                "syntactic": round(syntactic_sim, 3),
                "discourse": round(discourse_sim, 3),
            }

        # softmax over combined scores -> confidence percentages
        vals = np.array(list(scores.values()))
        exp = np.exp((vals - vals.max()) * 8)  # temperature sharpens gaps
        probs = exp / exp.sum()

        ranked = sorted(
            zip(scores.keys(), probs), key=lambda x: x[1], reverse=True
        )

        top_sender, top_prob = ranked[0]
        second_prob = ranked[1][1] if len(ranked) > 1 else 0
        is_uncertain = (
            top_prob < unknown_threshold or (top_prob - second_prob) < margin_threshold
        )

        dominant_category = (
            cats[int(np.argmax(query_cat_vec))]
            if any(has_category_signal(t) for t in window_texts)
            else None
        )
        sentence_type = dominant_sentence_type(syntactic_features(sentence))

        return {
            "sentence": sentence,
            "uncertain": bool(is_uncertain),
            "dominant_category": dominant_category,
            "sentence_type": sentence_type,
            "context_used": len(context),
            "weights_used": {k: round(v, 3) for k, v in weights.items()},
            "ranking": [
                {
                    "sender": s,
                    "confidence": round(float(p) * 100, 1),
                    "signals": breakdown[s],
                }
                for s, p in ranked
            ],
        }


if __name__ == "__main__":
    messages = parse_whatsapp("sample_chat.txt")
    engine = DetectiveEngine()
    engine.build(messages)

    test_sentences = [
        "omg no way that's crazy lol 😭",
        "we need to split the work into sections",
        "the deadline is tomorrow and I haven't started",
        "haha yeah for sure, sounds good",
        "I'm so worried about our exam tomorrow",
    ]

    for s in test_sentences:
        result = engine.investigate(s)
        print(json.dumps(result, indent=2))
        print("---")

    # Discourse demo: a near-featureless bare phrase, with vs. without the
    # surrounding conversation it actually appeared next to. Without context
    # it's close to a coin flip between all three (no lexical/topic signal
    # in "crazy right?" alone); with the preceding message + its known
    # sender, topical continuity + turn-taking pull it toward the right person.
    bare = "crazy right?"
    print("no context:", json.dumps(engine.investigate(bare)["ranking"], indent=2))
    print("with context:", json.dumps(
        engine.investigate(
            bare,
            context=[{"sender": "Aman", "text": "it was fun but I need to lock in now, the quiz is tomorrow"}],
        )["ranking"],
        indent=2,
    ))
