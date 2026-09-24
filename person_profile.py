"""Cross-source person profiles - "About this person" dossier.
Pools a person's messages across every imported source (different chats,
groups, platforms), keeping group vs DM behavior split per CLAUDE.md's
design principle (never average a person's behavior into one flat global
profile), and summarizes their overall texting personality: sentiment,
distinctive vocabulary, and a Big Five (OCEAN) style estimate.

The OCEAN scores are a structured summary of writing-style signals using
the standard psychological framework - not a clinical or diagnostic
assessment. That distinction matters: word-frequency heuristics can support
"this person writes with more exclamation marks and fun-topic messages than
most," they cannot support "this person has an anxiety disorder," and this
module deliberately never produces the latter kind of claim about a real,
identifiable person.
"""
import math
from collections import Counter
from statistics import mean, pstdev

from detective import (
    CATEGORY_LEXICONS, WORD_RE, category_features, emotion_features_batch,
    stylometric_features, syntactic_features,
)

STOPWORDS = {
    "a", "about", "above", "after", "again", "all", "am", "an", "and", "any",
    "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "could", "did", "do", "does",
    "doing", "don", "down", "during", "each", "few", "for", "from", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it",
    "its", "itself", "just", "me", "more", "most", "my", "myself", "no",
    "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other",
    "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she",
    "should", "so", "some", "such", "t", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "will", "with", "would", "you", "your", "yours", "yourself",
    "yourselves", "im", "ive", "ill", "youre", "dont", "didnt", "thats",
    "its", "got", "get", "one", "also", "really",
}

# Hinglish function words (Romanized Hindi/Urdu), so the vocabulary lists show what
# a person talks about instead of "hai", "tha", "nahi". Only words of 3+ letters
# matter (content_words already drops shorter ones). Deliberately NOT included:
# words that double as English content words, and the register markers that make
# someone sound like themselves ("bro", "bhai", "yaar", "arre").
HINGLISH_STOPWORDS = {
    "hai", "hain", "tha", "thi", "thay", "hoga", "hogi", "hota", "hoti", "hote", "hua", "hue", "hui",
    "raha", "rahi", "rahe", "gaya", "gayi", "gaye", "diya", "liya", "kiya", "karo", "karna", "karke",
    "toh", "nahi", "nahin", "nhi", "bhi", "aur", "lekin", "magar", "kyunki", "isliye", "phir", "warna",
    "kya", "kyun", "kyu", "kaise", "kaun", "kab", "kahan", "kitna", "kitne", "kitni",
    "main", "mai", "hum", "tum", "aap", "mera", "meri", "mere", "tera", "teri", "tere", "tumhara",
    "tumhari", "apna", "apni", "apne", "mujhe", "tujhe", "tumhe", "humein", "unhe", "usse", "isse",
    "yeh", "woh", "ye", "wo", "yahan", "wahan", "abhi", "jab", "tab", "bas", "sab", "kuch", "koi",
    "wala", "wali", "wale", "thoda", "thodi", "bahut", "mein", "sath", "saath", "liye", "baad",
}
STOPWORDS |= HINGLISH_STOPWORDS


def _words(text):
    return [w.lower().replace("'", "") for w in WORD_RE.findall(text)]


def content_words(text):
    return [w for w in _words(text) if w not in STOPWORDS and len(w) > 2]


def _entropy01(dist):
    """Normalized Shannon entropy of a distribution, 0 (all mass on one
    bucket) to 1 (spread evenly across every bucket).
    """
    total = sum(dist) or 1
    probs = [d / total for d in dist if d > 0]
    if len(probs) <= 1:
        return 0.0
    h = -sum(p * math.log2(p) for p in probs)
    h_max = math.log2(len(dist))
    return h / h_max if h_max else 0.0


def _clip01(x):
    return max(0.0, min(1.0, x))


def _word_stats(texts, global_freq, global_total):
    counter = Counter()
    for t in texts:
        counter.update(content_words(t))
    total = sum(counter.values()) or 1

    top_words = [{"word": w, "count": c} for w, c in counter.most_common(10)]

    # Words this person leans on far more than their share of the whole
    # corpus would predict - a keyness/idiolect measure, not raw frequency.
    # Requires >=2 uses so a single one-off word doesn't dominate.
    distinctive = []
    for word, count in counter.items():
        if count < 2:
            continue
        person_rate = count / total
        global_rate = global_freq.get(word, count) / global_total
        ratio = person_rate / max(global_rate, 1e-6)
        distinctive.append((word, count, ratio))
    distinctive.sort(key=lambda x: (-x[2], -x[1]))

    phrase_counter = Counter()
    for t in texts:
        words = _words(t)
        for i in range(len(words) - 1):
            phrase_counter[f"{words[i]} {words[i + 1]}"] += 1
    phrases = [
        {"phrase": p, "count": c}
        for p, c in phrase_counter.most_common(15)
        if c >= 2
    ][:6]

    return {
        "top_words": top_words,
        "distinctive_words": [
            {"word": w, "count": c, "ratio": round(r, 1)}
            for w, c, r in distinctive[:10]
        ],
        "frequent_phrases": phrases,
    }


def _compute_stats(texts):
    emo = emotion_features_batch(texts)
    style = [stylometric_features(t) for t in texts]
    syn = [syntactic_features(t) for t in texts]
    cat = [category_features(t) for t in texts]

    compounds = [e["compound"] for e in emo]
    sentiment = {
        "avg_compound": round(mean(compounds), 3),
        "volatility": round(pstdev(compounds), 3) if len(compounds) > 1 else 0.0,
        "pct_positive": round(100 * sum(1 for c in compounds if c > 0.2) / len(compounds), 1),
        "pct_negative": round(100 * sum(1 for c in compounds if c < -0.2) / len(compounds), 1),
        "pct_neutral": round(
            100 * sum(1 for c in compounds if -0.2 <= c <= 0.2) / len(compounds), 1
        ),
    }

    cat_dist = {c: mean(f[c] for f in cat) for c in CATEGORY_LEXICONS}

    return {
        "message_count": len(texts),
        "sentiment": sentiment,
        "avg_message_length": round(mean(f["len"] for f in style), 1),
        "avg_emoji_per_message": round(mean(f["emoji_count"] for f in style), 2),
        "avg_exclamations_per_message": round(mean(f["exclaim"] for f in style), 2),
        "question_rate": round(100 * mean(f["question"] for f in syn), 1),
        "imperative_rate": round(100 * mean(f["imperative"] for f in syn), 1),
        "self_reference_ratio": round(100 * mean(f["person_ratio"] for f in syn), 1),
        "topic_distribution": {c: round(v * 100, 1) for c, v in cat_dist.items()},
        "topic_diversity": round(_entropy01(list(cat_dist.values())), 2),
    }


def _estimate_traits(overall):
    """Heuristic Big Five (OCEAN) estimate - see module docstring for the
    boundary this deliberately stays inside.
    """
    s = overall
    fun_share = s["topic_distribution"].get("fun", 0) / 100
    structured_share = (
        s["topic_distribution"].get("studies", 0)
        + s["topic_distribution"].get("logistics", 0)
        + s["topic_distribution"].get("work", 0)
    ) / 100

    extraversion = _clip01(
        0.4 * min(s["avg_emoji_per_message"] / 1.5, 1)
        + 0.3 * min(s["avg_exclamations_per_message"] / 1.5, 1)
        + 0.3 * fun_share
    )
    conscientiousness = _clip01(0.5 * structured_share + 0.5 * (s["imperative_rate"] / 100))
    openness = _clip01(s["topic_diversity"])
    agreeableness = _clip01(
        0.5 * _clip01((s["sentiment"]["avg_compound"] + 1) / 2)
        + 0.5 * (1 - s["self_reference_ratio"] / 100)
    )
    neuroticism = _clip01(
        0.6 * min(s["sentiment"]["volatility"] / 0.6, 1) + 0.4 * (s["sentiment"]["pct_negative"] / 100)
    )

    return {
        "scores": {
            "extraversion": round(extraversion * 100),
            "conscientiousness": round(conscientiousness * 100),
            "openness": round(openness * 100),
            "agreeableness": round(agreeableness * 100),
            "neuroticism": round(neuroticism * 100),
        },
        "basis": {
            "extraversion": "emoji + exclamation frequency, share of lighthearted/fun-topic messages",
            "conscientiousness": "share of studies/work/logistics topics, rate of direct/organizing language",
            "openness": "how evenly messages spread across different topic domains",
            "agreeableness": "positivity baseline, how much they reference others vs. themselves",
            "neuroticism": "swings in emotional tone message-to-message, rate of negative-toned messages",
        },
        "disclaimer": "Estimated from writing-style patterns only - not a clinical or diagnostic assessment.",
    }


def _compare_contexts(group, dm):
    insights = []
    if abs(dm["sentiment"]["avg_compound"] - group["sentiment"]["avg_compound"]) > 0.1:
        direction = "more positive" if dm["sentiment"]["avg_compound"] > group["sentiment"]["avg_compound"] else "more negative"
        insights.append(
            f"Tone runs {direction} in DMs than in groups "
            f"({dm['sentiment']['avg_compound']:+.2f} vs {group['sentiment']['avg_compound']:+.2f} avg sentiment)."
        )
    if abs(dm["avg_emoji_per_message"] - group["avg_emoji_per_message"]) > 0.3:
        direction = "more" if dm["avg_emoji_per_message"] > group["avg_emoji_per_message"] else "less"
        insights.append(f"Uses {direction} emoji in DMs than in groups.")
    if abs(dm["self_reference_ratio"] - group["self_reference_ratio"]) > 10:
        direction = "more" if dm["self_reference_ratio"] > group["self_reference_ratio"] else "less"
        insights.append(f"Talks about themself {direction} in DMs than in groups.")
    if abs(dm["sentiment"]["volatility"] - group["sentiment"]["volatility"]) > 0.1:
        direction = "more emotionally variable" if dm["sentiment"]["volatility"] > group["sentiment"]["volatility"] else "more emotionally steady"
        insights.append(f"Comes across {direction} in DMs than in groups.")
    if not insights:
        insights.append("Behavior looks broadly consistent between groups and DMs so far.")
    return insights


def build_all_profiles(sources):
    """sources: list of {"label": str, "context": "group"|"dm", "messages": [...]}
    Returns {person_name: profile_dict}. Case-scoped: people are identified
    by exact sender-name match within this set of sources, same as ingestion's
    own resolution rule (see ingestion.py's _find_or_create_person).
    """
    names = set()
    for src in sources:
        names.update(m["sender"] for m in src["messages"])

    global_freq = Counter()
    for src in sources:
        for m in src["messages"]:
            global_freq.update(content_words(m["text"]))
    global_total = sum(global_freq.values()) or 1

    profiles = {}
    for name in names:
        person_sources = [
            {
                "label": src["label"],
                "context": src["context"],
                "messages": [m for m in src["messages"] if m["sender"] == name],
            }
            for src in sources
        ]
        profiles[name] = build_person_profile_from_sources(name, person_sources, global_freq, global_total)
    return profiles


def build_person_profile_from_sources(display_name, sources, global_freq, global_total):
    """Core aggregation, shared by both callers: build_all_profiles above
    (case-scoped, filters by exact name) and combined_profile.py's
    build_combined_profile (cross-case, filters by identifier id instead of
    name so a person with different raw names in different cases still
    pools correctly). `sources` messages must already be filtered to just
    this one person - this function does no filtering of its own.
    """
    all_texts = []
    by_context_texts = {}
    by_source = {}

    for src in sources:
        texts = [m["text"] for m in src["messages"]]
        if not texts:
            continue
        all_texts.extend(texts)
        by_context_texts.setdefault(src["context"], []).extend(texts)
        by_source[src["label"]] = {
            "context": src["context"],
            **_compute_stats(texts),
        }

    overall = _compute_stats(all_texts)
    by_context = {ctx: _compute_stats(texts) for ctx, texts in by_context_texts.items()}

    comparison = None
    if "group" in by_context and "dm" in by_context:
        comparison = _compare_contexts(by_context["group"], by_context["dm"])

    return {
        "name": display_name,
        "message_count": len(all_texts),
        "sources": list(by_source.keys()),
        "overall": overall,
        "by_context": by_context,
        "by_source": by_source,
        "vocabulary": _word_stats(all_texts, global_freq, global_total),
        "traits": _estimate_traits(overall),
        "context_comparison": comparison,
    }
