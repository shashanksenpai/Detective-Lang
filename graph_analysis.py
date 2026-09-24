"""Relationship graph computation - Phase 4 of the CLAUDE.md roadmap.

Edges are derived entirely from signals the attribution engine already
computes (category profiles, adjacent-turn structure) rather than a new
hand-authored metric. Case-scoped: nodes only mean "one real person" within
one case's own resolution, consistent with how every other case-scoped
endpoint works.

Deliberately NOT the combined cross-case view the original roadmap line
mentioned - that adds real complexity (cross-case edge computation, node
dedup across merged people) on top of an already-substantial first version.
Revisit once the case-scoped graph is actually being used.
"""
from statistics import mean

import numpy as np
from sqlmodel import Session

from db import engine as db_engine
from detective import CATEGORY_LEXICONS, emotion_features_batch
from engine_cache import get_engine, load_case_sources

MIN_ADJACENT_EXCHANGES = 2

# Tone of the exchanges between two people, from the mean sentiment compound of the
# replies they exchanged (hinglish_sentiment.py: P(pos) - P(neg); it reads wording,
# not sarcasm or subtext). The warm/tense cut-offs were set when this was VADER and
# re-checked against the Hinglish model's score distribution (similar mean and spread). "unclear" is not a tone: it means
# too few exchanges to call one, so a pair with two replies is never labelled
# "tense" on the strength of a single unlucky message.
WARM_ABOVE = 0.15
TENSE_BELOW = -0.05
TONE_MIN_EXCHANGES = 5

PROFESSIONAL_CATEGORIES = {"work", "studies", "logistics"}
PERSONAL_CATEGORIES = {"fun", "food"}


def tone_for(sentiment_between, interaction_count):
    """warm | neutral | tense | unclear - see WARM_ABOVE / TENSE_BELOW above."""
    if sentiment_between is None or interaction_count < TONE_MIN_EXCHANGES:
        return "unclear"
    if sentiment_between > WARM_ABOVE:
        return "warm"
    if sentiment_between < TENSE_BELOW:
        return "tense"
    return "neutral"


def _dominant_topic(profile):
    vec = _category_vector(profile)
    cats = list(CATEGORY_LEXICONS)
    return cats[int(np.argmax(vec))] if vec.sum() > 0 else None


def _category_vector(profile):
    return np.array([profile[c] for c in CATEGORY_LEXICONS])


def _cosine(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / denom)


def _adjacent_exchange_texts(sources):
    """Maps frozenset({person_a, person_b}) -> texts spoken at the moment
    one replies right after the other. The same adjacency structure
    DetectiveEngine.build() already walks for transition_counts, just
    keeping the actual text here instead of only a count.

    Walked one source at a time: the last message of one chat and the first
    of the next were never a reply to each other, but flattening every source
    into one list counted them as one.
    """
    exchanges = {}
    for src in sources:
        messages = src["messages"]
        for prev, curr in zip(messages, messages[1:]):
            if prev["sender"] == curr["sender"]:
                continue
            pair = frozenset({prev["sender"], curr["sender"]})
            exchanges.setdefault(pair, []).append(curr["text"])
    return exchanges


def build_graph(case_id: int):
    det_engine = get_engine(case_id)
    people = list(det_engine.centroids.keys())

    with Session(db_engine) as session:
        sources = load_case_sources(session, case_id)
    exchanges = _adjacent_exchange_texts(sources)

    nodes = [
        {
            "id": p,
            "message_count": len(det_engine.messages_by_sender[p]),
            "dominant_topic": _dominant_topic(det_engine.category_profiles[p]),
        }
        for p in people
    ]

    edges = []
    for i, p in enumerate(people):
        for q in people[i + 1:]:
            texts = exchanges.get(frozenset({p, q}), [])
            interaction_count = len(texts)
            uncertain = interaction_count < MIN_ADJACENT_EXCHANGES

            topic_overlap = _cosine(
                _category_vector(det_engine.category_profiles[p]),
                _category_vector(det_engine.category_profiles[q]),
            )
            sentiment_between = (
                mean(e["compound"] for e in emotion_features_batch(texts)) if not uncertain else None
            )

            edges.append({
                "source": p,
                "target": q,
                "interaction_count": interaction_count,
                "topic_overlap": round(topic_overlap, 3),
                "sentiment_between": round(sentiment_between, 3) if sentiment_between is not None else None,
                "tone": tone_for(sentiment_between, interaction_count),
                "uncertain": uncertain,
            })

    return {"nodes": nodes, "edges": edges}


def infer_relationship(case_id: int, people: list):
    det_engine = get_engine(case_id)
    for p in people:
        if p not in det_engine.category_profiles:
            raise ValueError(f"'{p}' is not a known person in this case")

    with Session(db_engine) as session:
        sources = load_case_sources(session, case_id)

    person_set = set(people)
    interaction_texts = [
        curr["text"]
        for src in sources
        for prev, curr in zip(src["messages"], src["messages"][1:])
        if prev["sender"] in person_set and curr["sender"] in person_set and prev["sender"] != curr["sender"]
    ]
    interaction_count = len(interaction_texts)
    uncertain = interaction_count < MIN_ADJACENT_EXCHANGES

    if uncertain:
        return {
            "people": people,
            "interaction_count": interaction_count,
            "uncertain": True,
            "dominant_shared_topic": None,
            "topic_overlap": None,
            "sentiment_between": None,
            "tone": "unclear",
            "relationship_lean": None,
        }

    category_vecs = [_category_vector(det_engine.category_profiles[p]) for p in people]
    avg_category = np.mean(category_vecs, axis=0)
    cats = list(CATEGORY_LEXICONS)
    dominant_shared_topic = cats[int(np.argmax(avg_category))] if avg_category.sum() > 0 else None

    pairwise_overlaps = [
        _cosine(category_vecs[i], category_vecs[j])
        for i in range(len(people)) for j in range(i + 1, len(people))
    ]
    topic_overlap = mean(pairwise_overlaps) if pairwise_overlaps else None

    sentiment_between = mean(e["compound"] for e in emotion_features_batch(interaction_texts))

    relationship_lean = None
    if dominant_shared_topic in PROFESSIONAL_CATEGORIES:
        relationship_lean = "professional"
    elif dominant_shared_topic in PERSONAL_CATEGORIES:
        relationship_lean = "personal"

    return {
        "people": people,
        "interaction_count": interaction_count,
        "uncertain": False,
        "dominant_shared_topic": dominant_shared_topic,
        "topic_overlap": round(topic_overlap, 3) if topic_overlap is not None else None,
        "sentiment_between": round(sentiment_between, 3),
        "tone": tone_for(sentiment_between, interaction_count),
        "relationship_lean": relationship_lean,
    }
