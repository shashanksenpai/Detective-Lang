"""Cross-case identity resolution - Phase 2 of the CLAUDE.md roadmap.

Suggests candidate matches between people in different cases and records
them as MergeSuggestion rows. Never merges anything automatically - see
server.py's accept/reject endpoints for the only place a merge actually
happens.

Tier notes:
- "hard": the roadmap's original "hard-match (phone/username)" assumed
  structured identifier fields that don't exist yet (only WhatsApp's raw
  sender-name string is captured today; Phase 3 brings real platform IDs).
  Until then, "hard" evidence is a raw sender string that looks like a
  phone number matching identically across two cases - strong but heuristic.
- "fuzzy": name similarity (edit-distance + token containment).
- "soft" (embedding/stylometric similarity): currently DISABLED, not just
  under-tuned - see SOFT_TIER_DISABLED's note by STYLOMETRIC_SIMILARITY_
  THRESHOLD below. Improvement Stage testing (eval_identity.py) proved
  neither signal separates true matches from unrelated people at this
  corpus size. Both scores are still computed and included in `evidence`
  for transparency, they just don't gate a suggestion on their own.
"""
import json
import re
from difflib import SequenceMatcher
from statistics import mean

import numpy as np
from sqlmodel import Session, select

from db import engine as db_engine
from detective import embed_texts, stylometric_features
from models import Identifier, MergeSuggestion, Message, PersonCase

NAME_SIMILARITY_THRESHOLD = 0.82

# Rate-per-100-words of common closed-class words - a classic authorship
# signal (Mosteller-Wallace / Burrows' Delta style): captures HOW someone
# writes, not WHAT they write about. This is what the soft tier actually
# needs, unlike topic embeddings (see stylometric_fingerprint below).
FUNCTION_WORDS = [
    "i", "you", "the", "a", "is", "it", "and", "to", "in", "that", "was",
    "for", "on", "but", "not", "so", "just", "really", "actually", "like",
    "what", "this", "if", "can", "will", "do", "my", "your", "we", "they",
    "he", "she", "no", "also", "then", "now", "still", "even", "though",
]

# SOFT_TIER_DISABLED: eval_identity.py's fixture (real people, permanent
# sample files) showed NEITHER embedding-centroid NOR this stylometric
# fingerprint cleanly separates true matches from unrelated people - a true
# negative ("Karan" vs "Aman") scored 0.913 stylometric similarity, higher
# than every true positive (max 0.851, one real person's own messages split
# in half). No threshold value fixes that; any cutoff either excludes every
# real match or admits false ones. Root cause looks like data volume, not
# algorithm choice: classic stylometric authorship techniques (Mosteller-
# Wallace / Burrows' Delta) are validated on thousands of words per author;
# these synthetic people have ~200-400 words total each, so each half of the
# true-positive split is only ~100-200 words - too little for function-word
# rates to be a stable per-person fingerprint. So the soft tier is disabled
# (see score_pair) rather than gated by a threshold that would just be
# overfit to this one 6-person fixture. Revisit once real usage provides
# meaningfully more messages per person - re-run eval_identity.py first to
# confirm separation actually improves before re-enabling.
STYLOMETRIC_SIMILARITY_THRESHOLD = None  # not used while the soft tier is disabled

PHONE_CHARS_RE = re.compile(r"^[\d\s\-()+]+$")


def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def _name_tokens(name: str):
    return set(re.findall(r"[a-z]+", normalize_name(name)))


def name_similarity(a: str, b: str) -> float:
    """Best of two measures: plain edit-distance ratio (catches typos/case
    differences) and token containment (catches "Karan" vs "Karan Mehta" -
    a first-name-only sender in one export vs a full name in another, which
    a pure edit-distance ratio penalizes too harshly for the length
    difference even though one name is fully contained in the other).
    """
    ratio = SequenceMatcher(None, normalize_name(a), normalize_name(b)).ratio()

    tokens_a, tokens_b = _name_tokens(a), _name_tokens(b)
    containment = 0.0
    if tokens_a and tokens_b:
        overlap = tokens_a & tokens_b
        if overlap:
            containment = len(overlap) / min(len(tokens_a), len(tokens_b))

    return max(ratio, containment)


def looks_like_phone(name: str) -> bool:
    stripped = name.strip()
    digits = re.sub(r"\D", "", stripped)
    return bool(PHONE_CHARS_RE.match(stripped)) and len(digits) >= 7


def normalize_phone(name: str) -> str:
    plus = "+" if name.strip().startswith("+") else ""
    return plus + re.sub(r"\D", "", name)


def _person_raw_names(session: Session, person_id: int):
    return list(
        session.exec(
            select(Identifier.raw_sender_name).where(Identifier.person_id == person_id).distinct()
        ).all()
    )


def _person_texts(session: Session, person_id: int):
    identifier_ids = session.exec(
        select(Identifier.id).where(Identifier.person_id == person_id)
    ).all()
    if not identifier_ids:
        return []
    return session.exec(
        select(Message.text).where(Message.identifier_id.in_(identifier_ids))
    ).all()


def person_centroid(session: Session, person_id: int):
    """Mean embedding of everything this person has said, across every case
    they currently belong to. None if they have no messages on file. Kept
    for evidence/transparency only - topic-embedding similarity no longer
    gates the soft tier (see stylometric_fingerprint below for why: it
    tracks topic/register, not authorship, and proved unreliable at
    small-corpus scale - unrelated people scored higher than a real match).
    """
    texts = _person_texts(session, person_id)
    if not texts:
        return None
    return np.mean(embed_texts(texts), axis=0)


def function_word_rates(texts):
    """Rate of each FUNCTION_WORDS entry per 100 words across `texts`."""
    total_words = 0
    counts = {w: 0 for w in FUNCTION_WORDS}
    for text in texts:
        words = [w.strip(".,!?;:\"'").lower() for w in text.split()]
        total_words += len(words)
        for w in words:
            if w in counts:
                counts[w] += 1
    total_words = total_words or 1
    return np.array([counts[w] / total_words * 100 for w in FUNCTION_WORDS])


def stylometric_fingerprint(texts):
    """Per-person authorship fingerprint: function-word rates (how often
    they lean on common closed-class words) combined with the existing
    stylometric_features (message length, emoji/punctuation/caps habits),
    averaged across their messages. This is what the soft tier compares now
    - it targets HOW someone writes, which is a more author-specific signal
    than topic embeddings (which target WHAT a message is about, and were
    never a good fit for cross-person identity matching - see
    person_centroid's docstring). Rough fixed scaling per dimension so no
    single feature dominates the cosine comparison - not learned, but good
    enough to be directionally sane.
    """
    if not texts:
        return None

    fw_rates = function_word_rates(texts) / 15.0

    feats = [stylometric_features(t) for t in texts]
    style_vec = np.array([
        mean(f["len"] for f in feats) / 20.0,
        mean(f["emoji_count"] for f in feats) / 2.0,
        mean(f["exclaim"] for f in feats) / 2.0,
        mean(f["question"] for f in feats) / 2.0,
        mean(f["caps_ratio"] for f in feats),
    ])

    return np.concatenate([fw_rates, style_vec])


def cosine_similarity(a, b):
    if a is None or b is None:
        return None
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / denom)


def person_stylometric_fingerprint(session: Session, person_id: int):
    return stylometric_fingerprint(_person_texts(session, person_id))


def _cached(cache, kind, compute, session, person_id):
    """Per-scan memo of a person's centroid/fingerprint. Without it
    score_pair recomputed (re-embedded!) each person once per *pair* they
    were in - 162 embedding passes to score 81 pairs on the Phase 5 paper-leak
    demo, ~30s per scan. Results are identical; only the repetition goes."""
    if cache is None:
        return compute(session, person_id)
    key = (kind, person_id)
    if key not in cache:
        cache[key] = compute(session, person_id)
    return cache[key]


def score_pair(session: Session, person_a_id: int, person_b_id: int, cache=None):
    """Returns (match_type, confidence, evidence_dict) for the strongest
    tier that clears its threshold, or None if nothing does. Evidence
    carries every tier's raw score (even sub-threshold ones) so the review
    UI can show why a suggestion was or wasn't made.
    """
    names_a = _person_raw_names(session, person_a_id)
    names_b = _person_raw_names(session, person_b_id)

    phone_match = None
    for na in names_a:
        if not looks_like_phone(na):
            continue
        for nb in names_b:
            if looks_like_phone(nb) and normalize_phone(na) == normalize_phone(nb):
                phone_match = normalize_phone(na)
                break
        if phone_match:
            break

    best_name_sim = max(
        (name_similarity(na, nb) for na in names_a for nb in names_b), default=0.0
    )

    embedding_sim = cosine_similarity(
        _cached(cache, 'centroid', person_centroid, session, person_a_id),
        _cached(cache, 'centroid', person_centroid, session, person_b_id),
    )
    stylometric_sim = cosine_similarity(
        _cached(cache, 'fingerprint', person_stylometric_fingerprint, session, person_a_id),
        _cached(cache, 'fingerprint', person_stylometric_fingerprint, session, person_b_id),
    )

    evidence = {
        "compared_names": {"a": names_a, "b": names_b},
        "name_similarity": round(best_name_sim, 3),
        "embedding_similarity": round(embedding_sim, 3) if embedding_sim is not None else None,
        "stylometric_similarity": round(stylometric_sim, 3) if stylometric_sim is not None else None,
        "phone_match": phone_match,
    }

    if phone_match:
        return "hard", 0.97, evidence
    if best_name_sim >= NAME_SIMILARITY_THRESHOLD:
        return "fuzzy", round(best_name_sim, 3), evidence
    # The "soft" tier deliberately never fires right now - see
    # SOFT_TIER_DISABLED's note. embedding_similarity/stylometric_similarity
    # are still computed and included in `evidence` above so a human
    # reviewing a fuzzy/hard suggestion can see them, and so this can be
    # re-enabled the moment real data volume justifies it.
    return None


def scan_for_matches(case_id: int):
    """Compares every person in this case against every person in every
    other case, recording a MergeSuggestion for any pair that clears a
    tier's threshold. Skips pairs that already have a suggestion (in any
    status) so a rejected call isn't silently re-litigated on the next scan.
    """
    with Session(db_engine) as session:
        this_case_people = set(
            session.exec(select(PersonCase.person_id).where(PersonCase.case_id == case_id)).all()
        )
        other_case_people = set(
            session.exec(select(PersonCase.person_id).where(PersonCase.case_id != case_id)).all()
        )

        cache = {}   # one scan: each person's centroid/fingerprint computed once
        for p in this_case_people:
            for q in other_case_people:
                if p == q:
                    continue
                lo, hi = (p, q) if p < q else (q, p)
                exists = session.exec(
                    select(MergeSuggestion).where(
                        MergeSuggestion.person_a_id == lo, MergeSuggestion.person_b_id == hi
                    )
                ).first()
                if exists:
                    continue

                result = score_pair(session, lo, hi, cache)
                if result is None:
                    continue
                match_type, confidence, evidence = result
                session.add(MergeSuggestion(
                    person_a_id=lo, person_b_id=hi, match_type=match_type,
                    confidence=confidence, evidence=json.dumps(evidence),
                ))

        session.commit()
