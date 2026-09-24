"""A small hand-curated lexicon of Romanized Hindi/Urdu emotion words, used as
extra input features by the Hinglish sentiment model (hinglish_sentiment.py).

Why it exists: the trained model learns sentiment words from tweets and
comments, so common chat emotion words it saw rarely (udaas, pareshan, naraaz,
dukh, lajawab, ...) were under-weighted, and "acha nahi" (the negator comes
AFTER the word in Hindi) was easy to misread. A lexicon states that knowledge
directly, is spelling-variant aware, and is explainable ("it read 'udaas' as
negative").

Deliberately conservative: a word is listed only if it is emotional in almost
every chat use. Ambiguous words that double as common English or neutral words
are left out ("dar" / "par" / "ban" / "kal" / "aansu" / "pagal"); the model
already covers them as n-grams. Strength 1 = ordinary, 2 = strong.
"""
import re
from typing import Dict, List, Tuple

import numpy as np

POSITIVE: Dict[str, float] = {
    # good / great
    "accha": 1, "acha": 1, "achha": 1, "achchha": 1, "achi": 1, "acchi": 1, "achhi": 1, "acche": 1, "achhe": 1,
    "badhiya": 1.5, "badiya": 1.5, "badhia": 1.5, "mast": 1.5, "zabardast": 2, "jhakaas": 2, "jhakkas": 2,
    "kamaal": 2, "kamal": 1.5, "shandaar": 2, "lajawab": 2, "lajwab": 2, "behtareen": 2, "behtar": 1, "umda": 1.5,
    "sahi": 0.6, "ghazab": 2, "gazab": 2, "faadu": 1.5, "dhamaal": 2, "dhamakedaar": 2, "bindaas": 1.5,
    "superb": 2, "awesome": 2, "amazing": 2, "perfect": 1.5, "excellent": 2, "best": 1.5, "legend": 1.5,
    # joy / fun
    "maza": 1.5, "mazaa": 1.5, "mazedaar": 2, "majedar": 2, "khush": 1.5, "khushi": 1.5, "khoosh": 1.5,
    "hansi": 1, "masti": 1, "jashn": 1.5, "celebrate": 1, "party": 0.5,
    # love / liking / beauty
    "pyaar": 1.5, "pyara": 1.5, "pyari": 1.5, "pyaare": 1.5, "mohabbat": 1.5, "pasand": 1, "sundar": 1.5,
    "khoobsurat": 2, "khubsurat": 2, "cute": 1, "love": 1.5, "loved": 1.5,
    # thanks / praise / wishes
    "shukriya": 1.5, "shukriyaa": 1.5, "dhanyavaad": 1.5, "dhanyawad": 1.5, "shukr": 1, "thanks": 1, "thank": 1,
    "mubarak": 1.5, "badhai": 1.5, "congrats": 1.5, "congratulations": 1.5, "shabash": 2, "shabaash": 2,
    "sabaash": 2, "wah": 1.5, "waah": 1.5, "wow": 1, "tarif": 1, "tareef": 1, "hausla": 1, "himmat": 1,
    "mashallah": 1.5, "alhamdulillah": 1.5, "kamyab": 1.5, "safal": 1.5, "garv": 1.5, "proud": 1.5,
    # relief / comfort / winning
    "sukoon": 1.5, "chain": 0.8, "aaram": 0.8, "raahat": 1.5, "relief": 1.5, "tasalli": 1, "jeet": 1.5,
    "jeeta": 1.5, "jeete": 1.5, "jeetna": 1.5, "khoobsurati": 1.5,
}
NEGATIVE: Dict[str, float] = {
    # bad / awful
    "bura": 1.5, "buri": 1.5, "bure": 1.5, "bekaar": 2, "bekar": 2, "bakwaas": 2, "bakwas": 2, "faltu": 1.5,
    "ganda": 1.5, "gandi": 1.5, "gande": 1.5, "kharab": 1.5, "kharaab": 1.5, "ghatiya": 2, "ghatia": 2,
    "worst": 2, "horrible": 2, "terrible": 2, "disgusting": 2, "awful": 2, "barbaad": 2, "tabah": 2,
    "boring": 1, "bore": 1, "irritating": 1.5, "irritate": 1.5,
    # anger / hate / annoyance
    "gussa": 2, "gusse": 2, "gussey": 2, "naraaz": 2, "naraz": 2, "nafrat": 2, "chidh": 1.5, "chidhna": 1.5,
    "hate": 2, "hated": 2, "bewakoof": 1.5, "harami": 2, "kamina": 2, "kameena": 2, "lanat": 2,
    # sadness / worry / fear
    "udaas": 2, "udas": 2, "dukhi": 2, "dukh": 2, "dukhaya": 2, "gham": 1.5, "rona": 1.5, "roya": 1.5,
    "royi": 1.5, "pareshan": 2, "pareshaan": 2, "parishan": 2, "tension": 1.5, "chinta": 1.5, "fikar": 1.5,
    "fikr": 1.5, "darr": 1.5, "darta": 1.5, "darti": 1.5, "dara": 1, "ghabra": 1.5, "ghabrahat": 1.5,
    "ghabraya": 1.5, "akela": 1, "akelapan": 1.5, "afsos": 1.5, "pachtava": 1.5, "sharam": 1.5, "sharminda": 1.5,
    "disappointed": 1.5, "frustrated": 1.5, "stressed": 1.5, "anxious": 1.5, "sad": 1.5, "upset": 1.5,
    # pain / trouble / harm
    "dard": 1.5, "takleef": 1.5, "takliif": 1.5, "mushkil": 1, "musibat": 2, "dhoka": 2, "jhooth": 1.5,
    "jhoota": 1.5, "jhuth": 1.5, "galat": 1.5, "galti": 1, "thak": 1, "thaka": 1, "thakan": 1.5, "toot": 1,
    "toota": 1.5, "tootna": 1.5, "bimaar": 1.5, "bukhar": 1, "nuksaan": 1.5, "nuksan": 1.5,
}
# two-word expressions (matched on adjacent tokens); (phrase) -> signed strength
PHRASES: Dict[str, float] = {
    "ban gaya": 1.5, "ban gayi": 1.5, "chha gaya": 1.5, "chha gayi": 1.5, "dil jeet": 1.5, "paisa vasool": 1.5,
    "dil khush": 2, "maza aa": 1.5, "mazaa aa": 1.5, "acha lag": 1, "achha lag": 1, "acchi lag": 1,
    "mood off": -1.5, "dil toot": -2, "dil dukha": -2, "sar dard": -1.2, "darr lag": -1.5, "dar lag": -1.5,
    "bura lag": -1.5, "buri lag": -1.5, "rona aa": -1.5, "gussa aa": -2,
}
# "na" is left out on purpose: in chat it is usually a tag question ("aa raha hai na?" = "you're coming, right?").
NEGATORS = {"nahi", "nahin", "nhi", "nahee", "mat", "not", "no", "never", "dont", "cant", "isnt", "wasnt", "aint", "didnt", "doesnt"}
INTENSIFIERS = {"bahut", "bohot", "bohut", "bhot", "bht", "boht", "kaafi", "ekdum", "bilkul", "behad", "full", "sabse", "zyada",
                "jyada", "so", "very", "really", "too", "extremely", "totally"}

_TOKEN_RE = re.compile(r"[a-z']+")
FEATURE_NAMES = ["Hinglish lexicon: positive", "Hinglish lexicon: negative", "emoji: positive", "emoji: negative"]

# Emoji, by how they read in FRIENDS' CHAT (not in tweets, where 😂 / 🔥 often mock). Acknowledgement
# emoji (thumbs-up, OK-hand, peace sign) and the polite/passive-aggressive slight smile are left out: they are
# not polarity.
# Only the unambiguous ones are listed; emoji that flip with context (the crying face, skull,
# pleading face, sweat-smile, folded hands, ...) are in hinglish_sentiment.AMBIGUOUS_EMOJI and
# count for nothing here - the words decide. All emoji are removed from the text the n-gram model
# and VADER see, so this list is the only way an emoji moves the score.
EMOJI_POSITIVE = set(
    "😂🤣😆😄😁😀😃😊😍🥰"
    "😘💖💕❤♥💗💓💞😻🤗🎉"
    "🎊🥳🔥💪👏🙌😎✨💯🌷"
    "🌸🌹☺😋😜😝🤩😇🏆🎂"
    "🫶💐"
)
EMOJI_NEGATIVE = set(
    "😡😠🤬😤😞😔😢😩😫😟"
    "😰😨😱😖😣💔👎😒🙄🤢"
    "🤮😷🤒💩😕"
)


def _tokens(text: str) -> List[str]:
    return [t.replace("'", "") for t in _TOKEN_RE.findall(text.lower())]


def matches(text: str) -> List[Tuple[str, float]]:
    """[(word or phrase, signed strength after negation/intensifier)] for every
    lexicon hit in the text - the explainable part of the feature."""
    toks = _tokens(text)
    out: List[Tuple[str, float]] = []
    i = 0
    while i < len(toks):
        phrase = toks[i] + " " + toks[i + 1] if i + 1 < len(toks) else None
        if phrase in PHRASES:
            value, width, label = PHRASES[phrase], 2, phrase
        elif toks[i] in POSITIVE and POSITIVE[toks[i]] > 0:
            value, width, label = POSITIVE[toks[i]], 1, toks[i]
        elif toks[i] in NEGATIVE:
            value, width, label = -NEGATIVE[toks[i]], 1, toks[i]
        else:
            i += 1
            continue
        # intensifier just before -> stronger
        if any(t in INTENSIFIERS for t in toks[max(0, i - 2):i]):
            value *= 1.4
        # negation: Hindi puts it AFTER ("acha nahi"), English before ("not good"); either flips it
        window = toks[max(0, i - 2):i] + toks[i + width:i + width + 2]
        if any(t in NEGATORS for t in window):
            value = -0.8 * value
        out.append((label, value))
        i += width
    return out


def emoji_counts(text: str) -> Tuple[int, int]:
    """(unambiguous positive emoji, unambiguous negative emoji) in the raw text."""
    return (sum(1 for c in text if c in EMOJI_POSITIVE), sum(1 for c in text if c in EMOJI_NEGATIVE))


def lexicon_features(texts, raw_texts=None) -> np.ndarray:
    """Four dense features per text: how much positive and negative emotion
    vocabulary it holds (after negation), and how many positive / negative
    unambiguous emoji it holds, each squashed with tanh. `raw_texts` are the
    same messages before emoji were stripped (defaults to `texts`)."""
    raw_texts = raw_texts if raw_texts is not None else texts
    rows = []
    for t, raw in zip(texts, raw_texts):
        hits = matches(t)
        pos = sum(v for _, v in hits if v > 0)
        neg = -sum(v for _, v in hits if v < 0)
        ep, en = emoji_counts(raw)
        rows.append([np.tanh(pos / 1.5), np.tanh(neg / 1.5), np.tanh(ep / 2.0), np.tanh(en / 2.0)])
    return np.array(rows, dtype=float)
