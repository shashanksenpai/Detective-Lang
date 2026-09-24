"""Datasets for training and evaluating the Hinglish sentiment model
(hinglish_sentiment.py). External data is downloaded once into
.sentiment_cache/ (git-ignored; it is not redistributed with the project).

Sources (all public, on Hugging Face):
- RTT1/SentiMix - the SemEval-2020 Task 9 Hindi-English code-mixed tweets
  (14k train / 3k dev / 3k test, Roman script; license: OpenRAIL). Mostly Indian
  political Twitter.
- shae2977/hinglish-youtube-sentiments-dataset - Hinglish YouTube comments on
  Indian entertainment/food videos (CC-BY-4.0).
- Abhishek4896/hindi-english-code-mixed-tweets-sentiment - ~500 short
  sentences (MIT). Templated, near-duplicate text: used for training only,
  never as evidence (any split of it scores ~100%).
- mteb/tweet_sentiment_extraction - English tweets, so the model does not lose
  English (license not stated by the source; used locally, not redistributed).
Plus, in this repo: sample_hinglish_chat_train.tsv (chat-register training
lines) and sample_hinglish_chat_labels.json (the blind-labeled chat set).

Splits are deterministic (fixed seeds) so a retrain reproduces the same
held-out sets.
"""
import csv
import json
import os
import random
import urllib.request
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".sentiment_cache")
HF = "https://huggingface.co/datasets/"
FILES = {
    "sentimix_train.txt": HF + "RTT1/SentiMix/resolve/main/train_14k_split_conll.txt",
    "sentimix_dev.txt": HF + "RTT1/SentiMix/resolve/main/dev_3k_split_conll.txt",
    "sentimix_test.txt": HF + "RTT1/SentiMix/resolve/main/Hindi_test_unalbelled_conll_updated.txt",
    "sentimix_test_labels.csv": HF + "RTT1/SentiMix/resolve/main/test_labels_hinglish.txt",
    "youtube.csv": HF + "shae2977/hinglish-youtube-sentiments-dataset/resolve/main/yt_hinglish_comments_dataset.csv",
    "tweets_small.csv": HF + "Abhishek4896/hindi-english-code-mixed-tweets-sentiment/resolve/main/hindi_english_code_mixed_tweets_sentiment.csv",
    "english_train.jsonl": HF + "mteb/tweet_sentiment_extraction/resolve/main/train.jsonl",
    "english_test.jsonl": HF + "mteb/tweet_sentiment_extraction/resolve/main/test.jsonl",
}
LABELS = ("negative", "neutral", "positive")
Rows = List[Tuple[str, str]]


def fetch_all() -> None:
    """Downloads any missing dataset file into the cache."""
    os.makedirs(CACHE, exist_ok=True)
    for name, url in FILES.items():
        path = os.path.join(CACHE, name)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            continue
        print(f"downloading {name} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(path, "wb") as out:
            out.write(resp.read())


def _p(name: str) -> str:
    return os.path.join(CACHE, name)


def _sentimix(path: str, labels_path: str = None) -> Rows:
    """CoNLL: 'meta<TAB>uid<TAB>label' then 'token<TAB>lang' lines; blank line between tweets."""
    labels_by_uid = {}
    if labels_path:
        with open(labels_path, encoding="utf-8") as f:
            next(f)
            for line in f:
                uid, label = line.strip().split(",")
                labels_by_uid[uid] = label.strip().lower()
    rows, uid, label, tokens = [], None, None, []

    def flush():
        if uid is not None and tokens:
            rows.append((" ".join(tokens), label or labels_by_uid.get(uid)))

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("meta"):
                flush()
                parts = line.split("\t")
                uid, label, tokens = parts[1], (parts[2].strip().lower() if len(parts) > 2 else None), []
            elif line.strip() and uid is not None:
                tokens.append(line.split("\t")[0])
        flush()
    return [(t, l) for t, l in rows if l in LABELS]


def _csv(path: str, text_col: str, label_col: str) -> Rows:
    with open(path, encoding="utf-8", newline="") as f:
        return [(r[text_col].strip(), r[label_col].strip().lower()) for r in csv.DictReader(f)
                if r[label_col].strip().lower() in LABELS and r[text_col].strip()]


def _english(path: str) -> Rows:
    with open(path, encoding="utf-8") as f:
        return [(d["text"].strip(), d["label_text"]) for d in map(json.loads, f) if d["text"].strip()]


def _split(rows: Rows, seed: int, train_fraction: float) -> Tuple[Rows, Rows]:
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    k = int(len(rows) * train_fraction)
    return rows[:k], rows[k:]


def chat_train_rows() -> Rows:
    with open(os.path.join(HERE, "sample_hinglish_chat_train.tsv"), encoding="utf-8") as f:
        return [(t.strip(), l) for l, t in (line.rstrip("\n").split("\t", 1) for line in f if line.strip())]


def chat_labeled() -> Dict[str, Rows]:
    """The blind-labeled chat messages, by split ('dev' / 'test')."""
    with open(os.path.join(HERE, "sample_hinglish_chat_labels.json"), encoding="utf-8") as f:
        msgs = json.load(f)["messages"]
    return {s: [(m["text"], m["label"]) for m in msgs if m["split"] == s] for s in ("dev", "test")}


CHAT_TRAIN_WEIGHT = 3.0


def load() -> Dict[str, Rows]:
    """Everything, already split. Keys: train (weights via train_weights()),
    and the held-out evaluation sets."""
    fetch_all()
    yt_train, yt_hold = _split(_csv(_p("youtube.csv"), "comment", "sentiment"), seed=11, train_fraction=0.7)
    tw_train, _tw_hold = _split(_csv(_p("tweets_small.csv"), "tweet", "sentiment"), seed=12, train_fraction=0.7)
    en_train, _ = _split(_english(_p("english_train.jsonl")), seed=13, train_fraction=12000 / 27481)
    chat = chat_labeled()
    return {
        "train_hinglish": _sentimix(_p("sentimix_train.txt")) + yt_train + tw_train,
        "train_english": en_train,
        "train_chat": chat_train_rows(),
        "sentimix_dev": _sentimix(_p("sentimix_dev.txt")),
        "sentimix_test": _sentimix(_p("sentimix_test.txt"), _p("sentimix_test_labels.csv")),
        "youtube_holdout": yt_hold,
        "english_test": _english(_p("english_test.jsonl"))[:2000],
        "chat_dev": chat["dev"],
        "chat_test": chat["test"],
    }


def training_set(data: Dict[str, Rows]) -> Tuple[Rows, List[float], List[bool]]:
    """(rows, sample weights, emoji_ok). emoji_ok is True only for the chat-register rows:
    the emoji features are learned from them alone (tweet/comment emoji mean something else)."""
    other = len(data["train_hinglish"]) + len(data["train_english"])
    rows = data["train_hinglish"] + data["train_english"] + data["train_chat"]
    weights = [1.0] * other + [CHAT_TRAIN_WEIGHT] * len(data["train_chat"])
    emoji_ok = [False] * other + [True] * len(data["train_chat"])
    return rows, weights, emoji_ok
