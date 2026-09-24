"""Evaluation harness for the sentiment scorer: plain VADER (what the app used
to run on) against the Hinglish model (hinglish_sentiment.py), on held-out
data neither has trained on.

    python eval_sentiment.py             # evaluate the current model
    python eval_sentiment.py --retrain   # download data if needed, retrain, save, evaluate

Each scorer is judged by its own natural decision rule: VADER by its standard
+-0.05 compound cut-off, the model by its most probable class. Held-out sets:
SentiMix test (Hindi-English tweets), a held-out slice of the Hinglish YouTube
comments, English tweets (a regression check - Hinglish support must not cost
English), and the blind-labeled chat messages (sample_hinglish_chat_labels.json:
'dev' may be studied, 'test' is the honest number; only 157 messages, 19 of them
negative, so per-class figures there are noisy).

Run this after any change to hinglish_sentiment.py, its training data, or the
preprocessing - same discipline as eval_attribution.py.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

import hinglish_sentiment as hs
import sentiment_data as sd

LABELS = sd.LABELS
EVAL_SETS = ["sentimix_test", "youtube_holdout", "english_test", "chat_dev", "chat_test"]


def metrics(y_true, y_pred):
    idx = {l: i for i, l in enumerate(LABELS)}
    cm = np.zeros((3, 3), int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    recall = [cm[i, i] / max(cm[i].sum(), 1) for i in range(3)]
    precision = [cm[i, i] / max(cm[:, i].sum(), 1) for i in range(3)]
    f1 = [2 * p * r / (p + r) if p + r else 0.0 for p, r in zip(precision, recall)]
    return {"n": int(cm.sum()), "accuracy": float(np.trace(cm) / cm.sum()), "macro_f1": float(np.mean(f1)),
            "recall": {l: float(r) for l, r in zip(LABELS, recall)}}


def vader_labels(texts):
    out = []
    for t in texts:
        c = hs._vader.polarity_scores(t)["compound"]
        out.append("negative" if c <= -0.05 else "positive" if c >= 0.05 else "neutral")
    return out


def model_labels(analyzer, texts):
    scores = analyzer.polarity_scores_batch(texts)
    return [LABELS[int(np.argmax([s["neg"], s["neu"], s["pos"]]))] for s in scores]


def evaluate(analyzer, data):
    results = {}
    for name in EVAL_SETS:
        texts = [t for t, _ in data[name]]
        y = [l for _, l in data[name]]
        results[name] = {"vader": metrics(y, vader_labels(texts)), "model": metrics(y, model_labels(analyzer, texts))}
    return results


def show(results):
    print(f"\n{'held-out set':16} {'n':>5} | {'VADER acc':>9} {'macroF1':>8} {'neg-rec':>8} | {'model acc':>9} {'macroF1':>8} {'neg-rec':>8} | {'F1 change':>9}")
    for name in EVAL_SETS:
        v, m = results[name]["vader"], results[name]["model"]
        print(f"{name:16} {v['n']:5} | {v['accuracy']*100:8.1f}% {v['macro_f1']*100:7.1f}% {v['recall']['negative']*100:7.0f}% "
              f"| {m['accuracy']*100:8.1f}% {m['macro_f1']*100:7.1f}% {m['recall']['negative']*100:7.0f}% | {(m['macro_f1']-v['macro_f1'])*100:+8.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true", help="download data, retrain, save the model, then evaluate")
    args = ap.parse_args()

    data = sd.load()
    if args.retrain:
        rows, weights, emoji_ok = sd.training_set(data)
        t0 = time.time()
        print(f"training on {len(rows)} examples ({len(data['train_hinglish'])} Hinglish, {len(data['train_english'])} English, "
              f"{len(data['train_chat'])} chat-register)...")
        artifact = hs.train_model(rows, weights, emoji_ok)
        hs.save_model(artifact)
        print(f"trained in {time.time()-t0:.0f}s -> {os.path.basename(hs.MODEL_PATH)} ({os.path.getsize(hs.MODEL_PATH)/1e6:.1f} MB)")
        hs._analyzer = None   # make get_analyzer() reload the fresh file

    analyzer = hs.get_analyzer()
    if analyzer.is_fallback:
        sys.exit("No trained model found (hinglish_sentiment_model.joblib). Run: python eval_sentiment.py --retrain")
    results = evaluate(analyzer, data)
    show(results)
    if args.retrain:
        with open(os.path.join(os.path.dirname(hs.MODEL_PATH), "sentiment_metrics.json"), "w", encoding="utf-8") as f:
            json.dump({"trained": time.strftime("%Y-%m-%d"), "results": results}, f, indent=1)
        print("\nwrote sentiment_metrics.json")


if __name__ == "__main__":
    main()
