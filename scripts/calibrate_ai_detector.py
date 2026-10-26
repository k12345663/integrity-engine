#!/usr/bin/env python
"""Fit the AI-likelihood combiner on external, public labelled corpora.

Data (downloaded to .cache/, never committed):
* HC3 (Hello-SimpleAI/HC3, CC BY-SA 4.0): human vs ChatGPT (Dec 2022)
  answers from the finance, wiki_csai and open_qa subsets.
* SAGE (EndLessTime/SAGE): human IvyPanda student essays and Medium articles
  vs summary-aligned rewrites by GPT-4.1, Claude 3.7 Sonnet, Gemini 2.5 Flash.

We sample texts of >= 120 words, compute the detector's raw signals and fit a
standardised logistic regression. Stored with the weights: 5-fold CV AUROC,
single-feature AUROCs, and cross-source transfer (train on one corpus, test
on the other), which quantifies how badly detectors age.

Our own sample_submissions are *not* used here, so the evaluation in
EVALUATION.md stays out-of-sample.

    python scripts/calibrate_ai_detector.py --per_class 150
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from integrity_engine.ai_detection import AIDetector  # noqa: E402
from integrity_engine.config import AIDetectionConfig  # noqa: E402
from integrity_engine.normalize import clean_text, split_sentences  # noqa: E402

HC3 = "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/{}.jsonl"
SUBSETS = ["finance", "wiki_csai", "open_qa"]
SAGE = "https://huggingface.co/datasets/EndLessTime/SAGE/resolve/main/SAGE.csv"
SAGE_DOMAINS = ["ivypanda_essays", "medium_articles"]
CACHE = ROOT / ".cache" / "hc3"
SAGE_CACHE = ROOT / ".cache" / "sage" / "SAGE.csv"


def load_subset(name: str):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.jsonl"
    if not path.exists():
        print(f"downloading HC3/{name} ...")
        urllib.request.urlretrieve(HC3.format(name), path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    human, ai = [], []
    for r in rows:
        for a in r.get("human_answers", [])[:1]:
            human.append(clean_text(a))
        for a in r.get("chatgpt_answers", [])[:1]:
            ai.append(clean_text(a))
    return human, ai


def load_sage():
    """Return {domain: (human_texts, ai_texts)} from the SAGE csv."""
    import csv

    if not SAGE_CACHE.exists():
        SAGE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        print("downloading SAGE (129 MB) ...")
        urllib.request.urlretrieve(SAGE, SAGE_CACHE)
    csv.field_size_limit(10**8)
    out = {d: ([], []) for d in SAGE_DOMAINS}
    with open(SAGE_CACHE, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["domain"] in out:
                out[row["domain"]][0 if row["model"] == "human" else 1].append(clean_text(row["generation"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_class", type=int, default=300, help="samples per class (split across subsets)")
    ap.add_argument("--min_words", type=int, default=120)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    texts, labels, origin = [], [], []

    def take(pool, n, label, source):
        pool = [t for t in pool if len(t.split()) >= args.min_words]
        for t in rng.sample(pool, min(n, len(pool))):
            texts.append(" ".join(t.split()[:450]))
            labels.append(label)
            origin.append(source)

    # 30% HC3 (older ChatGPT), 70% SAGE (current frontier models).
    n_hc3 = int(0.3 * args.per_class) // len(SUBSETS)
    n_sage = (args.per_class - n_hc3 * len(SUBSETS)) // len(SAGE_DOMAINS)
    for name in SUBSETS:
        human, ai = load_subset(name)
        take(human, n_hc3, 0, "hc3")
        take(ai, n_hc3, 1, "hc3")
    for domain, (human, ai) in load_sage().items():
        take(human, n_sage, 0, "sage")
        take(ai, n_sage, 1, "sage")
    print(f"{len(texts)} samples ({sum(labels)} AI / {len(labels) - sum(labels)} human)")

    det = AIDetector(AIDetectionConfig(calibration_file=None))
    feats = []
    for i, t in enumerate(texts):
        sig = det.signals(split_sentences(t), f"hc3-{i}")
        feats.append([sig.get(f) if sig.get(f) is not None else np.nan for f in AIDetector.FEATURES])
        if (i + 1) % 50 == 0:
            print(f"  features {i + 1}/{len(texts)}")
    X = np.array(feats, dtype=float)
    y = np.array(labels)
    col_mean = np.nanmean(X, axis=0)
    X = np.where(np.isnan(X), col_mean, X)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    cv = StratifiedKFold(5, shuffle=True, random_state=args.seed)
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    cv_auc = roc_auc_score(y, proba)
    single = {
        f: float(max(roc_auc_score(y, X[:, i]), 1 - roc_auc_score(y, X[:, i])))
        for i, f in enumerate(AIDetector.FEATURES)
    }
    acc = float(np.mean((proba >= 0.5) == y))
    src = np.array(origin)
    transfer = {}
    for train, test in (("hc3", "sage"), ("sage", "hc3")):
        tr, te = src == train, src == test
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)).fit(X[tr], y[tr])
        transfer[f"train_{train}_test_{test}"] = float(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    per_source_feature_auroc = {
        s_: {
            f: float(roc_auc_score(y[src == s_], -X[src == s_, i] if f == "log_perplexity" else X[src == s_, i]))
            for i, f in enumerate(AIDetector.FEATURES)
        }
        for s_ in ("hc3", "sage")
    }
    model.fit(X, y)
    scaler, lr = model[0], model[1]
    out = {
        "source": "HC3 + SAGE (GPT-4.1 / Claude 3.7 / Gemini 2.5)",
        "n_samples": int(len(y)),
        "features": list(AIDetector.FEATURES),
        "mean": scaler.mean_.tolist(),
        "std": scaler.scale_.tolist(),
        "coef": lr.coef_[0].tolist(),
        "intercept": float(lr.intercept_[0]),
        "cv_auroc": float(cv_auc),
        "cv_accuracy_at_0.5": acc,
        "single_feature_auroc": single,
        "per_source_feature_auroc (direction: higher=AI)": per_source_feature_auroc,
        "cross_source_transfer_auroc": transfer,
        "seed": args.seed,
    }
    dest = ROOT / "config" / "ai_calibration.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k not in ("mean", "std")}, indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
