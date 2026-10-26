#!/usr/bin/env python
"""Out-of-sample check of the paraphrase decision rule on public data.

Datasets (downloaded to .cache/public/, never committed):
* MRPC (GLUE): 1,725 test sentence pairs from news, human-labelled as
  paraphrase / not paraphrase.
* PAWS (labeled_final): 8,000 test pairs built by word swapping and back
  translation; negatives share almost all their words with the positive.

The engine's *fixed production thresholds* are applied without any tuning
on these datasets. Baselines (Jaccard, Levenshtein ratio, MiniLM cosine) get
a threshold tuned for F1 on each dataset's validation split, which favours
the baselines. Results go to evaluation/public_paraphrase.md / .json.

    python scripts/benchmark_public_paraphrase.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from integrity_engine.anchors import anchors_agree, raw_anchors  # noqa: E402
from integrity_engine.config import Config  # noqa: E402
from integrity_engine.index import HAVE_FAISS  # noqa: E402,F401  (macOS OpenMP import order)
from integrity_engine.normalize import jaccard, words  # noqa: E402

HF = "https://huggingface.co/datasets/{}/resolve/main/{}"
DATASETS = {
    "MRPC": ("nyu-mll/glue", "mrpc/{}-00000-of-00001.parquet", "sentence1", "sentence2"),
    "PAWS": ("google-research-datasets/paws", "labeled_final/{}-00000-of-00001.parquet", "sentence1", "sentence2"),
}
CACHE = ROOT / ".cache" / "public"


def load(name: str, split: str):
    import pyarrow.parquet as pq

    repo, pattern, a, b = DATASETS[name]
    path = CACHE / f"{name.lower()}_{split}.parquet"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        print(f"downloading {name} {split} ...")
        urllib.request.urlretrieve(HF.format(repo, pattern.format(split)), path)
    rows = pq.read_table(path).to_pylist()
    return [r[a] for r in rows], [r[b] for r in rows], np.array([int(r["label"]) for r in rows])


def prf(pred: np.ndarray, y: np.ndarray):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": round(p, 3),
        "recall": round(r, 3),
        "f1": round(2 * p * r / (p + r) if p + r else 0.0, 3),
        "accuracy": round(float((pred == y).mean()), 3),
    }


def best_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    cands = np.unique(np.round(scores, 3))
    return float(max(cands, key=lambda t: prf((scores >= t).astype(int), y)["f1"]))


def auroc(scores, y) -> float:
    from sklearn.metrics import roc_auc_score

    return round(float(roc_auc_score(y, scores)), 3)


def main() -> None:
    warnings.filterwarnings("ignore")
    from rapidfuzz import fuzz
    from sentence_transformers import CrossEncoder, SentenceTransformer

    cfg = Config().matching
    bi = SentenceTransformer(Config().embedding.model_name, device="cpu")
    ce = CrossEncoder(cfg.cross_encoder_model, device="cpu")

    def scores(s1, s2):
        a = bi.encode(s1, batch_size=128, normalize_embeddings=True)
        b = bi.encode(s2, batch_size=128, normalize_embeddings=True)
        cos = (a * b).sum(1)
        cross = np.array(ce.predict(list(zip(s1, s2)), batch_size=64, show_progress_bar=False))
        jac = np.array([jaccard(words(x), words(z)) for x, z in zip(s1, s2)])
        lev = np.array([fuzz.ratio(x, z) / 100 for x, z in zip(s1, s2)])
        anch = np.array([anchors_agree(raw_anchors(x) & raw_anchors(z)) for x, z in zip(s1, s2)])
        return {"cosine": cos, "cross": cross, "jaccard": jac, "levenshtein": lev, "anchor": anch}

    results = {}
    for name in DATASETS:
        vs1, vs2, vy = load(name, "validation")
        ts1, ts2, ty = load(name, "test")
        print(f"{name}: {len(ty)} test pairs ({ty.mean():.0%} positive)")
        v, t = scores(vs1, vs2), scores(ts1, ts2)
        rows = {}
        for feat in ("jaccard", "levenshtein", "cosine"):
            thr = best_threshold(v[feat], vy)
            rows[f"{feat} (threshold {thr:.2f} tuned on validation)"] = {
                **prf((t[feat] >= thr).astype(int), ty),
                "auroc": auroc(t[feat], ty),
            }
        rows["cross-encoder score (AUROC only)"] = {"auroc": auroc(t["cross"], ty)}
        # The engine's rule with its fixed production thresholds.
        direct = t["cosine"] >= cfg.sentence_threshold
        verified = (t["cosine"] >= cfg.window_prefilter) & (t["cross"] >= cfg.cross_encoder_threshold)
        anchored = verified & (t["anchor"] | (t["cross"] >= cfg.cross_encoder_strong))
        rows["engine: direct path only (cos >= 0.80, fixed)"] = prf(direct.astype(int), ty)
        rows["engine: direct OR cross-encoder (fixed)"] = prf((direct | verified).astype(int), ty)
        rows["engine: direct OR cross-encoder + anchor rule (fixed, default)"] = prf(
            (direct | anchored).astype(int), ty
        )
        results[name] = {"n_test": int(len(ty)), "positive_rate": round(float(ty.mean()), 3), "rows": rows}

    out = ROOT / "evaluation"
    out.mkdir(exist_ok=True)
    (out / "public_paraphrase.json").write_text(json.dumps(results, indent=2))
    lines = ["# Public paraphrase benchmark (auto-generated by scripts/benchmark_public_paraphrase.py)", ""]
    for name, res in results.items():
        lines += [
            f"## {name} test ({res['n_test']} pairs, {res['positive_rate']:.0%} paraphrases)",
            "",
            "| Method | Precision | Recall | F1 | Accuracy | AUROC |",
            "|---|---|---|---|---|---|",
        ]
        for label, m in res["rows"].items():
            lines.append(
                f"| {label} | {m.get('precision', '-')} | {m.get('recall', '-')} | {m.get('f1', '-')} | "
                f"{m.get('accuracy', '-')} | {m.get('auroc', '-')} |"
            )
        lines.append("")
    (out / "public_paraphrase.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
