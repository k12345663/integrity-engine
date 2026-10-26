#!/usr/bin/env python
"""Model-selection benchmark behind the embedding / cross-encoder choice.

Positives: the four paraphrased paragraph pairs (sub_03 -> sub_04).
Negatives: every cross-document paragraph pair from unrelated submissions
on the *same case* (the hard negatives a same-topic corpus produces).
A usable scorer needs min(positive) > max(negative), i.e. a positive margin.

    python scripts/benchmark_models.py
"""

from __future__ import annotations

import itertools
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import dataset_content as C  # noqa: E402
import numpy as np  # noqa: E402

import integrity_engine.index  # noqa: E402,F401  (macOS OpenMP import order)

MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-small-en-v1.5",
    "thenlper/gte-small",
    "cross:cross-encoder/stsb-distilroberta-base",
]


def main() -> None:
    warnings.filterwarnings("ignore")
    from sentence_transformers import CrossEncoder, SentenceTransformer

    pos = list(zip(C.C_PARAS, C.D_PARAS))
    groups = {
        "A": [C.A_SITUATION, C.A_ECONOMICS, C.A_RECOMMENDATION_X, C.A_RISK_Y, C.A_RISK_Z, C.A_RISK_3],
        "B": [C.B_INTRO, C.B_NUMBERS, C.B_CLOSE],
        "C": C.C_PARAS,
        "E": C.E_PARAS,
        "F": C.F_OWN,
        "G": C.G_AI,
        "H": C.H_COL_LEFT + C.H_COL_RIGHT,
        "I": C.I_PARAS,
        "O": C.O_AI,
        "L": [C.PITCH_11],
    }
    neg = [
        (x, y)
        for g1, g2 in itertools.combinations(groups, 2)
        if {g1, g2} != {"G", "O"}
        for x in groups[g1]
        for y in groups[g2]
    ]
    ai_ai = [(x, y) for x in C.G_AI for y in C.O_AI]

    print(f"{len(pos)} positive pairs, {len(neg)} same-topic negative pairs, {len(ai_ai)} AI-vs-AI pairs\n")
    print("| Model | Paraphrase min | Paraphrase mean | Negative max | Negative p99 | AI-vs-AI max | Margin |")
    print("|---|---|---|---|---|---|---|")
    for name in MODELS:
        if name.startswith("cross:"):
            model = CrossEncoder(name[6:])

            def score(pairs):
                return np.array(model.predict(pairs, show_progress_bar=False))
        else:
            model = SentenceTransformer(name)

            def score(pairs):
                a = model.encode([p[0] for p in pairs], normalize_embeddings=True)
                b = model.encode([p[1] for p in pairs], normalize_embeddings=True)
                return (a * b).sum(1)

        p, n, a = score(pos), score(neg), score(ai_ai)
        print(
            f"| {name.replace('cross:', '(cross-encoder) ')} | {p.min():.3f} | {p.mean():.3f} | {n.max():.3f} | "
            f"{np.percentile(n, 99):.3f} | {a.max():.3f} | {p.min() - n.max():+.3f} |"
        )


if __name__ == "__main__":
    main()
