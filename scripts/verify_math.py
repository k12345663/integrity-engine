#!/usr/bin/env python
"""Numerically verify the mathematical claims in docs/MATHEMATICS.md on the real system.

Every check uses the engine's own embedding model, sample corpus and calibration
file, so the results are a proof-by-measurement for *this* system, not for
textbook vectors.

    python scripts/verify_math.py            # writes evaluation/math_verification.json
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from integrity_engine.config import Config  # noqa: E402
from integrity_engine.index import VectorIndex  # noqa: E402  (also fixes macOS OpenMP order)
from integrity_engine.ingestion import discover, load_document  # noqa: E402

RESULTS = {}


def check(name: str, ok: bool, detail: str) -> None:
    RESULTS[name] = {"pass": bool(ok), "detail": detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def main() -> int:
    warnings.filterwarnings("ignore")
    from integrity_engine.embeddings import Embedder

    cfg = Config()
    input_dir = str(ROOT / "sample_submissions")
    docs = [load_document(p, input_dir, cfg) for p in discover(input_dir, cfg)]
    docs = [d for d in docs if d.is_usable]
    texts, owner = [], []
    for d in docs:
        for s in d.sentences:
            if s.n_words >= 6:
                texts.append(s.text)
                owner.append(d.doc_id)
    owner = np.array(owner)
    emb = Embedder(cfg.embedding)
    E = emb.encode(texts).astype(np.float64)  # n x d, rows are unit vectors
    n, d = E.shape
    print(f"{n} sentence embeddings of dimension {d} from {len(docs)} documents\n")

    # T1 unit norm
    norms = np.linalg.norm(E, axis=1)
    check("T1 unit norm", np.allclose(norms, 1, atol=1e-5), f"max | ||e|| - 1 | = {np.abs(norms - 1).max():.2e}")

    # T2 dot product of unit vectors == cosine
    raw = emb.model.encode(texts[:200], normalize_embeddings=False, convert_to_numpy=True).astype(np.float64)
    cos_raw = (raw @ raw.T) / np.outer(np.linalg.norm(raw, axis=1), np.linalg.norm(raw, axis=1))
    dot_unit = E[:200] @ E[:200].T
    check(
        "T2 dot(unit) = cosine(raw)",
        np.allclose(cos_raw, dot_unit, atol=1e-5),
        f"max difference = {np.abs(cos_raw - dot_unit).max():.2e}",
    )

    # T3 Cauchy-Schwarz range and symmetry of the Gram matrix
    G = E @ E.T
    check(
        "T3 -1 <= cos <= 1", G.max() <= 1 + 1e-6 and G.min() >= -1 - 1e-6, f"min = {G.min():.4f}, max = {G.max():.4f}"
    )
    check(
        "T3b Gram matrix symmetric, diag = 1",
        np.allclose(G, G.T) and np.allclose(np.diag(G), 1, atol=1e-5),
        f"max |G - G^T| = {np.abs(G - G.T).max():.2e}",
    )

    # T4 positive semi-definite Gram matrix and rank <= d
    eig = np.linalg.eigvalsh(G)
    rank = int((eig > 1e-8 * eig.max()).sum())
    check("T4 Gram matrix PSD", eig.min() > -1e-8, f"smallest eigenvalue = {eig.min():.2e}; rank = {rank} <= d = {d}")

    # T5 cosine <-> Euclidean distance: ||a-b||^2 = 2 - 2 cos
    i, j = np.triu_indices(min(n, 300), 1)
    lhs = np.sum((E[i] - E[j]) ** 2, axis=1)
    rhs = 2 - 2 * G[i, j]
    check("T5 ||a-b||^2 = 2 - 2cos", np.allclose(lhs, rhs, atol=1e-6), f"max error = {np.abs(lhs - rhs).max():.2e}")
    # consequence: inner-product ranking == nearest-neighbour ranking
    q = E[0]
    order_ip = np.argsort(-(E @ q))
    order_l2 = np.argsort(np.sum((E - q) ** 2, axis=1))
    check("T5b IP ranking == L2 ranking", np.array_equal(order_ip[:50], order_l2[:50]), "top-50 neighbours identical")

    # T6 angular triangle inequality on random triples
    rng = np.random.default_rng(0)
    trip = rng.integers(0, n, size=(20000, 3))
    ang = lambda a, b: np.arccos(np.clip(np.sum(E[a] * E[b], axis=1), -1, 1))  # noqa: E731
    gap = ang(trip[:, 0], trip[:, 1]) + ang(trip[:, 1], trip[:, 2]) - ang(trip[:, 0], trip[:, 2])
    check("T6 angle triangle inequality", gap.min() >= -1e-7, f"20,000 triples, min slack = {gap.min():.2e} rad")

    # T7 orthogonality / anisotropy: unrelated sentences are NOT orthogonal on average
    other = owner[:, None] != owner[None, :]
    cross = G[other]
    mu = E.mean(axis=0)
    check(
        "T7 anisotropy (mean cross-doc cosine > 0)",
        cross.mean() > 0,
        f"mean cosine between sentences of different files = {cross.mean():.3f} (orthogonal would be 0); "
        f"||mean vector|| = {np.linalg.norm(mu):.3f}",
    )
    # removing the common direction with an orthogonal projector P = I - u u^T
    u = mu / np.linalg.norm(mu)
    P = np.eye(d) - np.outer(u, u)
    check(
        "T7b P is an orthogonal projector",
        np.allclose(P @ P, P) and np.allclose(P, P.T) and abs(P @ u).max() < 1e-9,
        "P^2 = P, P^T = P, P u = 0",
    )
    Ec = E @ P
    Ec /= np.linalg.norm(Ec, axis=1, keepdims=True)
    Gc = Ec @ Ec.T
    check(
        "T7c projection reduces baseline similarity",
        abs(Gc[other].mean()) < abs(cross.mean()),
        f"mean cross-doc cosine {cross.mean():.3f} -> {Gc[other].mean():.3f} after removing the common direction",
    )

    # T8 random high-dimensional vectors are nearly orthogonal (concentration of measure)
    R = rng.standard_normal((2000, d))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    rc = (R @ R.T)[np.triu_indices(2000, 1)]
    check(
        "T8 random unit vectors ~ orthogonal",
        abs(rc.mean()) < 0.01 and abs(rc.std() - 1 / math.sqrt(d)) < 0.005,
        f"mean = {rc.mean():.4f}, std = {rc.std():.4f} vs theory 1/sqrt(d) = {1 / math.sqrt(d):.4f}",
    )

    # T9 FAISS inner-product search equals brute-force matrix product
    idx = VectorIndex(d)
    idx.add(E.astype(np.float32), list(range(n)))
    s, p = idx.search(E[:50].astype(np.float32), 5)
    brute = np.sort(G[:50], axis=1)[:, ::-1][:, :5]
    self_top = np.allclose(s[:, 0], 1, atol=1e-5)
    dup = int((p[:, 0] != np.arange(50)).sum())
    check(
        "T9 FAISS IndexFlatIP == exact cosine",
        np.allclose(s, brute, atol=1e-4) and self_top,
        f"max difference = {np.abs(s - brute).max():.2e}; top score is always 1 (the vector itself or an exact "
        f"duplicate - {dup}/50 ties with identical template sentences), which is why self-matches must be excluded",
    )

    # T10 logistic bounds under z-clipping, from the shipped calibration
    cal = json.loads((ROOT / "config" / "ai_calibration.json").read_text())
    zmax = cal["intercept"] + 3 * sum(abs(w) for w in cal["coef"])
    zmin = cal["intercept"] - 3 * sum(abs(w) for w in cal["coef"])
    sig = lambda z: 1 / (1 + math.exp(-z))  # noqa: E731
    check(
        "T10 AI score is bounded away from 0 and 100",
        sig(zmax) < 1 and sig(zmin) > 0,
        f"logit in [{zmin:.2f}, {zmax:.2f}] -> AI% in [{100 * sig(zmin):.1e}, 100 - {100 * (1 - sig(zmax)):.1e}]",
    )

    # T11 Clopper-Pearson interval for 3/3 detected relationships
    lower = 0.025 ** (1 / 3)
    check("T11 3/3 recall lower 95% bound", abs(lower - 0.292) < 0.001, f"(0.025)^(1/3) = {lower:.3f}")

    out = ROOT / "evaluation" / "math_verification.json"
    out.write_text(json.dumps({"n_vectors": n, "dimension": d, "checks": RESULTS}, indent=2))
    failed = [k for k, v in RESULTS.items() if not v["pass"]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed -> {out.relative_to(ROOT)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
