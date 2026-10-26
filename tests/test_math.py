"""Property tests for the mathematics in docs/MATHEMATICS.md (no ML models needed).

Where a property concerns the engine (coverage, template threshold, AI score
bounds, self-match), the test calls the engine's own code, not a re-implementation.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from integrity_engine.ai_detection import AIDetector
from integrity_engine.boilerplate import df_threshold
from integrity_engine.config import AIDetectionConfig, BoilerplateConfig
from integrity_engine.index import VectorIndex
from integrity_engine.models import Document, Sentence, SentencePair
from integrity_engine.normalize import jaccard
from integrity_engine.similarity import Passage, coverage

ROOT = Path(__file__).resolve().parent.parent
RNG = np.random.default_rng(42)
D = 384


def unit_rows(n, d=D):
    x = RNG.standard_normal((n, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_thm1_normalised_dot_is_cosine():
    raw = RNG.standard_normal((50, D)) * RNG.uniform(0.1, 10, size=(50, 1))
    cos = (raw @ raw.T) / np.outer(np.linalg.norm(raw, axis=1), np.linalg.norm(raw, axis=1))
    unit = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    assert np.allclose(np.linalg.norm(unit, axis=1), 1)
    assert np.allclose(unit @ unit.T, cos)


def test_thm2_cauchy_schwarz_and_self_is_maximum():
    E = unit_rows(200)
    G = E @ E.T
    assert G.max() <= 1 + 1e-12 and G.min() >= -1 - 1e-12
    assert np.allclose(np.diag(G), 1) and (G.argmax(axis=1) == np.arange(200)).all()


def test_thm3_gram_matrix_is_symmetric_psd_with_bounded_rank():
    E = unit_rows(60, d=20)  # n > d forces rank deficiency
    G = E @ E.T
    eig = np.linalg.eigvalsh(G)
    assert np.allclose(G, G.T) and eig.min() > -1e-9
    assert (eig > 1e-9).sum() <= 20


def test_thm4_distance_identity_and_identical_rankings():
    E = unit_rows(300)
    q = E[0]
    assert np.allclose(np.sum((E - q) ** 2, axis=1), 2 - 2 * (E @ q))
    assert np.array_equal(np.argsort(-(E @ q)), np.argsort(np.sum((E - q) ** 2, axis=1)))


def test_thm5_angular_triangle_inequality_and_chain_bound():
    E = unit_rows(300)
    t = RNG.integers(0, 300, size=(5000, 3))

    def ang(a, b):
        return np.arccos(np.clip(np.sum(E[a] * E[b], axis=1), -1, 1))

    assert (ang(t[:, 0], t[:, 1]) + ang(t[:, 1], t[:, 2]) - ang(t[:, 0], t[:, 2]) >= -1e-9).all()
    # chain bound: cos(a,b), cos(b,c) >= s  =>  cos(a,c) >= 2 s^2 - 1
    s = 0.95
    b = unit_rows(1)[0]
    for _ in range(200):
        a = b + 0.25 * unit_rows(1)[0]
        c = b + 0.25 * unit_rows(1)[0]
        a, c = a / np.linalg.norm(a), c / np.linalg.norm(c)
        if a @ b >= s and c @ b >= s:
            assert a @ c >= 2 * s * s - 1 - 1e-12


def test_thm6_random_unit_vectors_are_nearly_orthogonal():
    E = unit_rows(1500)
    c = (E @ E.T)[np.triu_indices(1500, 1)]
    assert abs(c.mean()) < 0.005
    assert abs(c.std() - 1 / math.sqrt(D)) < 0.003


def test_thm7_orthogonal_projector_removes_common_direction():
    common = unit_rows(1)[0]
    E = unit_rows(400) + 0.8 * common  # anisotropic cloud: all vectors lean one way
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    u = E.mean(axis=0)
    u /= np.linalg.norm(u)
    P = np.eye(D) - np.outer(u, u)
    assert np.allclose(P @ P, P) and np.allclose(P, P.T) and np.allclose(P @ u, 0)
    before = (E @ E.T)[np.triu_indices(400, 1)].mean()
    Ec = E @ P
    Ec /= np.linalg.norm(Ec, axis=1, keepdims=True)
    after = (Ec @ Ec.T)[np.triu_indices(400, 1)].mean()
    assert before > 0.3 and abs(after) < 0.02


def test_faiss_inner_product_equals_exact_cosine():
    E = unit_rows(100).astype(np.float32)
    idx = VectorIndex(D)
    idx.add(E, list(range(100)))
    scores, pos = idx.search(E[:10], 3)
    exact = np.sort(E[:10] @ E.T, axis=1)[:, ::-1][:, :3]
    assert np.allclose(scores, exact, atol=1e-5) and (pos[:, 0] == np.arange(10)).all()


def _doc(word_counts):
    d = Document(doc_id="q", path="q", modality="text")
    d.sentences = [Sentence(sid=i, text="x", n_words=w) for i, w in enumerate(word_counts)]
    return d


def _passage(sids, src):
    return Passage("q", src, [SentencePair(s, 0, 0.9) for s in sids], chunk_score=0.9)


def test_lemma8_coverage_bounds_and_no_double_counting():
    doc = _doc([10, 20, 30, 40])
    a = _passage([0, 1], "A")  # 30 words
    b = _passage([1, 2], "B")  # 50 words, overlaps A on sentence 1
    cov_a, cov_b, both = coverage(doc, [a]), coverage(doc, [b]), coverage(doc, [a, b])
    assert (cov_a, cov_b, both) == (30.0, 50.0, 60.0)  # union, not 30 + 50
    assert max(cov_a, cov_b) <= both <= min(100, cov_a + cov_b)
    assert coverage(doc, [_passage([0, 1, 2, 3], "C")]) == 100.0
    assert coverage(_doc([]), [a]) == 0.0


def test_prop10_copy_pair_is_never_template():
    cfg = BoilerplateConfig()
    for n in range(2, 5000, 37):
        assert 2 < df_threshold(n, cfg)
        assert df_threshold(n, cfg) == max(3, math.ceil(0.25 * n))


def test_jaccard_range_and_identity():
    assert jaccard(["a", "b"], ["a", "b"]) == 1.0
    assert jaccard(["a"], ["b"]) == 0.0
    assert 0.0 <= jaccard(["a", "b", "c"], ["b", "c", "d"]) <= 1.0


def test_lemma11_sigmoid_properties():
    sig = lambda z: 1 / (1 + math.exp(-z))  # noqa: E731
    zs = np.linspace(-20, 20, 401)
    vals = np.array([sig(z) for z in zs])
    assert ((vals > 0) & (vals < 1)).all() and (np.diff(vals) > 0).all()
    assert all(abs(sig(-z) - (1 - sig(z))) < 1e-12 for z in zs)


def test_prop12_ai_score_is_bounded_by_clipping():
    cal = json.loads((ROOT / "config" / "ai_calibration.json").read_text())
    bound = 3 * sum(abs(w) for w in cal["coef"])
    det = AIDetector(AIDetectionConfig())
    for sign in (1, -1):
        extreme = {
            "log_perplexity": -sign * 1e9,
            "burstiness": -sign * 1e9,
            "marker_rate": sign * 1e9,
            "classifier": sign * 1e9,
        }
        out = det.combine(extreme)
        contrib = out["logit_contributions"]  # rounded to 3 decimals by the engine
        assert all(abs(v) <= 3 * abs(w) + 1e-3 for v, w in zip(contrib.values(), cal["coef"]))
        z = out["intercept"] + sum(contrib.values())
        assert cal["intercept"] - bound - 1e-2 <= z <= cal["intercept"] + bound + 1e-2
        assert 0 < 1 / (1 + math.exp(-z)) < 1  # strictly inside (0, 1): never exactly 0% or 100%


@pytest.mark.parametrize("p,r", [(0.2, 0.9), (0.5, 0.5), (0.95, 0.84), (1.0, 0.1)])
def test_lemma13_f1_between_min_and_mean(p, r):
    f1 = 2 * p * r / (p + r)
    assert min(p, r) - 1e-12 <= f1 <= (p + r) / 2 + 1e-12


def test_prop14_clopper_pearson_three_of_three():
    lower = 0.025 ** (1 / 3)
    assert abs(lower**3 - 0.025) < 1e-12 and abs(lower - 0.2924) < 1e-4
