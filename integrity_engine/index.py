"""Vector index over normalised embeddings (FAISS, with a NumPy fallback).

``flat`` is exact brute-force inner product - right for a corpus of hundreds
to low millions of chunks. ``hnsw`` trades a little recall for sub-linear
search and is what the scaling brief proposes for 10k+ submissions.
"""

from __future__ import annotations

import logging
import os
import platform
from typing import Generic, List, Sequence, Tuple, TypeVar

import numpy as np

log = logging.getLogger(__name__)

T = TypeVar("T")

if platform.system() == "Darwin":
    # macOS wheels of torch and faiss each bundle their own libomp. Loading
    # both aborts ("libomp already initialized") or segfaults inside FAISS's
    # parallel search. We tolerate the duplicate runtime, load torch first and
    # keep FAISS single-threaded so the two runtimes never run concurrently.
    # Linux wheels share one runtime and need none of this.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        import torch  # noqa: F401  (import order matters, see above)
    except ImportError:  # pragma: no cover
        pass

try:
    import faiss  # type: ignore

    HAVE_FAISS = True
    if platform.system() == "Darwin":
        faiss.omp_set_num_threads(1)
except ImportError:  # pragma: no cover
    HAVE_FAISS = False


class VectorIndex(Generic[T]):
    def __init__(self, dim: int, kind: str = "flat"):
        self.dim = dim
        self.kind = kind
        self.keys: List[T] = []
        self._vectors: List[np.ndarray] = []
        self._index = None

    def add(self, vectors: np.ndarray, keys: Sequence[T]) -> None:
        assert len(vectors) == len(keys)
        if len(keys):
            self._vectors.append(np.ascontiguousarray(vectors, dtype=np.float32))
            self.keys.extend(keys)
        self._index = None

    def __len__(self) -> int:
        return len(self.keys)

    def _build(self) -> None:
        matrix = np.vstack(self._vectors) if self._vectors else np.zeros((0, self.dim), np.float32)
        if HAVE_FAISS:
            if self.kind == "hnsw":
                index = faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_INNER_PRODUCT)
                index.hnsw.efSearch = 128
            else:
                index = faiss.IndexFlatIP(self.dim)
            index.add(matrix)
            self._index = index
        else:
            self._index = matrix

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (scores, positions) of the ``k`` nearest neighbours per query."""
        if self._index is None:
            self._build()
        k = min(k, len(self.keys))
        if k == 0 or len(queries) == 0:
            return np.zeros((len(queries), 0), np.float32), np.zeros((len(queries), 0), np.int64)
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if HAVE_FAISS:
            return self._index.search(queries, k)
        sims = queries @ self._index.T
        idx = np.argsort(-sims, axis=1)[:, :k]
        return np.take_along_axis(sims, idx, axis=1), idx
