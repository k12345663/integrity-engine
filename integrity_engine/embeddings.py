"""Sentence-Transformers wrapper producing L2-normalised embeddings.

With unit-length vectors, the inner product equals cosine similarity, which
is what the FAISS ``IndexFlatIP`` computes.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from .config import EmbeddingConfig

log = logging.getLogger(__name__)


def _auto_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # pragma: no cover - torch always present in practice
        pass
    return "cpu"  # MPS is skipped on purpose: CPU results are bit-reproducible


class Embedder:
    def __init__(self, cfg: EmbeddingConfig):
        from sentence_transformers import SentenceTransformer

        self.cfg = cfg
        self.device = cfg.device or _auto_device()
        log.info("Loading embedding model %s on %s", cfg.model_name, self.device)
        self.model = SentenceTransformer(cfg.model_name, device=self.device)
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = get_dim()
        self.max_seq_length = self.model.max_seq_length

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer.tokenize(text))

    def encode(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self.model.encode(
            texts,
            batch_size=batch_size or self.cfg.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32)
