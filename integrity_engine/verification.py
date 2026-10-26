"""Cross-encoder verification of candidate passages (retrieve, then re-rank).

Bi-encoders (MiniLM) embed each text independently; that makes retrieval
cheap but compresses heavy paraphrases and same-topic text into the same
0.6-0.8 cosine band. A cross-encoder reads both passages *jointly* and is far
better at telling "same argument, reworded" from "same topic". It is ~100x
more expensive per pair, so it only scores the few window pairs that the
bi-encoder has already shortlisted.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)


class CrossEncoderVerifier:
    def __init__(self, model_name: Optional[str], batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._cache: Dict[Tuple[str, str], float] = {}
        self.calls = 0
        if model_name:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(model_name, device="cpu")
                log.info("Loaded cross-encoder %s", model_name)
            except Exception as exc:
                log.warning("Cross-encoder unavailable (%s); paraphrase verification falls back to cosine", exc)

    @property
    def available(self) -> bool:
        return self._model is not None

    def score(self, pairs: Sequence[Tuple[str, str]]) -> np.ndarray:
        todo = [p for p in pairs if p not in self._cache]
        if todo:
            self.calls += len(todo)
            scores = self._model.predict(list(todo), batch_size=self.batch_size, show_progress_bar=False)
            for p, s in zip(todo, np.atleast_1d(scores)):
                self._cache[p] = float(s)
        return np.array([self._cache[p] for p in pairs], dtype=np.float32)


def sentence_windows(sids: List[int], size: int) -> List[List[int]]:
    """Sliding windows of ``size`` consecutive sentence ids (stride 1)."""
    if len(sids) <= size:
        return [sids] if sids else []
    return [sids[i : i + size] for i in range(len(sids) - size + 1)]
