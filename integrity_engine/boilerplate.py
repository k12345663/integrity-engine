"""Detection of shared template text (case prompts, instructions, citations).

A sentence is boilerplate when a near-duplicate of it occurs in many distinct
documents. Copying between two students yields a document frequency (DF) of
2; a case prompt pasted by every team yields a DF close to the corpus size.
The DF threshold is ``max(min_doc_frequency, ceil(min_doc_fraction * N))``.

Organisers usually *know* the prompt, so a template directory can also be
supplied; any sentence matching it is boilerplate regardless of DF.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import BoilerplateConfig
from .index import VectorIndex
from .models import Document
from .normalize import clean_text, split_sentences

log = logging.getLogger(__name__)


def df_threshold(n_docs: int, cfg: BoilerplateConfig) -> int:
    return max(cfg.min_doc_frequency, math.ceil(cfg.min_doc_fraction * n_docs))


def load_template_sentences(template_dir: Optional[str]) -> List[str]:
    if not template_dir or not Path(template_dir).is_dir():
        return []
    sents: List[str] = []
    for path in sorted(Path(template_dir).glob("*.txt")):
        sents.extend(split_sentences(clean_text(path.read_text(encoding="utf-8"))))
    return sents


def mark_boilerplate(
    docs: List[Document],
    sentence_vectors: Dict[str, np.ndarray],
    cfg: BoilerplateConfig,
    template_vectors: Optional[np.ndarray] = None,
    min_words: int = 1,
) -> Dict[str, object]:
    """Set ``is_boilerplate`` / ``boilerplate_df`` on sentences in place."""
    usable = [d for d in docs if d.is_usable]
    threshold = df_threshold(len(usable), cfg)
    stats = {"df_threshold": threshold, "documents": len(usable), "sentences_marked": 0}
    if not cfg.enabled or not usable:
        return stats

    dim = next(iter(sentence_vectors.values())).shape[1]
    index: VectorIndex = VectorIndex(dim)
    for d in usable:
        index.add(sentence_vectors[d.doc_id], [(d.doc_id, s.sid) for s in d.sentences])

    k = min(len(index), 4 * len(usable) + 8)
    for d in usable:
        vecs = sentence_vectors[d.doc_id]
        scores, pos = index.search(vecs, k)
        tmpl = None
        if template_vectors is not None and len(template_vectors):
            tmpl = (vecs @ template_vectors.T).max(axis=1)
        for i, sent in enumerate(d.sentences):
            if sent.n_words < min_words:
                continue
            docs_with = {d.doc_id}
            for score, p in zip(scores[i], pos[i]):
                if p < 0 or score < cfg.near_duplicate_threshold:
                    break  # results are sorted by score
                docs_with.add(index.keys[p][0])
            sent.boilerplate_df = len(docs_with)
            from_template = tmpl is not None and tmpl[i] >= cfg.near_duplicate_threshold
            if len(docs_with) >= threshold or from_template:
                sent.is_boilerplate = True
                stats["sentences_marked"] += 1
    log.info(
        "Boilerplate: %d sentences marked (DF threshold %d over %d documents)",
        stats["sentences_marked"],
        threshold,
        len(usable),
    )
    return stats
