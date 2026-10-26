"""Sentence-aligned chunking with overlap.

Chunks are built from whole sentences so every chunk maps back to exact
sentence ids (and therefore pages / timestamps). Boilerplate and reference-list
sentences are excluded: they are reported separately and never inflate scores.
"""

from __future__ import annotations

from typing import Callable, List

from .config import ChunkingConfig
from .models import Chunk, Document, Sentence

TokenCounter = Callable[[str], int]


def approx_tokens(text: str) -> int:
    # ~1.3 word pieces per English word for MiniLM's WordPiece vocabulary.
    return int(len(text.split()) * 1.3) + 1


def chunk_sentences(
    doc_id: str, sentences: List[Sentence], cfg: ChunkingConfig, count_tokens: TokenCounter = approx_tokens
) -> List[Chunk]:
    chunks: List[Chunk] = []
    current: List[Sentence] = []
    tokens = 0

    def emit() -> None:
        text = " ".join(s.text for s in current)
        chunks.append(Chunk(len(chunks), doc_id, [s.sid for s in current], text, tokens))

    for sent in sentences:
        n = count_tokens(sent.text)
        if current and tokens + n > cfg.max_tokens:
            emit()
            current = current[-cfg.overlap_sentences :] if cfg.overlap_sentences else []
            tokens = sum(count_tokens(s.text) for s in current)
        current.append(sent)
        tokens += n

    if current:
        tail_is_new = not chunks or current[-1].sid != chunks[-1].sentence_ids[-1]
        if tail_is_new:
            if chunks and tokens < cfg.min_tokens:
                # Fold a short tail into the previous chunk instead of indexing a fragment.
                prev = chunks[-1]
                extra = [s for s in current if s.sid not in prev.sentence_ids]
                prev.sentence_ids.extend(s.sid for s in extra)
                prev.text = prev.text + " " + " ".join(s.text for s in extra)
                prev.n_tokens += sum(count_tokens(s.text) for s in extra)
            else:
                emit()
    return chunks


def chunk_document(doc: Document, cfg: ChunkingConfig, count_tokens: TokenCounter = approx_tokens) -> None:
    doc.chunks = chunk_sentences(doc.doc_id, doc.scoring_sentences(), cfg, count_tokens)
