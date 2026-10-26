"""Two-stage overlap detection: chunk retrieval, then sentence-level evidence.

Stage 1 (recall): every query chunk retrieves its ``top_k`` nearest chunks
from the FAISS index; neighbours from the query's own document are discarded
(self-match exclusion) and the rest are kept above a loose cosine threshold.

Stage 2 (precision + explainability), inside each candidate chunk pair:

* direct path - sentence pairs whose bi-encoder cosine clears
  ``sentence_threshold`` (verbatim copies, light edits, read-out transcripts);
* verified path - 3-sentence windows shortlisted by cosine and confirmed by a
  cross-encoder (paraphrases that reorder and reword whole arguments).

Aligned sentences are grouped into contiguous passages; weak, isolated
matches are dropped. The percentage score is the share of the query's
scoring words that fall inside accepted passages.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .anchors import anchors_agree
from .config import MatchingConfig
from .index import VectorIndex
from .models import Document, SentencePair
from .verification import CrossEncoderVerifier, sentence_windows


@dataclass
class Passage:
    query_doc: str
    matched_doc: str
    pairs: List[SentencePair]
    chunk_score: float
    kind: str = "peer"
    query_chunks: List[int] = field(default_factory=list)
    matched_chunks: List[int] = field(default_factory=list)

    @property
    def query_sids(self) -> List[int]:
        return [p.query_sid for p in self.pairs]

    @property
    def matched_sids(self) -> List[int]:
        return sorted({p.matched_sid for p in self.pairs})

    @property
    def sentence_score(self) -> float:
        return float(np.mean([p.score for p in self.pairs]))

    @property
    def methods(self) -> List[str]:
        return sorted({p.method for p in self.pairs})


@dataclass
class EmbeddedDoc:
    """A document plus its vectors. Sentence ids are row indices into
    ``doc.sentences`` and ``sentence_vecs``."""

    doc: Document
    sentence_vecs: np.ndarray
    chunk_vecs: np.ndarray
    window_sids: List[List[int]] = field(default_factory=list)
    window_vecs: Optional[np.ndarray] = None

    def window_text(self, w: int) -> str:
        return " ".join(self.doc.sentences[s].text for s in self.window_sids[w])


def build_windows(ed: EmbeddedDoc, encode, sizes: Sequence[int], min_words: int = 4) -> None:
    sids = [s.sid for s in ed.doc.scoring_sentences() if s.n_words >= min_words]
    windows: List[List[int]] = []
    for size in sizes:
        windows.extend(w for w in sentence_windows(sids, size) if w not in windows)
    ed.window_sids = windows
    ed.window_vecs = encode([ed.window_text(i) for i in range(len(ed.window_sids))])


def build_chunk_index(edocs: Sequence[EmbeddedDoc], kind: str = "flat") -> VectorIndex:
    dim = edocs[0].sentence_vecs.shape[1]
    index: VectorIndex = VectorIndex(dim, kind)
    for ed in edocs:
        index.add(ed.chunk_vecs, [(ed.doc.doc_id, c.cid) for c in ed.doc.chunks])
    return index


Best = Dict[int, Tuple[SentencePair, float, int, int]]


def _offer(best: Best, pair: SentencePair, chunk_score: float, qc: int, mc: int) -> None:
    """Keep the strongest evidence per query sentence; direct beats verified."""
    prev = best.get(pair.query_sid)
    rank = lambda p: (p.method == "direct", p.score)  # noqa: E731
    if prev is None or rank(pair) > rank(prev[0]):
        best[pair.query_sid] = (pair, chunk_score, qc, mc)


def _passages_from_pairs(
    query: EmbeddedDoc, matched_doc: str, best: Best, cfg: MatchingConfig, kind: str
) -> List[Passage]:
    """Group aligned query sentences into contiguous runs and filter weak ones."""
    order = {s.sid: i for i, s in enumerate(query.doc.scoring_sentences())}
    runs: List[List[int]] = []
    for sid in sorted(best, key=lambda sid: order[sid]):
        if runs and order[sid] - order[runs[-1][-1]] <= 2:  # tolerate one unmatched sentence
            runs[-1].append(sid)
        else:
            runs.append([sid])

    passages = []
    for run in runs:
        pairs = [best[sid][0] for sid in run]
        if len(run) < cfg.min_passage_sentences:
            p = pairs[0]
            strong = p.method == "direct" and p.score >= cfg.single_sentence_threshold
            if not strong or query.doc.sentences[p.query_sid].n_words < cfg.single_sentence_min_words:
                continue
        passages.append(
            Passage(
                query_doc=query.doc.doc_id,
                matched_doc=matched_doc,
                pairs=pairs,
                chunk_score=max(best[sid][1] for sid in run),
                kind=kind,
                query_chunks=sorted({best[sid][2] for sid in run}),
                matched_chunks=sorted({best[sid][3] for sid in run}),
            )
        )
    return passages


def find_passages(
    query: EmbeddedDoc,
    sources: Dict[str, EmbeddedDoc],
    index: VectorIndex,
    cfg: MatchingConfig,
    verifier: Optional[CrossEncoderVerifier] = None,
    kind: str = "peer",
) -> List[Passage]:
    """All accepted overlap passages between ``query`` and indexed sources."""
    if not query.doc.chunks or len(index) == 0:
        return []
    # Over-fetch so that discarding the query's own chunks still leaves top_k.
    scores, positions = index.search(query.chunk_vecs, cfg.top_k + len(query.doc.chunks))
    q_ok = {s.sid for s in query.doc.scoring_sentences() if s.n_words >= cfg.min_sentence_words}
    best: Dict[str, Best] = defaultdict(dict)

    for qc, (row_scores, row_pos) in enumerate(zip(scores, positions)):
        chunk_sids = set(query.doc.chunks[qc].sentence_ids)
        q_sids = [sid for sid in query.doc.chunks[qc].sentence_ids if sid in q_ok]
        q_wins = [i for i, w in enumerate(query.window_sids) if w[0] in chunk_sids]
        for score, p in zip(row_scores, row_pos):
            if p < 0 or score < cfg.chunk_candidate_threshold:
                break  # FAISS returns neighbours sorted by score
            m_doc, m_cid = index.keys[p]
            if m_doc == query.doc.doc_id:
                continue  # self-match exclusion
            src = sources[m_doc]
            m_chunk = src.doc.chunks[m_cid].sentence_ids
            m_sids = [sid for sid in m_chunk if src.doc.sentences[sid].n_words >= cfg.min_sentence_words]

            # Direct path: sentence-to-sentence cosine.
            if q_sids and m_sids:
                sims = query.sentence_vecs[q_sids] @ src.sentence_vecs[m_sids].T
                for i, q_sid in enumerate(q_sids):
                    j = int(np.argmax(sims[i]))
                    if sims[i, j] >= cfg.sentence_threshold:
                        _offer(
                            best[m_doc],
                            SentencePair(q_sid, m_sids[j], float(sims[i, j]), "direct"),
                            float(score),
                            qc,
                            m_cid,
                        )

            # Verified path: window shortlist -> cross-encoder confirmation.
            m_set = set(m_chunk)
            m_wins = [i for i, w in enumerate(src.window_sids) if m_set.intersection(w)]
            if not q_wins or not m_wins:
                continue
            wsims = query.window_vecs[q_wins] @ src.window_vecs[m_wins].T
            shortlist = []
            for i, qw in enumerate(q_wins):
                j = int(np.argmax(wsims[i]))
                if wsims[i, j] >= cfg.window_prefilter:
                    shortlist.append((qw, m_wins[j], float(wsims[i, j])))
            if not shortlist:
                continue
            if verifier is not None and verifier.available:
                ver = verifier.score([(query.window_text(a), src.window_text(b)) for a, b, _ in shortlist])
                threshold = cfg.cross_encoder_threshold
            else:
                ver = np.array([c for _, _, c in shortlist])
                threshold = cfg.window_cosine_fallback
            for (qw, mw, _), v in zip(shortlist, ver):
                if v < threshold:
                    continue
                mw_sids = src.window_sids[mw]
                if cfg.require_anchor and v < cfg.cross_encoder_strong:
                    q_anchors = set().union(*(query.doc.sentences[s].anchors for s in query.window_sids[qw]))
                    m_anchors = set().union(*(src.doc.sentences[s].anchors for s in mw_sids))
                    if not anchors_agree(q_anchors & m_anchors):
                        continue  # same topic, but no shared specific detail
                q_window = query.window_sids[qw]
                keep = set(q_window)
                if cfg.loo_attribution and len(q_window) > 1 and verifier is not None and verifier.available:
                    m_text = src.window_text(mw)
                    loo = verifier.score(
                        [(" ".join(query.doc.sentences[x].text for x in q_window if x != s), m_text) for s in q_window]
                    )
                    keep = {s for s, without in zip(q_window, loo) if without <= v + cfg.loo_margin}
                # Explain at sentence level: pair each query sentence with its
                # closest sentence in the matched window or chunk.
                targets = sorted(set(mw_sids) | set(m_sids))
                local = query.sentence_vecs[q_window] @ src.sentence_vecs[targets].T
                for i, q_sid in enumerate(q_window):
                    if q_sid not in q_ok or q_sid not in keep:
                        continue
                    j = int(np.argmax(local[i]))
                    pair = SentencePair(q_sid, targets[j], float(v), "verified", float(local[i, j]))
                    _offer(best[m_doc], pair, float(score), qc, m_cid)

    passages: List[Passage] = []
    for m_doc, pairs in best.items():
        passages.extend(_passages_from_pairs(query, m_doc, pairs, cfg, kind))
    passages.sort(key=lambda p: (-len(p.pairs) * p.sentence_score, -p.chunk_score))
    return passages


def coverage(doc: Document, passages: Sequence[Passage]) -> float:
    """Percentage of the document's scoring words covered by the passages."""
    total = sum(s.n_words for s in doc.scoring_sentences())
    if total == 0:
        return 0.0
    covered = {sid for p in passages for sid in p.query_sids}
    words = sum(s.n_words for s in doc.sentences if s.sid in covered)
    return round(100.0 * words / total, 1)
