"""Core data structures shared across the pipeline.

Every piece of text carries its provenance (file, page / timestamp, paragraph)
so that any match can be traced back to an exact, human-verifiable location.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional


@dataclass
class TextBlock:
    """A raw, cleaned block of text as produced by an extractor.

    For PDFs a block is a paragraph on a page; for DOCX a paragraph or table
    cell; for media a single ASR segment with start / end times (seconds).
    """

    text: str
    page: Optional[int] = None
    paragraph: Optional[int] = None
    start: Optional[float] = None
    end: Optional[float] = None
    section: Optional[str] = None  # e.g. "references"


@dataclass
class Sentence:
    sid: int  # index within the document
    text: str
    n_words: int
    page: Optional[int] = None
    paragraph: Optional[int] = None
    start: Optional[float] = None
    end: Optional[float] = None
    section: Optional[str] = None
    is_boilerplate: bool = False
    boilerplate_df: int = 1  # number of documents containing this sentence
    anchors: FrozenSet[str] = frozenset()  # rare numbers / names, see anchors.py


@dataclass
class Chunk:
    cid: int  # index within the document
    doc_id: str
    sentence_ids: List[int]
    text: str
    n_tokens: int


@dataclass
class Document:
    doc_id: str  # file name relative to the input dir
    path: str
    modality: str  # "pdf" | "docx" | "media" | "text"
    status: str = "ok"  # "ok" | "skipped" | "flagged"
    warnings: List[str] = field(default_factory=list)
    blocks: List[TextBlock] = field(default_factory=list)
    sentences: List[Sentence] = field(default_factory=list)
    chunks: List[Chunk] = field(default_factory=list)
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def is_usable(self) -> bool:
        return self.status == "ok" and bool(self.sentences)

    def scoring_sentences(self) -> List[Sentence]:
        """Sentences that count towards similarity percentages."""
        return [s for s in self.sentences if not s.is_boilerplate and s.section != "references"]


@dataclass
class SentencePair:
    query_sid: int
    matched_sid: int
    score: float
    method: str = "direct"  # "direct" (bi-encoder cosine) | "verified" (cross-encoder)
    cosine: Optional[float] = None  # sentence cosine, kept for verified pairs
