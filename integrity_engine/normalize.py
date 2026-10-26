"""Text cleaning, header/footer detection and sentence segmentation."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable, List, Optional, Sequence, Tuple

from .models import Document, Sentence, TextBlock

_LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}
_QUOTES = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": " - ", " ": " "}

PAGE_NUMBER_RE = re.compile(
    r"^\s*(?:page\s*)?[-–(\[]?\s*\d{1,4}\s*[-–)\]]?\s*(?:(?:of|/)\s*\d{1,4})?\s*$",
    re.IGNORECASE,
)
REFERENCES_HEADING_RE = re.compile(
    r"^\s*(?:\d+[.)]?\s*)?(references|bibliography|works cited|sources|citations)\s*:?\s*$",
    re.IGNORECASE,
)

# Abbreviations that end in a period but do not end a sentence.
_ABBREVIATIONS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "st",
    "vs",
    "etc",
    "e.g",
    "i.e",
    "fig",
    "no",
    "vol",
    "inc",
    "ltd",
    "co",
    "corp",
    "approx",
    "est",
    "dept",
    "u.s",
    "u.k",
    "al",
    "rs",
}
_SENT_BOUNDARY_RE = re.compile(r"([.!?]['\")\]]*)\s+(?=[\"'(\[]?[A-Z0-9])")
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*")


def clean_text(text: str) -> str:
    """Normalise unicode, fix hyphenated line breaks and collapse whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    for src, dst in {**_LIGATURES, **_QUOTES}.items():
        text = text.replace(src, dst)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)  # de-hyphenate wrapped words
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def words(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def layout_key(line: str) -> str:
    """Key used to spot repeated headers/footers ("Page 3" == "Page 7")."""
    return re.sub(r"\d+", "#", line.strip().lower())


def find_repeated_margin_lines(pages: Sequence[Sequence[str]], min_fraction: float = 0.5) -> set:
    """Return layout keys of margin lines repeated on >= ``min_fraction`` of pages.

    ``pages`` holds, per page, the candidate lines taken from the top and bottom
    margins. Only multi-page documents can have *repeated* headers.
    """
    if len(pages) < 2:
        return set()
    counts = Counter()
    for lines in pages:
        counts.update({layout_key(line) for line in lines if line.strip()})
    needed = max(2, int(round(min_fraction * len(pages))))
    return {k for k, c in counts.items() if c >= needed}


def split_sentences(text: str) -> List[str]:
    """Rule-based splitter that respects abbreviations and decimal numbers."""
    text = text.strip()
    if not text:
        return []
    pieces: List[str] = []
    start = 0
    for m in _SENT_BOUNDARY_RE.finditer(text):
        candidate = text[start : m.end(1)]
        last_token = candidate.rstrip(".!?'\")]").split()[-1:] or [""]
        token = last_token[0].lower().rstrip(".")
        if token in _ABBREVIATIONS or (len(token) == 1 and token.isalpha() and candidate.endswith(".")):
            continue
        pieces.append(candidate.strip())
        start = m.end()
    tail = text[start:].strip()
    if tail:
        pieces.append(tail)
    return [p for p in pieces if p]


def mark_reference_sections(blocks: List[TextBlock]) -> None:
    """Tag blocks after a "References" style heading so they are not scored."""
    in_refs = False
    for block in blocks:
        if REFERENCES_HEADING_RE.match(block.text):
            in_refs = True
            block.section = "references"
            continue
        if in_refs:
            block.section = "references"


def _media_streams(blocks: List[TextBlock]) -> List[Tuple[str, List[Tuple[int, int, TextBlock]]]]:
    """Join consecutive ASR segments so sentences can span segment borders."""
    text_parts: List[str] = []
    spans: List[Tuple[int, int, TextBlock]] = []
    pos = 0
    for b in blocks:
        t = b.text.strip()
        if not t:
            continue
        if text_parts:
            pos += 1
        spans.append((pos, pos + len(t), b))
        text_parts.append(t)
        pos += len(t)
    return [(" ".join(text_parts), spans)] if text_parts else []


def build_sentences(doc: Document) -> None:
    """Populate ``doc.sentences`` from ``doc.blocks``, keeping provenance."""
    sentences: List[Sentence] = []
    if doc.modality == "media":
        for stream, spans in _media_streams(doc.blocks):
            cursor = 0
            for sent in split_sentences(stream):
                idx = stream.find(sent, cursor)
                idx = cursor if idx < 0 else idx
                end = idx + len(sent)
                cursor = end
                covering = [b for s, e, b in spans if s < end and e > idx] or [spans[-1][2]]
                sentences.append(
                    Sentence(
                        sid=len(sentences),
                        text=sent,
                        n_words=len(words(sent)),
                        start=covering[0].start,
                        end=covering[-1].end,
                        paragraph=covering[0].paragraph,
                    )
                )
    else:
        for block in doc.blocks:
            for sent in split_sentences(block.text):
                n = len(words(sent))
                if n == 0:
                    continue
                sentences.append(
                    Sentence(
                        sid=len(sentences),
                        text=sent,
                        n_words=n,
                        page=block.page,
                        paragraph=block.paragraph,
                        section=block.section,
                    )
                )
    doc.sentences = sentences


def format_timestamp(seconds: Optional[float]) -> Optional[str]:
    if seconds is None:
        return None
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
