"""Specific-detail "anchors" used to confirm paraphrase evidence.

Someone paraphrasing a peer rewrites the wording but keeps the facts: the
numbers, place names and proper nouns. Two independent texts on the same
topic (two LLM answers to one prompt in particular) share the *topic* but
rarely the same specific details. An anchor is a number or a mid-sentence
capitalised word that appears in only a few documents of the corpus.

Numbers are normalised so "three hundred" == "300", "1.2 lakh" == "120,000",
"one crore" == "ten million" and "three percent" == "3%".
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, FrozenSet, Iterable, List, Set

from .models import Document

_UNITS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_SCALES = {
    "hundred": 100,
    "thousand": 1_000,
    "lakh": 100_000,
    "lakhs": 100_000,
    "million": 1_000_000,
    "crore": 10_000_000,
    "crores": 10_000_000,
    "billion": 1_000_000_000,
}
_TOKEN_RE = re.compile(r"\d[\d,]*(?:\.\d+)?|[A-Za-z][A-Za-z'\-]*")
_STOP_CAPS = {"I", "The", "A", "An", "In", "On", "If", "We", "Our", "It", "This", "That", "And", "But", "So"}


def extract_numbers(text: str) -> Set[str]:
    """All numeric values in ``text``, digits or words, as canonical strings."""
    values: List[float] = []
    current: float = None  # value being accumulated from words / digits
    total = 0.0

    def flush():
        nonlocal current, total
        if current is not None or total:
            values.append(total + (current or 0))
        current, total = None, 0.0

    for tok in _TOKEN_RE.findall(text):
        low = tok.lower()
        if tok[0].isdigit():
            flush()
            current = float(tok.replace(",", ""))
        elif low in _UNITS:
            if current is not None and current >= 20 and _UNITS[low] < 10:
                current += _UNITS[low]  # "forty five"
            else:
                if current is not None:
                    flush()
                current = float(_UNITS[low])
        elif low in _SCALES and current is not None:
            scale = _SCALES[low]
            if scale == 100:
                current *= 100
            else:
                total += current * scale
                current = None
        elif low in ("and", "-") and current is not None:
            continue
        else:
            flush()
    flush()
    return {f"#{v:g}" for v in values}


def raw_anchors(text: str) -> FrozenSet[str]:
    anchors = set(extract_numbers(text))
    tokens = text.split()
    for i, tok in enumerate(tokens):
        word = re.sub(r"[^A-Za-z\-]", "", tok)
        if i > 0 and word[:1].isupper() and word not in _STOP_CAPS and len(word) > 2:
            if not tokens[i - 1].endswith((".", "!", "?", ":")):
                anchors.add(word.lower())
    return frozenset(anchors)


def assign_anchors(docs: Iterable[Document], max_doc_fraction: float = 0.1) -> Dict[str, object]:
    """Keep only anchors that occur in few documents (df <= max(2, frac*N))."""
    docs = list(docs)
    per_sentence = {(d.doc_id, s.sid): raw_anchors(s.text) for d in docs for s in d.scoring_sentences()}
    df: Counter = Counter()
    for d in docs:
        df.update(set().union(*[per_sentence[(d.doc_id, s.sid)] for s in d.scoring_sentences()] or [set()]))
    max_df = max(2, math.ceil(max_doc_fraction * len(docs)))
    for d in docs:
        for s in d.scoring_sentences():
            s.anchors = frozenset(a for a in per_sentence[(d.doc_id, s.sid)] if df[a] <= max_df)
    return {"anchor_max_df": max_df, "distinct_anchors": len(df)}


def anchors_agree(shared: Set[str]) -> bool:
    """Enough shared specifics: one number, or two distinct rare names."""
    numbers = sum(a.startswith("#") for a in shared)
    return numbers >= 1 or len(shared) - numbers >= 2
