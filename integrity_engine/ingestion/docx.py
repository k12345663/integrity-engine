"""DOCX extraction preserving paragraph order, tables and (estimated) pages.

DOCX has no fixed pagination. We use explicit page breaks and Word's
``lastRenderedPageBreak`` markers when present; otherwise the page number is
estimated at ~450 words per page and the report says so.
"""

from __future__ import annotations

import zipfile

from ..models import Document, TextBlock
from ..normalize import PAGE_NUMBER_RE, clean_text, words
from .errors import ExtractionError

WORDS_PER_PAGE_ESTIMATE = 450
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _page_breaks_in(element, rendered: bool = False) -> int:
    """Count explicit page breaks, or Word's rendered breaks when ``rendered``.

    The two are never mixed: Word writes a rendered marker after each explicit
    break, so counting both would double-count.
    """
    breaks = 0
    for node in element.iter():
        if rendered:
            breaks += node.tag == f"{_W}lastRenderedPageBreak"
        elif node.tag == f"{_W}br" and node.get(f"{_W}type") == "page":
            breaks += 1
        elif node.tag == f"{_W}pageBreakBefore" and node.get(f"{_W}val", "true") != "false":
            breaks += 1
    return breaks


def _iter_body(document):
    """Yield paragraph texts and table cell texts in document order."""
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    for child in document.element.body.iterchildren():
        if child.tag == f"{_W}p":
            yield child, Paragraph(child, document).text
        elif child.tag == f"{_W}tbl":
            table = Table(child, document)
            for row in table.rows:
                seen = set()
                cells = []
                for cell in row.cells:  # merged cells repeat; de-duplicate
                    if id(cell._tc) in seen:
                        continue
                    seen.add(id(cell._tc))
                    cells.append(cell.text.strip())
                row_text = " | ".join(c for c in cells if c)
                if row_text:
                    yield child, row_text


def extract_docx(doc: Document) -> None:
    import docx

    if not zipfile.is_zipfile(doc.path):
        raise ExtractionError("DOCX is corrupted or not an Office Open XML file")
    try:
        document = docx.Document(doc.path)
    except Exception as exc:
        raise ExtractionError(f"DOCX could not be parsed: {exc}") from exc

    body = document.element.body
    rendered = _page_breaks_in(body) == 0
    explicit_breaks = _page_breaks_in(body, rendered) > 0
    page = 1
    words_seen = 0
    paragraph = 0
    last_element = None
    for element, raw in _iter_body(document):
        if explicit_breaks and element is not last_element:
            # Approximation: a paragraph containing a break is attributed to the
            # page after it (breaks normally sit in their own empty paragraph).
            page += _page_breaks_in(element, rendered)
        last_element = element
        text = clean_text(raw)
        if not text or PAGE_NUMBER_RE.match(text):
            continue
        paragraph += 1
        if not explicit_breaks:
            page = 1 + words_seen // WORDS_PER_PAGE_ESTIMATE
        words_seen += len(words(text))
        doc.blocks.append(TextBlock(text=text, page=page, paragraph=paragraph))

    if not explicit_breaks:
        doc.meta["page_numbers"] = "estimated"
    else:
        doc.meta["page_numbers"] = "rendered_breaks" if rendered else "explicit_breaks"
    if not doc.blocks:
        raise ExtractionError("DOCX contains no text")
