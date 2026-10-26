"""PDF text extraction with header/footer stripping and column-aware ordering."""

from __future__ import annotations

import logging
from typing import List, Tuple

from ..models import Document, TextBlock
from ..normalize import PAGE_NUMBER_RE, clean_text, find_repeated_margin_lines, layout_key
from .errors import ExtractionError

log = logging.getLogger(__name__)

MARGIN_FRACTION = 0.08  # top/bottom 8% of the page is header/footer territory

RawBlock = Tuple[float, float, float, float, str]


def _reading_order(blocks: List[RawBlock], page_width: float) -> List[RawBlock]:
    """Order blocks for multi-column layouts.

    Full-width blocks split the page into horizontal bands; inside a band the
    narrow blocks are read left column first, then right column.
    """
    blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
    mid = page_width / 2.0
    ordered: List[RawBlock] = []
    band: List[RawBlock] = []

    def flush() -> None:
        band.sort(key=lambda b: (0 if b[0] < mid - 5 else 1, b[1]))
        ordered.extend(band)
        band.clear()

    for b in blocks:
        width = b[2] - b[0]
        if width > 0.6 * page_width:
            flush()
            ordered.append(b)
        else:
            band.append(b)
    flush()
    return ordered


def extract_pdf(doc: Document) -> None:
    import pymupdf

    try:
        pdf = pymupdf.open(doc.path)
    except Exception as exc:  # pymupdf raises several exception types
        raise ExtractionError(f"PDF could not be opened (corrupted or not a PDF): {exc}") from exc

    with pdf:
        if pdf.needs_pass:
            raise ExtractionError("PDF is password protected; text cannot be extracted")
        doc.meta["pages"] = pdf.page_count
        pages: List[List[RawBlock]] = []
        margins: List[List[str]] = []
        image_pages = 0
        for page in pdf:
            h, w = page.rect.height, page.rect.width
            raw = [(b[0], b[1], b[2], b[3], b[4]) for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
            if not raw and page.get_images():
                image_pages += 1
            pages.append(raw)
            margins.append(
                [clean_text(b[4]) for b in raw if b[3] <= h * MARGIN_FRACTION or b[1] >= h * (1 - MARGIN_FRACTION)]
            )
            page_width = w

        repeated = find_repeated_margin_lines(margins)
        stripped = 0
        paragraph = 0
        for page_no, raw in enumerate(pages, start=1):
            h = pdf[page_no - 1].rect.height
            for b in _reading_order(raw, page_width):
                text = clean_text(b[4])
                in_margin = b[3] <= h * MARGIN_FRACTION or b[1] >= h * (1 - MARGIN_FRACTION)
                if in_margin and (layout_key(text) in repeated or PAGE_NUMBER_RE.match(text)):
                    stripped += 1
                    continue
                if PAGE_NUMBER_RE.match(text):
                    stripped += 1
                    continue
                paragraph += 1
                doc.blocks.append(TextBlock(text=text, page=page_no, paragraph=paragraph))

    doc.meta["header_footer_blocks_removed"] = stripped
    if not doc.blocks:
        if image_pages:
            raise ExtractionError(f"PDF has no text layer ({image_pages} image-only page(s)); OCR is required")
        raise ExtractionError("PDF contains no extractable text")
