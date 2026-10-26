"""File discovery and format dispatch. Never raises on a bad file."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from ..config import Config
from ..models import Document, TextBlock
from ..normalize import build_sentences, clean_text, mark_reference_sections
from .errors import ExtractionError

log = logging.getLogger(__name__)


def _extract_text_file(doc: Document) -> None:
    with open(doc.path, "r", encoding="utf-8", errors="replace") as fh:
        raw = fh.read()
    for i, para in enumerate((p for p in raw.split("\n\n") if p.strip()), start=1):
        doc.blocks.append(TextBlock(text=clean_text(para), page=1, paragraph=i))
    if not doc.blocks:
        raise ExtractionError("Text file is empty")


def discover(input_dir: str, cfg: Config) -> List[Path]:
    root = Path(input_dir)
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.name.startswith(".") and path.name != "dataset_manifest.json":
            files.append(path)
    return files


def load_document(path: Path, input_dir: str, cfg: Config) -> Document:
    doc_id = str(path.relative_to(input_dir))
    modality = cfg.supported_extensions.get(path.suffix.lower())
    doc = Document(doc_id=doc_id, path=str(path), modality=modality or "unsupported")
    doc.meta["size_bytes"] = path.stat().st_size

    if modality is None:
        doc.status = "skipped"
        doc.warnings.append(f"Unsupported file type '{path.suffix}'")
        log.warning("%s: unsupported file type, skipped", doc_id)
        return doc
    if doc.meta["size_bytes"] == 0:
        doc.status = "flagged"
        doc.warnings.append("File is empty (0 bytes)")
        log.warning("%s: empty file (0 bytes), flagged and skipped", doc_id)
        return doc

    try:
        if modality == "pdf":
            from .pdf import extract_pdf

            extract_pdf(doc)
        elif modality == "docx":
            from .docx import extract_docx

            extract_docx(doc)
        elif modality == "media":
            from .media import extract_media

            extract_media(doc, cfg.asr)
        else:
            _extract_text_file(doc)
        mark_reference_sections(doc.blocks)
        build_sentences(doc)
        if not doc.sentences:
            raise ExtractionError("No sentences could be extracted")
    except ExtractionError as exc:
        doc.status = "flagged"
        doc.warnings.append(str(exc))
        log.warning("%s: %s - flagged and skipped", doc_id, exc)
    except Exception as exc:  # defensive: one bad file must not stop the batch
        doc.status = "flagged"
        doc.warnings.append(f"Unexpected extraction error: {type(exc).__name__}: {exc}")
        log.exception("%s: unexpected extraction error", doc_id)
    return doc


__all__ = ["discover", "load_document", "ExtractionError"]
