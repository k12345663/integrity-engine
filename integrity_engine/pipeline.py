"""End-to-end orchestration: ingest -> normalise -> index -> match -> report."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from .ai_detection import AIDetector
from .anchors import assign_anchors
from .boilerplate import load_template_sentences, mark_boilerplate
from .chunking import chunk_document
from .config import Config
from .embeddings import Embedder
from .ingestion import discover, load_document
from .models import Document
from .reference_kb import distinctive_queries, load_local_kb, search_online
from .reporting import boilerplate_runs, passage_record, write_report, write_summary
from .scoring import build_explanation, risk_level
from .similarity import EmbeddedDoc, Passage, build_chunk_index, build_windows, coverage, find_passages
from .verification import CrossEncoderVerifier

log = logging.getLogger(__name__)


def _embed(doc: Document, embedder: Embedder, cfg: Config, sentence_vecs=None) -> EmbeddedDoc:
    if sentence_vecs is None:
        sentence_vecs = embedder.encode([s.text for s in doc.sentences])
    ed = EmbeddedDoc(doc, sentence_vecs, embedder.encode([c.text for c in doc.chunks]))
    build_windows(ed, embedder.encode, cfg.matching.window_sizes)
    return ed


def _per_source(doc: Document, passages: List[Passage], cfg: Config, label=lambda d: d) -> Dict[str, float]:
    by_src: Dict[str, List[Passage]] = {}
    for p in passages:
        by_src.setdefault(p.matched_doc, []).append(p)
    return {label(src): coverage(doc, ps) for src, ps in by_src.items()}


class IntegrityPipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.embedder: Optional[Embedder] = None
        self.ai: Optional[AIDetector] = None
        self.verifier: Optional[CrossEncoderVerifier] = None

    # ------------------------------------------------------------------
    def ingest(self, input_dir: str) -> List[Document]:
        files = discover(input_dir, self.cfg)
        log.info("Discovered %d files in %s", len(files), input_dir)
        docs = []
        for path in files:
            t0 = time.time()
            doc = load_document(path, input_dir, self.cfg)
            doc.meta["extraction_seconds"] = round(time.time() - t0, 2)
            log.info("%-40s %-6s %-8s %4d sentences", doc.doc_id, doc.modality, doc.status, len(doc.sentences))
            docs.append(doc)
        return docs

    def prepare(self, docs: List[Document]) -> Dict[str, EmbeddedDoc]:
        """Boilerplate marking, chunking and embedding of usable documents."""
        if self.embedder is None:
            self.embedder = Embedder(self.cfg.embedding)
        usable = [d for d in docs if d.is_usable]
        sent_vecs = {d.doc_id: self.embedder.encode([s.text for s in d.sentences]) for d in usable}
        template = load_template_sentences(self.cfg.boilerplate.template_dir)
        tmpl_vecs = self.embedder.encode(template) if template else None
        self.boilerplate_stats = (
            mark_boilerplate(
                docs, sent_vecs, self.cfg.boilerplate, tmpl_vecs, min_words=self.cfg.matching.min_sentence_words
            )
            if usable
            else {}
        )
        out = {}
        for d in usable:
            chunk_document(d, self.cfg.chunking, self.embedder.count_tokens)
            out[d.doc_id] = _embed(d, self.embedder, self.cfg, sent_vecs[d.doc_id])
        return out

    def reference_corpus(self, docs: Dict[str, EmbeddedDoc]) -> Dict[str, EmbeddedDoc]:
        ref_docs = load_local_kb(self.cfg.external.reference_dir)
        if self.cfg.external.online:
            known = {d.meta.get("title") for d in ref_docs}
            for ed in docs.values():
                ref_docs += search_online(
                    distinctive_queries(ed.doc, self.cfg.external.online_queries_per_doc), self.cfg.external, known
                )
        out = {}
        for d in ref_docs:
            chunk_document(d, self.cfg.chunking, self.embedder.count_tokens)
            if d.chunks:
                out[d.doc_id] = _embed(d, self.embedder, self.cfg)
        return out

    # ------------------------------------------------------------------
    def run(self, input_dir: str, output_dir: str, docs: Optional[List[Document]] = None) -> List[Dict[str, object]]:
        """Process ``input_dir`` (or pre-ingested ``docs``) and write reports."""
        t_start = time.time()
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        docs = docs if docs is not None else self.ingest(input_dir)
        embedded = self.prepare(docs) if any(d.is_usable for d in docs) else {}
        refs = self.reference_corpus(embedded) if embedded else {}
        self.anchor_stats = assign_anchors(
            [ed.doc for ed in embedded.values()] + [ed.doc for ed in refs.values()],
            self.cfg.matching.anchor_max_doc_fraction,
        )
        all_docs = {d.doc_id: d for d in docs}
        all_docs.update({k: v.doc for k, v in refs.items()})

        peer_index = build_chunk_index(list(embedded.values()), self.cfg.matching.index_type) if embedded else None
        ref_index = build_chunk_index(list(refs.values()), self.cfg.matching.index_type) if refs else None
        if self.verifier is None and embedded:
            self.verifier = CrossEncoderVerifier(self.cfg.matching.cross_encoder_model)
        if self.cfg.ai.enabled and self.ai is None:
            self.ai = AIDetector(self.cfg.ai)

        reports, matrix = [], {}
        for doc in docs:
            if doc.doc_id not in embedded:
                reports.append(self._unassessed_report(doc))
                continue
            ed = embedded[doc.doc_id]
            peer = find_passages(ed, embedded, peer_index, self.cfg.matching, self.verifier, "peer")
            ext = find_passages(ed, refs, ref_index, self.cfg.matching, self.verifier, "external") if ref_index else []
            ai = self.ai.analyse(doc) if self.ai else {"probability": None, "reason": "AI detection disabled"}
            report = self._report(doc, peer, ext, ai, all_docs)
            matrix[doc.doc_id] = report["score_details"]["per_source_peer"]
            reports.append(report)
            write_report(report, out_dir)
            log.info(
                "%-40s risk=%-6s peer=%5.1f%% ext=%5.1f%% ai=%s",
                doc.doc_id,
                report["total_integrity_risk"],
                report["scores"]["peer_similarity_percentage"],
                report["scores"]["external_similarity_percentage"],
                report["scores"]["ai_generated_probability"],
            )
        for r in reports:
            if r["status"] != "ok":
                write_report(r, out_dir)

        run_info = {
            "input_dir": str(input_dir),
            "documents_total": len(docs),
            "documents_ok": len(embedded),
            "documents_flagged": len(docs) - len(embedded),
            "reference_documents": len(refs),
            "boilerplate": getattr(self, "boilerplate_stats", {}),
            "anchors": getattr(self, "anchor_stats", {}),
            "embedding_model": self.cfg.embedding.model_name,
            "cross_encoder_model": self.cfg.matching.cross_encoder_model
            if self.verifier and self.verifier.available
            else None,
            "cross_encoder_pairs_scored": self.verifier.calls if self.verifier else 0,
            "runtime_seconds": time.time() - t_start,
            "config": self.cfg.to_dict(),
        }
        write_summary(reports, matrix, out_dir, run_info)
        log.info("Finished in %.1fs; reports written to %s", run_info["runtime_seconds"], out_dir)
        return reports

    # ------------------------------------------------------------------
    def _report(self, doc, peer, ext, ai, all_docs) -> Dict[str, object]:
        cfg = self.cfg
        peer_pct = coverage(doc, peer)
        ext_pct = coverage(doc, ext)
        ai_p = ai.get("probability")
        risk = risk_level(peer_pct, ext_pct, ai_p, cfg.risk)
        per_peer = _per_source(doc, peer, cfg)
        per_ext = _per_source(doc, ext, cfg, label=lambda d: all_docs[d].meta.get("title", d))

        peer_recs = [passage_record(p, all_docs) for p in peer[: cfg.matching.max_matches_reported]]
        ext_recs = [passage_record(p, all_docs) for p in ext[: cfg.matching.max_matches_reported]]
        same_modality = [
            r
            for r, p in zip(peer_recs, peer)
            if all_docs[p.matched_doc].modality == doc.modality
            or {all_docs[p.matched_doc].modality, doc.modality} <= {"pdf", "docx", "text"}
        ]
        cross_modal = [r for r in peer_recs if r not in same_modality]

        strongest = []
        for r in (peer_recs + ext_recs)[:2]:
            loc = r["location"]
            q = loc.get("query_timestamp") or f"p.{loc.get('query_page')}"
            m = loc.get("matched_timestamp") or (
                f"p.{loc['matched_page']}" if "matched_page" in loc else f"¶{loc.get('matched_paragraph')}"
            )
            strongest.append(
                f"{q} <-> {r['matched_file']} {m} ({r['match_type'].split(' ')[0]}, cos {r['similarity_score']:.2f})"
            )

        bp = boilerplate_runs(doc)
        details = {
            "scoring_words": sum(s.n_words for s in doc.scoring_sentences()),
            "boilerplate_words_excluded": sum(s.n_words for s in doc.sentences if s.is_boilerplate),
            "reference_words_excluded": sum(
                s.n_words for s in doc.sentences if s.section == "references" and not s.is_boilerplate
            ),
            "per_source_peer": per_peer,
            "per_source_external": per_ext,
            "top_peer_source": max(per_peer, key=per_peer.get) if per_peer else None,
            "chunks_indexed": len(doc.chunks),
            "thresholds": {
                "chunk_candidate": cfg.matching.chunk_candidate_threshold,
                "sentence_direct": cfg.matching.sentence_threshold,
                "cross_encoder": cfg.matching.cross_encoder_threshold,
                "min_passage_sentences": cfg.matching.min_passage_sentences,
            },
        }
        return {
            "document_id": doc.doc_id,
            "modality": doc.modality,
            "status": doc.status,
            "total_integrity_risk": risk,
            "scores": {
                "peer_similarity_percentage": peer_pct,
                "external_similarity_percentage": ext_pct,
                "ai_generated_probability": ai_p,
            },
            "peer_matches": same_modality,
            "multimodal_matches": cross_modal,
            "external_matches": ext_recs,
            "boilerplate_overlaps": bp,
            "ai_analysis": ai,
            "explanations": build_explanation(
                peer_pct, ext_pct, ai, per_peer, per_ext, strongest, details["boilerplate_words_excluded"], risk
            ),
            "score_details": details,
            "warnings": doc.warnings,
            "metadata": {k: v for k, v in doc.meta.items()},
        }

    @staticmethod
    def _unassessed_report(doc: Document) -> Dict[str, object]:
        return {
            "document_id": doc.doc_id,
            "modality": doc.modality,
            "status": doc.status,
            "total_integrity_risk": "UNASSESSED",
            "scores": {
                "peer_similarity_percentage": 0.0,
                "external_similarity_percentage": 0.0,
                "ai_generated_probability": None,
            },
            "peer_matches": [],
            "multimodal_matches": [],
            "external_matches": [],
            "boilerplate_overlaps": [],
            "ai_analysis": {"probability": None, "reason": "File could not be processed"},
            "explanations": "File could not be analysed and needs manual review: " + "; ".join(doc.warnings),
            "score_details": {
                "scoring_words": 0,
                "boilerplate_words_excluded": 0,
                "reference_words_excluded": 0,
                "per_source_peer": {},
                "per_source_external": {},
                "top_peer_source": None,
            },
            "warnings": doc.warnings,
            "metadata": doc.meta,
        }
