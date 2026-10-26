"""JSON + Markdown audit reports, and the batch summary matrix."""

from __future__ import annotations

import csv
import difflib
import json
from pathlib import Path
from typing import Dict, List, Optional

from .models import Document, Sentence
from .normalize import format_timestamp, jaccard, words
from .scoring import match_type
from .similarity import Passage

# -- passage serialisation -------------------------------------------------


def _location(doc: Document, sents: List[Sentence], prefix: str) -> Dict[str, object]:
    loc: Dict[str, object] = {}
    if doc.modality == "media":
        loc[f"{prefix}_timestamp"] = f"{format_timestamp(sents[0].start)} - {format_timestamp(sents[-1].end)}"
    elif doc.modality == "reference":
        loc[f"{prefix}_source_url"] = doc.meta.get("url")
        loc[f"{prefix}_paragraph"] = sents[0].paragraph
    else:
        pages = sorted({s.page for s in sents if s.page is not None})
        if pages:
            loc[f"{prefix}_page"] = pages[0]
            if len(pages) > 1:
                loc[f"{prefix}_page_range"] = f"{pages[0]}-{pages[-1]}"
            if doc.meta.get("page_numbers") == "estimated":
                loc[f"{prefix}_page_is_estimated"] = True
        loc[f"{prefix}_paragraph"] = sents[0].paragraph
    return loc


def passage_record(p: Passage, docs: Dict[str, Document]) -> Dict[str, object]:
    q_doc, m_doc = docs[p.query_doc], docs[p.matched_doc]
    q_sents = [q_doc.sentences[i] for i in p.query_sids]
    m_sents = [m_doc.sentences[i] for i in p.matched_sids]
    q_text = " ".join(s.text for s in q_sents)
    m_text = " ".join(s.text for s in m_sents)
    lex = jaccard(words(q_text), words(m_text))
    rec: Dict[str, object] = {
        "matched_file": m_doc.meta.get("title") if m_doc.modality == "reference" else m_doc.doc_id,
        "similarity_score": round(p.chunk_score, 3),
        "sentence_similarity": round(p.sentence_score, 3),
        "evidence": " + ".join(
            {"direct": "sentence cosine", "verified": "cross-encoder verified paraphrase"}[m] for m in p.methods
        ),
        "lexical_overlap_jaccard": round(lex, 3),
        "match_type": match_type(lex),
        "query_passage": q_text,
        "matched_passage": m_text,
        "location": {**_location(q_doc, q_sents, "query"), **_location(m_doc, m_sents, "matched")},
        "sentence_alignment": [
            {
                "query_sentence": q_doc.sentences[pr.query_sid].text,
                "matched_sentence": m_doc.sentences[pr.matched_sid].text,
                "score": round(pr.score, 3),
                "method": pr.method,
                "sentence_cosine": round(pr.cosine if pr.cosine is not None else pr.score, 3),
            }
            for pr in p.pairs
        ],
        "modalities": f"{q_doc.modality} -> {m_doc.modality}",
        "query_words": sum(s.n_words for s in q_sents),
    }
    return rec


def boilerplate_runs(doc: Document) -> List[Dict[str, object]]:
    runs: List[List[Sentence]] = []
    for s in doc.sentences:
        if s.is_boilerplate:
            if runs and runs[-1][-1].sid == s.sid - 1:
                runs[-1].append(s)
            else:
                runs.append([s])
    return [
        {
            "passage": " ".join(s.text for s in run),
            "words": sum(s.n_words for s in run),
            "shared_with_documents": max(s.boilerplate_df for s in run),
            "location": _location(doc, run, "query"),
        }
        for run in runs
    ]


# -- markdown --------------------------------------------------------------


def highlight_shared(a: str, b: str, min_run: int = 3) -> str:
    """Bold runs of >= ``min_run`` consecutive words of ``a`` that also occur in ``b``."""
    a_tokens = a.split()
    norm = lambda t: "".join(ch for ch in t.lower() if ch.isalnum())  # noqa: E731
    sm = difflib.SequenceMatcher(None, [norm(t) for t in a_tokens], [norm(t) for t in b.split()], autojunk=False)
    bold = [False] * len(a_tokens)
    for block in sm.get_matching_blocks():
        if block.size >= min_run:
            for i in range(block.a, block.a + block.size):
                bold[i] = True
    out, i = [], 0
    while i < len(a_tokens):
        if bold[i]:
            j = i
            while j < len(a_tokens) and bold[j]:
                j += 1
            out.append("**" + " ".join(a_tokens[i:j]) + "**")
            i = j
        else:
            out.append(a_tokens[i])
            i += 1
    return " ".join(out)


def _fmt_loc(loc: Dict[str, object], prefix: str) -> str:
    if f"{prefix}_timestamp" in loc:
        return f"video {loc[f'{prefix}_timestamp']}"
    if f"{prefix}_source_url" in loc:
        return f"{loc[f'{prefix}_source_url']} (¶{loc.get(f'{prefix}_paragraph')})"
    page = loc.get(f"{prefix}_page_range") or loc.get(f"{prefix}_page")
    est = " (est.)" if loc.get(f"{prefix}_page_is_estimated") else ""
    return f"page {page}{est}, ¶{loc.get(f'{prefix}_paragraph')}"


def _md_match(n: int, rec: Dict[str, object], label: str) -> List[str]:
    loc = rec["location"]
    lines = [
        f"### {n}. {label}: `{rec['matched_file']}` - {rec['match_type']}",
        "",
        f"Chunk cosine **{rec['similarity_score']:.2f}** | sentence evidence **{rec['sentence_similarity']:.2f}** "
        f"({rec['evidence']}) | word overlap (Jaccard) {rec['lexical_overlap_jaccard']:.2f} | {rec['modalities']}",
        "",
        f"**This submission** - {_fmt_loc(loc, 'query')}",
        "",
        "> " + highlight_shared(rec["query_passage"], rec["matched_passage"]),
        "",
        f"**Matched source** - `{rec['matched_file']}`, {_fmt_loc(loc, 'matched')}",
        "",
        "> " + highlight_shared(rec["matched_passage"], rec["query_passage"]),
        "",
        "<details><summary>Sentence-level alignment</summary>",
        "",
        "| # | This submission | Source | score (method) |",
        "|---|---|---|---|",
    ]
    for i, al in enumerate(rec["sentence_alignment"], 1):
        q = al["query_sentence"].replace("|", "/")
        m = al["matched_sentence"].replace("|", "/")
        detail = "cosine" if al["method"] == "direct" else f"cross-encoder; sentence cos {al['sentence_cosine']:.2f}"
        lines.append(f"| {i} | {q} | {m} | {al['score']:.2f} ({detail}) |")
    lines += ["", "</details>", ""]
    return lines


def render_markdown(report: Dict[str, object]) -> str:
    s = report["scores"]
    ai = s["ai_generated_probability"]
    lines = [
        f"# Integrity report: `{report['document_id']}`",
        "",
        f"**Overall risk: {report['total_integrity_risk']}**  ",
        f"Modality: {report['modality']} | Status: {report['status']}",
        "",
        "| Signal | Score |",
        "|---|---|",
        f"| Internal peer similarity | {s['peer_similarity_percentage']:.1f}% |",
        f"| External (web/reference) match | {s['external_similarity_percentage']:.1f}% |",
        f"| AI-content likelihood (advisory) | {'n/a' if ai is None else f'{ai:.1f}%'} |",
        "",
        "## Summary for reviewers",
        "",
        report["explanations"],
        "",
    ]
    if report.get("warnings"):
        lines += ["## Processing warnings", ""] + [f"- {w}" for w in report["warnings"]] + [""]

    n = 0
    sections = [
        ("peer_matches", "Peer match", "Peer matches (same modality)"),
        ("multimodal_matches", "Cross-modal match", "Cross-modal matches (video/audio <-> document)"),
        ("external_matches", "External match", "External reference matches"),
    ]
    for key, label, title in sections:
        if report.get(key):
            lines += [f"## {title}", ""]
            for rec in report[key]:
                n += 1
                lines += _md_match(n, rec, label)

    if report.get("boilerplate_overlaps"):
        lines += [
            "## Shared template text (excluded from scores)",
            "",
            "These passages appear in many submissions (case prompt, instructions, citations) and are not counted.",
            "",
        ]
        for b in report["boilerplate_overlaps"]:
            text = b["passage"] if len(b["passage"]) < 300 else b["passage"][:297] + "..."
            lines.append(
                f"- ({b['words']} words, in {b['shared_with_documents']} documents, "
                f"{_fmt_loc(b['location'], 'query')}) {text}"
            )
        lines.append("")

    a = report.get("ai_analysis") or {}
    lines += ["## AI-content analysis", ""]
    if a.get("probability") is None:
        lines += [f"Not estimated: {a.get('reason', 'n/a')}", ""]
    else:
        lines.append(f"Method: {a.get('method', 'n/a')}.")
        lines.append("")
        if a.get("logit_contributions"):
            lines += ["| Signal | Contribution (log-odds; + = more AI-like) |", "|---|---|"]
            for k, v in a["logit_contributions"].items():
                lines.append(f"| {k} | {v:+.2f} |")
        else:
            lines += ["| Component | Sub-score |", "|---|---|"]
            for k, v in a.get("components", {}).items():
                lines.append(f"| {k} | {v:.1f}% |")
        sig = a.get("signals", {})
        lines += [
            "",
            f"Raw signals: GPT-2 perplexity {sig.get('perplexity', float('nan')):.1f}, burstiness "
            f"{sig.get('burstiness', float('nan')):.2f}, "
            f"assistant-phrase rate {sig.get('marker_rate', 0):.2f}/100 words, "
            f"classifier {sig.get('classifier') if sig.get('classifier') is not None else 'n/a'}.",
        ]
        if a.get("caveat"):
            lines.append(f"Caveat: {a['caveat']}.")
        lines.append("")
    d = report["score_details"]
    lines += [
        "## Scoring basis",
        "",
        f"- Words scored: {d['scoring_words']} (template words excluded: {d['boilerplate_words_excluded']}, "
        f"reference-list words excluded: {d['reference_words_excluded']})",
        f"- Document metadata: {json.dumps(report['metadata'])}",
        "",
    ]
    return "\n".join(lines)


# -- batch outputs ---------------------------------------------------------


def write_report(report: Dict[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = report["document_id"].replace("/", "__")
    (out_dir / f"{stem}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / f"{stem}.md").write_text(render_markdown(report), encoding="utf-8")


def write_summary(
    reports: List[Dict[str, object]],
    matrix: Dict[str, Dict[str, float]],
    out_dir: Path,
    run_info: Dict[str, object],
) -> None:
    names = [r["document_id"] for r in reports if r["status"] == "ok"]
    with open(out_dir / "similarity_matrix.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["query \\ source"] + names)
        for a in names:
            w.writerow([a] + ["" if a == b else matrix.get(a, {}).get(b, 0.0) for b in names])

    summary = {
        "run": run_info,
        "documents": [
            {
                "document_id": r["document_id"],
                "status": r["status"],
                "total_integrity_risk": r["total_integrity_risk"],
                **r["scores"],
                "top_peer_source": r["score_details"].get("top_peer_source"),
                "warnings": r["warnings"],
            }
            for r in reports
        ],
        "peer_similarity_matrix": matrix,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Batch integrity summary",
        "",
        f"Processed {run_info['documents_total']} files: {run_info['documents_ok']} analysed, "
        f"{run_info['documents_flagged']} flagged/skipped. Runtime {run_info['runtime_seconds']:.1f}s.",
        "",
        "| Document | Status | Risk | Peer % | External % | AI % | Top peer source |",
        "|---|---|---|---|---|---|---|",
    ]
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "UNASSESSED": 3}
    for r in sorted(reports, key=lambda r: (order.get(r["total_integrity_risk"], 9), r["document_id"])):
        s = r["scores"]
        ai = s["ai_generated_probability"]
        lines.append(
            f"| [{r['document_id']}]({r['document_id'].replace('/', '__')}.md) | {r['status']} | "
            f"{r['total_integrity_risk']} | {s['peer_similarity_percentage']:.1f} | "
            f"{s['external_similarity_percentage']:.1f} | {'n/a' if ai is None else f'{ai:.0f}'} | "
            f"{r['score_details'].get('top_peer_source') or '-'} |"
        )
    flagged = [r for r in reports if r["status"] != "ok"]
    if flagged:
        lines += ["", "## Flagged / skipped files", ""]
        lines += [f"- `{r['document_id']}`: {'; '.join(r['warnings'])}" for r in flagged]

    if names:
        lines += [
            "",
            "## Peer similarity matrix",
            "",
            "Cell = % of the **row** document's scored text covered by passages matched in the **column** document.",
            "",
            "| # | Document |",
            "|---|---|",
        ]
        lines += [f"| D{i} | {n} |" for i, n in enumerate(names, 1)]
        lines += [
            "",
            "| row \\ col | " + " | ".join(f"D{i}" for i in range(1, len(names) + 1)) + " |",
            "|---" * (len(names) + 1) + "|",
        ]
        for i, a in enumerate(names, 1):
            cells = []
            for b in names:
                v: Optional[float] = None if a == b else matrix.get(a, {}).get(b, 0.0)
                cells.append("-" if v is None else (f"**{v:.0f}**" if v >= 10 else (f"{v:.0f}" if v else ".")))
            lines.append(f"| D{i} | " + " | ".join(cells) + " |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
