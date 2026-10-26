#!/usr/bin/env python
"""Evaluate the engine against the synthetic ground truth.

    python evaluate.py                      # run pipeline + metrics
    python evaluate.py --skip_ablations     # metrics only (faster)

Writes evaluation/results.json and evaluation/results.md. EVALUATION.md
interprets these numbers.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

PEER_TYPES = ("exact_copy", "paraphrase", "cross_modal")


# ----------------------------------------------------------------- helpers
def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)}


def wer(ref: str, hyp: str) -> float:
    norm = lambda t: re.sub(r"[^a-z0-9 ]", " ", t.lower()).split()  # noqa: E731
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / max(1, len(r))


def auroc(scores: List[float], labels: List[int]) -> Optional[float]:
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 3)


def pair_scores(reports: Dict[str, dict]) -> Dict[frozenset, float]:
    """Symmetric pair score = max coverage in either direction."""
    out: Dict[frozenset, float] = {}
    ok = [d for d, r in reports.items() if r["status"] == "ok"]
    for a, b in itertools.combinations(ok, 2):
        sa = reports[a]["score_details"]["per_source_peer"].get(b, 0.0)
        sb = reports[b]["score_details"]["per_source_peer"].get(a, 0.0)
        out[frozenset((a, b))] = max(sa, sb)
    return out


def peer_confusion(reports, manifest, threshold: float = 0.0) -> Dict[str, object]:
    truth = {
        frozenset((r["source"], r["target"])): r["type"] for r in manifest["relationships"] if r["type"] in PEER_TYPES
    }
    scores = pair_scores(reports)
    tp = fp = fn = tn = 0
    errors = []
    for pair, score in scores.items():
        pred = score > threshold
        actual = pair in truth
        tp += pred and actual
        fp += pred and not actual
        fn += actual and not pred
        tn += not pred and not actual
        if pred != actual:
            errors.append({"pair": sorted(pair), "score": score, "type": "FP" if pred else "FN"})
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, **prf(tp, fp, fn), "pairs": len(scores), "errors": errors}


def load_reports(report_dir: Path) -> Dict[str, dict]:
    out = {}
    for p in report_dir.glob("*.json"):
        if p.name == "summary.json":
            continue
        r = json.loads(p.read_text())
        out[r["document_id"]] = r
    return out


# ----------------------------------------------------------------- metrics
def evaluate(reports: Dict[str, dict], manifest: dict) -> Dict[str, object]:
    files = manifest["files"]
    res: Dict[str, object] = {}

    # 1. Ingestion robustness
    rows = []
    for name, meta in files.items():
        r = reports.get(name)
        actual = r["status"] if r else "missing"
        rows.append(
            {
                "file": name,
                "expected": meta["expected_status"],
                "actual": actual,
                "ok": (actual == "ok") == (meta["expected_status"] == "ok"),
                "warning": "; ".join(r["warnings"]) if r else "",
            }
        )
    res["ingestion"] = {
        "rows": rows,
        "accuracy": sum(r["ok"] for r in rows) / len(rows),
        "pipeline_completed": len(reports) == len(files),
    }

    # 2. Peer / cross-modal pair detection
    res["peer_pairs"] = peer_confusion(reports, manifest)
    by_type = {}
    scores = pair_scores(reports)
    for rel in manifest["relationships"]:
        if rel["type"] in PEER_TYPES:
            by_type[rel["type"]] = {
                "pair": [rel["source"], rel["target"]],
                "pair_score": scores.get(frozenset((rel["source"], rel["target"])), 0.0),
            }
    res["peer_by_type"] = by_type

    # 3. Localisation: does the evidence point at the right page / a timestamp?
    loc = []
    for rel in manifest["relationships"]:
        if rel["type"] not in PEER_TYPES:
            continue
        tgt = reports[rel["target"]]
        recs = [m for m in tgt["peer_matches"] + tgt["multimodal_matches"] if m["matched_file"] == rel["source"]]
        exp_page = rel.get("source_location", {}).get("page")
        pages = {m["location"].get("matched_page") for m in recs}
        stamps = [m["location"].get("query_timestamp") for m in recs if m["location"].get("query_timestamp")]
        loc.append(
            {
                "relationship": rel["type"],
                "target": rel["target"],
                "evidence_passages": len(recs),
                "expected_source_page": exp_page,
                "reported_source_pages": sorted(p for p in pages if p),
                "page_correct": exp_page in pages if exp_page else None,
                "video_timestamps": stamps,
            }
        )
    res["localisation"] = loc

    # 4. External similarity (doc level, threshold = MEDIUM band 10%)
    ext_truth = {r["target"] for r in manifest["relationships"] if r["type"] == "external_copy"}
    tp = fp = fn = tn = 0
    ext_rows = []
    for name, r in reports.items():
        if r["status"] != "ok":
            continue
        pred = r["scores"]["external_similarity_percentage"] >= 10
        actual = name in ext_truth
        tp += pred and actual
        fp += pred and not actual
        fn += actual and not pred
        tn += not pred and not actual
        if pred or actual:
            ext_rows.append(
                {
                    "file": name,
                    "external_pct": r["scores"]["external_similarity_percentage"],
                    "sources": list(r["score_details"]["per_source_external"]),
                    "truth": actual,
                }
            )
    res["external"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, **prf(tp, fp, fn), "rows": ext_rows}

    # 5. AI-generation indicator
    ai_rows, s_, l_ = [], [], []
    for name, r in reports.items():
        p = r["scores"]["ai_generated_probability"]
        if p is None:
            continue
        label = int(files[name]["ai_generated"])
        ai_rows.append({"file": name, "probability": p, "truth": bool(label)})
        s_.append(p)
        l_.append(label)
    tp = sum(p >= 50 and y for p, y in zip(s_, l_))
    fp = sum(p >= 50 and not y for p, y in zip(s_, l_))
    fn = sum(p < 50 and y for p, y in zip(s_, l_))
    tn = sum(p < 50 and not y for p, y in zip(s_, l_))
    res["ai"] = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        **prf(tp, fp, fn),
        "auroc": auroc(s_, l_),
        "rows": sorted(ai_rows, key=lambda x: -x["probability"]),
        "not_estimated": [
            n for n, r in reports.items() if r["status"] == "ok" and r["scores"]["ai_generated_probability"] is None
        ],
    }

    # 6. Boilerplate isolation (Scenario 4)
    tmpl_files = manifest["boilerplate"]["files"]
    bp_rows = []
    for name in tmpl_files:
        r = reports[name]
        bp_rows.append(
            {
                "file": name,
                "template_words_excluded": r["score_details"]["boilerplate_words_excluded"],
                "peer_pct": r["scores"]["peer_similarity_percentage"],
                "risk": r["total_integrity_risk"],
            }
        )
    truth_pairs = {frozenset((x["source"], x["target"])) for x in manifest["relationships"]}
    tmpl_only = [p for p in itertools.combinations(tmpl_files, 2) if frozenset(p) not in truth_pairs]
    res["boilerplate"] = {
        "rows": bp_rows,
        "template_only_pairs": len(tmpl_only),
        "template_only_pairs_flagged": sum(scores.get(frozenset(p), 0) > 0 for p in tmpl_only),
    }

    # 7. Risk bands vs expectation
    involved = {x for r in manifest["relationships"] for x in (r["source"], r["target"])}
    risk_rows = []
    for name, meta in files.items():
        if meta["expected_status"] != "ok":
            expected = "UNASSESSED"
        elif name in involved:
            expected = "HIGH"
        elif meta["ai_generated"]:
            expected = "MEDIUM"
        else:
            expected = "LOW"
        actual = reports[name]["total_integrity_risk"]
        risk_rows.append({"file": name, "expected": expected, "actual": actual, "ok": expected == actual})
    res["risk"] = {"rows": risk_rows, "accuracy": round(sum(r["ok"] for r in risk_rows) / len(risk_rows), 3)}
    return res


def sentence_level(reports, manifest, docs) -> Dict[str, object]:
    """Sentence-level P/R: is each flagged sentence really copied/paraphrased?

    Ground truth comes from the passage variables named in the manifest; a
    document sentence is "copied" if it fuzzy-matches (partial ratio >= 85)
    one of the passages planted in that document. Template sentences are
    excluded from scoring and therefore from this metric.
    """
    import dataset_content as C
    from rapidfuzz import fuzz

    truth: Dict[str, List[str]] = {}
    for rel in manifest["relationships"]:
        for role in ("source", "target"):
            names = rel.get(f"{role}_passages", [])
            texts = [t for n in names for t in (getattr(C, n) if isinstance(getattr(C, n), list) else [getattr(C, n)])]
            if not rel[role].startswith("ref:"):
                truth.setdefault(rel[role], []).extend(texts)
    per_doc, TP, FP, FN = [], 0, 0, 0
    for d in docs:
        if d.doc_id not in reports or reports[d.doc_id]["status"] != "ok":
            continue
        r = reports[d.doc_id]
        flagged = {
            a["query_sentence"]
            for m in r["peer_matches"] + r["multimodal_matches"] + r["external_matches"]
            for a in m["sentence_alignment"]
        }
        tp = fp = fn = 0
        for s in d.scoring_sentences():
            if s.n_words < 6:
                continue
            actual = any(fuzz.partial_ratio(s.text.lower(), t.lower()) >= 85 for t in truth.get(d.doc_id, []))
            pred = s.text in flagged
            tp += pred and actual
            fp += pred and not actual
            fn += actual and not pred
        TP += tp
        FP += fp
        FN += fn
        if tp or fp or fn:
            per_doc.append({"file": d.doc_id, "tp": tp, "fp": fp, "fn": fn, **prf(tp, fp, fn)})
    return {"micro": {"tp": TP, "fp": FP, "fn": FN, **prf(TP, FP, FN)}, "per_doc": per_doc}


def lexical_baseline(reports, manifest) -> Dict[str, object]:
    """Classic shingle baseline: pairs sharing >= 3 word 5-grams are 'copies'.

    Also reports Jaccard / Levenshtein ratio for each true pair's evidence,
    showing why string matching misses paraphrase (Scenario 2).
    """
    from rapidfuzz import fuzz

    from integrity_engine.normalize import jaccard, words

    texts = {}
    for name, r in reports.items():
        if r["status"] == "ok":
            texts[name] = r["_text"]
    shingles = {
        n: {tuple(w[i : i + 5]) for i in range(len(w) - 4)} for n, w in ((n, words(t)) for n, t in texts.items())
    }
    truth = {frozenset((r["source"], r["target"])) for r in manifest["relationships"] if r["type"] in PEER_TYPES}
    tp = fp = fn = tn = 0
    for a, b in itertools.combinations(texts, 2):
        pred = len(shingles[a] & shingles[b]) >= 3
        act = frozenset((a, b)) in truth
        tp += pred and act
        fp += pred and not act
        fn += act and not pred
        tn += not pred and not act
    evidence = []
    for rel in manifest["relationships"]:
        if rel["type"] not in PEER_TYPES:
            continue
        tgt = reports[rel["target"]]
        for m in (tgt["peer_matches"] + tgt["multimodal_matches"])[:3]:
            if m["matched_file"] != rel["source"]:
                continue
            evidence.append(
                {
                    "type": rel["type"],
                    "semantic_chunk_cosine": m["similarity_score"],
                    "sentence_evidence": m["sentence_similarity"],
                    "evidence": m["evidence"],
                    "jaccard": round(jaccard(words(m["query_passage"]), words(m["matched_passage"])), 3),
                    "levenshtein_ratio": round(fuzz.ratio(m["query_passage"], m["matched_passage"]) / 100, 3),
                }
            )
    return {
        "shingle_baseline": {"tp": tp, "fp": fp, "fn": fn, "tn": tn, **prf(tp, fp, fn)},
        "evidence_lexical_vs_semantic": evidence,
    }


# ----------------------------------------------------------------- runner
def main() -> int:
    ap = argparse.ArgumentParser()
    # Relative defaults keep machine-specific paths out of the committed outputs.
    ap.add_argument("--input_dir", default=os.path.relpath(ROOT / "sample_submissions"))
    ap.add_argument("--out", default=os.path.relpath(ROOT / "evaluation"))
    ap.add_argument("--skip_ablations", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    from integrity_engine.config import Config, project_root
    from integrity_engine.pipeline import IntegrityPipeline
    from integrity_engine.verification import CrossEncoderVerifier

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(args.input_dir) / "dataset_manifest.json").read_text())

    cfg = Config()
    cfg.external.reference_dir = os.path.relpath(project_root() / cfg.external.reference_dir)
    base = IntegrityPipeline(cfg)
    t0 = time.time()
    docs = base.ingest(args.input_dir)
    t_ingest = time.time() - t0
    base_docs = copy.deepcopy(docs)
    reports_list = base.run(args.input_dir, str(out / "reports"), docs=docs)
    reports = {r["document_id"]: r for r in reports_list}
    for d in docs:
        if d.doc_id in reports:
            reports[d.doc_id]["_text"] = " ".join(s.text for s in d.scoring_sentences())

    results = {"main": evaluate(reports, manifest), "timing": {"ingest_seconds": round(t_ingest, 1)}}
    results["lexical"] = lexical_baseline(reports, manifest)
    results["sentence_level"] = sentence_level(reports, manifest, docs)

    # ASR quality on the read-out clip (reference = the script that was synthesised)
    import dataset_content as C

    asr = {}
    for name, ref in (
        ("pitch_10_sanjay.mp4", f"{C.PITCH_10_INTRO} {C.A_RECOMMENDATION_X} {C.PITCH_10_OUTRO}"),
        ("pitch_11_leela.mp4", C.PITCH_11),
    ):
        d = next(x for x in docs if x.doc_id == name)
        hyp = " ".join(b.text for b in d.blocks)
        asr[name] = {"wer": round(wer(ref, hyp), 3), "transcript": hyp}
    results["asr"] = asr

    # AI robustness probe: strip assistant phrases from an AI text and re-score.
    from integrity_engine.ai_detection import _MARKER_RE

    ai_doc = next(x for x in docs if x.doc_id == "sub_07_ishaan.docx")
    original = [s.text for s in ai_doc.scoring_sentences()]
    stripped = [re.sub(r"\s+", " ", _MARKER_RE.sub("", s)).strip(" ,") for s in original]
    probe = {}
    for label, sents in (("original", original), ("markers_removed", stripped)):
        sig = base.ai.signals(sents, label)
        probe[label] = {
            "probability": base.ai.combine(sig)["probability"],
            "marker_rate": round(sig["marker_rate"], 2),
            "perplexity": round(sig.get("perplexity", float("nan")), 1),
        }
    results["ai_robustness_probe"] = probe

    if not args.skip_ablations:

        def variant(label: str, **overrides) -> Dict[str, object]:
            c = copy.deepcopy(cfg)
            c.ai.enabled = False
            for dotted, value in overrides.items():
                section, key = dotted.split("__")
                setattr(getattr(c, section), key, value)
            p = IntegrityPipeline(c)
            p.embedder = base.embedder
            p.verifier = base.verifier if c.matching.cross_encoder_model else CrossEncoderVerifier(None)
            vdocs = copy.deepcopy(base_docs)
            reps = {r["document_id"]: r for r in p.run(args.input_dir, str(out / "ablation_tmp"), docs=vdocs)}
            m = evaluate(reps, manifest)
            sent = sentence_level(reps, manifest, vdocs)["micro"]
            return {
                "label": label,
                **{k: m["peer_pairs"][k] for k in ("tp", "fp", "fn", "precision", "recall", "f1")},
                "sentence": sent,
                "external_f1": m["external"]["f1"],
                "template_only_pairs_flagged": m["boilerplate"]["template_only_pairs_flagged"],
                "errors": m["peer_pairs"]["errors"],
            }

        abl = [
            variant("full system (default)"),
            variant("no boilerplate filter", boilerplate__enabled=False),
            variant("no cross-encoder (bi-encoder only)", matching__cross_encoder_model=None),
            variant("no anchor rule", matching__require_anchor=False),
            variant("no leave-one-out attribution", matching__loo_attribution=False),
            variant("3-sentence windows only", matching__window_sizes=[3]),
            variant(
                "v1 (3-sentence windows, no leave-one-out)", matching__loo_attribution=False, matching__window_sizes=[3]
            ),
            variant(
                "single-sentence evidence allowed",
                matching__min_passage_sentences=1,
                matching__single_sentence_threshold=0.80,
                matching__single_sentence_min_words=6,
            ),
        ]
        sweep_sent = [
            variant(f"direct threshold {t}", matching__sentence_threshold=t) for t in (0.70, 0.75, 0.80, 0.85, 0.90)
        ]
        sweep_ce = [
            variant(f"cross-encoder threshold {t}", matching__cross_encoder_threshold=t)
            for t in (0.50, 0.55, 0.60, 0.65, 0.70)
        ]
        results["ablations"] = abl
        results["sweep_sentence_threshold"] = sweep_sent
        results["sweep_cross_encoder_threshold"] = sweep_ce

    for r in reports.values():
        r.pop("_text", None)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    (out / "results.md").write_text(render(results))
    print((out / "results.md").read_text())
    return 0


def render(res: Dict[str, object]) -> str:
    m = res["main"]
    L = ["# Evaluation results (auto-generated by evaluate.py)", ""]
    L += ["## Ingestion", "", "| File | Expected | Actual | OK | Warning |", "|---|---|---|---|---|"]
    for r in m["ingestion"]["rows"]:
        L.append(f"| {r['file']} | {r['expected']} | {r['actual']} | {'yes' if r['ok'] else 'NO'} | {r['warning']} |")
    pp = m["peer_pairs"]
    L += [
        "",
        f"## Peer / cross-modal pair detection ({pp['pairs']} unordered pairs)",
        "",
        "| | Predicted match | Predicted no match |",
        "|---|---|---|",
        f"| **Actual match** | TP = {pp['tp']} | FN = {pp['fn']} |",
        f"| **Actual no match** | FP = {pp['fp']} | TN = {pp['tn']} |",
        "",
        f"Precision {pp['precision']:.2f} - Recall {pp['recall']:.2f} - F1 {pp['f1']:.2f}",
        "",
    ]
    L += ["| Relationship | Pair | Pair score (max coverage %) |", "|---|---|---|"]
    for t, v in m["peer_by_type"].items():
        L.append(f"| {t} | {v['pair'][0]} <-> {v['pair'][1]} | {v['pair_score']:.1f} |")
    L += [
        "",
        "### Localisation",
        "",
        "| Relationship | Target | Passages | Expected source page | Reported | Correct | Video timestamps |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in m["localisation"]:
        L.append(
            f"| {r['relationship']} | {r['target']} | {r['evidence_passages']} | {r['expected_source_page']} | "
            f"{r['reported_source_pages']} | {r['page_correct']} | {', '.join(r['video_timestamps'])} |"
        )
    sl = res.get("sentence_level")
    if sl:
        mi = sl["micro"]
        L += [
            "",
            "### Sentence-level evidence quality (micro-averaged over all analysed documents)",
            "",
            f"TP {mi['tp']} / FP {mi['fp']} / FN {mi['fn']} - Precision {mi['precision']:.2f}, "
            f"Recall {mi['recall']:.2f}, F1 {mi['f1']:.2f}",
            "",
            "| File | TP | FP | FN | Precision | Recall |",
            "|---|---|---|---|---|---|",
        ]
        for x in sl["per_doc"]:
            L.append(f"| {x['file']} | {x['tp']} | {x['fp']} | {x['fn']} | {x['precision']:.2f} | {x['recall']:.2f} |")
    e = m["external"]
    L += [
        "",
        "## External (web/reference) similarity",
        "",
        f"TP {e['tp']} / FP {e['fp']} / FN {e['fn']} / TN {e['tn']} - "
        f"Precision {e['precision']:.2f}, Recall {e['recall']:.2f}",
        "",
    ]
    for r in e["rows"]:
        L.append(f"- {r['file']}: {r['external_pct']}% (sources: {', '.join(r['sources'])}; truth={r['truth']})")
    a = m["ai"]
    L += [
        "",
        "## AI-generation indicator (threshold 50%)",
        "",
        f"TP {a['tp']} / FP {a['fp']} / FN {a['fn']} / TN {a['tn']} - Precision {a['precision']:.2f}, "
        f"Recall {a['recall']:.2f}, AUROC {a['auroc']}",
        "",
        "| File | AI % | Truth |",
        "|---|---|---|",
    ]
    for r in a["rows"]:
        L.append(f"| {r['file']} | {r['probability']:.1f} | {'AI' if r['truth'] else 'human'} |")
    L.append(f"\nNot estimated (too little text): {', '.join(a['not_estimated']) or '-'}")
    pr = res.get("ai_robustness_probe", {})
    if pr:
        L += [
            "",
            "### Robustness probe: assistant phrases removed from sub_07",
            "",
            "| Variant | AI % | Marker rate | Perplexity |",
            "|---|---|---|---|",
        ]
        for k, v in pr.items():
            L.append(f"| {k} | {v['probability']:.1f} | {v['marker_rate']} | {v['perplexity']} |")
    b = m["boilerplate"]
    L += [
        "",
        "## Template / boilerplate isolation (Scenario 4)",
        "",
        "| File | Template words excluded | Peer % | Risk |",
        "|---|---|---|---|",
    ]
    for r in b["rows"]:
        L.append(f"| {r['file']} | {r['template_words_excluded']} | {r['peer_pct']} | {r['risk']} |")
    L.append(
        f"\nTemplate-only pairs flagged as peer matches: "
        f"{b['template_only_pairs_flagged']} / {b['template_only_pairs']}"
    )
    r = m["risk"]
    L += ["", f"## Risk band agreement: {r['accuracy']:.0%}", "", "| File | Expected | Actual |", "|---|---|---|"]
    for x in r["rows"]:
        L.append(f"| {x['file']} | {x['expected']} | {x['actual']}{'' if x['ok'] else ' (mismatch)'} |")
    lex = res["lexical"]
    sb = lex["shingle_baseline"]
    L += [
        "",
        "## Lexical baseline vs semantic engine",
        "",
        f"Word 5-gram shingle baseline (>= 3 shared shingles): TP {sb['tp']} / FP {sb['fp']} / FN {sb['fn']} - "
        f"Precision {sb['precision']:.2f}, Recall {sb['recall']:.2f}",
        "",
        "| Type | Chunk cosine | Sentence evidence | Evidence | Jaccard | Levenshtein ratio |",
        "|---|---|---|---|---|---|",
    ]
    for x in lex["evidence_lexical_vs_semantic"]:
        L.append(
            f"| {x['type']} | {x['semantic_chunk_cosine']:.2f} | {x['sentence_evidence']:.2f} | {x['evidence']} | "
            f"{x['jaccard']:.2f} | {x['levenshtein_ratio']:.2f} |"
        )
    L += ["", "## ASR", "", "| Clip | WER | Transcript |", "|---|---|---|"]
    for k, v in res["asr"].items():
        L.append(f"| {k} | {v['wer']:.1%} | {v['transcript']} |")
    if "ablations" in res:
        L += [
            "",
            "## Ablations",
            "",
            "Pair level = document pairs (66); sentence level = flagged sentences vs planted passages.",
            "",
            "| Variant | Pair TP/FP/FN | Pair F1 | Sentence P | Sentence R | Sentence F1 | External F1 "
            "| Template pairs flagged |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for x in res["ablations"]:
            se = x["sentence"]
            L.append(
                f"| {x['label']} | {x['tp']}/{x['fp']}/{x['fn']} | {x['f1']:.2f} | {se['precision']:.2f} | "
                f"{se['recall']:.2f} | {se['f1']:.2f} | {x['external_f1']:.2f} | {x['template_only_pairs_flagged']} |"
            )
        for key, title in (
            ("sweep_sentence_threshold", "Direct-path sentence threshold sweep"),
            ("sweep_cross_encoder_threshold", "Cross-encoder threshold sweep"),
        ):
            L += [
                "",
                f"### {title}",
                "",
                "| Setting | Pair TP/FP/FN | Pair F1 | Sentence P | Sentence R | Sentence F1 | Pair errors |",
                "|---|---|---|---|---|---|---|",
            ]
            for x in res[key]:
                errs = (
                    "; ".join(f"{e['type']} {'/'.join(p.split('_')[1] for p in e['pair'])}" for e in x["errors"]) or "-"
                )
                se = x["sentence"]
                L.append(
                    f"| {x['label']} | {x['tp']}/{x['fp']}/{x['fn']} | {x['f1']:.2f} | {se['precision']:.2f} | "
                    f"{se['recall']:.2f} | {se['f1']:.2f} | {errs} |"
                )
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
