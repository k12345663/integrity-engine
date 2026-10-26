#!/usr/bin/env python
"""CLI entry point.

python run_integrity_check.py --input_dir ./sample_submissions --output_dir ./reports
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Multimodal submission integrity & plagiarism checker")
    ap.add_argument("--input_dir", required=True, help="Folder of submissions (PDF, DOCX, MP4, ...)")
    ap.add_argument("--output_dir", default="./reports", help="Where JSON/Markdown reports are written")
    ap.add_argument("--config", default=None, help="Optional YAML config (see config/default.yaml)")
    ap.add_argument("--reference_dir", default=None, help="External reference corpus (default: reference_corpus/)")
    ap.add_argument("--template_dir", default=None, help="Folder of known prompt/template .txt files")
    ap.add_argument("--online", action="store_true", help="Also search Wikipedia's free API for external matches")
    ap.add_argument("--no_ai", action="store_true", help="Disable AI-content estimation")
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value, e.g. --set matching.sentence_threshold=0.8",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(out_dir / "pipeline.log", mode="w")],
    )
    for noisy in ("httpx", "urllib3", "sentence_transformers", "transformers", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    from integrity_engine.config import load_config, project_root
    from integrity_engine.pipeline import IntegrityPipeline

    cfg = load_config(args.config, args.set)
    ref_dir = Path(args.reference_dir or cfg.external.reference_dir)
    if not ref_dir.is_absolute() and not ref_dir.exists():
        ref_dir = project_root() / ref_dir  # default resolves relative to the repo
    cfg.external.reference_dir = str(ref_dir)
    if args.template_dir:
        cfg.boilerplate.template_dir = args.template_dir
    if args.online:
        cfg.external.online = True
    if args.no_ai:
        cfg.ai.enabled = False

    if not Path(args.input_dir).is_dir():
        logging.error("Input directory %s does not exist", args.input_dir)
        return 2
    reports = IntegrityPipeline(cfg).run(args.input_dir, str(out_dir))
    high = sum(r["total_integrity_risk"] == "HIGH" for r in reports)
    print(f"\n{len(reports)} files processed, {high} HIGH risk. Summary: {out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
