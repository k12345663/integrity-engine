#!/usr/bin/env python
"""Fetch public Wikipedia extracts into the local reference knowledge base.

The snapshot is committed (reference_corpus/wikipedia_snapshot.json) so runs
are reproducible offline. Content is CC BY-SA 4.0; each record keeps its URL.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from integrity_engine.reference_kb import fetch_wikipedia  # noqa: E402

TOPICS = [
    "Micromobility",
    "Scooter-sharing system",
    "Electric scooter",
    "Bicycle-sharing system",
    "Battery swapping",
    "Customer lifetime value",
    "Customer acquisition cost",
    "Network effect",
    "Unit economics",
    "Porter's five forces analysis",
    "SWOT analysis",
    "Tier 2 city",
    "Pune",
]
MAX_CHARS = 7000  # keep the snapshot small; intro + first sections


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "reference_corpus" / "wikipedia_snapshot.json"
    out.parent.mkdir(exist_ok=True)
    records = []
    today = dt.date.today().isoformat()
    for title in TOPICS:
        try:
            rec = fetch_wikipedia(title, timeout=15)
        except Exception as exc:
            print(f"! {title}: {exc}")
            continue
        if not rec:
            print(f"! {title}: not found")
            continue
        text = rec["text"]
        # Drop trailing reference/see-also sections and cut at a paragraph border.
        for marker in ("\n== See also", "\n== References", "\n== External links", "\n== Notes"):
            text = text.split(marker)[0]
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS].rsplit("\n", 1)[0]
        lines = [ln for ln in text.split("\n") if ln.strip() and not ln.strip().startswith("==")]
        rec["text"] = "\n".join(lines)
        rec["retrieved"] = today
        records.append(rec)
        print(f"+ {rec['title']}: {len(rec['text'])} chars")
    out.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(records)} records to {out}")


if __name__ == "__main__":
    main()
