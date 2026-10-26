"""External reference corpus ("simulated web").

The default source is a local knowledge base of public Wikipedia extracts
(``reference_corpus/*.json``, CC BY-SA 4.0, fetched by
``scripts/fetch_reference_kb.py``). With ``external.online=true`` the engine
also queries the free Wikipedia search API using each submission's most
distinctive sentences; any network failure falls back to the local KB.
"""

from __future__ import annotations

import hashlib
import json
import logging
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .config import ExternalConfig
from .models import Document, TextBlock
from .normalize import build_sentences, clean_text

log = logging.getLogger(__name__)

WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "integrity-engine/1.0 (academic prototype)"


def _doc_from_record(rec: Dict[str, str]) -> Document:
    doc = Document(doc_id=f"ref:{rec['title']}", path=rec.get("url", ""), modality="reference")
    doc.meta.update({k: rec.get(k) for k in ("title", "url", "license", "retrieved")})
    paragraphs = [p for p in rec["text"].split("\n") if p.strip()]
    for i, para in enumerate(paragraphs, start=1):
        doc.blocks.append(TextBlock(text=clean_text(para), paragraph=i))
    build_sentences(doc)
    return doc


def load_local_kb(reference_dir: str) -> List[Document]:
    root = Path(reference_dir)
    docs = []
    if not root.is_dir():
        log.warning("Reference corpus %s not found; external similarity will be 0", reference_dir)
        return docs
    for path in sorted(root.glob("*.json")):
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
            for rec in records if isinstance(records, list) else [records]:
                docs.append(_doc_from_record(rec))
        except Exception as exc:
            log.warning("Skipping malformed reference file %s: %s", path.name, exc)
    log.info("Loaded %d reference documents from %s", len(docs), reference_dir)
    return docs


def _get_json(params: Dict[str, str], timeout: float) -> dict:
    url = f"{WIKI_API}?{urllib.parse.urlencode({**params, 'format': 'json'})}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_wikipedia(title: str, timeout: float = 8.0, intro_only: bool = False) -> Optional[Dict[str, str]]:
    params = {
        "action": "query",
        "prop": "extracts|info",
        "explaintext": "1",
        "inprop": "url",
        "titles": title,
        "redirects": "1",
    }
    if intro_only:
        params["exintro"] = "1"
    data = _get_json(params, timeout)
    for page in data.get("query", {}).get("pages", {}).values():
        if page.get("extract"):
            return {
                "title": page["title"],
                "url": page.get("fullurl", f"https://en.wikipedia.org/wiki/{page['title'].replace(' ', '_')}"),
                "license": "CC BY-SA 4.0",
                "text": page["extract"],
            }
    return None


def search_online(queries: Iterable[str], cfg: ExternalConfig, known_titles: set) -> List[Document]:
    """Search Wikipedia for each query; return newly found reference docs."""
    cache = Path(cfg.online_cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    found: List[Document] = []
    for q in queries:
        key = cache / (hashlib.sha1(q.encode()).hexdigest() + ".json")
        try:
            if key.exists():
                records = json.loads(key.read_text())
            else:
                hits = (
                    _get_json({"action": "query", "list": "search", "srsearch": q, "srlimit": "2"}, cfg.online_timeout)
                    .get("query", {})
                    .get("search", [])
                )
                records = [r for r in (fetch_wikipedia(h["title"], cfg.online_timeout) for h in hits) if r]
                key.write_text(json.dumps(records))
        except Exception as exc:  # network down, rate limited, ...
            log.warning("Online reference search failed (%s); using local KB only", exc)
            return found
        for rec in records:
            if rec["title"] not in known_titles:
                known_titles.add(rec["title"])
                found.append(_doc_from_record(rec))
    return found


def distinctive_queries(doc: Document, n: int) -> List[str]:
    """Longest scoring sentences make the most specific search queries."""
    sents = sorted(doc.scoring_sentences(), key=lambda s: -s.n_words)[:n]
    return [" ".join(s.text.split()[:14]) for s in sents]
