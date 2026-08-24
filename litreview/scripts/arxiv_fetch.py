"""Supplement OpenAlex retrieval with arXiv (covers very recent preprints OpenAlex
hasn't indexed yet -- important since the assignment asks for recent coverage)."""
import re
import sys
import time
import xml.etree.ElementTree as ET

import requests

ARXIV_API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom"}

QUERIES = [
    'all:"diffusion language model"',
    'all:"discrete diffusion" AND all:"text generation"',
    'all:"masked diffusion" AND all:"language model"',
    'all:"non-autoregressive" AND all:"text generation"',
    'all:"semi-autoregressive" AND all:"generation"',
    'all:"continual pre-training" AND all:"language model"',
    'all:"adapting" AND all:"autoregressive" AND all:"diffusion"',
]


def fetch(query, max_results=25):
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    r = requests.get(ARXIV_API, params=params, timeout=20)
    if r.status_code != 200:
        return []
    root = ET.fromstring(r.text)
    out = []
    for entry in root.findall("a:entry", NS):
        title = (entry.findtext("a:title", default="", namespaces=NS) or "").strip().replace("\n", " ")
        summary = (entry.findtext("a:summary", default="", namespaces=NS) or "").strip().replace("\n", " ")
        arxiv_id_full = entry.findtext("a:id", default="", namespaces=NS) or ""
        m = re.search(r"abs/([\w.\-]+?)(v\d+)?$", arxiv_id_full)
        arxiv_id = m.group(1) if m else arxiv_id_full
        published = entry.findtext("a:published", default="", namespaces=NS) or ""
        year = int(published[:4]) if published[:4].isdigit() else None
        authors = [a.findtext("a:name", default="", namespaces=NS) for a in entry.findall("a:author", NS)]
        out.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "abstract": summary,
            "year": year,
            "authors": authors,
            "doi": f"https://doi.org/10.48550/arxiv.{arxiv_id}",
        })
    return out


def main():
    seen = {}
    for q in QUERIES:
        results = fetch(q)
        print(f"  query={q!r} -> {len(results)} results", file=sys.stderr)
        for r in results:
            seen[r["arxiv_id"]] = r
        time.sleep(1.0)  # be polite to arXiv
    print(f"Total unique arXiv candidates: {len(seen)}", file=sys.stderr)
    from common import save_json
    save_json("../data/arxiv_candidates.json", list(seen.values()))


if __name__ == "__main__":
    main()
