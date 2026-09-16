"""
Citation Agent (RAG-style literature grounding for candidate limitations).

Per LimitGen (arXiv:2507.02694 Sec 4): "we prompt LLMs to query the Semantic
Scholar API to retrieve papers related to the one under review... to enrich
their domain understanding" -- and per the Multi-Agent paper's Citation Agent
(arXiv:2601.11578 Sec IV.D): collects Cited-In and Cited-By papers to check
whether a candidate limitation is confirmed, contradicted, or already
addressed elsewhere in the literature.

Deviation, documented: Semantic Scholar's unauthenticated API returns HTTP 429
almost immediately under real use (this exact failure mode is already logged
in litreview/REPORT.md, independently rediscovered as project-docs M-19 for a
different script) -- so this uses OpenAlex (keyless, no rate-limit issue in
practice) with the same "search -> read title/abstract -> judge groundedness"
role LimitGen assigns to the retrieval step. This mirrors, not reuses, the
substitution litreview/ already made for the same reason -- a fresh instance
of the same environment constraint, not copied code (no import from litreview/).

Deliberately independent of litreview/data/candidates_topK.json: every query
here is posed fresh, targeted at a specific candidate-limitation hypothesis,
not drawn from the SOTA assignment's retrieval pool.
"""
import re
import sys
import time
import xml.etree.ElementTree as ET

import requests

OPENALEX = "https://api.openalex.org"
HEADERS = {"User-Agent": "limitations-assignment-pipeline/1.0 (mailto:research-project@example.com)"}
ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_NS = {"a": "http://www.w3.org/2005/Atom"}


def arxiv_search(query, max_results=8):
    params = {"search_query": query, "start": 0, "max_results": max_results,
              "sortBy": "relevance", "sortOrder": "descending"}
    try:
        r = requests.get(ARXIV_API, params=params, timeout=20)
    except requests.RequestException:
        return []
    if r.status_code != 200:
        return []
    root = ET.fromstring(r.text)
    out = []
    for entry in root.findall("a:entry", ARXIV_NS):
        title = (entry.findtext("a:title", default="", namespaces=ARXIV_NS) or "").strip().replace("\n", " ")
        summary = (entry.findtext("a:summary", default="", namespaces=ARXIV_NS) or "").strip().replace("\n", " ")
        published = entry.findtext("a:published", default="", namespaces=ARXIV_NS) or ""
        year = int(published[:4]) if published[:4].isdigit() else None
        arxiv_id_full = entry.findtext("a:id", default="", namespaces=ARXIV_NS) or ""
        m = re.search(r"abs/([\w.\-]+?)(v\d+)?$", arxiv_id_full)
        out.append({
            "title": title, "year": year, "abstract": summary[:400],
            "cited_by_count": None, "doi": f"arXiv:{m.group(1)}" if m else arxiv_id_full,
        })
    return out


def search(query, per_page=8, date_from="2023-01-01"):
    params = {
        "search": query,
        "per-page": per_page,
        "sort": "relevance_score:desc",
        "filter": f"from_publication_date:{date_from},has_abstract:true",
    }
    for attempt in range(4):
        try:
            r = requests.get(f"{OPENALEX}/works", params=params, headers=HEADERS, timeout=20)
        except requests.RequestException:
            time.sleep(2)
            continue
        if r.status_code == 200:
            return r.json().get("results", [])
        time.sleep(2)
    return []


def reconstruct_abstract(inv_index):
    if not inv_index:
        return ""
    positions = {}
    max_pos = 0
    for word, idxs in inv_index.items():
        for i in idxs:
            positions[i] = word
            max_pos = max(max_pos, i)
    return " ".join(positions.get(i, "") for i in range(max_pos + 1))


QUERIES = {
    "A1_annealing_at_scale": [
        "attention mask annealing diffusion language model adaptation scale",
        "causal to bidirectional attention transition diffusion language model",
    ],
    "A8_adaptation_generalization": [
        "adapting autoregressive language model diffusion language model architecture",
        "continual pretraining diffusion language model beyond LLaMA",
    ],
    "A6_instruction_tuning_diffusion": [
        "instruction tuning diffusion language model",
        "chain of thought reasoning diffusion language model",
    ],
    "A4_diffusion_inference_efficiency": [
        "diffusion language model inference efficiency KV cache",
        "diffusion language model decoding speed FLOPs comparison autoregressive",
    ],
}

ARXIV_QUERIES = {
    "A1_annealing_at_scale": [
        'all:"attention mask annealing" AND all:"diffusion"',
        'all:"bidirectional attention" AND all:"diffusion language model" AND all:"7B"',
    ],
    "A8_adaptation_generalization": [
        'all:"adapting" AND all:"autoregressive" AND all:"diffusion language model"',
        'all:"continual pre-training" AND all:"diffusion language model"',
    ],
    "A6_instruction_tuning_diffusion": [
        'all:"instruction tuning" AND all:"diffusion language model"',
        'all:"chain-of-thought" AND all:"diffusion language model"',
    ],
    "A4_diffusion_inference_efficiency": [
        'all:"diffusion language model" AND all:"KV cache"',
        'all:"diffusion language model" AND all:"inference efficiency"',
    ],
}


def main():
    out = []
    for label, queries in QUERIES.items():
        print(f"\n=== {label} (OpenAlex) ===", file=sys.stderr)
        seen = {}
        for q in queries:
            results = search(q)
            print(f"  query={q!r} -> {len(results)} results", file=sys.stderr)
            for w in results:
                seen[w["id"]] = w
            time.sleep(0.3)
        entries = []
        for w in list(seen.values())[:10]:
            entries.append({
                "title": w.get("title") or w.get("display_name") or "",
                "year": w.get("publication_year"),
                "abstract": reconstruct_abstract(w.get("abstract_inverted_index"))[:400],
                "cited_by_count": w.get("cited_by_count", 0),
                "doi": w.get("doi"),
            })

        print(f"=== {label} (arXiv) ===", file=sys.stderr)
        seen_arxiv = {}
        for q in ARXIV_QUERIES.get(label, []):
            results = arxiv_search(q)
            print(f"  query={q!r} -> {len(results)} results", file=sys.stderr)
            for w in results:
                seen_arxiv[w["title"]] = w
            time.sleep(1.0)
        entries.extend(list(seen_arxiv.values())[:10])

        out.append({"label": label, "candidates": entries})
        for e in entries:
            print(f"    [{e['year']}] {e['title'][:80]} ({e['doi']})", file=sys.stderr)

    with open("../data/citation_agent_results.txt", "w", encoding="utf-8") as f:
        for group in out:
            f.write(f"\n=== {group['label']} ===\n")
            for e in group["candidates"]:
                f.write(f"[{e['year']}] {e['title']}\n")
                f.write(f"  cited_by={e['cited_by_count']} doi={e['doi']}\n")
                f.write(f"  abstract: {e['abstract']}\n\n")
    print("\nWrote ../data/citation_agent_results.txt", file=sys.stderr)


if __name__ == "__main__":
    main()
