"""Stage 1 -- assemble ResearchAgent's literature input (paper sec 3.2, Table 6).

  target paper   DiffuLLaMA: title + abstract, from litreview's extracted full text
  related papers the paper uses "studies that have cited the target paper", narrowed
                 "based on their similarities of abstracts with the core paper".
                 OpenAlex indexes zero citing works for DiffuLLaMA (verified in the
                 limitations assignment), so the citing-side set is the 46 works that
                 mention DiffuLLaMA by name, retrieved there via OpenAlex search.
                 They are ranked by TF-IDF abstract similarity; the top n are used.
  entity corpus  every abstract available: target + citing-side works + litreview's
                 48 references. Used only to build the knowledge store (stage 2).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (DATA, IDEATION, ROOT, TfidfCosine, clean_pdf_text,  # noqa: E402
                    read_json, read_text, tokenize, write_json)


def _key(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", title.lower())


def dedupe(papers: list[dict]) -> list[dict]:
    """Drop repeat titles. OpenAlex holds duplicate records for some works (e.g. two
    for 'dLLM: Simple Diffusion Language Modeling'), and a few works appear in both
    source pools. The first occurrence, which carries the richer source, is kept."""
    seen, out = set(), []
    for p in papers:
        k = _key(p["title"])
        if k and k not in seen:
            seen.add(k)
            out.append(p)
    return out


def target_paper() -> dict:
    meta = read_json(ROOT / "litreview/data/anchor_related_work_groundtruth.json")["anchor_paper"]
    full = read_text(ROOT / "litreview/data/anchor_fulltext.txt")
    m = re.search(r"A\s?BSTRACT\s*\n(.*?)\n\s*1\s+I\s?NTRODUCTION", full, re.S)
    if not m:
        raise SystemExit("could not locate the abstract in anchor_fulltext.txt")
    return {"title": meta["title"], "abstract": clean_pdf_text(m.group(1)),
            "arxiv_id": meta.get("arxiv_id"), "venue": meta.get("venue")}


def citing_side() -> list[dict]:
    out = []
    for c in read_json(ROOT / "limitations/data/rag_corpus.json"):
        if c["provenance"] == "cited_in":
            continue
        title = c["source_title"].strip()
        abstract = c["text"][len(title):].lstrip(". ").strip()
        if len(abstract) < 80:          # no usable abstract
            continue
        out.append({"title": title, "abstract": abstract, "year": c.get("year"),
                    "source": "openalex_mention_search"})
    return out


def references() -> list[dict]:
    return [{"title": r["title"], "abstract": r.get("abstract") or "", "year": r.get("year"),
             "source": "litreview_top48"}
            for r in read_json(ROOT / "litreview/data/candidates_topK.json")]


def main() -> int:
    cfg = read_json(IDEATION / "config.json")
    target = target_paper()
    citing = dedupe(citing_side())

    model = TfidfCosine([tokenize(p["title"] + " " + p["abstract"]) for p in citing])
    sims = model.scores(tokenize(target["title"] + " " + target["abstract"]))
    order = sorted(range(len(citing)), key=lambda i: -sims[i])
    related = [{**citing[i], "similarity": round(sims[i], 4)} for i in order[:cfg["n_related_papers"]]]

    corpus = [{"title": target["title"], "abstract": target["abstract"], "source": "target"}]
    corpus = dedupe(corpus + citing + references())

    write_json(DATA / "literature.json", {"target": target, "related": related,
                                          "entity_corpus": corpus})
    print(f"target        : {target['title']}")
    print(f"abstract      : {len(target['abstract'])} chars")
    print(f"citing-side   : {len(citing)} with abstracts -> top {len(related)} by similarity")
    for r in related:
        print(f"   {r['similarity']:.3f}  ({r['year']}) {r['title'][:78]}")
    print(f"entity corpus : {len(corpus)} papers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
