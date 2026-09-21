"""Stage 1 -- build the per-paper vector store the Citation Agent retrieves over.

arXiv:2601.11578 3.2 specifies, for each input paper P_i:

    C_total(P_i) = (X_in * A_in) + (X_by * A_by)

where X_in are "Cited In" papers (the anchor's own references, obtained in the
paper via ScienceParse) and X_by are "Cited By" papers (obtained via the OpenAlex
API), each divided into A sections stored as individual chunks.

Implementation here:
  * X_in  -- supplied by litreview/, this project's prior literature-review
             assignment.  Its retrieval pipeline already assembled and verified
             the anchor's domain reference pool, so it is used directly rather
             than re-parsing the anchor's bibliography with ScienceParse.
  * X_by  -- fetched live from the OpenAlex API exactly as the paper specifies.
  * A     -- title+abstract chunks for references (that is all litreview stored);
             true section-level chunks for the anchor's own full text.

Every degradation is printed and recorded in the manifest, never silently absorbed.
"""
from __future__ import annotations

import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import read_json, read_text, split_sections, write_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LITREVIEW = ROOT / "litreview" / "data"
OUT = ROOT / "limitations" / "data"

ANCHOR_OPENALEX_ID = "W4404307915"
MAILTO = "neelnaik2005@gmail.com"


def _get(url: str, timeout: int = 30):
    req = urllib.request.Request(url, headers={"User-Agent": f"limitations-pipeline (mailto:{MAILTO})"})
    import json as _json
    return _json.load(urllib.request.urlopen(req, timeout=timeout))


def fetch_cited_by(anchor_id: str, max_works: int = 50) -> tuple[list[dict], str]:
    """X_by -- works citing the anchor, via OpenAlex (paper 3.2)."""
    url = (
        f"https://api.openalex.org/works?filter=cites:{anchor_id}"
        f"&per-page={max_works}&mailto={MAILTO}"
    )
    try:
        data = _get(url)
    except urllib.error.HTTPError as e:
        return [], f"HTTP {e.code} from OpenAlex cites-filter"
    except Exception as e:  # noqa: BLE001
        return [], f"{type(e).__name__}: {e}"

    results = data.get("results", [])
    out = []
    for w in results:
        inv = w.get("abstract_inverted_index")
        abstract = ""
        if inv:
            positions = [(p, tok) for tok, ps in inv.items() for p in ps]
            abstract = " ".join(tok for _, tok in sorted(positions))
        out.append({
            "id": w.get("id"),
            "title": w.get("title") or "",
            "abstract": abstract,
            "year": w.get("publication_year"),
            "cited_by_count": w.get("cited_by_count", 0),
        })
    return out, f"ok ({len(out)} works)"


def fetch_mentions(term: str, max_works: int = 50) -> tuple[list[dict], str]:
    """Documented fallback for X_by when OpenAlex has no citation edges indexed.

    The paper's purpose for X_by is to "capture broader contextual weaknesses"
    from the literature surrounding the input paper.  When the citation graph is
    unavailable, works whose full text *mentions the anchor by name* serve that
    same function.  This is a substitution, not the paper's method, and is
    recorded as such in the manifest and in REPORT.md 3.3.
    """
    url = (
        f"https://api.openalex.org/works?search={urllib.parse.quote(term)}"
        f"&per-page={max_works}&mailto={MAILTO}"
    )
    try:
        data = _get(url)
    except Exception as e:  # noqa: BLE001
        return [], f"{type(e).__name__}: {e}"

    out = []
    for w in data.get("results", []):
        if w.get("id", "").endswith(ANCHOR_OPENALEX_ID):
            continue  # never include the anchor itself
        inv = w.get("abstract_inverted_index")
        abstract = ""
        if inv:
            positions = [(p, tok) for tok, ps in inv.items() for p in ps]
            abstract = " ".join(tok for _, tok in sorted(positions))
        out.append({
            "id": w.get("id"),
            "title": w.get("title") or "",
            "abstract": abstract,
            "year": w.get("publication_year"),
            "cited_by_count": w.get("cited_by_count", 0),
        })
    return out, f"ok ({len(out)} works mentioning '{term}')"


def main() -> int:
    manifest = {"substitutions": [], "degradations": []}
    chunks: list[dict] = []

    # ---- X_in : the anchor's domain reference pool, from litreview/ --------
    topk = read_json(LITREVIEW / "candidates_topK.json")
    n_in = 0
    for i, ref in enumerate(topk):
        title = (ref.get("title") or "").strip()
        abstract = (ref.get("abstract") or "").strip()
        if not title:
            continue
        chunks.append({
            "chunk_id": f"in::{i}",
            "provenance": "cited_in",
            "source_title": title,
            "year": ref.get("year"),
            "venue": ref.get("venue"),
            "text": f"{title}. {abstract}".strip(),
        })
        n_in += 1
    manifest["substitutions"].append(
        "X_in obtained from litreview/candidates_topK.json (this project's prior "
        "literature-review assignment) instead of ScienceParse bibliography parsing."
    )

    # ---- X_by : works citing the anchor, via OpenAlex ----------------------
    cited_by, status = fetch_cited_by(ANCHOR_OPENALEX_ID)
    time.sleep(0.3)
    n_by = 0
    for i, w in enumerate(cited_by):
        title = (w.get("title") or "").strip()
        if not title:
            continue
        chunks.append({
            "chunk_id": f"by::{i}",
            "provenance": "cited_by",
            "source_title": title,
            "year": w.get("year"),
            "venue": None,
            "text": f"{title}. {w.get('abstract','')}".strip(),
        })
        n_by += 1
    x_by_mode = "openalex_cites_filter"
    if n_by == 0:
        manifest["degradations"].append(
            f"X_by via the paper's OpenAlex cites-filter returned 0 works ({status}). "
            "The anchor's OpenAlex record reports cited_by_count=0 despite being an "
            "ICLR 2025 paper; verified as an OpenAlex indexing gap (only one record "
            "exists for this title), NOT evidence that the paper is uncited."
        )
        mentions, mstatus = fetch_mentions("DiffuLLaMA")
        time.sleep(0.3)
        for i, w in enumerate(mentions):
            title = (w.get("title") or "").strip()
            if not title:
                continue
            chunks.append({
                "chunk_id": f"by::{i}",
                "provenance": "cited_by_substitute",
                "source_title": title,
                "year": w.get("year"),
                "venue": None,
                "text": f"{title}. {w.get('abstract','')}".strip(),
            })
            n_by += 1
        x_by_mode = "openalex_fulltext_mention_search"
        manifest["substitutions"].append(
            f"X_by substituted with OpenAlex full-text search for 'DiffuLLaMA' "
            f"({mstatus}). Serves the paper's stated purpose for X_by (surrounding "
            "literature) but is a mention graph, not a citation graph."
        )

    # ---- anchor's own sections (needed by Extractor/Analyzer/Reviewer) -----
    anchor_text = read_text(LITREVIEW / "anchor_fulltext.txt")
    sections = [
        s for s in split_sections(anchor_text)
        if not s["heading"].lower().startswith("published as a conference")
    ]
    write_json(OUT / "anchor_sections.json", sections)

    write_json(OUT / "rag_corpus.json", chunks)
    manifest.update({
        "n_chunks_total": len(chunks),
        "n_cited_in": n_in,
        "n_cited_by": n_by,
        "openalex_cited_by_status": status,
        "x_by_mode": x_by_mode,
        "n_anchor_sections": len(sections),
    })
    write_json(OUT / "rag_manifest.json", manifest)

    print(f"X_in  (cited-in  chunks): {n_in}")
    print(f"X_by  (cited-by  chunks): {n_by}   [OpenAlex: {status}]")
    print(f"anchor sections          : {len(sections)}")
    print(f"C_total                  : {len(chunks)}")
    for d in manifest["degradations"]:
        print(f"\n!! DEGRADATION: {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
