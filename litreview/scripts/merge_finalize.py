"""Merge OpenAlex-sourced candidates with arXiv-sourced candidates (for recency),
apply the same relevance gate + quality enrichment + rank-averaging reranking,
and produce the final top-K reference set."""
import re
import sys
import time

from common import OPENALEX, get, tfidf_vectors, cosine, percentile_rank, save_json, load_json, score_pool
from rerank import relevance_score, RELEVANCE_GATE, norm_title

FINAL_K = 35  # legacy flat cut, kept for reference; real selection uses the tiered split below
FOUNDATIONAL_K = 36  # pre-2025 papers ranked among themselves
RECENT_K = 12  # 2025-2026 papers ranked among themselves
STRATIFY_YEAR = 2025


def enrich_from_openalex(doi):
    data = get(f"{OPENALEX}/works/doi:{doi.split('doi.org/')[-1]}")
    if not data:
        return None
    return data


def main():
    oa = load_json("../data/candidates_all.json")
    arxiv = load_json("../data/arxiv_candidates.json")

    # score arXiv candidates first (cheap), only enrich the ones that pass the gate
    pool = []
    for w in oa:
        w["source"] = "openalex"
        pool.append(w)

    print(f"Scoring {len(arxiv)} arXiv candidates for relevance...", file=sys.stderr)
    kept_arxiv = []
    for a in arxiv:
        score = relevance_score(a["title"], a["abstract"])
        if score >= RELEVANCE_GATE:
            a["llm_relevance_precomputed"] = score
            kept_arxiv.append(a)
    print(f"  {len(kept_arxiv)}/{len(arxiv)} arXiv candidates pass relevance gate", file=sys.stderr)

    print("Enriching relevant arXiv candidates with OpenAlex citation/author/venue data...", file=sys.stderr)
    for i, a in enumerate(kept_arxiv):
        data = enrich_from_openalex(a["doi"])
        if data:
            authorships = data.get("authorships") or []
            first_author = authorships[0]["author"] if authorships else {}
            source = (data.get("primary_location") or {}).get("source") or {}
            a["id"] = data["id"]
            a["cited_by_count"] = data.get("cited_by_count", 0)
            a["first_author_id"] = first_author.get("id")
            a["venue"] = source.get("display_name") or "arXiv preprint"
            a["venue_id"] = source.get("id")
            a["referenced_works"] = data.get("referenced_works") or []
        else:
            a["id"] = a["doi"]
            a["cited_by_count"] = 0
            a["first_author_id"] = None
            a["venue"] = "arXiv preprint"
            a["venue_id"] = None
            a["referenced_works"] = []
        a["source"] = "arxiv"
        a["subarea"] = "arxiv-recency-supplement"
        a["author_hindex"] = 0  # filled below if resolvable
        pool.append(a)
        if (i + 1) % 10 == 0:
            print(f"  enriched {i+1}/{len(kept_arxiv)}", file=sys.stderr)
        time.sleep(0.15)

    # fetch author h-index / venue reputation for arXiv-sourced items that resolved an author id
    from retrieve import authors_hindex, sources_reputation
    need_auth = [w["first_author_id"] for w in pool if w.get("source") == "arxiv" and w.get("first_author_id")]
    need_venue = [w["venue_id"] for w in pool if w.get("source") == "arxiv" and w.get("venue_id")]
    hmap = authors_hindex(need_auth) if need_auth else {}
    vmap = sources_reputation(need_venue) if need_venue else {}
    for w in pool:
        if w.get("source") == "arxiv":
            w["author_hindex"] = hmap.get(w.get("first_author_id"), 0)
            w["venue_hindex"] = vmap.get(w.get("venue_id"), 0)

    # score relevance uniformly (openalex ones weren't scored yet in this script)
    for w in pool:
        if "llm_relevance_precomputed" in w:
            w["llm_relevance"] = w["llm_relevance_precomputed"]
        else:
            w["llm_relevance"] = relevance_score(w["title"], w["abstract"])

    # dedupe by normalized title, prefer higher citation count, but keep the most
    # complete metadata (OpenAlex record) when tied
    best = {}
    for w in pool:
        key = norm_title(w["title"])
        if not key:
            continue
        if key not in best or w.get("cited_by_count", 0) > best[key].get("cited_by_count", 0):
            best[key] = w
    merged = list(best.values())
    print(f"Merged pool after dedup: {len(merged)}", file=sys.stderr)

    gated = [w for w in merged if w["llm_relevance"] >= RELEVANCE_GATE]
    print(f"After relevance gate (>= {RELEVANCE_GATE}): {len(gated)}", file=sys.stderr)

    foundational = score_pool([w for w in gated if (w.get("year") or 0) < STRATIFY_YEAR])
    recent = score_pool([w for w in gated if (w.get("year") or 0) >= STRATIFY_YEAR])
    print(f"Foundational (pre-{STRATIFY_YEAR}) pool: {len(foundational)}; "
          f"Recent ({STRATIFY_YEAR}+) pool: {len(recent)}", file=sys.stderr)

    all_scored = foundational + recent
    all_scored.sort(key=lambda w: w["final_score"], reverse=True)
    save_json("../data/candidates_merged_reranked.json", all_scored)

    top_k = foundational[:FOUNDATIONAL_K] + recent[:RECENT_K]
    top_k.sort(key=lambda w: w["final_score"], reverse=True)
    save_json("../data/candidates_topK.json", top_k)
    print(f"\nFinal stratified selection: {len(foundational[:FOUNDATIONAL_K])} foundational + "
          f"{len(recent[:RECENT_K])} recent = {len(top_k)} total", file=sys.stderr)
    for w in top_k:
        src = w.get("source", "?")
        print(f"  [{w['final_score']:.3f} rel={w['llm_relevance']:.1f} {src}] {w['title'][:70]} ({w['year']}) cit={w.get('cited_by_count',0)}", file=sys.stderr)


if __name__ == "__main__":
    main()
